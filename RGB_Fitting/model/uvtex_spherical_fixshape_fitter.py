import torch
import numpy as np
import torch.nn.functional as F

from .optimizers import SphericalOptimizer
from .losses import perceptual_loss, photo_loss, vgg_loss, latents_geocross_loss
from utils.preprocess_utils import estimate_norm_torch
from utils.data_utils import tensor2np, draw_mask, draw_landmarks, img3channel


class Fitter:

    def __init__(self,
                 facemodel,
                 tex_gan,
                 renderer,
                 net_recog,
                 net_vgg,
                 logger,
                 input_data,
                 init_coeffs,
                 init_latents_z=None,
                 **kwargs):
        # parametric face model
        self.facemodel = facemodel
        # texture gan
        self.tex_gan = tex_gan
        # renderer
        self.renderer = renderer
        # the recognition model
        self.net_recog = net_recog.eval().requires_grad_(False)
        # the vgg model
        self.net_vgg = net_vgg.eval()

        # set fitting args
        self.w_feat = kwargs['w_feat'] if 'w_feat' in kwargs else 0.
        self.w_color = kwargs['w_color'] if 'w_color' in kwargs else 0.
        self.w_vgg = kwargs['w_vgg'] if 'w_vgg' in kwargs else 0.
        self.w_reg_latent = kwargs['w_reg_latent'] if 'w_reg_latent' in kwargs else 0.
        self.initial_lr = kwargs['initial_lr'] if 'initial_lr' in kwargs else 0.1
        self.lr_rampdown_length = kwargs['lr_rampdown_length'] if 'lr_rampdown_length' in kwargs else 0.25
        self.total_step = kwargs['total_step'] if 'total_step' in kwargs else 100
        self.print_freq = kwargs['print_freq'] if 'print_freq' in kwargs else 10
        self.visual_freq = kwargs['visual_freq'] if 'visual_freq' in kwargs else 10

        # input data for supervision
        self.input_img = input_data['img']
        self.skin_mask = input_data['skin_mask']
        self.parse_mask = input_data['parse_mask']
        self.gt_lm = input_data['lm']
        self.trans_m = input_data['M']
        with torch.no_grad():
            recog_output = self.net_recog(self.input_img, self.trans_m)
        self.input_img_feat = recog_output

        # init coeffs
        self.coeffs_opt = init_coeffs
        self.coeffs_opt.requires_grad = False  # fix shape
        self.coeffs_opt_dict = self.facemodel.split_coeff(self.coeffs_opt)

        # init latents
        if init_latents_z is not None:
            self.latents_z_opt = init_latents_z
            self.latents_z_opt.requires_grad = True
        else:
            self.latents_z_opt = self.tex_gan.get_init_z_latents()
        self.latents_w_opt = self.tex_gan.map_z_to_w(self.latents_z_opt)

        # optimization
        self.optimizer = SphericalOptimizer([self.latents_z_opt],
                                            torch.optim.Adam,
                                            betas=(0.9, 0.999),
                                            lr=self.initial_lr)
        self.initial_lr_list = [self.initial_lr]
        self.now_step = 0

        # logger
        self.logger = logger

    def update_learning_rate(self):
        t = float(self.now_step) / self.total_step
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        for i, param_group in enumerate(self.optimizer.opt.param_groups):
            lr = self.initial_lr_list[i] * lr_ramp
            param_group['lr'] = lr

    def forward(self):
        # forward face model
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.coeffs_opt_dict)
        # forward texture gan
        self.latents_w_opt = self.tex_gan.map_z_to_w(self.latents_z_opt)
        self.pred_uv_map = self.tex_gan.synth_uv_map(self.latents_w_opt)
        # render full head
        vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.latents_z_opt.size()[0], 1, 1)
        render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
        self.render_head_mask, _, self.render_head = \
            self.renderer(self.pred_vertex, self.facemodel.head_buf, feat=render_feat, uv_map=self.pred_uv_map)
        # render front face
        self.render_face_mask, _, self.render_face = \
            self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)

    def compute_losses(self):
        # initial loss
        self.loss_names = ['all']
        self.loss_all = 0.
        # inset front face with input image
        render_face_mask = self.render_face_mask.detach()
        render_face = self.render_face * render_face_mask + (1 - render_face_mask) * self.input_img
        # id feature loss
        if self.w_feat > 0:
            assert self.net_recog.training == False
            if self.pred_lm.shape[1] == 68:
                pred_trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])
            else:
                pred_trans_m = self.trans_m
            pred_feat = self.net_recog(render_face, pred_trans_m)
            self.loss_feat = perceptual_loss(pred_feat, self.input_img_feat)
            self.loss_all += self.w_feat * self.loss_feat
            self.loss_names.append('feat')
        # color loss
        if self.w_color > 0:
            loss_face_mask = render_face_mask * self.parse_mask * self.skin_mask
            self.loss_color = photo_loss(render_face, self.input_img, loss_face_mask)
            self.loss_all += self.w_color * self.loss_color
            self.loss_names.append('color')
        # vgg loss, using the same render_face(face_mask) with color loss
        if self.w_vgg > 0:
            loss_face_mask = render_face_mask * self.parse_mask
            render_face_vgg = render_face * loss_face_mask
            input_face_vgg = self.input_img * loss_face_mask
            self.loss_vgg = vgg_loss(render_face_vgg, input_face_vgg, self.net_vgg)
            self.loss_all += self.w_vgg * self.loss_vgg
            self.loss_names.append('vgg')
        # w latent geocross regression
        if self.w_reg_latent > 0:
            self.loss_reg_latent = latents_geocross_loss(self.latents_w_opt)
            self.loss_all += self.w_reg_latent * self.loss_reg_latent
            self.loss_names.append('reg_latent')

    def optimize_parameters(self):
        self.update_learning_rate()
        self.forward()
        self.compute_losses()
        self.optimizer.opt.zero_grad()
        self.loss_all.backward()
        self.optimizer.step()
        self.now_step += 1

    def gather_visual_img(self):
        # input data
        input_img = tensor2np(self.input_img[:1, :, :, :])
        skin_img = img3channel(tensor2np(self.skin_mask[:1, :, :, :]))
        parse_mask = tensor2np(self.parse_mask[:1, :, :, :], dst_range=1.0)
        gt_lm = self.gt_lm[0, :, :].detach().cpu().numpy()
        # predict data
        pre_uv_img = tensor2np(F.interpolate(self.pred_uv_map, size=input_img.shape[:2], mode='area')[:1, :, :, :])
        pred_face_img = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * self.input_img
        pred_face_img = tensor2np(pred_face_img[:1, :, :, :])
        pred_head_img = tensor2np(self.render_head[:1, :, :, :])
        pred_lm = self.pred_lm[0, :, :].detach().cpu().numpy()
        # draw mask and landmarks
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm[..., 1] = pred_face_img.shape[0] - 1 - gt_lm[..., 1]
        pred_lm[..., 1] = pred_face_img.shape[0] - 1 - pred_lm[..., 1]
        lm_img = draw_landmarks(pred_face_img, gt_lm, color='b')
        lm_img = draw_landmarks(lm_img, pred_lm, color='r')
        # combine visual images
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img, pred_face_img, pred_head_img, pre_uv_img],
                                     axis=1)
        return combine_img

    def gather_loss_log_str(self):
        loss_log = {}
        loss_str = ''
        for name in self.loss_names:
            loss_value = float(getattr(self, 'loss_' + name))
            loss_log[f'loss/{name}'] = loss_value
            loss_str += f'[loss/{name}: {loss_value:.5f}]'
        return loss_log, loss_str

    def iterate(self):
        for _ in range(self.total_step):
            # optimize
            self.optimize_parameters()
            # print log
            if self.now_step % self.print_freq == 0 or self.now_step == self.total_step:
                loss_log, loss_str = self.gather_loss_log_str()
                now_lr = self.optimizer.opt.param_groups[0]['lr']
                self.logger.write_tb_scalar(['lr'], [now_lr], self.now_step)
                self.logger.write_tb_scalar(loss_log.keys(), loss_log.values(), self.now_step)
                self.logger.write_txt_log(f'[step {self.now_step}/{self.total_step}] [lr:{now_lr:.7f}] {loss_str}')
            # save intermediate visual results
            if self.now_step % self.visual_freq == 0 or self.now_step == self.total_step:
                vis_img = self.gather_visual_img()
                self.logger.write_tb_images([vis_img], ['vis'], self.now_step)

        final_coeffs = self.coeffs_opt.detach().clone()
        final_latents_z = self.latents_z_opt.detach().clone()
        final_latents_w = self.latents_w_opt.detach().clone()
        return final_coeffs, final_latents_z, final_latents_w
