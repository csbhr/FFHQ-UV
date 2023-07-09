import os
import torch
import numpy as np
import torch.nn.functional as F

from .hifi3dpp import ParametricFaceModel
from .renderer_nvdiffrast import MeshRenderer
from . import uvtex_spherical_fixshape_fitter, uvtex_wspace_shape_joint_fitter
from network import texgan_cropface630resize1024
from network.recog import define_net_recog
from network.recon_deep3d import define_net_recon_deep3d
from network.stylegan2 import dnnlib
from utils.data_utils import setup_seed, tensor2np, np2tensor, draw_mask, draw_landmarks, img3channel, read_img
from utils.mesh_utils import unwrap_vertex_to_uv, blend_uv_with_template, write_mesh_obj, write_mtl


class FitModel:

    def __init__(self, cpk_dir, topo_dir, texgan_model_name, loose_tex=False, lm86=False, device='cuda'):
        self.args_model = {
            # face model and renderer
            'fm_model_file': os.path.join(topo_dir, 'hifi3dpp_model_info.mat'),
            'unwrap_info_file': os.path.join(topo_dir, 'unwrap_1024_info.mat'),
            'hair_mask_file': os.path.join(topo_dir, 'hair_mask.png'),
            'minor_blend_mask_file': os.path.join(topo_dir, 'minor_valid_whole_mask.png'),
            'major_blend_mask_file': os.path.join(topo_dir, 'major_valid_whole_mask.png'),
            'template_uv_file': os.path.join(topo_dir, 'template_base_uv.png'),
            'output_uv_size': 1024,
            'camera_distance': 10.,
            'focal': 1015.,
            'center': 112.,
            'znear': 5.,
            'zfar': 15.,
            # texture gan
            'texgan_model_file': os.path.join(cpk_dir, f'texgan_model/{texgan_model_name}'),
            # deep3d nn inference model
            'net_recon': 'resnet50',
            'net_recon_path': os.path.join(cpk_dir, 'deep3d_model/epoch_latest.pth'),
            # recognition model
            'net_recog': 'r50',
            'net_recog_path': os.path.join(cpk_dir, 'arcface_model/ms1mv3_arcface_r50_fp16_backbone.pth'),
            # vgg model
            'net_vgg_path': os.path.join(cpk_dir, 'vgg_model/vgg16.pt'),
        }
        self.args_s2_search_uvtex_spherical_fixshape = {
            'w_feat': 10.0,
            'w_color': 10.0,
            'w_vgg': 100.0,
            'w_reg_latent': 0.05,
            'initial_lr': 0.1,
            'lr_rampdown_length': 0.25,
            'total_step': 100,
            'print_freq': 5,
            'visual_freq': 10,
        }
        self.args_s3_optimize_uvtex_shape_joint = {
            'w_feat': 0.2,
            'w_color': 1.6,
            'w_reg_id': 2e-4,
            'w_reg_exp': 1.6e-3,
            'w_reg_gamma': 10.0,
            'w_reg_latent': 0.05,
            'w_lm': 2e-3,
            'initial_lr': 0.01,
            'tex_lr_scale': 1.0 if loose_tex else 0.05,
            'lr_rampdown_length': 0.4,
            'total_step': 200,
            'print_freq': 10,
            'visual_freq': 20,
        }

        self.args_names = ['model', 's2_search_uvtex_spherical_fixshape', 's3_optimize_uvtex_shape_joint']

        # parametric face model
        self.facemodel = ParametricFaceModel(fm_model_file=self.args_model['fm_model_file'],
                                             unwrap_info_file=self.args_model['unwrap_info_file'],
                                             camera_distance=self.args_model['camera_distance'],
                                             focal=self.args_model['focal'],
                                             center=self.args_model['center'],
                                             lm86=lm86,
                                             device=device)

        # texture gan
        self.tex_gan = texgan_cropface630resize1024.TextureGAN(model_path=self.args_model['texgan_model_file'],
                                                               base_uv_path=self.args_model['template_uv_file'],
                                                               blend_mask_path=self.args_model['minor_blend_mask_file'],
                                                               hair_mask_path=self.args_model['hair_mask_file'],
                                                               output_uv_size=self.args_model['output_uv_size'],
                                                               device=device)
        # read blend template and mask
        self.hair_mask = read_img(self.args_model['hair_mask_file'],
                                  resize=(self.args_model['output_uv_size'], self.args_model['output_uv_size']),
                                  dst_range=1.)
        self.major_blend_mask = read_img(self.args_model['major_blend_mask_file'],
                                         resize=(self.args_model['output_uv_size'], self.args_model['output_uv_size']),
                                         dst_range=1.)
        self.template_uv = read_img(self.args_model['template_uv_file'],
                                    resize=(self.args_model['output_uv_size'], self.args_model['output_uv_size']))

        # deep3d nn reconstruction model
        fc_info = {
            'id_dims': self.facemodel.id_dims,
            'exp_dims': self.facemodel.exp_dims,
            'tex_dims': self.facemodel.tex_dims
        }
        self.net_recon_deep3d = define_net_recon_deep3d(net_recon=self.args_model['net_recon'],
                                                        use_last_fc=False,
                                                        fc_dim_dict=fc_info,
                                                        pretrained_path=self.args_model['net_recon_path'])
        self.net_recon_deep3d = self.net_recon_deep3d.eval().requires_grad_(False)

        # renderer
        fov = 2 * np.arctan(self.args_model['center'] / self.args_model['focal']) * 180 / np.pi
        self.renderer = MeshRenderer(fov=fov,
                                     znear=self.args_model['znear'],
                                     zfar=self.args_model['zfar'],
                                     rasterize_size=int(2 * self.args_model['center']))

        # the recognition model
        self.net_recog = define_net_recog(net_recog=self.args_model['net_recog'],
                                          pretrained_path=self.args_model['net_recog_path'])
        self.net_recog = self.net_recog.eval().requires_grad_(False)

        # the vgg model
        with dnnlib.util.open_url(self.args_model['net_vgg_path']) as f:
            self.net_vgg = torch.jit.load(f).eval()

        # coeffs and latents
        self.pred_coeffs = None
        self.pred_latents_w = None
        self.pred_latents_z = None

        self.to(device)
        self.device = device

    def to(self, device):
        self.device = device
        self.facemodel.to(device)
        self.tex_gan.to(device)
        self.net_recon_deep3d.to(device)
        self.renderer.to(device)
        self.net_recog.to(device)
        self.net_vgg.to(device)

    def infer_render(self, is_uv_tex=True):
        # forward face model
        self.pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.pred_coeffs_dict)
        if is_uv_tex:
            # forward texture gan
            # blend with template using minor mask, and resize to 'output_uv_size'
            self.pred_uv_map = self.tex_gan.synth_uv_map(self.pred_latents_w)

            # blend with template using major mask
            pred_uv_map_major_blend = blend_uv_with_template(tensor2np(self.pred_uv_map[:1, :, :, :]),
                                                             self.template_uv,
                                                             self.hair_mask,
                                                             self.major_blend_mask)
            self.pred_uv_map = np2tensor(pred_uv_map_major_blend).type_as(self.pred_uv_map)

            # render front face
            vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.pred_coeffs.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)
        else:
            # render front face
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

    def visualize(self, input_data, is_uv_tex=True):
        # input data
        input_img = tensor2np(input_data['img'][:1, :, :, :])
        skin_img = img3channel(tensor2np(input_data['skin_mask'][:1, :, :, :]))
        parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
        gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
        # predict data
        pred_face_img = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * input_data['img']
        pred_face_img = tensor2np(pred_face_img[:1, :, :, :])
        pred_lm = self.pred_lm[0, :, :].detach().cpu().numpy()
        # draw mask and landmarks
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm[..., 1] = pred_face_img.shape[0] - 1 - gt_lm[..., 1]
        pred_lm[..., 1] = pred_face_img.shape[0] - 1 - pred_lm[..., 1]
        lm_img = draw_landmarks(pred_face_img, gt_lm, color='b')
        lm_img = draw_landmarks(lm_img, pred_lm, color='r')
        # combine visual images
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img, pred_face_img], axis=1)
        if is_uv_tex:
            pre_uv_img = tensor2np(F.interpolate(self.pred_uv_map, size=input_img.shape[:2], mode='area')[:1, :, :, :])
            combine_img = np.concatenate([combine_img, pre_uv_img], axis=1)
        return combine_img

    def visualize_3dmmtex_as_uv(self):
        tex_vertex = self.pred_tex[0, :, :].detach().cpu().numpy()
        unwrap_uv_idx_v_idx = self.facemodel.unwrap_uv_idx_v_idx.detach().cpu().numpy()
        unwrap_uv_idx_bw = self.facemodel.unwrap_uv_idx_bw.detach().cpu().numpy()
        tex_uv = unwrap_vertex_to_uv(tex_vertex, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw) * 255.
        return tex_uv

    def save_mesh(self, path, mesh_name, mlt_name=None, uv_name=None, is_uv_tex=True):
        pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        pred_id_vertex, pred_exp_vertex, pred_alb_tex = self.facemodel.compute_for_mesh(pred_coeffs_dict)
        if is_uv_tex:
            assert mlt_name is not None and uv_name is not None
            write_mtl(os.path.join(path, mlt_name), uv_name)
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
        else:
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
        write_mesh_obj(mesh_info=id_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_id{mesh_name[-4:]}'))
        write_mesh_obj(mesh_info=exp_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_exp{mesh_name[-4:]}'))

    def save_coeffs(self, path, coeffs_name, is_uv_tex=True):
        # coeffs & landmarks
        coeffs_info = {'coeffs': self.pred_coeffs, 'lm68': self.pred_lm}
        if is_uv_tex:
            coeffs_info['latents_w'] = self.pred_latents_w
            coeffs_info['latents_z'] = self.pred_latents_z
        torch.save(coeffs_info, os.path.join(path, coeffs_name))

    def gather_args_str(self):
        args_str = '\n'
        for name in self.args_names:
            args_dict = getattr(self, 'args_' + name)
            args_str += f'----------------- Args-{name} ---------------\n'
            for k, v in args_dict.items():
                args_str += '{:>30}: {:<30}\n'.format(str(k), str(v))
        args_str += '----------------- End -------------------'
        return args_str

    def fitting(self, input_data, logger):
        # fix random seed
        setup_seed(123)

        # print args
        logger.write_txt_log(self.gather_args_str())

        # save the input data
        torch.save(input_data, os.path.join(logger.vis_dir, f'input_data.pt'))

        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------

        logger.write_txt_log('Stage 1 getting initial coeffs by Deep3D NN inference.')
        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        logger.write_disk_images([vis_img], ['stage1_vis'])
        logger.write_disk_images([vis_tex_uv], ['stage1_vis_3dmmtex_as_uv'])
        self.save_mesh(path=logger.vis_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)

        #--------- Stage 2 - search UV tex on a spherical surface with fixed shape ---------

        logger.write_txt_log('Start stage 2 searching UV tex on a spherical surface with fixed shape.')
        logger.reset_prefix(prefix='s2_search_uvtex_spherical_fixshape')
        fitter = uvtex_spherical_fixshape_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=None,
                                                        **self.args_s2_search_uvtex_spherical_fixshape)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 2 searching UV tex on a spherical surface with fixed shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage2_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage2_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage2_mesh.obj',
                       mlt_name='stage2_mesh.mlt',
                       uv_name='stage2_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage2_coeffs.pt')

        # the output of textgan without blending
        pred_uv_map_no_blend = self.tex_gan.synth_uv_map(self.pred_latents_w, is_blend=False)
        logger.write_disk_images([tensor2np(pred_uv_map_no_blend[:1, :, :, :])], ['stage2_uv_no_blend'])

        #--------- Stage 3 - jointly optimize UV tex and shape ---------

        logger.write_txt_log('Start stage 3 jointly optimize UV tex and shape.')
        logger.reset_prefix(prefix='s3_optimize_uvtex_shape_joint')
        fitter = uvtex_wspace_shape_joint_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=self.pred_latents_z,
                                                        **self.args_s3_optimize_uvtex_shape_joint)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 3 jointly optimize UV tex and shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage3_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage3_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage3_mesh.obj',
                       mlt_name='stage3_mesh.mlt',
                       uv_name='stage3_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage3_coeffs.pt')

        # the output of textgan
        pred_uv_map_no_blend = self.tex_gan.synth_uv_map(self.pred_latents_w, is_blend=False)
        logger.write_disk_images([tensor2np(pred_uv_map_no_blend[:1, :, :, :])], ['stage3_uv_no_blend'])
