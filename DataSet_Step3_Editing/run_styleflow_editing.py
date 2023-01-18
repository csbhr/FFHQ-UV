import os
import json
import time
import numpy as np
import torch
import PIL.Image as Image

from options.test_options import TestOptions
from utils import Build_model
from module.flow import cnf
from expression_recognition import Exp_Recog_API

attribute_idx = {
    'Light': {
        'attr_idx': (0, 8),
        'w_idxs': [(7, 11)]
    },
    'Gender': {
        'attr_idx': 9,
        'w_idxs': [(0, 7)]
    },
    'Glasses': {
        'attr_idx': 10,
        'w_idxs': [(2, 3)]
    },
    'Yaw': {
        'attr_idx': 11,
        'w_idxs': [(0, 3)]
    },
    'Pitch': {
        'attr_idx': 12,
        'w_idxs': [(0, 3)]
    },
    'Baldness': {
        'attr_idx': 13,
        'w_idxs': [(0, 5)]
    },
    'Beard': {
        'attr_idx': 14,
        'w_idxs': [(5, 9)]
    },
    'Age': {
        'attr_idx': 15,
        'w_idxs': [(4, 7)]
    },
    'Expression': {
        'attr_idx': 16,
        'w_idxs': [(0, 17)]
    },
}


def replace_w_space(attr_name, current, target):
    w_idxs = attribute_idx[attr_name]['w_idxs']
    new_w = current.clone().detach()
    for a, b in w_idxs:
        new_w[0][a:b + 1] = target[0][a:b + 1]
    return new_w


class StyleFlow:

    def __init__(self, args):

        super().__init__()
        self.device = args.device

        # StyleGAN model
        self.stylegan_model = Build_model(args)

        # StyleFlow model
        self.prior = cnf(512, '512-512-512-512-512', 17, 1)
        self.prior.load_state_dict(torch.load(args.flow_model_path))
        self.prior.eval()

        # Expression direction
        self.exp_direct = torch.load(args.exp_direct_path).to(self.device)
        self.exp_direct = self.exp_direct.unsqueeze(1).repeat([1, 18, 1])

        # Expression recognition model
        self.exp_recon_model = Exp_Recog_API(model_path=args.exp_recognition_path)

        self.zero_padding = torch.zeros(1, 18, 1).to(self.device)
        self.z_current = None  # tensor [1, 18, 512]
        self.w_current = None  # tensor [1, 18, 512]
        self.attr_current = None  # tensor [1, 17, 1, 1]
        self.GAN_image = None  # array [1024, 1024, 3]

    def set_latents(self, curr_w, cur_light, cur_attr):
        '''
        curr_w: tensor [1, 18, 512]
        cur_light: array [1, 9, 1, 1]
        cur_attr: array [8, 1]
        '''
        self.w_current = curr_w.to(self.device)  # [1, 18, 512]
        self.attr_current = torch.cat(
            [
                torch.from_numpy(cur_light).type(torch.FloatTensor),
                torch.from_numpy(cur_attr).type(torch.FloatTensor).unsqueeze(0).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)  # [1, 17, 1, 1]

        # get z from w
        self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]  # [1, 18, 512]
        self.GAN_image = self.stylegan_model.generate_im_from_w_space(
            self.w_current.detach().cpu().numpy())[0]  # [1024, 1024, 3] array

        return self.GAN_image, self.w_current, self.attr_current

    def change_light(self, lightingvec, keep_change=True):
        '''
        lightingvec: array [1, 9, 1, 1]
        keep_change: bool, if True, keep the changes to current state
        '''

        # replace attributes, :9 dim is light, and replace them by target light
        attr_target = self.attr_current.clone().detach().to(self.device)
        attr_target[:, :9] = torch.from_numpy(lightingvec).type(torch.FloatTensor).to(self.device)

        # get w_target
        w_target = self.prior(self.z_current, attr_target, logpx=self.zero_padding, reverse=True)[0]

        # replace w, in w+ space, only replace 7-11 layers
        w_target = replace_w_space(attr_name='Light', current=self.w_current, target=w_target)
        GAN_image = self.stylegan_model.generate_im_from_w_space(w_target.detach().cpu().numpy())[0]

        if keep_change:
            self.w_current = w_target.clone().detach().to(self.device)
            self.attr_current = attr_target.clone().detach().to(self.device)
            self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]
            self.GAN_image = GAN_image

        return GAN_image, w_target, attr_target

    def change_exp(self, exp_alpha, keep_change=True):
        '''
        exp_alpha: array [1]
        keep_change: bool, if True, keep the changes to current state
        '''

        # replace attributes
        attr_target = self.attr_current.clone().detach().to(self.device)
        attr_idx = attribute_idx['Expression']['attr_idx']
        attr_target[:, attr_idx, 0, 0] = torch.from_numpy(np.array([0])).type(torch.FloatTensor).to(self.device)

        # get w_target
        exp_alpha = float(exp_alpha)
        w_target_walk = self.w_current + exp_alpha * self.exp_direct

        # replace w, in w+ space, only replace 7-11 layers
        w_target = replace_w_space(attr_name='Expression', current=self.w_current, target=w_target_walk)
        GAN_image = self.stylegan_model.generate_im_from_w_space(w_target.detach().cpu().numpy())[0]

        # Facial expression recognition for determining whether to stop or not
        n_loop = 0
        alpha_loop = 0.05 * exp_alpha
        while self.exp_recon_model(GAN_image)[0] == 'Happy' and n_loop < 5:
            n_loop = n_loop + 1
            w_target_walk = w_target_walk + alpha_loop * self.exp_direct
            w_target = replace_w_space(attr_name='Expression', current=self.w_current, target=w_target_walk)
            GAN_image = self.stylegan_model.generate_im_from_w_space(w_target.detach().cpu().numpy())[0]

        if keep_change:
            self.w_current = w_target.clone().detach().to(self.device)
            self.attr_current = attr_target.clone().detach().to(self.device)
            self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]
            self.GAN_image = GAN_image

        return GAN_image, w_target, attr_target

    def change_attr(self, attr_name, attr_value, keep_change=True):
        '''
        attr_name: str
        attr_value: array [1]
        keep_change: bool, if True, keep the changes to current state
        '''

        # replace attributes
        attr_target = self.attr_current.clone().detach().to(self.device)
        attr_idx = attribute_idx[attr_name]['attr_idx']
        attr_target[:, attr_idx, 0, 0] = torch.from_numpy(attr_value).type(torch.FloatTensor).to(self.device)

        # get w_target
        w_target = self.prior(self.z_current, attr_target, logpx=self.zero_padding, reverse=True)[0]

        # replace w, in w+ space, only replace 7-11 layers
        w_target = replace_w_space(attr_name=attr_name, current=self.w_current, target=w_target)
        GAN_image = self.stylegan_model.generate_im_from_w_space(w_target.detach().cpu().numpy())[0]

        if keep_change:
            self.w_current = w_target.clone().detach().to(self.device)
            self.attr_current = attr_target.clone().detach().to(self.device)
            self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]
            self.GAN_image = GAN_image

        return GAN_image, w_target, attr_target


def set_normal(sf_model, proj_data_dir, save_dir, fn, edit_items):
    basename = fn[:fn.rfind('.')]
    os.makedirs(os.path.join(save_dir, basename), exist_ok=True)

    cur_latent = torch.load(os.path.join(proj_data_dir, 'latents', f'{basename}.pt'))
    cur_light = np.load(os.path.join(proj_data_dir, 'lights', f'{basename}.npy'))
    attr = json.load(open(os.path.join(proj_data_dir, 'attributes', f'{basename}.json')))
    cur_attr = np.array([
        [attr['Gender']],
        [attr['Glasses']],
        [attr['Yaw']],
        [attr['Pitch']],
        [attr['Baldness']],
        [attr['Beard']],
        [attr['Age']],
        [attr['Expression']],
    ])

    # initialize
    img_in, w_in, attr_in = sf_model.set_latents(cur_latent, cur_light, cur_attr)
    img_out, w_new, attr_new = img_in, w_in, attr_in

    if 'delight' in edit_items:
        target_light = np.array([[
            [[1.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
        ]])
        # change lighting
        img_out, w_new, attr_new = sf_model.change_light(target_light)

    if 'norm_attr' in edit_items:
        target_attr = {
            'Glasses': np.array([0]),  # no glass
            'Yaw': np.array([0]),  # 0 degree
            'Pitch': np.array([0]),  # 0 degree
            'Baldness': np.array([1]),  # no hair
            'Expression': np.array([0]),  # no expression
        }
        # change attributes
        for attr_name in target_attr.keys():
            if attr_name == 'Expression':
                # walk along the exp direction
                img_out, w_new, attr_new = sf_model.change_exp(exp_alpha=target_attr[attr_name] - 2 * attr[attr_name])
            else:
                img_out, w_new, attr_new = sf_model.change_attr(attr_name=attr_name, attr_value=target_attr[attr_name])

    front_result_img = Image.fromarray(img_out, 'RGB')
    front_result_latent = {'latent': w_new, 'attribute': attr_new}
    front_result_img.save(os.path.join(save_dir, basename, f'{basename}_front.png'))
    torch.save(front_result_latent, os.path.join(save_dir, basename, f'{basename}_front_latent.pt'))

    if 'multi_yaw' in edit_items:
        # turn left, and given right face
        yaw = np.array([-30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Yaw', attr_value=yaw, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, basename, f'{basename}_right.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, basename, f'{basename}_right_latent.pt'))
        # turn right, and given left face
        yaw = np.array([30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Yaw', attr_value=yaw, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, basename, f'{basename}_left.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, basename, f'{basename}_left_latent.pt'))

    if 'multi_pitch' in edit_items:
        # turn down, and given up face
        pitch = np.array([-30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Pitch', attr_value=pitch, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, basename, f'{basename}_up.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, basename, f'{basename}_up_latent.pt'))
        # turn up, and given down face
        pitch = np.array([30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Pitch', attr_value=pitch, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, basename, f'{basename}_down.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, basename, f'{basename}_down_latent.pt'))


if __name__ == '__main__':
    '''Usage
    cd ./DataSet_Step3_Editing
    python run_styleflow_editing.py \
        --proj_data_dir ../examples/dataset_examples \
        --network_pkl ../checkpoints/stylegan_model/stylegan2-ffhq-config-f.pkl \
        --flow_model_path ../checkpoints/styleflow_model/modellarge10k.pt \
        --exp_direct_path ../checkpoints/styleflow_model/expression_direction.pt \
        --exp_recognition_path ../checkpoints/exprecog_model/FacialExpRecognition_model.t7 \
        --edit_items delight,norm_attr,multi_yaw
    '''

    args = TestOptions().parse()

    # ----------------------- Define Inference Parameters -----------------------
    flow_model_path = args.flow_model_path
    output_edit_dir = os.path.join(args.proj_data_dir, 'edit')
    edit_items = args.edit_items.split(',')
    device = args.device

    os.makedirs(output_edit_dir, exist_ok=True)

    # ----------------------- Load StyleFlow Model -----------------------
    sf_model = StyleFlow(args)
    print(f'StyleFlow model successfully loaded from {flow_model_path}.')

    # ----------------------- Perform Editing -----------------------
    fnames = sorted(os.listdir(os.path.join(args.proj_data_dir, 'attributes')))
    for fn in fnames:
        basename = fn[:fn.rfind('.')]

        tic = time.time()

        set_normal(sf_model, args.proj_data_dir, output_edit_dir, fn, edit_items)

        toc = time.time()
        print('Editing {} done, took {:.4f} seconds.'.format(fn, toc - tic))

    print('Editing process done!')
