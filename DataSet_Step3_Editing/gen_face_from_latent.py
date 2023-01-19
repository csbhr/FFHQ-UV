import os
import time
import torch
import pickle
import argparse
import numpy as np
import PIL.Image as Image

import dnnlib as dnnlib
import dnnlib.tflib as tflib


def load_networks(path):
    stream = open(path, 'rb')
    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    return G, D, Gs


class StyleGAN2_Model:

    def __init__(self, network_pkl):

        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = load_networks(network_pkl)
        self.Gs = Gs
        self.Gs_syn_kwargs = dnnlib.EasyDict()
        self.Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_syn_kwargs.randomize_noise = False
        self.Gs_syn_kwargs.minibatch_size = 4
        self.noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        rnd = np.random.RandomState(0)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars})

    def generate_im_from_random_seed(self, seed=22, truncation_psi=0.5):
        Gs = self.Gs
        seeds = [seed]
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
            images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        return images

    def generate_im_from_z_space(self, z, truncation_psi=0.5):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi  # [height, width]

        images = self.Gs.run(z, None, **Gs_kwargs)
        return images

    def generate_im_from_w_space(self, w):
        images = self.Gs.components.synthesis.run(w, **self.Gs_syn_kwargs)
        return images


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="gen_face_from_latent")
    parser.add_argument(
        "--latent_dir",
        type=str,
        required=True,
        help="The directory of the download multi-view face latent codes.",
    )
    parser.add_argument(
        "--save_face_dir",
        type=str,
        required=True,
        help="The directory for saving generated multi-view face images.",
    )
    parser.add_argument(
        "--stylegan_network_pkl",
        type=str,
        default='./stylegan2-ffhq-config-f.pkl',
        help="The path of the offical pretrained StyleGAN2 network pkl file.",
    )
    args = parser.parse_args()

    # ----------------------- Define Inference Parameters -----------------------
    latent_dir = args.latent_dir
    save_face_dir = args.save_face_dir
    stylegan_network_pkl = args.stylegan_network_pkl

    os.makedirs(save_face_dir, exist_ok=True)

    # ----------------------- Load StyleGAN Model -----------------------
    stylegan_model = StyleGAN2_Model(stylegan_network_pkl)

    # ----------------------- Perform Editing -----------------------
    clip_names = sorted([cn for cn in os.listdir(latent_dir) if os.path.isdir(os.path.join(latent_dir, cn))])
    for cn in clip_names:
        os.makedirs(os.path.join(save_face_dir, cn), exist_ok=True)

        fnames = sorted(os.listdir(os.path.join(latent_dir, cn)))
        for fn in fnames:
            os.makedirs(os.path.join(save_face_dir, cn, fn), exist_ok=True)
            tic = time.time()

            left_latent = torch.load(os.path.join(latent_dir, cn, fn, f'{fn}_left_latent.pt'),
                                     map_location='cpu')['latent']
            front_latent = torch.load(os.path.join(latent_dir, cn, fn, f'{fn}_front_latent.pt'),
                                      map_location='cpu')['latent']
            right_latent = torch.load(os.path.join(latent_dir, cn, fn, f'{fn}_right_latent.pt'),
                                      map_location='cpu')['latent']

            left_face = stylegan_model.generate_im_from_w_space(left_latent.detach().cpu().numpy())[0]
            front_face = stylegan_model.generate_im_from_w_space(front_latent.detach().cpu().numpy())[0]
            right_face = stylegan_model.generate_im_from_w_space(right_latent.detach().cpu().numpy())[0]

            Image.fromarray(left_face, 'RGB').save(os.path.join(save_face_dir, cn, fn, f'{fn}_left.png'))
            Image.fromarray(front_face, 'RGB').save(os.path.join(save_face_dir, cn, fn, f'{fn}_front.png'))
            Image.fromarray(right_face, 'RGB').save(os.path.join(save_face_dir, cn, fn, f'{fn}_right.png'))

            toc = time.time()
            print('Generate {}/{} done, took {:.4f} seconds.'.format(cn, fn, toc - tic))

        print(f'Generate clip {cn} done!')

    print(f'Generate all done!')
