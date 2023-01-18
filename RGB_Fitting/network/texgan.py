import os
import numpy as np
import torch
import torch.nn as nn

from .stylegan2.networks import Generator


class TextureGAN(nn.Module):

    def __init__(self, model_path, device='cuda'):
        super(TextureGAN, self).__init__()

        generator_kwargs = {
            'z_dim': 512,  # Input latent (Z) dimensionality.
            'c_dim': 0,  # Conditioning label (C) dimensionality.
            'w_dim': 512,  # Intermediate latent (W) dimensionality.
            'img_resolution': 1024,  # Output resolution.
            'img_channels': 3,  # Number of output color channels.
            'mapping_kwargs': {  # Arguments for MappingNetwork.
                'num_layers': 8,  # Number of mapping layers.
                'embed_features': None,  # Label embedding dimensionality, None = same as w_dim.
                'layer_features': None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
                'activation': 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
                'lr_multiplier': 0.01,  # Learning rate multiplier for the mapping layers.
                'w_avg_beta': 0.995,  # Decay for tracking the moving average of W during training, None = do not track.
            },
            'synthesis_kwargs': {  # Arguments for SynthesisNetwork.
                'channel_base': 32768,  # Overall multiplier for the number of channels.
                'channel_max': 512,  # Maximum number of channels in any layer.
                'num_fp16_res': 4,  # Use FP16 for the N highest resolutions.
                'conv_clamp': 256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                'architecture': 'skip',  # Architecture: 'orig', 'skip', 'resnet'.
                'resample_filter': [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                'use_noise': True,  # Enable noise input?
                'activation': 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
                'fp16_channels_last': False,  # Use channels-last memory format with FP16?
                'kernel_size': 3,  # Convolution kernel size.
            }
        }
        # generator
        self.G = Generator(**generator_kwargs)
        self.G.load_state_dict(torch.load(model_path))
        self.G = self.G.eval().requires_grad_(False).to(device)

        # noise_buffer
        self.noise_bufs = {name: buf for (name, buf) in self.G.synthesis.named_buffers() if 'noise_const' in name}

        # noise_buffer keep
        self.noise_bufs_keep = {name: buf.detach().clone() for (name, buf) in self.noise_bufs.items()}

        # the w avg and std
        w_stats_path = model_path[:-4] + '.pt'
        if os.path.isfile(w_stats_path):
            load_pt = torch.load(w_stats_path)
            self.w_avg = load_pt['w_avg'].float().to(device)
            self.w_std = load_pt['w_std'].float().to(device)
        else:
            with torch.no_grad():
                w_avg_samples = 100000
                z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
                w_samples = self.G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
                w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
                w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
                w_std = np.std(w_samples, axis=0, keepdims=True)  # [1, 1, C]
                self.w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device)
                self.w_std = torch.tensor(w_std, dtype=torch.float32, device=device)
                torch.save({'w_avg': self.w_avg, 'w_std': self.w_std}, w_stats_path)

        self.device = device

    def reset_noise_bufs(self):
        for k in self.noise_bufs.keys():
            self.noise_bufs[k][:] = self.noise_bufs_keep[k].detach().clone()

    def get_init_z_latents(self):
        '''
        return an init latent code, it is z space;
        if opt_noise, also return the noise buffer for opt
        '''
        z_opt = np.random.RandomState(123).randn(1, self.G.mapping.num_ws, self.G.z_dim)  # [1, 18, C]
        z_opt = torch.tensor(z_opt, dtype=torch.float32, device=self.device, requires_grad=True)
        return z_opt

    def map_z_to_w(self, z_in):
        '''apply learned linear mapping to match latent distribution to that of the mapping network'''
        w_in = z_in * self.w_std + self.w_avg
        return w_in

    def inverse_w_to_z(self, w_in):
        '''inverse learned linear mapping'''
        z_in = (w_in - self.w_avg) / self.w_std
        return z_in

    def synth_uv_map(self, in_w):
        _, num_w, _ = in_w.size()
        if num_w == 1:
            in_w = in_w.repeat([1, self.G.mapping.num_ws, 1])
        uvmap = self.G.synthesis(in_w, noise_mode='const')
        uvmap = (uvmap + 1) * 0.5
        return uvmap
