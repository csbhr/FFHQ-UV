import os
import numpy as np
import skimage
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stylegan2.networks import Generator
from utils.data_utils import read_img, np2tensor, tensor2np, img2mask


def match_color_in_yuv(src_tex, dst_tex, mask):
    '''
    Match color (src_tex -> dst_tex) in YUV color space.

    Args:
        src_tex: numpy.array (unwrap_size, unwrap_size, 3). The source UV texture (template texture).
        dst_tex: numpy.array (unwrap_size, unwrap_size, 3). The target UV texture (input texture).
        mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of valid region.
    Returns:
        match_tex: numpy.array (unwrap_size, unwrap_size, 3). The matched UV texture.
    '''
    # rgb -> yuv
    dst_tex_yuv = skimage.color.convert_colorspace(dst_tex, "rgb", "yuv")
    src_tex_yuv = skimage.color.convert_colorspace(src_tex, "rgb", "yuv")
    # status
    is_valid = mask[:, :, 0] > 0.5
    mu_dst = np.mean(dst_tex_yuv[is_valid], axis=0, keepdims=True)
    std_dst = np.std(dst_tex_yuv[is_valid], axis=0, keepdims=True)
    mu_src = np.mean(src_tex_yuv[is_valid], axis=0, keepdims=True)
    std_src = np.std(src_tex_yuv[is_valid], axis=0, keepdims=True)
    # match
    match_tex_yuv = (src_tex_yuv - mu_src) / std_src
    match_tex_yuv = (match_tex_yuv / 1.5) * std_dst + mu_dst
    # yuv -> rgb
    match_tex = skimage.color.convert_colorspace(match_tex_yuv, "yuv", "rgb")
    match_tex = np.clip(match_tex, 0, 255)
    return match_tex


def blend_uv_with_template(res_uv, template_uv, blend_mask, hair_mask):
    b, _, _, _ = res_uv.size()
    res_uv_np = tensor2np(res_uv)

    # match color
    template_uv_tensor = np2tensor(template_uv).type_as(res_uv).repeat((b, 1, 1, 1))
    template_uv_match_color = match_color_in_yuv(src_tex=template_uv, dst_tex=res_uv_np, mask=blend_mask)
    template_uv_match_color_tensor = np2tensor(template_uv_match_color).type_as(res_uv).repeat((b, 1, 1, 1))

    # blend with template
    blend_mask_blur = cv2.blur(blend_mask, (49, 49), 0)
    blend_mask_blur_tensor = np2tensor(blend_mask_blur, src_range=1.0).repeat((b, 1, 1, 1))
    blend_uv = res_uv * blend_mask_blur_tensor + template_uv_match_color_tensor * (1 - blend_mask_blur_tensor)

    # cover hair
    hair_mask_erode = cv2.blur(hair_mask, (23, 23), 0)
    hair_mask_erode = img2mask(hair_mask_erode, thre=1., mode='greater-equal')
    hair_mask_blur = cv2.blur(hair_mask_erode, (23, 23), 0)
    hair_mask_blur_tensor = np2tensor(hair_mask_blur, src_range=1.0).repeat((b, 1, 1, 1))
    blend_result = template_uv_tensor * hair_mask_blur_tensor + blend_uv * (1 - hair_mask_blur_tensor)

    return blend_result


class TextureGAN(nn.Module):

    def __init__(self, model_path, base_uv_path, blend_mask_path, hair_mask_path, output_uv_size=1024, device='cuda'):
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

        self.hair_mask = read_img(hair_mask_path, resize=(1664, 1664), dst_range=1.)
        self.blend_mask = read_img(blend_mask_path, resize=(1664, 1664), dst_range=1.)
        self.base_uv = read_img(base_uv_path, resize=(1664, 1664))
        self.base_uv_tensor = np2tensor(self.base_uv, device=device)
        self.output_uv_size = output_uv_size
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

    def synth_uv_map(self, in_w, is_blend=True):
        _, num_w, _ = in_w.size()
        if num_w == 1:
            in_w = in_w.repeat([1, self.G.mapping.num_ws, 1])
        uvmap = self.G.synthesis(in_w, noise_mode='const')
        uvmap = (uvmap + 1) * 0.5

        if is_blend:
            # resize and fill
            b, _, _, _ = uvmap.size()
            full_uvmap = self.base_uv_tensor.detach().clone().repeat((b, 1, 1, 1))
            full_uvmap[:, :, 187:1211, 317:1341] = uvmap
            blend_uvmap = blend_uv_with_template(res_uv=full_uvmap,
                                                template_uv=self.base_uv,
                                                blend_mask=self.blend_mask,
                                                hair_mask=self.hair_mask)
            if self.output_uv_size != 1664:
                blend_uvmap = F.interpolate(blend_uvmap,
                                            size=(self.output_uv_size, self.output_uv_size),
                                            mode='bilinear',
                                            align_corners=False)
            return blend_uvmap
        else:
            return uvmap
