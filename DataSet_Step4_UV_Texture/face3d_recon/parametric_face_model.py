import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat


class ParametricFaceModel(nn.Module):

    def __init__(self, model_path, device='cuda'):
        super(ParametricFaceModel, self).__init__()

        # load 3DMM model from mat file
        model = loadmat(model_path)

        # mean face shape. [1, 3*20481]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # mean face texture. [1, 3*20481]
        self.mean_tex = model['meantex'].astype(np.float32)
        # recenter
        mean_shape = self.mean_shape.reshape([-1, 3])
        mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
        self.mean_shape = mean_shape.reshape([1, -1])

        # identity basis. [3*20481, 532]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*20481, 45]
        self.exp_base = model['exBase'].astype(np.float32)
        # texture basis. [3*20481, 439]
        self.tex_base = model['texBase'].astype(np.float32)

        self.id_dims = self.id_base.shape[1]
        self.exp_dims = self.exp_base.shape[1]
        self.tex_dims = self.tex_base.shape[1]

        # UV coordinates [82810, 2], 0~512
        self.uv_idx = model['uv_idx'].astype(np.int64)
        # UV index for each vertex [20481, 1], 0~(512*512)
        self.vtx_uv_idx = model['vtx_uv_idx'].astype(np.int64)
        # UV coordinates for each vertex [20481, 2], 0~512
        self.vtx_vt = model['vtx_vt'].astype(np.float32)

        # texture list, [20792, 2], 0~1
        self.vt_list = model['vt_list'].astype(np.float32)
        # vertex index for each texture [20792, 1], 0~20481
        self.vt_vtx_idx = model['vt_vtx_idx'].astype(np.int64)

        # vertex indices for each *facial* triangular face [18318, 3]
        self.face_buf = model['tri'].astype(np.int64) - 1
        # vertex indices for each *full head* triangular face [40832, 3]
        self.head_buf = model['head_tri'].astype(np.int64) - 1
        # vertex indices for 68 landmarks [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        # texture indices for each *facial* triangular face [18318, 3]
        self.tri_vt = model['tri_vt'].astype(np.int64) - 1
        # texture indices for each *full head* triangular face [40832, 3]
        self.head_tri_vt = model['head_tri_vt'].astype(np.int64) - 1

        # face indices for each vertex that lies in. starts from 0. [20481, 8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1

        self.np2tensor()
        self.to(device)

    def np2tensor(self):
        '''
        Transfer numpy.array to torch.Tensor.
        '''
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value))

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == torch.__name__:
                setattr(self, key, value.to(device))

    def compute_shape(self, id_coeff, exp_coeff):
        '''
        Compute the 3D face shape from 3DMM by coefficients.

        Args:
            id_coeff: torch.Tensor, (B, 532). The identity coeffs.
            exp_coeff: torch.Tensor, (B, 45). The expression coeffs.
        Returns:
            face_shape: torch.Tensor, (B, 20481, 3). The 3D face shape.
            id_shape: torch.Tensor, (B, 20481, 3). The 3D face shape (neutral).
        '''
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = (id_part + exp_part + self.mean_shape).reshape([batch_size, -1, 3])
        id_shape = (id_part + self.mean_shape).reshape([batch_size, -1, 3])
        return face_shape, id_shape

    def compute_texture(self, tex_coeff, normalize=True):
        '''
        Compute the 3D face texture from 3DMM by coefficients.

        Args:
            tex_coeff: torch.Tensor, (B, 439). The texture coeffs.
            normalize: bool. If true, the value will be normalized to 0~1.
        Returns:
            face_texture: torch.Tensor, (B, 20481, 3). The 3D face texture.
        '''
        batch_size = tex_coeff.shape[0]
        tex_part = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff)
        face_texture = (tex_part + self.mean_tex).reshape([batch_size, -1, 3])
        if normalize:
            face_texture = face_texture / 255.
        return face_texture
