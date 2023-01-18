import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat


def pinhole_projection(focal, center):
    return np.array([focal, 0, center, 0, focal, center, 0, 0, 1]).reshape([3, 3]).astype(np.float32).transpose()


class SH:

    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1 / np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class ParametricFaceModel(nn.Module):

    def __init__(self,
                 fm_model_file='./hifi3dpp_model_info.mat',
                 unwrap_info_file='./unwrap_1024_info.mat',
                 recenter=True,
                 camera_distance=10.,
                 focal=1015.,
                 center=112.,
                 init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 lm86=False,
                 device='cuda'):
        super(ParametricFaceModel, self).__init__()

        # load 3DMM model from mat file
        model = loadmat(fm_model_file)

        # mean face shape. [1, 3*20481]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # mean face texture. [1, 3*20481]
        self.mean_tex = model['meantex'].astype(np.float32)

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([1, -1])

        # identity basis. [3*20481, 532]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*20481, 45]
        self.exp_base = model['exBase'].astype(np.float32)
        # texture basis. [3*20481, 439]
        self.tex_base = model['texBase'].astype(np.float32)

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
        if lm86:
            self.keypoints = np.squeeze(
                np.array([
                    11592, 10911, 11580, 10681, 2653, 11654, 4566, 18648, 6648, 16094, 2022, 6476, 27, 5499, 6400, 5730,
                    6411, 11877, 3454, 18705, 18707, 3484, 12056, 3513, 12133, 12130, 939, 16153, 16151, 909, 6712,
                    6965, 6969, 968, 6893, 14083, 4644, 4649, 3933, 12559, 12384, 4647, 12775, 1388, 2109, 2104, 8983,
                    7227, 7416, 2107, 7636, 741, 7450, 686, 7279, 13894, 8793, 12647, 7511, 14364, 9270, 12451, 12084,
                    7273, 6920, 7306, 13226, 4302, 735, 1757, 8108, 7917, 13216, 8098, 16646, 19200, 13077, 7950, 13175,
                    726, 8049, 19180, 7865, 16626, 19203, 16649
                ])).astype(np.int64)
        else:
            self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        # texture indices for each *facial* triangular face [18318, 3]
        self.tri_vt = model['tri_vt'].astype(np.int64) - 1
        # texture indices for each *full head* triangular face [40832, 3]
        self.head_tri_vt = model['head_tri_vt'].astype(np.int64) - 1

        # face indices for each vertex that lies in. starts from 0. [20481, 8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1

        # vertex indices for pre-defined skin region to compute reflectance loss
        self.skin_mask = np.squeeze(model['skinmask'])

        # the pin-hole project matrix (camera intrinsic parameters)
        self.pinhole_proj = pinhole_projection(focal, center)

        # the SH lighting coefficients
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        self.SH = SH()

        # the distance of camera position
        self.camera_distance = camera_distance

        # the dims of coeffs
        self.id_dims = self.id_base.shape[1]
        self.exp_dims = self.exp_base.shape[1]
        self.tex_dims = self.tex_base.shape[1]

        # load unwrap info from mat file
        unwrap_info = loadmat(unwrap_info_file)
        self.unwrap_uv_idx_bw = unwrap_info['uv_idx_bw'].astype(np.float32)
        self.unwrap_uv_idx_v_idx = unwrap_info['uv_idx_v_idx'].astype(np.float32)

        self.np2tensor()
        self.to(device)
        self.device = device

    def np2tensor(self):
        '''
        Transfer numpy.array to torch.Tensor.
        '''
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value))

    def tensor2np(self):
        '''
        Transfer torch.Tensor to numpy.array.
        '''
        for key, value in self.__dict__.items():
            if type(value).__module__ == torch.__name__:
                setattr(self, key, value.detach().cpu().numpy())

    def to(self, device):
        '''
        Move to device.
        '''
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == torch.__name__:
                setattr(self, key, value.to(device))

    def split_coeff(self, coeffs):
        '''
        Split the estimated coeffs.
        '''
        id_coeffs = coeffs[:, :self.id_dims]
        exp_coeffs = coeffs[:, self.id_dims:self.id_dims + self.exp_dims]
        tex_coeffs = coeffs[:, self.id_dims + self.exp_dims:self.id_dims + self.exp_dims + self.tex_dims]
        angles = coeffs[:,
                        self.id_dims + self.exp_dims + self.tex_dims:self.id_dims + self.exp_dims + self.tex_dims + 3]
        gammas = coeffs[:, self.id_dims + self.exp_dims + self.tex_dims + 3:self.id_dims + self.exp_dims +
                        self.tex_dims + 3 + 27]
        translations = coeffs[:, self.id_dims + self.exp_dims + self.tex_dims + 3 + 27:]

        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    def combine_coeff(self, coeffs_dict):
        '''
        Combine the estimated coeffs.
        '''
        coeffs = torch.cat([
            coeffs_dict['id'],
            coeffs_dict['exp'],
            coeffs_dict['tex'],
            coeffs_dict['angle'],
            coeffs_dict['gamma'],
            coeffs_dict['trans'],
        ],
                           dim=1)
        return coeffs

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
        id_face_shape = (id_part + self.mean_shape).reshape([batch_size, -1, 3])
        exp_face_shape = (id_part + exp_part + self.mean_shape).reshape([batch_size, -1, 3])
        return id_face_shape, exp_face_shape

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

    def compute_norm(self, face_shape):
        '''
        Compute the normal vector according the shape.

        Args:
            face_shape: torch.Tensor, (B, N, 3). The vertices (world coordinate system).
        Returns:
            vertex_norm: torch.Tensor, (B, N, 3). The normal vector for each vertex.
        '''
        v1 = face_shape[:, self.head_buf[:, 0]]  # (B, M, 3)
        v2 = face_shape[:, self.head_buf[:, 1]]  # (B, M, 3)
        v3 = face_shape[:, self.head_buf[:, 2]]  # (B, M, 3)

        e1 = v1 - v2  # (B, M, 3)
        e2 = v2 - v3  # (B, M, 3)
        face_norm = torch.cross(e1, e2, dim=-1)  # (B, M, 3)
        face_norm = F.normalize(face_norm, dim=-1, p=2)  # (B, M, 3)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).type_as(face_norm)],
                              dim=1)  # (B, M+1, 3)

        vertex_norm = face_norm[:, self.point_buf]  # (B, N, 8, 3)
        vertex_norm = torch.sum(vertex_norm, dim=2)  # (B, N, 3)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)  # (B, N, 3)
        return vertex_norm

    def compute_shading(self, face_norm, gamma):
        '''
        Compute the shading according the normal vectors and SH lighting coefficients.

        Args:
            face_norm: torch.Tensor, (B, N, 3). The normal vector for each vertex.
            gamma: torch.Tensor, (B, 27). The SH lighting coefficients (R,G,B channels).
        Returns:
            face_shading: torch.Tensor, (B, N, 3). The shading for each vertex.
        '''
        batch_size = gamma.shape[0]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
            a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device), -a[1] * c[1] * face_norm[..., 1:2],
            a[1] * c[1] * face_norm[..., 2:], -a[1] * c[1] * face_norm[..., :1],
            a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:], 0.5 * a[2] * c[2] / np.sqrt(3.) *
            (3 * face_norm[..., 2:]**2 - 1), -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:], 0.5 * a[2] * c[2] *
            (face_norm[..., :1]**2 - face_norm[..., 1:2]**2)
        ],
                      dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_shading = torch.cat([r, g, b], dim=-1)
        return face_shading

    def compute_rotation(self, angles):
        '''
        Get rotation matrix (internal camera parameters).

        Args:
            angles: torch.Tensor, (B, 3). The estimated angle.
        Returns:
            rot: torch.Tensor, (B, 3, 3). The rotation matrix.
        '''
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).type_as(angles)
        zeros = torch.zeros([batch_size, 1]).type_as(angles)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat(
            [ones, zeros, zeros, zeros,
             torch.cos(x), -torch.sin(x), zeros,
             torch.sin(x), torch.cos(x)], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([torch.cos(y), zeros,
                           torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros,
                           torch.cos(y)],
                          dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat(
            [torch.cos(z), -torch.sin(z), zeros,
             torch.sin(z), torch.cos(z), zeros, zeros, zeros, ones], dim=1).reshape([batch_size, 3, 3])

        rot = (rot_z @ rot_y @ rot_x).permute(0, 2, 1)
        return rot

    def transform(self, face_shape, rot, trans):
        '''
        From world coordinates to camera coordinates according to R, T.

        Args:
            face_shape: torch.Tensor, (B, N, 3). The input vertices in the world coordinate system.
            rot: torch.Tensor, (B, 3, 3). The rotation matrix.
            trans: torch.Tensor, (B, 3). The translation matrix.
        Returns:
            face_shape: torch.Tensor, (B, N, 3). The output vertices in the camera coordinate system.
        '''
        return face_shape @ rot + trans.unsqueeze(1)

    def to_camera(self, face_shape):
        '''
        Move to camera position along side the z-axis
        '''
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        '''
        From camera coordinates to imaging coordinates.

        Args:
            face_shape: torch.Tensor, (B, N, 3). The input vertices in the camera coordinate system.
        Returns:
            face_proj: torch.Tensor, (B, N, 2). The output vertices in the imaging coordinate system.
        '''
        # to image_plane
        face_proj = face_shape @ self.pinhole_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj

    def get_landmarks(self, face_proj):
        '''
        Get landmarks in imaging coordinate system.

        Args:
            face_proj: torch.Tensor, (B, N, 2). The vertices in the imaging coordinate system.
        Returns:
            keypoints: torch.Tensor, (B, 68, 2). The 68 landmark vertices in the imaging coordinate system.
        '''
        return face_proj[:, self.keypoints]

    def compute_for_render(self, coeffs_dict):
        '''
        Compute the 3D (shape, texture, shading, color, landmarks) for differentiable renderer with nvdiffrast.
        
        The 3D information is in camera coordinate system (After camera external parameters R, T).
        But the output landmarks are in the imaging corrdinate system (After pin-hole camera intrinsic parameters).

        Args:
            coeffs_dict: A dict of torch.Tensor, which has keys 'id', 'exp', 'tex', 'angle', 'gamma', 'trans'.
        Returns:
            face_vertex: torch.Tensor, (B, N, 3). The vertices in the camera coordinate system.
            face_texture: torch.Tensor, (B, N, 3). The texture for each vertex.
            face_shading: torch.Tensor, (B, N, 3). The rendered shading for each vertex.
            face_color: torch.Tensor, (B, N, 3). The rendered color for each vertex.
            landmarks: torch.Tensor, (B, 68, 2). The 68 landmarks in the imaging coordinate system.
        '''

        # camera external parameters (R, T)
        rotation = self.compute_rotation(coeffs_dict['angle'])
        translation = coeffs_dict['trans']

        # the vertices in the camera coordinate system
        _, face_shape = self.compute_shape(coeffs_dict['id'], coeffs_dict['exp'])
        face_shape_transformed = self.transform(face_shape, rotation, translation)
        face_vertex = self.to_camera(face_shape_transformed)

        # the render color for each vertex
        face_texture = self.compute_texture(coeffs_dict['tex'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_shading = self.compute_shading(face_norm_roted, coeffs_dict['gamma'])
        face_color = face_shading * face_texture

        # the 68 landmarks in the imaging coordinate system
        face_proj = self.to_image(face_vertex)
        landmarks = self.get_landmarks(face_proj)

        return face_vertex, face_texture, face_shading, face_color, landmarks

    def compute_for_mesh(self, coeffs_dict):
        '''
        Compute the 3D (id/exp shape without pose, albedo texture) for saving mesh obj file.

        Args:
            coeffs_dict: A dict of torch.Tensor, which has keys 'id', 'exp', 'tex'.
        Returns:
            face_id_vertex: torch.Tensor, (B, N, 3). The vertices of id shape.
            face_exp_vertex: torch.Tensor, (B, N, 3). The vertices of exp shape.
            face_alb_texture: torch.Tensor, (B, N, 3). The albedo texture for each vertex.
        '''
        face_id_vertex, face_exp_vertex = self.compute_shape(coeffs_dict['id'], coeffs_dict['exp'])
        face_alb_texture = self.compute_texture(coeffs_dict['tex'])
        return face_id_vertex, face_exp_vertex, face_alb_texture
