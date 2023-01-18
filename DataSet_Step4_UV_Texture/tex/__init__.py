import cv2
import numpy as np
from scipy.io import loadmat
import skimage

from utils import read_img, img2mask
from .tex_func import remap_tex_from_input2D, fill_facial_region, blend_with_template
from .poisson_blend import poisson_blend, get_laplacian_matrices


class Tex_API:

    def __init__(self,
                 base_uv_path,
                 facial_base_uv_path,
                 unwrap_info_path,
                 unwrap_info_mask_path,
                 mouth_mask_path,
                 hair_mask_path,
                 center_face_mask_path,
                 nosal_base_mask_path,
                 nostril_mask_path,
                 major_front_mask_path,
                 major_left_mask_path,
                 major_right_mask_path,
                 minor_front_mask_path,
                 minor_left_mask_path,
                 minor_right_mask_path,
                 unwrap_size=1024):
        '''
        Args:
            base_uv_path: str. The file path of template UV map for other region.
            facial_base_uv_path: str. The file path of template UV map for facial invisible region.
            unwrap_info_path: str. The file path of unwrap information.
            unwrap_info_mask_path: str. The file path of unwrap mask.
            mouth_mask_path: str. The file path of mouth mask.
            hair_mask_path: str. The file path of hair mask.
            center_face_mask_path: str. The file path of center face mask.
            nosal_base_mask_path: str. The file path of nosal base mask.
            nostril_mask_path: str. The file path of nostril mask.
            major_front_mask_path: str. The file path of major front valid region mask.
            major_left_mask_path: str. The file path of major left valid region mask.
            major_right_mask_path: str. The file path of major right valid region mask.
            minor_front_mask_path: str. The file path of minor front valid region mask.
            minor_left_mask_path: str. The file path of minor left valid region mask.
            minor_right_mask_path: str. The file path of minor right valid region mask.
            unwrap_size: int. The image size of unwrap texture map.
        '''

        assert unwrap_size == 1024

        # template UV map
        self.base_uv = read_img(base_uv_path, resize=(unwrap_size, unwrap_size))
        self.facial_base_uv = read_img(facial_base_uv_path, resize=(unwrap_size, unwrap_size))

        # unwrap information
        unwrap_info = loadmat(unwrap_info_path)
        self.unwrap_uv_idx_bw = unwrap_info['uv_idx_bw'].astype(np.float32)
        self.unwrap_uv_idx_v_idx = unwrap_info['uv_idx_v_idx'].astype(np.float32)
        self.unwrap_info_mask = read_img(unwrap_info_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.unwrap_uv_idx_bw = self.unwrap_uv_idx_bw * self.unwrap_info_mask
        self.unwrap_uv_idx_v_idx = self.unwrap_uv_idx_v_idx * self.unwrap_info_mask

        # mouth, hair, center face, nosal base region
        self.mouth_mask = read_img(mouth_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.hair_mask = read_img(hair_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.center_face_mask = read_img(center_face_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.nosal_base_mask = read_img(nosal_base_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.nostril_mask = read_img(nostril_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)

        # major valid region
        self.major_front_mask = read_img(major_front_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.major_left_mask = read_img(major_left_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.major_right_mask = read_img(major_right_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.major_whole_mask = (self.major_front_mask + self.major_left_mask + self.major_right_mask > 0.5).astype(
            np.float32)

        # minor valid region
        self.minor_front_mask = read_img(minor_front_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.minor_left_mask = read_img(minor_left_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.minor_right_mask = read_img(minor_right_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.minor_whole_mask = (self.minor_front_mask + self.minor_left_mask + self.minor_right_mask > 0.5).astype(
            np.float32)

        # for fill eyes and mouth using poisson blending
        self.unreliable_coord_poisson = {
            'l_eye': (330, 400, 560, 710),
            'r_eye': (330, 400, 310, 460),
            'nosal_base': (465, 525, 440, 585),
            'mouth': (555, 620, 410, 610)
        }
        self.unreliable_laplacian_matrices = {}
        for k in self.unreliable_coord_poisson.keys():
            coo = self.unreliable_coord_poisson[k]
            self.unreliable_laplacian_matrices[k] = get_laplacian_matrices(coo[1] - coo[0], coo[3] - coo[2])

        self.unwrap_size = unwrap_size

    def solve_unreliable_parts_using_poisson(self, input_tex, unrely_mask):
        '''
        Solve unreliable parts (eyes and mouth) using poisson blending.

        Args:
            input_tex: numpy.array (unwrap_size, unwrap_size, 3). The input UV texture map.
            unrely_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of unreliable texture.
        Returns:
            rely_tex: numpy.array (unwrap_size, unwrap_size, 3). The solved reliable UV texture map.
            solve_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of region that solved by poisson.
        '''
        solve_mask = np.zeros_like(unrely_mask)
        rely_tex = input_tex.copy()
        for k in self.unreliable_coord_poisson.keys():
            coo = self.unreliable_coord_poisson[k]
            tex_part = input_tex[coo[0]:coo[1], coo[2]:coo[3]]
            fill_part = self.facial_base_uv[coo[0]:coo[1], coo[2]:coo[3]]

            # prevent the mask from reaching the boundary
            bl = 2
            mask_part_ori = unrely_mask[coo[0]:coo[1], coo[2]:coo[3]]
            mask_part = np.zeros_like(mask_part_ori)
            mask_part[bl:-bl, bl:-bl] = mask_part_ori[bl:-bl, bl:-bl]

            identity_mat, laplacian_mat = self.unreliable_laplacian_matrices[k]
            res_part = poisson_blend(template_img=tex_part,
                                     input_img=fill_part,
                                     mask=mask_part,
                                     laplacian_mat=laplacian_mat,
                                     identity_mat=identity_mat)
            rely_tex[coo[0]:coo[1], coo[2]:coo[3]] = res_part
            solve_mask[coo[0]:coo[1], coo[2]:coo[3]] = np.ones_like(solve_mask[coo[0]:coo[1], coo[2]:coo[3]])
        return rely_tex, solve_mask

    def __call__(self, left_img, front_img, right_img, left_seg_mask, front_seg_mask, right_seg_mask, left_projXY,
                 front_projXY, right_projXY, left_norm, front_norm, right_norm):
        '''
        Unwrap UV texture from input 2D image.

        Args:
            left_img: numpy.array (h, w, 3). The left input 2D image.
            front_img: numpy.array (h, w, 3). The front input 2D image.
            right_img: numpy.array (h, w, 3). The right input 2D image.
            left_seg_mask: numpy.array (h, w, 3). The parsing mask of left facial parts (without eyes and mouth).
            front_seg_mask: numpy.array (h, w, 3). The parsing mask of front facial parts (without eyes and mouth).
            right_seg_mask: numpy.array (h, w, 3). The parsing mask of right facial parts (without eyes and mouth).
            left_projXY: numpy.array (N, 2). The project XY coordinates (h x w) for each vertex of left face.
            front_projXY: numpy.array (N, 2). The project XY coordinates (h x w) for each vertex of front face.
            right_projXY: numpy.array (N, 2). The project XY coordinates (h x w) for each vertex of right face.
            left_norm: numpy.array (N, 3). The normal vector for each vertex of left face.
            front_norm: numpy.array (N, 3). The normal vector for each vertex of front face.
            right_norm: numpy.array (N, 3). The normal vector for each vertex of right face.
        Returns:
            unwrap_uv_tex: numpy.array (unwrap_size, unwrap_size, 3). The unwarpped UV texture map.
        '''

        # remap texture from input 2D image to UV map
        left_remap_tex, left_remap_mask = remap_tex_from_input2D(input_img=left_img,
                                                                 seg_mask=left_seg_mask,
                                                                 projXY=left_projXY,
                                                                 norm=left_norm,
                                                                 unwrap_uv_idx_v_idx=self.unwrap_uv_idx_v_idx,
                                                                 unwrap_uv_idx_bw=self.unwrap_uv_idx_bw)
        front_remap_tex, front_remap_mask = remap_tex_from_input2D(input_img=front_img,
                                                                   seg_mask=front_seg_mask,
                                                                   projXY=front_projXY,
                                                                   norm=front_norm,
                                                                   unwrap_uv_idx_v_idx=self.unwrap_uv_idx_v_idx,
                                                                   unwrap_uv_idx_bw=self.unwrap_uv_idx_bw)
        right_remap_tex, right_remap_mask = remap_tex_from_input2D(input_img=right_img,
                                                                   seg_mask=right_seg_mask,
                                                                   projXY=right_projXY,
                                                                   norm=right_norm,
                                                                   unwrap_uv_idx_v_idx=self.unwrap_uv_idx_v_idx,
                                                                   unwrap_uv_idx_bw=self.unwrap_uv_idx_bw)
        save_remap_masks = {
            'left': left_remap_mask[..., 0],
            'front': front_remap_mask[..., 0],
            'right': right_remap_mask[..., 0]
        }

        # fill facial invisible region: left -> right -> front
        fill_tex, left_fill_mask = fill_facial_region(template_tex=self.facial_base_uv,
                                                      input_tex=left_remap_tex,
                                                      tex_mask=left_remap_mask,
                                                      major_valid_mask=self.major_left_mask,
                                                      minor_valid_mask=self.minor_left_mask,
                                                      mouth_mask=self.mouth_mask)
        fill_tex, right_fill_mask = fill_facial_region(template_tex=fill_tex,
                                                       input_tex=right_remap_tex,
                                                       tex_mask=right_remap_mask,
                                                       major_valid_mask=self.major_right_mask,
                                                       minor_valid_mask=self.minor_right_mask,
                                                       mouth_mask=self.mouth_mask)
        fill_tex, front_fill_mask = fill_facial_region(template_tex=fill_tex,
                                                       input_tex=front_remap_tex,
                                                       tex_mask=front_remap_mask,
                                                       major_valid_mask=self.major_front_mask,
                                                       minor_valid_mask=self.minor_front_mask,
                                                       mouth_mask=self.mouth_mask)
        fill_mask = img2mask(left_fill_mask + right_fill_mask + front_fill_mask, thre=0.5)

        # calculate unreliable parts mask
        # from fill_mask, can obtain the incompleted eyes, mouth, and nosal base region
        unrely_mask = (1 - fill_mask) * self.center_face_mask
        # for nosal base region, the nostril has double vision, which need to detect the nostril region
        fill_tex_yuv = skimage.color.convert_colorspace(fill_tex, "rgb", "yuv")
        tex_dark_mask = img2mask(np.mean(fill_tex_yuv[:, :, 0]) - fill_tex_yuv[:, :, 0], thre=0.0)
        tex_dark_mask_dilate = img2mask(cv2.blur(tex_dark_mask, (5, 5), 0), thre=0.0)
        unrely_nostril_mask = tex_dark_mask_dilate * self.nosal_base_mask
        unrely_mask = img2mask(unrely_mask + unrely_nostril_mask + self.nostril_mask, thre=0.5)

        # solve unreliable parts (eyes, mouth, and nosal base)
        rely_tex, solve_mask = self.solve_unreliable_parts_using_poisson(fill_tex, unrely_mask)
        rely_mask = img2mask(fill_mask + solve_mask, thre=0.5)

        # blend with template
        uv_tex_result = blend_with_template(template_tex=self.base_uv,
                                            input_tex=rely_tex,
                                            tex_mask=rely_mask,
                                            major_valid_mask=self.major_whole_mask,
                                            minor_valid_mask=self.minor_whole_mask,
                                            hair_mask=self.hair_mask)

        return uv_tex_result, save_remap_masks
