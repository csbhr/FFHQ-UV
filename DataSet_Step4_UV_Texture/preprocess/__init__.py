import cv2
import numpy as np
from scipy.io import loadmat

from third_party import Landmark68_API, FaceParsing_API
from utils import np2pillow, pillow2np, img2mask
from .preprocess_func import (extract_lm5_from_lm68, POS, resize_crop_img, resize_crop_img_retain_hr,
                              trans_projXY_back_to_ori_coord)


class Preprocess_API:

    def __init__(self,
                 lm_detector_path,
                 mtcnn_path,
                 lm68_3d_path,
                 parsing_pth,
                 resnet18_path,
                 target_size=224,
                 rescale_factor=102.,
                 device='cuda'):
        '''
        Args:
            lm_detector_path: str. The pretrained landmark detector model.
            mtcnn_path: str. The pretrained mtcnn model.
            lm68_3d_path: str. The file path of pre-saved 3D 68 landmarks.
            parsing_pth: str. The pretrained face parsing model.
            resnet18_path: str. The pretrained resnet18 model.
            target_size: int. The size of the target output image.
            rescale_factor: float. The rescale factor.
            device: str. The device.
        '''

        self.lm68_model = Landmark68_API(lm_detector_path=lm_detector_path, mtcnn_path=mtcnn_path)
        self.lm68_3d = loadmat(lm68_3d_path)['lm']
        self.parsing_model = FaceParsing_API(parsing_pth=parsing_pth, resnet18_path=resnet18_path, device=device)
        self.target_size = target_size
        self.rescale_factor = rescale_factor

    def __call__(self, input_img):
        '''
        Preprocess image for 3D face reconstruction.

        Args:
            input_img: numpy.array, float, (h, w, 3). The input image.
        Returns:
            tar_img: numpy.array, float, (target_size, target_size, 3). The aligned image.
            hr_img: numpy.array, float, (h, w, 3). The aligned HR image.
            trans_params: numpy.array, (5). Contains w0, h0, s, t0, t1, target_size.
            lm68_2d: numpy.array, float, (68, 2). The coordinates of 68 landmarks (on h x w range).
            seg_mask: numpy.array, float, (h, w, 3). The mask of face which excludes eyes and mouth.
        '''
        # detect 68 landmarks
        lm68_2d = self.lm68_model(input_img)
        # Not detect faces
        if lm68_2d is None:
            return None, None, None, None, None, None
        lm68_2d = lm68_2d.astype(np.float32)

        # face parsing
        require_part = ['face', 'l_eye', 'r_eye', 'mouth']
        seg_mask_dict, seg_result = self.parsing_model(input_img, require_part=require_part)
        face_mask = seg_mask_dict['face']

        # exclude eyes and mouth, where the mask of eyes is dilated
        ex_mouth_mask = 1 - seg_mask_dict['mouth']
        ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
        ex_eye_mask_erode = cv2.blur(ex_eye_mask, (11, 11), 0)
        ex_eye_mask_erode = img2mask(ex_eye_mask_erode, thre=1., mode='greater-equal')
        seg_mask = face_mask * ex_mouth_mask * ex_eye_mask_erode

        input_img = np2pillow(input_img)
        w0, h0 = input_img.size

        # extract 5 landmarks from 68 landmarks
        lm5_2d = extract_lm5_from_lm68(lm68_2d)
        lm5_3d = extract_lm5_from_lm68(self.lm68_3d)

        # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
        t, s = POS(lm5_2d, lm5_3d)
        s = self.rescale_factor / s
        trans_params = np.array([w0, h0, s, t[0], t[1], self.target_size])

        # processing the image
        tar_img = resize_crop_img(input_img, trans_params)
        hr_img = resize_crop_img_retain_hr(input_img, trans_params)

        # PIL.Image -> numpy.array
        tar_img = pillow2np(tar_img)
        hr_img = pillow2np(hr_img)

        return tar_img, hr_img, trans_params, lm68_2d, seg_mask, seg_result

    def trans_projXY_back_to_ori_coord(self, projXY, trans_params):
        ''' Transfer project XY coordinates from (224x224) back to (w0 x h0).

        Args:
            projXY: numpy.array, (N, 2). The project XY coordinates (224x224).
            trans_params: numpy.array, (6). Contains w0, h0, s, t0, t1, target_size.
        Returns:
            projXY_ori: numpy.array, (N, 2). The project XY coordinates (w0 x h0).
        '''
        return trans_projXY_back_to_ori_coord(projXY, trans_params)
