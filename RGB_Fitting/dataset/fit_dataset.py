import torch
import numpy as np
from scipy.io import loadmat

from utils.data_utils import read_img, img3channel, img2mask, np2pillow, pillow2np, np2tensor
from utils.preprocess_utils import align_img, estimate_norm
from third_party import Landmark68_API, SkinMask_API, FaceParsing_API


class FitDataset:

    def __init__(self, lm_detector_path, mtcnn_detector_path, parsing_model_pth, parsing_resnet18_path, lm68_3d_path,
                 batch_size, device):
        self.lm68_model = Landmark68_API(lm_detector_path=lm_detector_path, mtcnn_path=mtcnn_detector_path)
        self.skin_model = SkinMask_API()
        self.parsing_model = FaceParsing_API(parsing_pth=parsing_model_pth,
                                             resnet18_path=parsing_resnet18_path,
                                             device=device)
        self.lm68_3d = loadmat(lm68_3d_path)['lm']
        self.batch_size = batch_size
        self.device = device

    def get_input_data(self, img_path):
        with torch.no_grad():
            input_img = read_img(img_path)
            raw_img = np2pillow(input_img)

            # detect 68 landmarks
            raw_lm = self.lm68_model(input_img)
            if raw_lm is None:
                return None
                
            raw_lm = raw_lm.astype(np.float32)

            # calculate skin attention mask
            raw_skin_mask = self.skin_model(input_img, return_uint8=True)
            raw_skin_mask = img3channel(raw_skin_mask)
            raw_skin_mask = np2pillow(raw_skin_mask)

            # face parsing mask
            require_part = ['face', 'l_eye', 'r_eye', 'mouth']
            seg_mask_dict, _ = self.parsing_model(input_img, require_part=require_part)
            face_mask = seg_mask_dict['face']
            ex_mouth_mask = 1 - seg_mask_dict['mouth']
            ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
            raw_parse_mask = face_mask * ex_mouth_mask * ex_eye_mask
            raw_parse_mask = np2pillow(raw_parse_mask, src_range=1.0)

            # alignment
            trans_params, img, lm, skin_mask, parse_mask = align_img(raw_img, raw_lm, self.lm68_3d, raw_skin_mask,
                                                                     raw_parse_mask)

            # to tensor
            _, H = img.size
            M = estimate_norm(lm, H)
            img_tensor = np2tensor(pillow2np(img), device=self.device)
            skin_mask_tensor = np2tensor(pillow2np(skin_mask), device=self.device)[:, :1, :, :]
            parse_mask_tensor = np2tensor(pillow2np(parse_mask), device=self.device)[:, :1, :, :]
            lm_tensor = torch.tensor(np.array(lm).astype(np.float32)).unsqueeze(0).to(self.device)
            M_tensor = torch.tensor(np.array(M).astype(np.float32)).unsqueeze(0).to(self.device)

            return {
                'img': img_tensor,
                'skin_mask': skin_mask_tensor,
                'parse_mask': parse_mask_tensor,
                'lm': lm_tensor,
                'M': M_tensor,
                'trans_params': trans_params
            }
