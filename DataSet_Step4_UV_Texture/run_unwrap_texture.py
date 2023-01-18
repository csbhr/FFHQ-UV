import os
import time
import numpy as np
import argparse
import torch

from utils import read_img, save_img, np2tensor
from preprocess import Preprocess_API
from face3d_recon import Face3d_Recon_API
from tex import Tex_API


def unwrap_one_texture(preprocess_model, face3d_model, tex_model, left_img_path, front_img_path, right_img_path,
                       device):

    # ---------------------- Step 1. Preprocess ----------------------

    # read image
    left_img = read_img(left_img_path)
    front_img = read_img(front_img_path)
    right_img = read_img(right_img_path)

    # preprocess to get aligned images, 2D 68 landmarks, and face_parsing map
    left_align_img, left_align_hr_img, left_trans_params, \
        left_lm68_2d, left_seg_mask, left_seg_result = preprocess_model(left_img)
    front_align_img, front_align_hr_img, front_trans_params, \
        front_lm68_2d, front_seg_mask, front_seg_result = preprocess_model(front_img)
    right_align_img, right_align_hr_img, right_trans_params, \
        right_lm68_2d, right_seg_mask, right_seg_result = preprocess_model(right_img)

    # Not detect faces
    if left_align_img is None or front_align_img is None or right_align_img is None:
        return None, None, None, None, None, None

    save_lm68_2d = {'left': left_lm68_2d, 'front': front_lm68_2d, 'right': right_lm68_2d}
    save_seg_result = {'left': left_seg_result, 'front': front_seg_result, 'right': right_seg_result}
    save_trans_params = {'left': left_trans_params, 'front': front_trans_params, 'right': right_trans_params}

    # ---------------------- Step 2. Reconstruct 3D shape ----------------------

    # 3D face recon
    left_coeffs = face3d_model.pred_coeffs(np2tensor(left_align_img, device=device))
    front_coeffs = face3d_model.pred_coeffs(np2tensor(front_align_img, device=device))
    right_coeffs = face3d_model.pred_coeffs(np2tensor(right_align_img, device=device))

    save_face3d_coeffs = {'left': left_coeffs, 'front': front_coeffs, 'right': right_coeffs}

    # ---------------------- Step 3. Remap texture & Blend with template ----------------------

    # unwrap texture
    left_projXY, left_norm = face3d_model.compute_224projXY_norm_by_pin_hole(left_coeffs)
    front_projXY, front_norm = face3d_model.compute_224projXY_norm_by_pin_hole(front_coeffs)
    right_projXY, right_norm = face3d_model.compute_224projXY_norm_by_pin_hole(right_coeffs)
    left_projXY, left_norm = left_projXY[0].cpu().numpy(), left_norm[0].cpu().numpy()
    front_projXY, front_norm = front_projXY[0].cpu().numpy(), front_norm[0].cpu().numpy()
    right_projXY, right_norm = right_projXY[0].cpu().numpy(), right_norm[0].cpu().numpy()

    left_projXY = preprocess_model.trans_projXY_back_to_ori_coord(left_projXY, left_trans_params)
    front_projXY = preprocess_model.trans_projXY_back_to_ori_coord(front_projXY, front_trans_params)
    right_projXY = preprocess_model.trans_projXY_back_to_ori_coord(right_projXY, right_trans_params)

    unwrap_uv_tex, save_remap_masks = tex_model(left_img, front_img, right_img, left_seg_mask, front_seg_mask,
                                                right_seg_mask, left_projXY, front_projXY, right_projXY, left_norm,
                                                front_norm, right_norm)

    return unwrap_uv_tex, save_lm68_2d, save_seg_result, save_trans_params, save_face3d_coeffs, save_remap_masks


def main(opt):

    # ----------------------- Define Inference Parameters -----------------------
    input_images_dir = os.path.join(opt.proj_data_dir, opt.input_folder)
    output_texture_dir = os.path.join(opt.proj_data_dir, opt.save_folder)
    device = opt.device
    os.makedirs(output_texture_dir, exist_ok=True)

    # ----------------------- Load models -----------------------

    # init preprocess model
    preprocess_model = Preprocess_API(lm_detector_path=os.path.join(opt.ckp_dir, 'lm_model/68lm_detector.pb'),
                                      mtcnn_path=os.path.join(opt.ckp_dir, 'mtcnn_model/mtcnn_model.pb'),
                                      lm68_3d_path=os.path.join(opt.topo_dir, 'similarity_Lm3D_all.mat'),
                                      parsing_pth=os.path.join(opt.ckp_dir, 'parsing_model/79999_iter.pth'),
                                      resnet18_path=os.path.join(opt.ckp_dir, 'resnet_model/resnet18-5c106cde.pth'),
                                      target_size=224,
                                      rescale_factor=102.,
                                      device=device)

    # init face3d model
    face3d_model = Face3d_Recon_API(pfm_model_path=os.path.join(opt.topo_dir, 'hifi3dpp_model_info.mat'),
                                    recon_model_path=os.path.join(opt.ckp_dir, 'deep3d_model/epoch_latest.pth'),
                                    focal=opt.focal,
                                    camera_distance=opt.camera_distance,
                                    device=device)

    # init unwrap texture model
    tex_model = Tex_API(base_uv_path=os.path.join(opt.topo_dir, 'template_base_uv.png'),
                        facial_base_uv_path=os.path.join(opt.topo_dir, 'template_base_uv.png'),
                        unwrap_info_path=os.path.join(opt.topo_dir, 'unwrap_1024_info.mat'),
                        unwrap_info_mask_path=os.path.join(opt.topo_dir, 'unwrap_1024_info_mask.png'),
                        mouth_mask_path=os.path.join(opt.topo_dir, 'mouth_constract_mask.png'),
                        hair_mask_path=os.path.join(opt.topo_dir, 'hair_mask.png'),
                        nosal_base_mask_path=os.path.join(opt.topo_dir, 'nosal_base_mask.png'),
                        nostril_mask_path=os.path.join(opt.topo_dir, 'nostril_mask.png'),
                        center_face_mask_path=os.path.join(opt.topo_dir, 'center_face_mask.png'),
                        major_front_mask_path=os.path.join(opt.topo_dir, 'major_valid_front_mask.png'),
                        major_left_mask_path=os.path.join(opt.topo_dir, 'major_valid_left_mask.png'),
                        major_right_mask_path=os.path.join(opt.topo_dir, 'major_valid_right_mask.png'),
                        minor_front_mask_path=os.path.join(opt.topo_dir, 'minor_valid_front_mask.png'),
                        minor_left_mask_path=os.path.join(opt.topo_dir, 'minor_valid_left_mask.png'),
                        minor_right_mask_path=os.path.join(opt.topo_dir, 'minor_valid_right_mask.png'),
                        unwrap_size=1024)

    # ----------------------- Start Inference -----------------------

    failed_list = []

    img_folder_names = sorted(os.listdir(input_images_dir))
    for fname in img_folder_names:

        left_img_path = os.path.join(input_images_dir, fname, f'{fname}_left.png')
        front_img_path = os.path.join(input_images_dir, fname, f'{fname}_front.png')
        right_img_path = os.path.join(input_images_dir, fname, f'{fname}_right.png')

        tic = time.time()

        uv_result, lm68_2d, seg_result, trans_params, face3d_coeffs, remap_masks = unwrap_one_texture(
            preprocess_model, face3d_model, tex_model, left_img_path, front_img_path, right_img_path, device)

        if uv_result is None:
            print(f'>> Unwrap Texture Failed: {fname}')
            failed_list.append(fname)
            continue

        out_folder = os.path.join(output_texture_dir, fname)
        os.makedirs(out_folder, exist_ok=True)

        # save 68 landmarks, parsing results, remap texture masks, and 3DMM coefficient
        torch.save(lm68_2d, os.path.join(out_folder, f'{fname}_final_lm68_2d.pt'))
        save_img(np.stack([seg_result['left'], seg_result['front'], seg_result['right']], axis=-1),
                 os.path.join(out_folder, f'{fname}_final_seg_result.png'))
        torch.save(trans_params, os.path.join(out_folder, f'{fname}_final_trans_params.pt'))
        torch.save(face3d_coeffs, os.path.join(out_folder, f'{fname}_final_coeffs.pt'))
        save_img(np.stack([remap_masks['left'], remap_masks['front'], remap_masks['right']], axis=-1),
                 os.path.join(out_folder, f'{fname}_final_remap_masks.png'),
                 src_range=1.0)

        # save uv texture
        save_img(uv_result, os.path.join(out_folder, f'{fname}_final_uv.png'))

        toc = time.time()
        print('Unwrap texture {} took {:.4f} seconds.'.format(fname, toc - tic))

    print('Unwrap texture process done!')
    print(f'>> Failed List: {str(failed_list)}')


if __name__ == '__main__':
    '''Usage
    cd ./DataSet_Step4_UV_Texture
    python run_unwrap_texture.py \
        --proj_data_dir ../examples/dataset_examples \
        --ckp_dir ../checkpoints \
        --topo_dir ../topo_assets
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--proj_data_dir',
        type=str,
        required=True,
        help="The directory of the project data, which should inculde '$input_folder' sub-directory.",
    )
    parser.add_argument(
        '--ckp_dir',
        type=str,
        default='../checkpoints',
        help='folder for checkpoints.',
    )
    parser.add_argument(
        '--topo_dir',
        type=str,
        default='../assets_topo',
        help='folder for assets of topo.',
    )
    parser.add_argument("--input_folder", type=str, default='edit', help="The name of the input folder.")
    parser.add_argument("--save_folder", type=str, default='unwrap_texture', help="The name of the saved folder.")
    parser.add_argument("--device", type=str, default='cuda', help="The device, optional: cup/cuda.")
    parser.set_defaults(
        focal=1015.0,
        camera_distance=10.0,
    )
    opt = parser.parse_args()

    main(opt)
