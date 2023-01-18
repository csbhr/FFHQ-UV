import os
import time
import argparse
import dlib
import torch
import torchvision.transforms as transforms

from utils.common import tensor2im
from utils.alignment import align_face
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.


def run_alignment(image_path, shape_predictor_model_path):
    predictor = dlib.shape_predictor(shape_predictor_model_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image


def det_num_faces(image_path):
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(image_path)
    dets = detector(img, 1)
    return len(dets)


def batch_inversion(args):
    # ----------------------- Define Inference Parameters -----------------------
    e4e_model_path = args.e4e_model_path
    shape_predictor_model_path = args.shape_predictor_model_path
    input_images_dir = os.path.join(args.proj_data_dir, 'images')
    output_latents_dir = os.path.join(args.proj_data_dir, 'latents')
    output_inversions_dir = os.path.join(args.proj_data_dir, 'inversions')
    device = args.device

    os.makedirs(output_latents_dir, exist_ok=True)
    os.makedirs(output_inversions_dir, exist_ok=True)

    # ----------------------- Load Pretrained Model -----------------------
    ckpt = torch.load(e4e_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = e4e_model_path
    opts = argparse.Namespace(**opts)
    net = pSp(opts).eval().to(device)

    # ----------------------- Setup Data Transformations -----------------------
    image_transformer = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # ----------------------- Perform Inversion -----------------------
    fnames = sorted(os.listdir(input_images_dir))
    for fn in fnames:
        basename = fn[:fn.rfind('.')]

        tic = time.time()

        # detect if the image contains faces
        if det_num_faces(os.path.join(input_images_dir, fn)) < 1:
            print(f'>> Skip {fn}, there is no face detected in this image!')
            continue

        # align face
        input_image = run_alignment(image_path=os.path.join(input_images_dir, fn),
                                    shape_predictor_model_path=shape_predictor_model_path)

        # preprocess image
        transformed_image = image_transformer(input_image)

        # inversion
        images, latents = net(transformed_image.unsqueeze(0).to(device).float(),
                              randomize_noise=False,
                              resize=False,
                              return_latents=True)

        torch.save(latents, os.path.join(output_latents_dir, f"{basename}.pt"))
        tensor2im(images[0]).save(os.path.join(output_inversions_dir, f"{basename}.png"))

        # detect if the inversed image contains faces
        if det_num_faces(os.path.join(output_inversions_dir, f"{basename}.png")) < 1:
            os.unlink(os.path.join(output_latents_dir, f"{basename}.pt"))
            os.unlink(os.path.join(output_inversions_dir, f"{basename}.png"))
            print(f'>> Drop {fn}, there is no face detected in the inversed image!')

        toc = time.time()
        print('Inverse {} done, took {:.4f} seconds.'.format(fn, toc - tic))

    print('Inversion process done!')


if __name__ == "__main__":
    '''Usage
    cd ./DataSet_Step1_Inversion
    python run_e4e_inversion.py \
        --proj_data_dir ../examples/dataset_examples \
        --e4e_model_path ../checkpoints/e4e_model/e4e_ffhq_encode.pt \
        --shape_predictor_model_path ../checkpoints/dlib_model/shape_predictor_68_face_landmarks.dat
    '''
    parser = argparse.ArgumentParser(description="e4e_inversion")
    parser.add_argument(
        "--proj_data_dir",
        type=str,
        required=True,
        help="The directory of the project data, which should inculde 'images' sub-directory.",
    )
    parser.add_argument(
        "--e4e_model_path",
        type=str,
        default='../checkpoints/e4e_model/e4e_ffhq_encode.pt',
        help="The path of the pretrained e4e model.",
    )
    parser.add_argument(
        "--shape_predictor_model_path",
        type=str,
        default='../checkpoints/dlib_model/shape_predictor_68_face_landmarks.dat',
        help="The path of the dlib shape predictor model (68 face landmarks).",
    )
    parser.add_argument("--device", type=str, default='cuda', help="The device, optional: cup/cuda.")
    args = parser.parse_args()

    batch_inversion(args)
