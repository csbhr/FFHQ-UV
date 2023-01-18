import os
import sys
import time
import numpy as np
from PIL import Image
from argparse import Namespace
import torch
import torchvision.transforms as transforms
import dlib

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from utils.alignment import align_face
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.
from editings import latent_editor


def run_alignment(image_path, shape_predictor_model_path):
    predictor = dlib.shape_predictor(shape_predictor_model_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def display_alongside_source_image(result_image, source_image, resize_dims):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))],
                         axis=1)
    return Image.fromarray(res)


def demo():
    #----------------------- Define Inference Parameters -----------------------
    experiment_type = 'ffhq_encode'  # ['ffhq_encode', 'cars_encode', 'horse_encode', 'church_encode']
    model_path = "pretrained_models/e4e_ffhq_encode.pt"
    shape_predictor_model_path = "pretrained_models/shape_predictor_68_face_landmarks.dat"
    image_path = "data/images/input_img.jpg"
    output_path = "data/demo_output"

    os.makedirs(output_path, exist_ok=True)

    #----------------------- Load Pretrained Model -----------------------
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts).eval().cuda()
    print('Model successfully loaded!')

    #----------------------- Setup Data Transformations -----------------------
    image_transformer = None
    if experiment_type == 'cars_encode':
        image_transformer = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        resize_dims = (256, 192)
    else:
        image_transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        resize_dims = (256, 256)

    #----------------------- Preprocess Input Image -----------------------
    if experiment_type == "ffhq_encode":
        input_image = run_alignment(image_path=image_path, shape_predictor_model_path=shape_predictor_model_path)
    else:
        input_image = Image.open(image_path).convert("RGB")
    input_image.resize(resize_dims)
    input_image.save(os.path.join(output_path, "input_image.png"))

    transformed_image = image_transformer(input_image)

    #----------------------- Perform Inference -----------------------
    with torch.no_grad():
        tic = time.time()
        images, latents = net(transformed_image.unsqueeze(0).to("cuda").float(),
                              randomize_noise=False,
                              return_latents=True)
        if experiment_type == 'cars_encode':
            images = images[:, :, 32:224, :]
        result_image, latent = images[0], latents[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    os.makedirs(os.path.join(output_path, 'inference'), exist_ok=True)
    display_result_image = display_alongside_source_image(tensor2im(result_image), input_image, resize_dims)
    display_result_image.save(os.path.join(output_path, 'inference', "inference_image.png"))
    torch.save(latents, os.path.join(output_path, 'inference', "inference_latents.pt"))

    #----------------------- Perform Editing -----------------------
    is_cars = experiment_type == 'cars_encode'
    editor = latent_editor.LatentEditor(net.decoder, is_cars)

    os.makedirs(os.path.join(output_path, 'editing'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'editing', 'interfacegan'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'editing', 'ganspace'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'editing', 'sefa'), exist_ok=True)

    #--------- InterFaceGAN ---------
    interfacegan_directions = {
        'ffhq_encode': {
            'age': 'editings/interfacegan_directions/age.pt',
            'smile': 'editings/interfacegan_directions/smile.pt',
            'pose': 'editings/interfacegan_directions/pose.pt'
        }
    }
    available_interfacegan_directions = None
    if experiment_type in interfacegan_directions:  # List supported directions for the current experiment
        available_interfacegan_directions = interfacegan_directions[experiment_type]
        print(list(available_interfacegan_directions.keys()))
    # As an example, we currently released the age and smile directions for the FFHQ StyleGAN Generator.
    interfacegan_direction = torch.load(available_interfacegan_directions["age"]).cuda()
    # For a single edit:
    result = editor.apply_interfacegan(latents, interfacegan_direction, factor=-3).resize(resize_dims)
    display_result_image = display_alongside_source_image(result, input_image, resize_dims)
    display_result_image.save(os.path.join(output_path, 'editing', 'interfacegan', "single_edit_age.png"))
    # For a range of editings
    display_result_image = editor.apply_interfacegan(latents, interfacegan_direction, factor_range=(-5, 5))
    display_result_image.save(os.path.join(output_path, 'editing', 'interfacegan', "range_edit_age.png"))

    #--------- GANSpace ---------
    # Here we provide the editings for the cars domain as displayed in the paper, as well as several examples for the facial domain,
    # taken from the official GANSpace repository.
    if experiment_type == 'ffhq_encode':
        ganspace_pca = torch.load('editings/ganspace_pca/ffhq_pca.pt')
        directions = {
            'eye_openness': (54, 7, 8, 20),
            'smile': (46, 4, 5, -20),
            'trimmed_beard': (58, 7, 9, 20),
            'white_hair': (57, 7, 10, -24),
            'lipstick': (34, 10, 11, 20)
        }
    elif experiment_type == 'cars_encode':
        ganspace_pca = torch.load('editings/ganspace_pca/cars_pca.pt')
        directions = {
            "Viewpoint I": (0, 0, 5, 2),
            "Viewpoint II": (0, 0, 5, -2),
            "Cube": (16, 3, 6, 25),
            "Color": (22, 9, 11, -8),
            "Grass": (41, 9, 11, -18),
        }
    print(f'Available Editings: {list(directions.keys())}')
    display_result_image = editor.apply_ganspace(
        latents, ganspace_pca, [directions["white_hair"], directions["eye_openness"], directions["smile"]])
    display_result_image.save(os.path.join(output_path, 'editing', 'ganspace', "edited.png"))

    #--------- SeFa ---------
    # Note that each model behaves differently to the selected editing parameters.
    # We encourage the user to try out different configurations, using different indices, start/end_distance, etc.
    # In the paper, we used start and end distance of -15.0, +15.0 over the horses and churches domains.
    # See code at editings/sefa.py for further options.
    display_result_image = editor.apply_sefa(latents,
                                             indices=[2, 3, 4, 5],
                                             start_distance=0.,
                                             end_distance=15.0,
                                             step=3)
    display_result_image.save(os.path.join(output_path, 'editing', 'sefa', "edited.png"))

    #--------- Styleflow ---------
    # Note, that for Styleflow editings,
    # one need to save the output latent codes and load them over the official StyleFlow repository:
    # torch.save(latents, 'latents.pt')


if __name__ == "__main__":
    demo()
