#!/bin/bash


######################### Configuration #########################
# latent_dir: the directory of the download face latents
# save_face_dir: the directory for saving the generated face images
# stylegan_network_pkl: the checkpoint of StyleGAN2 can be download from http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl
#################################################################
latent_dir=../examples/face_latent_examples/face_latents
save_face_dir=../examples/face_latent_examples/face_images
stylegan_network_pkl=../checkpoints/stylegan_model/stylegan2-ffhq-config-f.pkl


########################## Run Script ###########################
# Read the face latents in ${latent_dir}
# Save the generated face images in ${save_face_dir}
#################################################################
cd ./DataSet_Step3_Editing
python gen_face_from_latent.py \
    --latent_dir ${latent_dir} \
    --save_face_dir ${save_face_dir} \
    --stylegan_network_pkl ${stylegan_network_pkl}
