#!/bin/bash


######################### Configuration #########################
# proj_data_dir: the directory of the dataset project, which contains a "images" sub-folder
# checkpoints_dir: the directory of the used checkpoints
# topo_assets_dir: the directory of the topo assets, e.g., 3DMM, masks, etc.
#################################################################
proj_data_dir=../examples/dataset_examples
checkpoints_dir=../checkpoints
topo_assets_dir=../topo_assets


####################### Step 1. Inversion #######################
# Read the original images in "images" sub-folder
# Save the inverted latents in "latents" sub-folder
# Save the inverted images in "inversions" sub-folder
#################################################################
cd ./DataSet_Step1_Inversion
python run_e4e_inversion.py \
    --proj_data_dir ${proj_data_dir} \
    --e4e_model_path ${checkpoints_dir}/e4e_model/e4e_ffhq_encode.pt \
    --shape_predictor_model_path ${checkpoints_dir}/dlib_model/shape_predictor_68_face_landmarks.dat


################### Step 2. Detect Attributes ###################
# Read the inverted images in "inversions" sub-folder
# Save the detected light attributes in "lights" sub-folder
# Save the detected other attributes in "attributes" and "attributes_ms_api" sub-folders
#################################################################
cd ../DataSet_Step2_Det_Attributes
python run_dpr_light.py \
    --proj_data_dir ${proj_data_dir} \
    --dpr_model_path ${checkpoints_dir}/dpr_model/trained_model_03.t7
python run_ms_api_attr.py \
    --proj_data_dir ${proj_data_dir}


##################### Step 3. Face Editing ######################
# Read the inverted latents in "latents" sub-folder
# Read the detected attributes in "lights" and "attributes" sub-folders
# Save the edited multi-view face images and latents in "edit" sub-folder
#################################################################
cd ../DataSet_Step3_Editing
python run_styleflow_editing.py \
    --proj_data_dir ${proj_data_dir} \
    --network_pkl ${checkpoints_dir}/stylegan_model/stylegan2-ffhq-config-f.pkl \
    --flow_model_path ${checkpoints_dir}/styleflow_model/modellarge10k.pt \
    --exp_direct_path ${checkpoints_dir}/styleflow_model/expression_direction.pt \
    --exp_recognition_path ${checkpoints_dir}/exprecog_model/FacialExpRecognition_model.t7 \
    --edit_items delight,norm_attr,multi_yaw


################### Step 4. Unwrap UV Texture ###################
# Read the edited multi-view face images in "edit" sub-folder
# Save the unwrapped texture UV-map in "unwrap_texture" sub-folder
#################################################################
cd ../DataSet_Step4_UV_Texture
python run_unwrap_texture.py \
    --proj_data_dir ${proj_data_dir} \
    --ckp_dir ${checkpoints_dir} \
    --topo_dir ${topo_assets_dir}
