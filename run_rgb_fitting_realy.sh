#!/bin/bash
set -e


######################### Configuration #########################
# input_dir: the directory of the input images
# output_dir: the directory of the output results
# checkpoints_dir: the directory of the used checkpoints
# topo_assets_dir: the directory of the topo assets, e.g., 3DMM, masks, etc.
#################################################################
input_dir=../examples/fitting_realy/inputs
output_dir=../examples/fitting_realy/outputs
checkpoints_dir=../checkpoints
topo_assets_dir=../topo_assets


########################## RGB Fitting ##########################
# Read the processed data in ${input_dir}
# Save the output results in ${output_dir}
#################################################################
cd ./RGB_Fitting
python step2_fit_processed_data_realy.py \
    --input_dir ${input_dir} \
    --output_dir ${output_dir} \
    --checkpoints_dir ${checkpoints_dir} \
    --topo_dir ${topo_assets_dir} \
    --texgan_model_name texgan_ffhq_uv.pth
