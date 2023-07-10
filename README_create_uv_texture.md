# Create facial UV-texture dataset


## Run source codes
- Prepare a directory of dataset project, which contains a "images" subfolder.
- Put the original facial images into the "images" subfolder.
- Modify the configuration and then run the following script to create the facial UV-texture dataset.
```
sh run_ffhq_uv_dataset.sh  # Please refer to this script for detailed configuration
```


## Details of each step

### Step 0 - Preparation
```
proj_data_dir=../examples/dataset_examples
checkpoints_dir=../checkpoints
topo_assets_dir=../topo_assets
```
- Put the original facial images into "images" subfolder of the dataset project.
- The checkpoints and topology assets can be downloaded from [here](./README_ckp_topo.md).

### Step 1 - Inversion
```
cd ./DataSet_Step1_Inversion
python run_e4e_inversion.py \
    --proj_data_dir ${proj_data_dir} \
    --e4e_model_path ${checkpoints_dir}/e4e_model/e4e_ffhq_encode.pt \
    --shape_predictor_model_path ${checkpoints_dir}/dlib_model shape_predictor_68_face_landmarks.dat
```
- Based on [Designing an Encoder for StyleGAN Image Manipulation (e4e)](https://github.com/omertov/encoder4editing).
- Given a facial image, using the encoder [e4e](https://github.com/omertov/encoder4editing) to map the facial image to the latent code of [StyleGAN2](https://github.com/NVlabs/stylegan2).

### Step 2 - Detect attributes of the inverted faces
```
cd ../DataSet_Step2_Det_Attributes
python run_dpr_light.py \
    --proj_data_dir ${proj_data_dir} \
    --dpr_model_path ${checkpoints_dir}/dpr_model/trained_model_03.t7
python run_ms_api_attr.py \
    --proj_data_dir ${proj_data_dir}
```
- Based on [Deep Single-Image Portrait Relighting (DPR)](https://github.com/zhhoper/DPR) and [Microsoft Face API](https://azure.microsoft.com/en-in/products/cognitive-services/face/).
- The [DPR](https://github.com/zhhoper/DPR) is used to detect 9-dimensional spherical harmonic lighting coefficients (see [example (.npy file)](./examples/dataset_examples/lights/01223.npy)).
- The [Microsoft Face API](https://azure.microsoft.com/en-in/products/cognitive-services/face/) is used to detect attributes: Age, Baldness, Beard, Expression, Gender, Glasses, Pitch, Yaw (see [example (.json file)](./examples/dataset_examples/attributes/01223.json)).
- **Microsoft Face API is not accessible for new users**, one can find an alternative API, or manually fill in the [json file](./examples/dataset_examples/attributes/01223.json) to avoid this step.
- **We provide the detected facial attributes of the FFHQ dataset we used**, please find details from [here](https://github.com/csbhr/FFHQ-UV/blob/main/README_dataset.md#ffhq-uv-dataset-project-details).


### Step 3 - StyleGAN-based facial image editing 
```
cd ../DataSet_Step3_Editing
python run_styleflow_editing.py \
    --proj_data_dir ${proj_data_dir} \
    --network_pkl ${checkpoints_dir}/stylegan_model/stylegan2-ffhq-config-f.pkl \
    --flow_model_path ${checkpoints_dir}/styleflow_model/modellarge10k.pt \
    --exp_direct_path ${checkpoints_dir}/styleflow_model/expression_direction.pt \
    --exp_recognition_path ${checkpoints_dir}/exprecog_model/FacialExpRecognition_model.t7 \
    --edit_items delight,norm_attr,multi_yaw
```
- Based on [StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows](https://github.com/RameenAbdal/StyleFlow).
- Edit the following properties in order: Lighting, Glasses, Yaw, Pitch, Baldness, Expression, Multi-view (Yaw).


### Step 4 - UV-texture extraction, correction & completion
```
cd ../DataSet_Step4_UV_Texture
python run_unwrap_texture.py \
    --proj_data_dir ${proj_data_dir} \
    --ckp_dir ${checkpoints_dir} \
    --topo_dir ${topo_assets_dir}
```
- Using our trained Deep3D model to predict 3D shapes of multi-view facial images.
- Extract facial textures from multi-view facial images.
- Perform texture correction & completion to generate high-quality UV-texture maps robustly.
