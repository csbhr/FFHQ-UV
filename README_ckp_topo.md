# Download

- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).
- OneDrive: [download link](https://t1h0q-my.sharepoint.com/:f:/g/personal/csbhr_t1h0q_onmicrosoft_com/Em2_9wf4ZD9Bm2JVbnBZKn0B8WuFStiMHu07IYCPRLy7Hw?e=dNwuVW)


# The checkpoints used in FFHQ-UV project

## File structure

```
|--FFHQ-UV  
    |--checkpoints 
        |--arcface_model
            |--ms1mv3_arcface_r50_fp16_backbone.pth
        |--deep3d_model
            |--epoch_latest.pth
        |--dlib_model
            |--shape_predictor_68_face_landmarks.dat
        |--dpr_model
            |--trained_model_03.t7
        |--e4e_model
            |--e4e_ffhq_encode.pt
        |--exprecog_model
            |--FacialExpRecognition_model.t7
        |--lm_model
            |--68lm_detector.pb
        |--mtcnn_model
            |--mtcnn_model.pb
        |--parsing_model
            |--79999_iter.pth
        |--resnet_model
            |--resnet18-5c106cde.pth
        |--styleflow_model
            |--modellarge10k.pt
            |--expression_direction.pt
        |--stylegan_model
            |--stylegan2-ffhq-config-f.pkl
        |--texgan_model
            |--texgan_ffhq_uv.pth
            |--texgan_cropface630resize1024_ffhq_uv_interpolate.pth
        |--vgg_model
            |--vgg16.pt
```

## The introduction of these checkpoints
- arcface_model: Arcface backbone, download from [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#ms1mv3).
- deep3d_model: Our trained Deep3D model with shape basis [HiFi3D++](https://github.com/czh-98/REALY).
- dlib_model: The 68 landmarks detector from [dlib](http://dlib.net/).
- dpr_model: The checkpoint of [DPR](https://github.com/zhhoper/DPR).
- e4e_model: The checkpoint of [encoder4editing](https://github.com/omertov/encoder4editing).
- exprecog_model: The face expression recognition from [this repo](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch).
- lm_model: The 68 landmarks detector from [Deep3D](https://github.com/sicxu/Deep3DFaceRecon_pytorch).
- mtcnn_model: The 5 point detector from [MTCNN](https://github.com/ipazc/mtcnn).
- parsing_model: The face parsing from [this repo](https://github.com/zllrunning/face-parsing.PyTorch).
- resnet_model: ResNet backbone, download from [this link](https://download.pytorch.org/models/resnet18-5c106cde.pth).
- styleflow_model: The checkpoint of [StyleFlow](https://github.com/RameenAbdal/StyleFlow) and the expression editing direction found by SVM.
- stylegan_model: The checkpoint of [StyleGAN2](https://github.com/NVlabs/stylegan2).
- texgan_model: Our Texture GAN models trained on FFHQ-UV dataset or FFHQ-UV-Interpolate dataset.
- vgg_model: VGG backbone, download from [this link](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt).



# The topology assets used in FFHQ-UV project

## File structure

```
|--FFHQ-UV  
    |--topo_assets 
        |--center_face_mask.png
        |--hair_mask.png
        |--major_valid_front_mask.png
        |--major_valid_left_mask.png
        |--major_valid_right_mask.png
        |--major_valid_whole_mask.png
        |--minor_valid_front_mask.png
        |--minor_valid_left_mask.png
        |--minor_valid_right_mask.png
        |--minor_valid_whole_mask.png
        |--mouth_constract_mask.png
        |--nosal_base_mask.png
        |--nostril_mask.png
        |--template_base_uv.png
        |--unwrap_1024_info_mask.png
        |--hifi3dpp_model_info.mat
        |--hifi3dpp_mean_face.obj
        |--similarity_Lm3D_all.mat
        |--unwrap_1024_info.mat
```

## The introduction of these topology assets
- *_mask.png: The masks used in texture UV-map unwrapping.
- hifi3dpp_model_info.mat: The [HiFi3D++](https://github.com/czh-98/REALY) face model information.
- hifi3dpp_mean_face.obj: The mean face of the [HiFi3D++](https://github.com/czh-98/REALY) 3DMM shape basis.
- similarity_Lm3D_all.mat: The 68 3D landmarks on the mesh.
- unwrap_1024_info.mat: The coordinates used in texture UV-map unwrapping.