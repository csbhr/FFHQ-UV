## The checkpoints used in FFHQ-UV project

#### Download

- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).
- OneDrive: [download link](https://gdutgz-my.sharepoint.com/:f:/g/personal/csbhr_gdutgz_onmicrosoft_com/EroU0mA5LfBCqcYyr7FSvjgBxBXTUlHEZmYAQRhI2m6S6A?e=hFbejA)

#### File structure

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
        |--vgg_model
            |--vgg16.pt
```

#### The introduction of these checkpoints
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
- texgan_model: Our Texture GAN model trained on FFHQ-UV dataset.
- vgg_model: VGG backbone, download from [this link](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt).
