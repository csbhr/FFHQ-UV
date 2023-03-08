# FFHQ-UV

### FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction
By [Haoran Bai](https://csbhr.github.io/), [Di Kang](https://scholar.google.com.hk/citations?user=2ztThPwAAAAJ&hl=zh-CN), Haoxian Zhang, [Jinshan Pan](https://jspan.github.io/), and [Linchao Bao](https://linchaobao.github.io/)  
*In CVPR 2023 [[Paper: https://arxiv.org/abs/2211.13874]](https://arxiv.org/abs/2211.13874)*



![teaser](./demos/teaser.png)

**FFHQ-UV** is a large-scale facial UV-texture dataset that contains over **50,000** high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions.

The dataset is derived from a large-scale face image dataset namely [FFHQ](https://github.com/NVlabs/ffhq-dataset), with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps.


## TODO
- Since Microsoft Face API is not accessible for new users, in order to make it easier for others to reproduce our work, we will provide project details of the FFHQ-UV dataset creation pipeline in the near future, including inverted latents/faces, detected face attributes, sample correspondence, etc.


## Updates
[2022-02-28] This paper will appear in CVPR 2023.  
[2022-01-19] The source codes are available.  
[2022-12-16] The OneDrive download link is available.  
[2022-12-16] The AWS CloudFront download link is offline.  
[2022-12-06] The script for generating face images from latent codes is available.  
[2022-12-02] The latent codes and attributes of the multi-view normalized face images are available.  
[2022-12-02] The FFHQ-UV-Interpolate dataset is available.  
[2022-12-01] The FFHQ-UV dataset is available.  
[2022-11-28] The paper is available [here](https://arxiv.org/abs/2211.13874).   


## Dependencies
- Linux + Anaconda
- CUDA 10.0 + CUDNN 7.6.0
- Python 3.7
- dlib: `pip install dlib`
- PyTorch 1.7.1: `pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2`
- TensorBoard: `pip install tensorboard`
- TensorFlow 1.15.0: `pip install tensorflow-gpu==1.15.0`
- MS Face API: `pip install --upgrade azure-cognitiveservices-vision-face`
- OpenCV (should be higher than version 4.5): `pip install opencv-python`
- Other packages: `pip install tqdm scikit-image pillow imageio matplotlib mxnet Ninja google-auth google-auth-oauthlib click requests pyspng imageio-ffmpeg==0.4.3 scikit-learn torchdiffeq==0.0.1 flask kornia==0.2.0 lmdb psutil dominate rtree`
- PyTorch3D and Nvdiffrast:
```
mkdir thirdparty
cd thirdparty
git clone https://github.com/facebookresearch/iopath
git clone https://github.com/facebookresearch/fvcore
git clone https://github.com/facebookresearch/pytorch3d
git clone https://github.com/NVlabs/nvdiffrast
conda install -c bottler nvidiacub
pip install -e iopath
pip install -e fvcore
pip install -e pytorch3d
pip install -e nvdiffrast
```


## Dataset

#### Download
- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).
- OneDrive: [download link](https://gdutgz-my.sharepoint.com/:f:/g/personal/csbhr_gdutgz_onmicrosoft_com/EroU0mA5LfBCqcYyr7FSvjgBxBXTUlHEZmYAQRhI2m6S6A?e=hFbejA)

#### Dataset file structure
```
|--FFHQ-UV  
    |--ffhq-uv  # FFHQ-UV dataset
    |--ffhq-uv-face-latents  # The normalized face images' latent codes of FFHQ-UV dataset
    |--ffhq-uv-face-attributes  # The normalized face images' attributes of FFHQ-UV dataset
    |--ffhq-uv-interpolate    # FFHQ-UV-Interpolate dataset
    |--ffhq-uv-interpolate-face-latents  # The normalized face images' latent codes of FFHQ-UV-Interpolate dataset
    |--ffhq-uv-interpolate-face-attributes  # The normalized face images' attributes of FFHQ-UV-Interpolate dataset
```

#### FFHQ-UV-Interpolate dataset
- FFHQ-UV-Interpolate is a variant of FFHQ-UV. Please refer to [this readme](./README_ffhq_uv_interpolate.md) for details.

#### The latent codes and attributes of the normalized face images
- We provide the latent codes of the multi-view normalized face images which are used for extracting texture UV-maps. Along with the latent codes, we also provide the attributes (gender, age, beard) of each face, which are detected by [Microsoft Face API](https://azure.microsoft.com/en-in/products/cognitive-services/face/).
- One can generate face images from download latent codes by using the following script.
```
sh run_gen_face_from_latent.sh  # Please refer to this script for detailed configuration
```


## Run source codes

#### Download checkpoints and topology assets
- Please refer to [this readme](./README_checkpoints.md) for details of checkpoints.
- Please refer to [this readme](./README_topo_assets.md) for details of topology assets.

#### Create facial UV-texture dataset
- Prepare a directory of dataset project, which contains a "images" subfolder.
- Put the original images into the "images" subfolder.
- Modify the configuration and then run the following script to create the facial UV-texture dataset.
```
sh run_ffhq_uv_dataset.sh  # Please refer to this script for detailed configuration
```

#### RGB fitting
- Put the input images into a folder.
- Modify the configuration and then run the following script for fitting.
```
run_rgb_fitting.sh  # Please refer to this script for detailed configuration
```


## Citation
```
@InProceedings{Bai_2023_CVPR,
  title={FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction},
  author={Bai, Haoran and Kang, Di and Zhang, Haoxian and Pan, Jinshan and Bao, Linchao},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  month={June},
  year={2023}
}
```
