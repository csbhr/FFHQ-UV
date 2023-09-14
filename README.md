# FFHQ-UV

### FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction
By [Haoran Bai](https://csbhr.github.io/), [Di Kang](https://scholar.google.com.hk/citations?user=2ztThPwAAAAJ&hl=zh-CN), Haoxian Zhang, [Jinshan Pan](https://jspan.github.io/), and [Linchao Bao](https://linchaobao.github.io/)  
*In CVPR 2023 [[Paper: https://arxiv.org/abs/2211.13874]](https://arxiv.org/abs/2211.13874)*  
*Rendering demos [[YouTube video]](https://youtu.be/dXFRJODJlNY)*



![teaser](./demos/teaser.png)

**FFHQ-UV** is a large-scale facial UV-texture dataset that contains over **50,000** high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions.

The dataset is derived from a large-scale face image dataset namely [FFHQ](https://github.com/NVlabs/ffhq-dataset), with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps.


## Updates
[2023-07-11] A solution for using our UV-texture maps on a [FLAME](https://flame.is.tue.mpg.de/) mesh is available [[here]](./README_flame2hifi3d.md).  
[2023-07-10] A more detailed description and a new version of the RGB fitting process is available [[here]](./README_rgb_fitting.md).  
[2023-07-10] A more detailed description of the facial UV-texture dataset creation pipeline is available [[here]](./README_create_uv_texture.md).  
[2023-03-17] The source codes for adding eyeballs into head mesh are available [[here]](./README.md#add-eyeballs-into-head-mesh).  
[2023-03-16] The project details of the FFHQ-UV dataset creation pipeline are released [[here]](./README_dataset.md#ffhq-uv-dataset-project-details).  
[2023-03-16] The OneDrive download link was updated and the file structures have been reorganized.  
[2023-02-28] This paper will appear in CVPR 2023.  
[2023-01-19] The source codes are available, refer to [[here]](./README.md#run-source-codes) for quickly running.  
[2022-12-16] The OneDrive download link is available.  
[2022-12-16] The AWS CloudFront download link is offline.  
[2022-12-06] The script for generating face images from latent codes is available.  
[2022-12-02] The latent codes of the multi-view normalized face images are available.  
[2022-12-02] The FFHQ-UV-Interpolate dataset is available.  
[2022-12-01] The FFHQ-UV dataset is available [[here]](./README_dataset.md).  
[2022-11-28] The paper is available [[here]](https://arxiv.org/abs/2211.13874).   


## Dependencies
- Linux + Anaconda
- CUDA 10.0 + CUDNN 7.6.0
- Python 3.7
- dlib: `pip install dlib`
- PyTorch 1.7.1: `pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2`
- TensorBoard: `pip install tensorboard`
- TensorFlow 1.15.0: `pip install tensorflow-gpu==1.15.0`
- MS Face API: `pip install --upgrade azure-cognitiveservices-vision-face`
- Other packages: `pip install tqdm scikit-image opencv-python pillow imageio matplotlib mxnet Ninja google-auth google-auth-oauthlib click requests pyspng imageio-ffmpeg==0.4.3 scikit-learn torchdiffeq==0.0.1 flask kornia==0.2.0 lmdb psutil dominate rtree`
- **Important: OpenCV's version needs to be higher than 4.5, otherwise it will not work well.**
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


## Get dataset
- Please refer to this [[README]](./README_dataset.md) to download the dataset and learn more.


## Run source codes

### Download checkpoints and topology assets
- Please refer to this [[README]](./README_ckp_topo.md) for details of checkpoints and topology assets.

### Create FFHQ-UV dataset
- Please refer to this [[README]](./README_create_uv_texture.md) for details of running facial UV-texture dataset creation pipeline.
- **Microsoft Face API is not accessible for new users**, one can find an alternative API, or manually fill in the [json file](./examples/dataset_examples/attributes/01223.json) to avoid this step.
- **We provide the detected facial attributes of the FFHQ dataset we used**, please find details from [here](https://github.com/csbhr/FFHQ-UV/blob/main/README_dataset.md#ffhq-uv-dataset-project-details).

### RGB fitting
- Please refer to this [[README]](./README_rgb_fitting.md) for details of running RGB fitting process.
- We provide a new version of RGB fitting, where the FFHQ-UV-Interpolate dataset is used and the GAN-based texture decoder only generates facial textures. More details can found [here](./README_rgb_fitting.md#a-new-version-of-rgb-fitting).

### Add eyeballs into head mesh
- Prepare a head mesh with [HiFi3D++](https://github.com/czh-98/REALY) topology, which is without eyeballs.
- Modify the configuration and then run the following script to add eyeballs into head mesh.
```
sh run_mesh_add_eyeball.sh  # Please refer to this script for detailed configuration
```


## Generate a UV-texture map from a single facial image

There are two ways to generate a UV-texture map from a given facial image:
1. Facial editing + texture unwrapping (Section 3.1 of the paper)
2. RGB fitting (Section 4.2 of the paper)

#### 1. UV-texture map from: facial editing + texture unwrapping
- The FFHQ-UV dataset is created from the FFHQ dataset in this way.
- See source codes of [facial UV-texture dataset creation pipeline](./README.md#create-ffhq-uv-dataset),  which including GAN inversion, attribute detection, StyleGAN-based editing, and texture unwrapping steps.
- Advantages:
  - The generated textures are directly extracted from facial images, which are detailed and with high-quality.
- Disadvantages:
  - The GAN inversion step would be failed for some samples.
  - The attribute detection step requires the Microsoft Face API, which is no longer accessible.
  - The StyleGAN-based editing would change the ID of the faces, resulting in textures with low-fidelity.

#### 2. UV-texture map from: RGB fitting
- This is the proposed 3D face reconstruction algorithm, which uses FFHQ-UV dataset to train a nonlinear texture basis.
- See source codes of [RGB fitting](./README.md#rgb-fitting).
- Advantages:
  - The generated textures are fitted based on the supervision of input faces, which are with high-fidelity.
- Disadvantages:
  - The textures are generated by a GAN-based texture decoder, sometimes with less detail than texture maps in the FFHQ-UV dataset. This is mainly due to the limitations of the nonlinear texture basis design and training. 
  - Future work will improve the generation capabilities of the nonlinear texture basis, in order to take full advantage of the high-quality and detailed UV texture maps in the FFHQ-UV dataset.



## Using our UV-texture maps on a FLAME mesh
- We provide a solution for using our UV-texture maps on a [FLAME](https://flame.is.tue.mpg.de/) mesh, please refer to this [[README]](./README_flame2hifi3d.md) for details.


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

## Acknowledgments
This implementation builds upon the awesome works done by Tov et al. ([e4e](https://github.com/omertov/encoder4editing)), Zhou et al. ([DPR](https://github.com/zhhoper/DPR)), Abdal et al. ([StyleFlow](https://github.com/RameenAbdal/StyleFlow)), and Karras et al. ([StyleGAN2](https://github.com/NVlabs/stylegan2), [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch), [FFHQ](https://github.com/NVlabs/ffhq-dataset)).

This work is based on [HiFi3D++](https://github.com/czh-98/REALY) topology, and was supported by Tencent AI Lab.
