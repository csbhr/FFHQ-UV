# FFHQ-UV

### FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction
By [Haoran Bai](https://csbhr.github.io/), [Di Kang](https://scholar.google.com.hk/citations?user=2ztThPwAAAAJ&hl=zh-CN), Haoxian Zhang, [Jinshan Pan](https://jspan.github.io/), and [Linchao Bao](https://linchaobao.github.io/)  
*In CVPR 2023 [[Paper: https://arxiv.org/abs/2211.13874]](https://arxiv.org/abs/2211.13874)*  
*Rendering demos [[YouTube video]](https://youtu.be/dXFRJODJlNY)*



![teaser](./demos/teaser.png)

**FFHQ-UV** is a large-scale facial UV-texture dataset that contains over **50,000** high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions.

The dataset is derived from a large-scale face image dataset namely [FFHQ](https://github.com/NVlabs/ffhq-dataset), with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps.


## Updates
[2022-03-17] The source codes for adding eyeballs into head mesh are available [[here]](./README.md#add-eyeballs-into-head-mesh).  
[2022-03-16] The project details of the FFHQ-UV dataset creation pipeline are released [[here]](./README_dataset.md#ffhq-uv-dataset-project-details).  
[2022-03-16] The OneDrive download link was updated and the file structures have been reorganized.  
[2022-02-28] This paper will appear in CVPR 2023.  
[2022-01-19] The source codes are available, refer to [[here]](./README.md#run-source-codes) for quickly running.  
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

- Please refer to this [README](./README_dataset.md) to download the dataset and learn more.


## Run source codes

#### Download checkpoints and topology assets
- Please refer to this [README](./README_ckp_topo.md) for details of checkpoints and topology assets.

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

#### Add eyeballs into head mesh
- Prepare a head mesh, which is without eyeballs.
- Modify the configuration and then run the following script to add eyeballs into head mesh.
```
sh run_mesh_add_eyeball.sh  # Please refer to this script for detailed configuration
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
