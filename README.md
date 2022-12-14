# FFHQ-UV

#### [Paper](https://arxiv.org/abs/2211.13874) | [Project Page](https://csbhr.github.io/projects/ffhq-uv/)
### FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction
By [Haoran Bai](https://csbhr.github.io/), [Di Kang](https://scholar.google.com.hk/citations?user=2ztThPwAAAAJ&hl=zh-CN), Haoxian Zhang, [Jinshan Pan](https://jspan.github.io/), and [Linchao Bao](https://linchaobao.github.io/)



![teaser](./demos/teaser.png)

**FFHQ-UV** is a large-scale facial UV-texture dataset that contains over **50,000** high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions.

The dataset is derived from a large-scale face image dataset namely [FFHQ](https://github.com/NVlabs/ffhq-dataset), with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps.


## Updates
[2022-12-16] The OneDrive download link is available.  
[2022-12-16] The AWS CloudFront download link is offline.  
[2022-12-06] The script for generating face images from latent codes is available.  
[2022-12-02] The latent codes and attributes of the multi-view normalized face images are available.  
[2022-12-02] The FFHQ-UV-Interpolate dataset is available.  
[2022-12-01] The FFHQ-UV dataset is available.  
[2022-11-28] The paper is available [here](https://arxiv.org/abs/2211.13874).   


## Dataset Downloads

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

#### The latent codes and attributes of the normalized face images
- We provide the latent codes of the multi-view normalized face images which are used for extracting texture UV-maps. Along with the latent codes, we also provide the attributes (gender, age, beard) of each face, which are detected by [Microsoft Face API](https://azure.microsoft.com/en-in/products/cognitive-services/face/).
- One can generate face images from download latent codes by using the following script. The environment installation can refer to [StyleFlow](https://github.com/RameenAbdal/StyleFlow).
```
# the checkpoint of StyleGAN2 can be download from http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl
python gene_face_from_latent.py --latent_dir ./latent_dir --save_face_dir ./save_face_dir --stylegan_network_pkl ./stylegan2-ffhq-config-f.pkl
```


## FFHQ-UV-Interpolate

**FFHQ-UV-Interpolate** is a variant of FFHQ-UV. It is based on latent space interpolation, which is with compromised diversity but higher quality and larger scale (**100,000** UV-maps).

We adopt the following main steps to obtain FFHQ-UV-Interpolate from FFHQ-UV:
- Automatic data filtering considering BS Error, valid texture area ratio, expression detection, etc.
- Sample classification considering attributes such as gender, age, beard, etc.
- Latent space interpolation within each sample category.

Some quantitative comparisons between FFHQ-UV and FFHQ-UV-Interpolate (the values of ID std. are divided by the value of FFHQ):  
|  Dataset   | ID std. $\uparrow$ | # UV-maps $\uparrow$ | BS Error $\downarrow$ |
|  ----  | ----  | ----  | ----  |
| FFHQ-UV  | 90.06% | 54,165 | 7.293 |
| FFHQ-UV-Interpolate  | 80.12% | 100,000 | 4.490 |


## Citation
```
@article{FFHQ-UV,
  title={FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction},
  author={Bai, Haoran and Kang, Di and Zhang, Haoxian and Pan, Jinshan and Bao, Linchao},
  journal={arXiv preprint arXiv:2211.13874},
  year={2022}
}
```
