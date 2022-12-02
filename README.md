# FFHQ-UV

#### [Paper](https://arxiv.org/abs/2211.13874) | [Project Page](https://github.com/csbhr/FFHQ-UV)
### FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction
By [Haoran Bai](https://csbhr.github.io/), [Di Kang](https://scholar.google.com.hk/citations?user=2ztThPwAAAAJ&hl=zh-CN), Haoxian Zhang, [Jinshan Pan](https://jspan.github.io/), and [Linchao Bao](https://linchaobao.github.io/)



![teaser](./assets/teaser.png)

**FFHQ-UV** is a large-scale facial UV-texture dataset that contains over **50,000** high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions.

The dataset is derived from a large-scale face image dataset namely [FFHQ](https://github.com/NVlabs/ffhq-dataset), with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps.


## Updates
[2022-12-02] The latent codes of the multi-view normalized face images are available.  
[2022-12-02] The FFHQ-UV-Interpolate dataset is available.  
[2022-12-01] The FFHQ-UV dataset is available.  
[2022-11-28] The paper is available [here](https://arxiv.org/abs/2211.13874).   


## Dataset Downloads

#### Standard FFHQ-UV dataset
- AWS CloudFront: Using the download script.
```
python download_ffhq_uv.py --dataset ffhq-uv-standard --dst_dir ./save_dir
```
- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).

#### FFHQ-UV-Interpolate dataset
- AWS CloudFront: Using the download script.
```
python download_ffhq_uv.py --dataset ffhq-uv-interpolate --dst_dir ./save_dir
```
- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).

#### The latent codes of the normalized face images
- We provide the latent codes of the multi-view normalized face images which are used for extracting texture UV-maps. One can generate face images by [StyleGAN2](https://github.com/NVlabs/stylegan2) using the pre-trained [checkpoint](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl).
- Along with the latent codes, we also provide the attributes (gender, age, beard) of each face, which are detected by [Microsoft Face API](https://azure.microsoft.com/en-in/products/cognitive-services/face/).
- AWS CloudFront: Using the download script.
```
# for Standard FFHQ-UV dataset
python download_ffhq_uv.py --dataset ffhq-uv-standard-face-latent --dst_dir ./save_dir
# for FFHQ-UV-Interpolate dataset
python download_ffhq_uv.py --dataset ffhq-uv-interpolate-face-latent --dst_dir ./save_dir
```
- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).


## FFHQ-UV-Interpolate

**FFHQ-UV-Interpolate** is a variant of FFHQ-UV. It is based on latent space interpolation, which is with compromised diversity but higher quality and larger scale (**100,000** UV-maps).

We adopt the following main steps to obtain FFHQ-UV-Interpolate from FFHQ-UV:
- Automatic data filtering considering BS Error, valid texture area ratio, expression detection, etc.
- Sample classification considering attributes such as gender, age, beard, etc.
- Latent space interpolation within each sample category.

More detailed descriptions and source codes will be released later.

Some quantitative comparisons between FFHQ-UV and FFHQ-UV-Interpolate (the values of ID std. are divided by the value of FFHQ):  
|  Dataset   | ID std. $\uparrow$ | # UV-maps $\uparrow$ | BS Error $\downarrow$ |
|  ----  | ----  | ----  | ----  |
| FFHQ-UV  | 90.06% | 54,165 | 7.293 |
| FFHQ-UV-Interpolate  | 80.12% | 100,000 | 4.522 |


## Citation
```
@article{FFHQ-UV,
  title={FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction},
  author={Bai, Haoran and Kang, Di and Zhang, Haoxian and Pan, Jinshan and Bao, Linchao},
  journal={arXiv preprint arXiv:2211.13874},
  year={2022}
}
```
