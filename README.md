# FFHQ-UV

#### [Paper](https://arxiv.org/abs/2211.13874) | [Project Page](https://github.com/csbhr/FFHQ-UV)
### FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction
By [Haoran Bai](https://csbhr.github.io/), Di Kang, Haoxian Zhang, [Jinshan Pan](https://jspan.github.io/), and [Linchao Bao](https://linchaobao.github.io/)



![teaser](./assets/teaser.png)

**FFHQ-UV** is a large-scale facial UV-texture dataset that contains over **50,000** high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions.

The dataset is derived from a large-scale face image dataset namely [FFHQ](https://github.com/NVlabs/ffhq-dataset), with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps.


## Updates
[2022-11-28] The paper is available [here](https://arxiv.org/abs/2211.13874).  
[2022-11-27] The download link for the dataset will be available in a few days.  


## Citation
```
@article{FFHQ-UV,
  title={FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction},
  author={Bai, Haoran and Kang, Di and Zhang, Haoxian and Pan, Jinshan and Bao, Linchao},
  journal={arXiv preprint arXiv:2211.13874},
  year={2022}
}
```
