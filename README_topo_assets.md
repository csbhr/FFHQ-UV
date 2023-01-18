## The topo assets used in FFHQ-UV project

#### Download

- Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).
- OneDrive: [download link](https://gdutgz-my.sharepoint.com/:f:/g/personal/csbhr_gdutgz_onmicrosoft_com/EroU0mA5LfBCqcYyr7FSvjgBxBXTUlHEZmYAQRhI2m6S6A?e=hFbejA)

#### File structure

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
        |--similarity_Lm3D_all.mat
        |--unwrap_1024_info.mat
```

#### The introduction of these checkpoints
- *_mask.png: The masks used in texture UV-map unwrapping.
- hifi3dpp_model_info.mat: The [HiFi3D++](https://github.com/czh-98/REALY) face model information.
- similarity_Lm3D_all.mat: The 68 3D landmarks on the mesh.
- unwrap_1024_info.mat: The coordinates used in texture UV-map unwrapping.
