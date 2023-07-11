# RGB Fitting


## RGB fitting from a single face image
- Put the input images into a folder.
- Modify the configuration and then run the following script for fitting.
```
run_rgb_fitting.sh  # Please refer to this script for detailed configuration
```


## Reproducing REALY benchmark results
- For the REALY benchmark results, all the fitting-based methods compared in the paper (Table 5), used 86 landmarks during the fitting process.
- We provide the preprocessed data of REALY benchmark, which contains the detected 86 landmarks. The file (named "*fitting_realy.zip*") can be downloaded from:
    - Baidu Netdisk: [download link](https://pan.baidu.com/s/1BbvlTuhlD_PEtT3QZ_ja2g) (extract code: 5wbi).
    - OneDrive: [download link](https://t1h0q-my.sharepoint.com/:f:/g/personal/csbhr_t1h0q_onmicrosoft_com/Em2_9wf4ZD9Bm2JVbnBZKn0B8WuFStiMHu07IYCPRLy7Hw?e=dNwuVW)
- Modify the configuration and then run the following script for fitting REALY images.
```
run_rgb_fitting_realy.sh  # Please refer to this script for detailed configuration
```
- To calculate the evaluation metrics of REALY benchmark, please follow the instructions of the [official repository](https://github.com/czh-98/REALY). One should sign **Agreement** for downloading the benchmark data.


## A new version of RGB fitting
- The changes of RGB fitting process
    - Instead of using FFHQ-UV dataset, we use the FFHQ-UV-Interpolate dataset to train the GAN-based texture decoder.
    - When training the GAN-based texture decoder, instead of generating the entire UV-texture map (1024x1024), this version only generates the texture of the facial area (cut out the 630x630 facial area, and then resize to 1024x1024).
    - During RGB fitting process, the output texture map is first blended with template UV-texture map, and then resized back to original resolution.
- Modify the configuration and then run the following script for fitting.
```
run_rgb_fitting_cropface630resize1024.sh  # Please refer to this script for detailed configuration
```
- In some samples, this version is able to generate UV-texture maps with more details, such as the sample below.

![fitting samples](./demos/imgs/new_version_of_rgb_fitting.png)



