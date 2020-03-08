# REDNet
implementation of REDNet (CVPR 2020)

“A Novel Recurrent Encoder-Decoder Structure for Large-Scale Multi-view Stereo Reconstruction from An Open Aerial Dataset”(arxiv：https://arxiv.org/abs/2003.00637)

The proposed network was trained and tested on a single NVIDIA TITAN RTX 2080Ti (24G).

## Requirements
CUDA 10.0 <br/>
cudnn 7.6 <br/>
python 3.6 <br/>
tensorflow-gpu 1.13.1 <br/>
numpy 1.18 <br/>
opencv-python 4.1 <br/>


## Data Preparation
1. Download the WHU MVS dataset.  http://gpcv.whu.edu.cn/data/WHU_dataset/WHU_MVS_dataset.zip.<br/>
2. Unzip the dataset to the ```WHU_MVS_dataset``` folder. <br/>

## Train
1. In ```train.py```, set ```data_root``` to your train data path ```YOUR_PATH/WHU_MVS_dataset/train```<br/>
2. Train REDNet (TITAN RTX 2080Ti 24G):<br/>
```
python train.py
```

## Test
1. Download the pre-trained REDNet model (https://pan.baidu.com/s/13BfLJ3sNfQL_933wZjR8PA, code：ohqx)<br/>
Unzip it to ```MODEL_FOLDER```folder.<br/>
2. In ```test.py```, set ```dense_folder``` to your test data path ```YOUR_PATH/WHU_MVS_dataset/test```, set ```model_dir``` to your model path ```MODEL_FOLDER```, set depth sample number ```max_d```.<br/>
3. Run REDNet：<br/>
```
python test.py 
```

The test outputs are stored in ```YOUR_PATH/WHU_MVS_dataset/test/depths_rednet/```, including depth map ```XXX_init.pfm```, probability map ```XXX_prob.pfm```, scaled images ```XXX.jpg``` and camera parameters ```XXX.txt```.<br/>


## Predict 
If you want to apply REDNet to your own data, please structure your data into such a folder.<br/>

### Image Files
All image files are stored in the ```Images``` folder.<br/>
### Camera files
All camera files are stored in the ```Cams```folder.<br/>
The text file contains the camera extrinsic, intrinsic and the depth range:<br/>
```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

f(pixel)  x0(pixel)  y0(pixel)

DEPTH_MIN   DEPTH_MAX   DEPTH_INTERVAL
IMAGE_INDEX 0 0 0 0 WIDTH HEIGHT
```
Make sure the camera extrinsic is ```Twc [Rwc|twc]```, camera orientation is ```XrightYup```.<br/>

1. In ```viewselection.py```, set ```dense_folder``` to your data path.<br/>
2. Run:
```
python viewselection.py
```
The output file ```viewpair.txt``` is stored in your data path.<br/>

3. In ```predict.py```, set ```dense_folder``` to your data path. Set ```model_dir``` to your model path ```MODEL_FOLDER```, set depth sample number ```max_d``` and image size ```max_w```, ```max_h```.<br/>
4. Run:
```
python predict.py
```
The outputs are stored in ```YOUR_DATA_PATH/depths_rednet/```.<br/>

We provided the script ```fusion.py``` to apply depth map filter for post-processing, which converted the per-view depth maps to 3D point cloud.<br/>

5. In ```fusion.py```, set ```dense_folder``` to your data path.<br/>
6. Run:
```
python fusion.py
```
Final point clouds are stored in ```YOUR_DATA_PATH/rednet_fusion/```.<br/>


### Reference
This project is based on the implementation of ```R-MVSNet```. Thank Yao, Yao for providing the source code (https://github.com/YoYo000/MVSNet)
