# Centernet-like-detector

##  Dependencies
 - CUDA 11.3
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0
 - mmcv-full 1.5.0
 - numpy 1.22.4

## Datasets
- SODA-D: [OneDrvie](https://nwpueducn-my.sharepoint.com/:f:/g/personal/gcheng_nwpu_edu_cn/EhXUvvPZLRRLnmo0QRmd4YUBvDLGMixS11_Sr6trwJtTrQ?e=PellK6); [BaiduNetDisk](https://pan.baidu.com/s/1aqmqkG_GzDKBTM_NK5ecqA?pwd=SODA)


The data preparation for SODA differs slightly from that of conventional object detection datasets, as it requires the initial step of splitting the original images. 
Srcipts to obtain sub-images of SODA-D can be found at `tools/img_split`. 


## Install
This repository is build on **MMDetection 2.26.0**  which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.
```
git clone https://github.com/forgivehat/CenterNet-Like-Detector.git centernet
cd centernet
pip install -v -e .
```


## Training
 - Single GPU:
```
python ./tools/train.py ${CONFIG_FILE} 
```

 - Multiple GPUs:
```
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

##  Evaluation
 - Single GPU:
```
python ./tools/test.py ${CONFIG_FILE} ${WORK_DIR} --eval bbox
```

 - Multiple GPUs:
```
bash ./tools/dist_test.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --eval bbox
```