# TransPose


## 0. Model  

- Architecture  

![Architecture](figure/transpose_architecture.png)  


- Table  

|Name|Backbone|Input size|Output size|  
|-----|-----|-----|-----|  
|transpose_r|Resnet50|(3, 256, 192)|(joints, 64, 48)|    
|transpose_h|HRNet-S-W32<br>HRNet-S-W48|(3, 256, 192)|(joints, 64, 48)|    


- Details : [documentation](#3)  




## 1. Installation  

### 1-1. PyTorch (CUDA=11.3)  
- PyTorch>=1.6 and torchvision>=0.7  
``` bash
$ pip install pytorch torchvision cudatoolkit=11.3 -c pytorch
// or 'pip3 install pytorch torchvision cudatoolkit=11.3 -c pytorch'
```

### 1-2. Another tools  
``` bash
$ pip install -r requirements.txt
// or 'pip3 install -r requirements.txt'
```

### 1-3. Make (for 'nms')
``` bash
$ cd lib
$ make
```



## 2. Dataset  

### 2-1. Download Dataset for train/test  

- [MPII](http://human-pose.mpi-inf.mpg.de/)
- [COCO](https://cocodataset.org/#home)
- Markany(chungju) : [Google Drive(MSP)]()  
- NIA : [Google Drive(MSP)]()  


### 2-2. Preparation (optional)
```txt
${POSE_ROOT}/data/
|-- coco
|	|-- annotations
|	|   |-- person_keypoints_train2017.json
|	|   `-- person_keypoints_val2017.json
|	|-- person_detection_results
|	|   |-- COCO_val2017_detections_AP_H_56_person.json
|	|   `-- COCO_test-dev2017_detections_AP_H_609_person.json
|	`-- images
|		|-- train2017
|		|   |-- 000000000009.jpg
|		|   |-- ... 
|		`-- val2017
|			|-- 000000000139.jpg
|			|-- ... 
|-- mpii
|	|-- anno
|	|   |-- gt_valid.mat
|	|   |-- mpii_human_pose_v1_u12_1.mat
|	|   |-- test.json
|	|   |-- train.json
|	|   |-- trainval.json
|	|   `-- valid.json
|	|-- images
|	|   |-- 000001163.jpg
|	|   |-- 000003072.jpg
|	|   |-- ...
|-- nia_markany  
|   |--  ...
```


## 3. Train & Testing  

### 3-0. Config File  
- request [MSP](mailto:parkms@markany.com)(@parkms) and put them into 'experiments' folder (download : [google drive](https://drive.google.com/drive/folders/14v0iAJQMO07P8AWKxxoNrpmJC9_6iht9?usp=sharing))  
```txt
${POSE_ROOT}/experiments/
|-- coco
|	|-- transpose_r
|	|   |-- TP_R_256x192_d256_h1024_enc3_mh8.yaml
|	|   `-- TP_R_256x192_d256_h1024_enc4_mh8.yaml
|	|-- transpose_h
|	|   |-- TP_H_w32_256x192_stage3_1_4_d64_h128_relu_enc4_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d64_h128_relu_enc4_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc5_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml
|	|   `-- TP_H_w48_256x192_stage3_1_4_d192_h384_relu_enc4_mh1.yaml
|-- mpii
|	|-- transpose_r
|	|   |-- TP_R_256x192_d256_h1024_enc3_mh8.yaml
|	|   `-- TP_R_256x192_d256_h1024_enc4_mh8.yaml
|	|-- transpose_h
|	|   |-- TP_H_w32_256x192_stage3_1_4_d64_h128_relu_enc4_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d64_h128_relu_enc4_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc5_mh1.yaml
|	|   |-- TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml
|	|   `-- TP_H_w48_256x192_stage3_1_4_d192_h384_relu_enc4_mh1.yaml
|-- nia_markany
|   |-- TP_R_256x192_d256_h1024_enc4_mh8_nia_markany.yaml
|   |-- ...
```

### 3-1. Pretrained model weights for backbone or TransPose   

- download from [TransPose github](https://github.com/yangsenius/TransPose) for coco format  
- or request [MSP](mailto:parkms@markany.com)(@parkms) and put them into 'weights' folder (download : [google drive](https://drive.google.com/drive/folders/1aiWYHOYiyFlGagNoiKKV_EnOu7XPCpYR?usp=sharing))  
```txt
${POSE_ROOT}/weights
|-- TransPose								# (essential) for TransPose
|	|-- coco  
|	|	|-- tp_h_48_256x192_enc6_d96_h192_mh1.pth
|	|	|-- tp_r_256x192_enc3_d256_h1024_mh8.pth
|	|	|-- tp_r_256x192_enc4_d256_h1024_mh8.pth
|	`-- mpii  
|		`-- tp_r_256x192_enc4_d256_h1024_mh8_mpii.pth
|-- hrnet_imagenet							# (optional) for backbone
|	|-- hrnet_w32-36af842e.pth
|	`-- hrnet_w48-8ef0771d.pth
|-- resnet_imagenet							# (optional) for backbone
|	`-- resnet50-19c8e357.pth
`-- CenterNet								# (optional) for two-stage demo (object detection)
	`-- best_mAP.pth
```	

### 3-2. Train  

- command : python tools/train.py --cfg [cfg path]
- example  

```bash
python tools/train.py --cfg experiments/mpii/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml  			// MPII 
python tools/train.py --cfg experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml				// COCO
python tools/train.py --cfg experiments/nia_markany/TP_R_256x192_d256_h1024_enc4_mh8_nia_markany.yaml		// NIA (by MarkAny)
...
```

### 3-3. Test    

- command : python tools/test.py --cfg [cfg path] TEST.USE_GT_BBOX [True/False]  
- "USE_GT_BBOX" 태그는 evaluation metric을 적용하기 위해 필요한 정보가 있을 경우 사용. (예 : MPII 데이터셋의 경우, 머리 Bbox가 필요)  

```bash
python tools/test.py --cfg experiments/mpii/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml TEST.USE_GT_BBOX True				// MPII  
python tools/test.py --cfg experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml TEST.USE_GT_BBOX True				// COCO  
python tools/test.py --cfg experiments/nia_markany/TP_R_256x192_d256_h1024_enc4_mh8_nia_markany.yaml TEST.USE_GT_BBOX True		// NIA (by MarkAny)
...
```



## 4. Official Performance  (reference : [TransPose github](https://github.com/yangsenius/TransPose))

### 4-1. Model  

| Model          | Backbone    | #Attention layers |  d   |  h   | #Heads | #Params | AP (coco val gt bbox) | Download |
| -------------- | ----------- | :---------------: | :--: | :--: | :----: | :-----: | :-------------------: | :------: |
| TransPose-R-A3 | ResNet-S    |         3         | 256  | 1024 |   8    |  5.2Mb  |         73.8          | [model](https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc3_d256_h1024_mh8.pth) |
| TransPose-R-A4 | ResNet-S    |         4         | 256  | 1024 |   8    |  6.0Mb  |         75.1          | [model](https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc4_d256_h1024_mh8.pth) |
| TransPose-H-S  | HRNet-S-W32 |         4         |  64  | 128  |   1    |  8.0Mb  |         76.1          | [model](https://github.com/yangsenius/TransPose/releases/download/Hub/tp_h_32_256x192_enc4_d64_h128_mh1.pth) |
| TransPose-H-A4 | HRNet-S-W48 |         4         |  96  | 192  |   1    | 17.3Mb  |         77.5          | [model](https://github.com/yangsenius/TransPose/releases/download/Hub/tp_h_48_256x192_enc4_d96_h192_mh1.pth) |
| TransPose-H-A6 | HRNet-S-W48 |         6         |  96  | 192  |   1    | 17.5Mb  |         78.1          | [model](https://github.com/yangsenius/TransPose/releases/download/Hub/tp_h_48_256x192_enc6_d96_h192_mh1.pth) |


### 4-2. Result on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|     Model      | Input size | FPS* | GFLOPs | AP    | Ap .5 | AP .75 | AP (M) | AP (L) |  AR   | AR .5 | AR .75 | AR (M) | AR (L) |
| :------------: | :--------: | :--: | :----: | ----- | ----- | :----: | :----: | :----: | :---: | :---: | :----: | :----: | :----: |
| TransPose-R-A3 |  256x192   | 141  |  8.0   | 0.717 | 0.889 | 0.788  | 0.680  | 0.786  | 0.771 | 0.930 | 0.836  | 0.727  | 0.835  |
| TransPose-R-A4 |  256x192   | 138  |  8.9   | 0.726 | 0.891 | 0.799  | 0.688  | 0.798  | 0.780 | 0.931 | 0.845  | 0.735  | 0.844  |
| TransPose-H-S  |  256x192   |  45  |  10.2  | 0.742 | 0.896 | 0.808  | 0.706  | 0.810  | 0.795 | 0.935 | 0.855  | 0.752  | 0.856  |
| TransPose-H-A4 |  256x192   |  41  |  17.5  | 0.753 | 0.900 | 0.818  | 0.717  | 0.821  | 0.803 | 0.939 | 0.861  | 0.761  | 0.865  |
| TransPose-H-A6 |  256x192   |  38  |  21.8  | 0.758 | 0.901 | 0.821  | 0.719  | 0.828  | 0.808 | 0.939 | 0.864  | 0.764  | 0.872  |


### 4-3. Results on COCO test-dev2017 with detector having human AP of 60.9 on COCO test-dev2017 dataset

| Model          | Input size | #Params | GFLOPs | AP    | Ap .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
| -------------- | ---------- | ------- | ------ | ----- | ----- | ------ | ------ | ------ | ----- | ----- | ------ | ------ | ------ |
| TransPose-H-S  | 256x192    | 8.0M    | 10.2   | 0.734 | 0.916 | 0.811  | 0.701  | 0.793  | 0.786 | 0.950 | 0.856  | 0.745  | 0.843  |
| TransPose-H-A4 | 256x192    | 17.3M   | 17.5   | 0.747 | 0.919 | 0.822  | 0.714  | 0.807  | 0.799 | 0.953 | 0.866  | 0.758  | 0.854  |
| TransPose-H-A6 | 256x192    | 17.5M   | 21.8   | 0.750 | 0.922 | 0.823  | 0.713  | 0.811  | 0.801 | 0.954 | 0.867  | 0.759  | 0.859  |



## 5. Demo  

### 5-1. Image (one stage : bounding box image)

- modify config file (TEST : MODEL_FILE)   

<img src="figure/example_configFile.jpg" width="300" height="200">  


- command : python demo/demo_image_oneStage.py --cfg-keypoint [cfg path] --image-folder [image folder path]  

``` bash
$ python demo/demo_image_oneStage.py --cfg-keypoint ./experiments/mpii/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml --image-folder ./data/image_one
```


### 5-2. Image (two stage : full image)

- modify config file (same 5-1)  

- command : python demo/demo_image_twoStage.py --cfg-keypoint [cfg path for keypoint] --cfg-detection [cfg path for detection] --weights-detection [weight path for detection] --conf-thresh [float] --image-folder [image folder path]

```text
#===== Argument =====#
--cfg-keypoint 			# config file path for keypoint
--cfg-detection			# config file path for detection (usually CenterNet)
--weights-detection		# weight path for detection
--conf-thresh			# threshold value for detection
--image-folder			# image folder for test images
```

- example  

``` bash
$ python demo/demo_image_twoStage.py --cfg-keypoint ./experiments/mpii/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml --cfg-detection ./experiments/markany/argos_dark_model_s_v6.5.5.yaml --weights-detection ./weights/best_mAP.pth --image-folder ./data/image_multi --conf-thresh 0.4 --flag-save True
``` 



### 5-3. Video (two stage)

- 5-3-1. modify config file (same 5-1-1)  

- 5-3-2. command : python demo/demo_video.py --cfg-keypoint [cfg path for keypoint] --cfg-detection [cfg path for detection] --weights-detection [weight path for detection] --conf-thresh [float] --video-folder [video folder path]

```text
#===== Argument =====#
--cfg-keypoint 			# config file path for keypoint
--cfg-detection			# config file path for detection (usually CenterNet)
--weights-detection		# weight path for detection
--conf-thresh			# threshold value for detection
--video-folder			# video folder for test videos
```

- example  

``` bash
$ python demo/demo_video.py --cfg-keypoint ./experiments/mpii/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml --cfg-detection ./experiments/markany/argos_dark_model_s_v6.5.5.yaml --weights-detection ./weights/best_mAP.pth --video-folder ./data/video --conf-thresh 0.4
```



## 6. Result  

### 6.1. image  

```text
detection model : CenterNet  
keypoint model : TransPose_R_256x192_d256_h1024_enc4_mh8
```

<img src="figure/result_demo_01.jpg">  
<img src="figure/result_demo_02.jpg">  


### 6.2. video  

```text
detection model : CenterNet  
keypoint model : TransPose_R_256x192_d256_h1024_enc4_mh8
```

<img src="figure/result_demo_03.gif" width="60%">  



### if any question, contact [MSP](mailto:parkms@markany.com)(@parkms)