# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# Modifier : MSP (parkms@markany.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from tkinter import EW
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from visualize import update_config, add_path

lib_path = os.path.join('lib')
add_path(lib_path)


import dataset as dataset
from config import cfg
import models
import os
import torchvision.transforms as T
from lib.utils.utils import detect_image, detect_keypoint, draw_bboxes
from lib.utils.vis import plt_config, plot_poses 

# ===== 객체 탐지 프로젝트 ===== #
from models.centernet import CenterNet
from models.darkcenternet import DarkCenterNet
from common.parser import parse_yaml            
from common.utils import load_weights
# ============================= #


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import argparse

# Set argument
parser = argparse.ArgumentParser()
# argument for keypoint
parser.add_argument('--cfg-keypoint', type=str, default="./experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml")
parser.add_argument('--model-name', type=str, default="T-H-A4")
# argument for detection
parser.add_argument('--cfg-detection', type=str, default="./experiments/markany/")
parser.add_argument('--weights-detection', type=str, required=True)
parser.add_argument('--conf-thresh', type=float, default=0.3)
# argument for video
parser.add_argument('--video-folder', type=str, default="./data", help='video folder')        
parser.add_argument('--save-folder', type=str, default="./result", help='save folder')        
parser.add_argument('--write-fps', type=bool, default=False, help='write "fps" on image.')        
args = parser.parse_args()


# Check exist
assert os.path.exists(args.cfg_keypoint), "Check keypoint config file : %s" %args.cfg_keypoint
assert os.path.exists(args.cfg_detection), "Check detection config file : %s" %args.cfg_detection
assert os.path.exists(args.video_folder), "Check video folder : %s" %args.video_folder
model_type = ['T-R', 'T-H','T-H-L','T-R-A4', 'T-H-A6', 'T-H-A5', 'T-H-A4' ,'T-R-A4-DirectAttention']
assert args.model_name in model_type, "check model name. {}".format(model_type)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


# Load videos
video_list = os.listdir(args.video_folder)
video_list = [video for video in video_list if '.avi' in video or '.mp4' in  video]


# Model for keypoint
f = open(args.cfg_keypoint, 'r')
update_config(cfg, args.cfg_keypoint)
device = torch.device('cuda')
model_keypoint = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
    cfg, is_train=True
)
if cfg.TEST.MODEL_FILE:
    print("=====> loading model from {}".format(cfg.TEST.MODEL_FILE))
    model_keypoint.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
else:
    raise ValueError("=====> please choose one .pth in cfg.TEST.MODEL_FILE")
model_keypoint.to(device)
print("=====> complete loading keypoint model : %s" %(cfg.MODEL.NAME))


# Model for detection
model_detection_option = parse_yaml(args.cfg_detection)
model_detection = DarkCenterNet(model_detection_option) if "yolox" in model_detection_option["MODEL"]["BACKBONE"] else CenterNet(model_detection_option)
model_detection_name = "DarkCenterNet" if "yolox" in model_detection_option["MODEL"]["BACKBONE"] else "CenterNet"
model_detection.to(device)
weights_detection = load_weights(args.weights_detection)
model_detection.load_state_dict(weights_detection['model_state_dict'])
print("=====> complete loading detection model : %s" %(model_detection_name))


# Check keypoint #parameters
numParam_detection = sum([p.numel() for p in model_detection.parameters()])
numParam_keypoint = sum([p.numel() for p in model_keypoint.parameters()])
print("=====> detection model parameters : {:.3f}M".format(numParam_detection/1000*2))
print("=====> keypoints model parameters : {:.3f}M".format(numParam_keypoint/1000*2))


# Normailze
normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# Transform (for keypoint model input)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


# Inference
with torch.no_grad():
    model_detection.eval()
    model_keypoint.eval()        

    # dummy
    dummy_input_detection = torch.randn((1, 3, 608, 608)).to(device)
    dummy_input_keypoint = torch.randn((1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).to(device)
    for _ in range(3):
        model_detection(dummy_input_detection)
        model_keypoint(dummy_input_keypoint)

    print("==========> start demo videos =========")
    for video_path in video_list:
        # Load one video
        print("=====> video : %s" %(video_path))        
        video = cv2.VideoCapture(os.path.join(args.video_folder, video_path))
        _, frame = video.read()
        video_save_path = os.path.join(args.save_folder, video_path.split('.')[0] + "_keypoints.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        output_video = cv2.VideoWriter(video_save_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        
        while(True):
            # get one image
            ret, img = video.read()
            
            if ret:
                # bboxes (only person)
                bboxes = detect_image(img, model_detection, model_detection_option, device, args.conf_thresh)
                
                # keypoint
                keypoints = detect_keypoint(img, transform, bboxes, model_keypoint, device, cfg)
                
                # draw bboxes
                img = draw_bboxes(img, bboxes)
                # draw keypoints
                for keypoint in keypoints:
                    img = plot_poses(img,
                                     [keypoint],
                                     config=plt_config(cfg.DATASET.DATASET),
                                     save_path=None,
                                     dataset_name=cfg.DATASET.DATASET)                
                
                # write images
                output_video.write(img)
            else:
                break
            
        output_video.release()
        video.release()
            
