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
from lib.core.inference import get_final_preds
from lib.utils.vis import convert_dstKeypoints_orgKeypoints, plot_poses, plt_config     # by MSP


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import argparse

# Set argument
parser = argparse.ArgumentParser()
parser.add_argument('--cfg-keypoint', type=str, default="./experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml")
parser.add_argument('--model-name', type=str, default="T-H-A4")
parser.add_argument('--image-folder', type=str, default="./data", help='image folder')        
parser.add_argument('--save-folder', type=str, default="./result", help='save folder')        
parser.add_argument('--write-fps', type=bool, default=False, help='write "fps" on image.')        
args = parser.parse_args()

# Check exist
assert os.path.exists(args.cfg_keypoint), "Check config file : %s" %args.cfg_keypoint
assert os.path.exists(args.image_folder), "Check image folder : %s" %args.image_folder
model_type = ['T-R', 'T-H','T-H-L','T-R-A4', 'T-H-A6', 'T-H-A5', 'T-H-A4' ,'T-R-A4-DirectAttention']
assert args.model_name in model_type, "check model name. {}".format(model_type)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

# Load images
img_list = os.listdir(args.image_folder)


# Model
f = open(args.cfg_keypoint, 'r')
update_config(cfg, args.cfg_keypoint)
device = torch.device('cuda')
model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
    cfg, is_train=True
)

# Normailze
normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Transform (for making same input tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

print("=====> cfg.TEST.MODEL_FILE : {}".format(cfg.TEST.MODEL_FILE))

# Load model (pretrained)
if cfg.TEST.MODEL_FILE:
    print("=====> loading model from {}".format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
else:
    raise ValueError("=====> please choose one .pth in cfg.TEST.MODEL_FILE")
model.to(device)

# Check #parameters
numParam = sum([p.numel() for p in model.parameters()])
print("=====> model params : {:.3f}M".format(numParam/1000*2))


# Inference
with torch.no_grad():
    model.eval()
    
    # dummy
    dummy_input_keypoint = torch.randn((1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).to(device)
    for _ in range(3):
        model(dummy_input_keypoint)
    
    print("==========> start demo images =========")
    for img_path in img_list:
        print("=====> image : %s" %(img_path))                
        img_name = img_path.split('.')[0]
        save_path = os.path.join(args.save_folder, img_path.split('.')[0] + "_keypoints.jpg")
        img_path = os.path.join(args.image_folder, img_path)
        
        # Load one image (img -> numpy -> torch)
        img_numpy = cv2.imread(
            img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )        
        img_save = img_numpy.copy()
        img_numpy = cv2.resize(img_numpy, (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])), 
                               interpolation=cv2.INTER_CUBIC)
        if cfg.DATASET.COLOR_RGB:
            img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
        img_torch = transform(img_numpy)       
        img_torch = torch.cat([img_torch.to(device)]).unsqueeze(0)
        
        # Output
        outputs = model(img_torch)              # heatmaps (height/4, width/4)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
                
        preds, maxvals = get_final_preds(       # keypoints (preds, masvals : (1, 16, 2))
            cfg, output.clone().cpu().numpy(), None, None, transform_back=False
        )
                
        query_locations = np.array([p*4+0.5 for p in preds[0]])     # apply coordinate [heatmap(H/4, W/4) => original image(H, W)]

        """
        # save GT
        output = output.clone().cpu().numpy().squeeze()
        for ch in range(output.shape[0]):
            save_heat_name = os.path.join(args.save_folder, 'heatmap',  img_name + '_' + str(ch) + ".jpg")
            cv2.imwrite(save_heat_name, output[ch] * 255.)
        print("output = {}".format(output.shape))
        """

        # dst_keypoints -> org_keypoints
        org_keypoints = convert_dstKeypoints_orgKeypoints(query_locations, img_save, img_numpy)

        # save
        img_save = plot_poses(img_save, 
                              [org_keypoints], 
                              config=plt_config(cfg.DATASET.DATASET), 
                              save_path=None, dataset_name=cfg.DATASET.DATASET)
        cv2.imwrite(save_path, img_save)
