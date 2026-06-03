# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# modifier : MSP (parkms@markany.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import logging
import time
from collections import namedtuple
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn


from utils.vis import convert_dstKeypoints_orgKeypoints     # by MSP
from core.inference import get_final_preds      # by MSP


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=[0,0],#list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details



# =========== 객체 탐지 프로젝트 ========== #
def detect_image(img, model, model_option, device, conf_thresh):
    input_tensor = to_tensor(img, model_option).to(device)
    
    bboxes = model(input_tensor).cpu() # [1, N, M]
    bboxes = bboxes[0] # [N, M]
    bboxes = bboxes[bboxes[:, 4] > conf_thresh]

    # only person
    bboxes = bboxes[bboxes[:, 5] == 0]

    return bboxes


def to_tensor(img, model_option):
    img = img.copy()
    
    img_w = model_option["MODEL"]["INPUT_SIZE"]["WIDTH"]
    img_h = model_option["MODEL"]["INPUT_SIZE"]["HEIGHT"]

    img = cv2.resize(img, dsize=(img_w, img_h))
    img = img[..., ::-1].transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = torch.tensor(img, dtype=torch.float32)/255.
    
    return img

def draw_bboxes(img, bboxes, masking=False):
    img_h, img_w = img.shape[:2]
    for cx, cy, w, h, conf, idx in bboxes:
        cx = int(cx.item() * img_w)
        cy = int(cy.item() * img_h)
        w = int(w.item() * img_w)
        h = int(h.item() * img_h)

        tx = max(0, cx - w//2)
        ty = max(0, cy - h//2)

        if masking:
            roi = img[ty:ty+h, tx:tx+w]
            resize_rate = 5

            roi_resize = cv2.resize(roi, (w//resize_rate, h//resize_rate))
            roi_resize = cv2.resize(roi_resize, (w, h), interpolation=cv2.INTER_AREA)
            
            if roi.shape != roi_resize.shape:
                roi_resize = roi[0:h, 0:w]

            img[ty:ty+h, tx:tx+w] = roi_resize
        else:
            cv2.rectangle(img, (tx,ty), (tx+w,ty+h), (255,0,0), 3)
            # draw_border(img, (tx,ty), (tx+w,ty+h), (0,0,200), 3, 0, 30) # 모서리 박스 그리기

    return img
# ======================================== #

# by MSP
def detect_keypoint(img, transform, bboxes, model, device, cfg):
    img_h, img_w = img.shape[:2]
    
    keypoint_list = []
    for cx, cy, w, h, conf, idx in bboxes:
        # get (top_left, top_right, width, height) according to person
        cx, cy = int(cx.item() * img_w), int(cy.item() * img_h)
        w, h = int(w.item() * img_w), int(h.item() * img_h)
        tx, ty = max(0, cx - w//2), max(0, cy - h//2)
        
        # input person (numpy -> tensor)
        img_person = img[ty:ty+h, tx:tx+w, :].copy()
        img_resize = cv2.resize(img_person, (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]), 
                                interpolation=cv2.INTER_CUBIC)
        img_input = img_resize.copy()
        if cfg.DATASET.COLOR_RGB:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = transform(img_input)
        img_input = torch.cat([img_input.to(device)]).unsqueeze(0)
        
        # heatmap
        outputs = model(img_input)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
        # keypoints
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), None, None, transform_back=False
        )
        query_locations = np.array([p*4+0.5 for p in preds[0]])
        org_keypoints = convert_dstKeypoints_orgKeypoints(query_locations, img_person, img_resize)
        org_keypoints[:, 0] += tx
        org_keypoints[:, 1] += ty
        
        keypoint_list.append(org_keypoints)
        
    return keypoint_list