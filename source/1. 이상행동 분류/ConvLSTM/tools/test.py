import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pprint
import time
import os, sys
import cv2

import _init_paths
from models.NIA_LSTM import LSTM_NIA
from dataset.NIADataset import NIADataset_mp4
from utils.utils import create_logger

from collections import OrderedDict
import argparse


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def test(model, test_loader, test_dataset):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    model.eval()
    
    num_samples = len(test_dataset)
    all_preds = np.zeros(
        (num_samples, 1), dtype=np.float32
    )
    num_total = 0.
    num_correct = 0.
    
    mp4_list = []
    mp4_name = 'None'
    mp4_sum = 0.
    mp4_count = 0
    mp4_time = 0.
    
    total_accuracy = 0.
    total_count = 0
    
    with torch.no_grad():
        end = time.time()
        
        for i, (input, label, img_path) in enumerate(test_loader):
            img_path = img_path[0]
            idx = img_path.rfind('_')
            if mp4_name == 'None':
                mp4_name = img_path[:idx]
                mp4_sum = 0.
                mp4_count = 0
                mp4_time = 0.
            elif mp4_name != img_path[:idx]:
                msg = 'video : {2} \t' \
                      'time : {0:.06f} \t' \
                      'score : {1:.06f} \t'.format(
                        round(mp4_time, 6), round(mp4_sum/mp4_count, 6), mp4_name+'.mp4')
                # print(msg)
                logger.info(msg)
                total_accuracy += mp4_sum/mp4_count
                total_count += 1
                mp4_name = img_path[:idx]
                mp4_sum = 0.
                mp4_count = 0
                mp4_time = 0.              
                
            
            num_total += 1
            # compute output
            input = input.to(device)
            output = model(input)
            label = torch.LongTensor(label).to(device)
            
            #print(torch.argmax(output).item() == label.item())
            mp4_count += 1
            if torch.argmax(output).item() == label.item():
                num_correct += 1
                mp4_sum += 1 
            num_images = input.size(0)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            mp4_time += (time.time() - end)
            end = time.time()

            
            # if i % 1 == 0:
                # accuracy = num_correct / num_total
                # msg = 'Test: [{0}/{1}]\t' \
                      # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      # 'accuracy : {2}'.format(
                          # i, len(test_loader), accuracy, batch_time=batch_time)
                # msg = 'image: [{0}/{1}] img_path\t' \
                      # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      # 'accuracy : {2}'.format(
                          # i, len(test_loader), accuracy, img_path=img_path, batch_time=batch_time)
                # print(msg)
                
        msg = 'video : {2} \t' \
              'time : {0} \t' \
              'score : {1} \t'.format(
                round(mp4_time, 6), round(mp4_sum/mp4_count, 6), mp4_name+'.mp4')
        # print(msg)
        logger.info(msg)
        total_accuracy += mp4_sum/mp4_count
        total_count += 1
               
        # print("=====> total accuracy : {}".format(num_correct/num_total))
        print("=====> total score : {}".format(total_accuracy/total_count))
        logger.info("=====> total score : {}".format(total_accuracy/total_count))
        
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test abnormal event')
    # general
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--logDir', type=str, default="", help='log directory')
    
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger, final_output_dir = create_logger('valid')

    logger.info(f"command : python {' '.join(sys.argv)}\n")

    logger.info(f"=> {args}")


    # model
    model = LSTM_NIA(
        hidden_size = 256,
        num_classes = 8,
        num_layers = 1,
        backbone = 'resnet50',
        device = device
    )
    state = torch.load('weights/model_final.pth')['state_dict']
    keys = state.keys()
    values = state.values()
    new_keys = []
    for key in keys:
        new_key = key[7:]
        new_keys.append(new_key)
    new_dict = OrderedDict(list(zip(new_keys, values)))
    model.load_state_dict(new_dict)

    logger.info("=> loading model from ./weights/model_final.pth")

    model = model.to(device)

    # data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # test dataset & data loader
    root = args.path
    test_dataset = NIADataset_mp4(
        root = root,
        image_set = 'test',
        fps = 3,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )
    print("len(test_dataset) = {}".format(len(test_dataset)))
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 4
    )    
    
    test(
        model = model,
        test_dataset = test_dataset,
        test_loader = test_loader
    )    
