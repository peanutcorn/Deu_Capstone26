from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch

import os

import json
from pprint import pprint



event_start_list = ["fall_start", "broken_start", "fire_start", "smoke_start", "abandon_start", "theft_start", "fight_start"]
event_end_list = ["fall_end", "broken_end", "fire_end", "smoke_end", "abandon_end", "theft_end", "fight_end"]
event_dict = {
    'none' : 0,
    'fall' : 1,
    'broken' : 2,
    'fire' : 3,
    'smoke' : 4,
    'abandon' : 5,
    'theft' : 6,
    'fight' : 7
}


class NIAEvent(torch.utils.data.Dataset):
    def __init__(self, root, fps=3):
        print(root)
        self.root = root
        self.fps = fps
        # just one data
        self.db = self._get_db(self.root)
        # sequences data
        self._split_sequence()
        
    def __len__(self):
        return len(self.db['image'])    
    
    '''
    # === load just one data === #
    def __getitem__(self, idx):
        # read image
        img_path = self.db['image'][idx]
        print(img_path)
        img_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        return img_numpy, self.db['label'][idx]
    '''
    
    # === load sequences data === #
    def __getitem__(self, idx):
        imgs, labels = [], []
        for img_path, label in zip(self.db['image'][idx], self.db['label'][idx]):
            img_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            imgs.append(img_numpy)
            labels.append(label)
        imgs = np.array(imgs)
        
        return imgs, labels[-1]
    
    def _get_db(self, data_dir):
        print("===> loading %s dataset." %data_dir)
        
        assert os.path.exists(data_dir), "Wrong data path : {}".format(data_dir)
        
        db = {
            'image' : [],
            'label' : []
        }
                
        for data in os.listdir(data_dir):
            imgs, labels = self._get_one_data(os.path.join(data_dir, data))
            db['image'].extend(imgs), db['label'].extend(labels)
            
        return db
        
    
    def _get_one_data(self, data_path):
        img_dir = os.path.join(data_path, "images")        
        json_path = os.path.join(data_path, os.path.basename(data_path).upper() + ".json")
        
        # get event, start/end time
        event, start_time, end_time = self._get_event2frame(json_path)
        
        # get label
        save_img_list, save_event_list = self._create_eventLabel(os.path.join(data_path, "images"), event, start_time, end_time)
        
        return save_img_list, save_event_list
        
        
    def _get_event2frame(self, json_path):
        '''
            Param : json_path
            Return : event, start_frame, end_frame
        '''        
        with open(json_path, "r") as file:
            json_data = json.load(file)
            
            track_data = json_data['track']
            #print(len(json_data))
            #pprint(json_data)
            
            event, start_frame, end_frame = "none", 0, 0
            for track_one in track_data:
                if track_one['label'] in event_start_list:
                    event = track_one['label'].split('_')[0]
                    start_frame = (track_one['shape'][0]['frame'])
                if track_one['label'] in event_end_list:
                    end_frame = (track_one['shape'][0]['frame'])
                    
            return event, int(start_frame), int(end_frame)
        
        
    def _create_eventLabel(self, img_dir, event, start_time, end_time):
        img_list = os.listdir(img_dir)
        img_list = sorted(img_list)
        
        save_img_list = []
        save_event_list = []
        for img in img_list:
            idx = int(img.split('.')[0].split('_')[-1])
            
            if (idx >= start_time) and (idx <= end_time):
                event_num = event_dict[event]
            else:
                event_num = 0       # none
        
            save_img_list.append(os.path.join(self.root, img_dir, img))
            save_event_list.append(event_num)
            
        return save_img_list, save_event_list

    def _split_sequence(self):
        '''
            Param : fps
            Return : 
                'image' : (num_total/fps, fps)
                'label' : (num_total/fps)
        '''
        img_list = self.db['image']
        label_list = self.db['label']
        
        img_2d, label_2d = [], []
        img_tmp, label_tmp = [], []
        for img_one, label_one in zip(img_list, label_list):
            img_tmp.append(img_one)
            label_tmp.append(label_one)
            if len(img_tmp) == self.fps:
                img_2d.append(img_tmp)
                label_2d.append(label_tmp)
                img_tmp, label_tmp = [], []
        
        self.db['image'] = img_2d
        self.db['label'] = label_2d




if __name__ == "__main__":

    
    root = '/mnt/f/NIA'
    nia = NIAEvent(root)
    
    img, label = nia[0]
    print(label)
    print(len(img))
    print(len(nia))
    print(img.shape)        # fps, h, w, 3
    
    
    
        