from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch

import os


def getOneImageFromMP4(image_file):
    video_file = image_file[:image_file.rfind('/')] + '.mp4'
    frame_num = int(image_file.split('/')[-1].split('.png')[0][-6:])
    cap = cv2.VideoCapture(video_file)

    assert cap.isOpened(), \
        'There is no video : {}' %video_file

    num = 1
    while(cap.isOpened()):
        _ret, frame = cap.read()
        
        if frame_num == num:
            data_numpy = frame
            break
        else:
            num += 1
    cap.release()
    cv2.destroyAllWindows()
    return frame


# final directory
class NIADataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', fps=3, transform=None):
        self.db = self._get_db(image_set)
        self.root = root
        self.fps = fps
        self.transform = transform
        self.db = self._check_exists()
                
    def __len__(self):
        return len(self.db)    
    
    
    def __getitem__(self, idx):
        images = torch.zeros((self.fps, 3, 224, 224), dtype=torch.float32)
        label = int(self.db[idx].split(',')[-1])
        
        for i, img_path in enumerate(self.db[idx].split(',')[:self.fps]):            
            img_path = img_path.replace('\\', '/')
            img_numpy = cv2.imread(os.path.join(self.root, img_path), cv2.IMREAD_COLOR)
            img_numpy = cv2.resize(img_numpy, (224, 224), interpolation=cv2.INTER_LINEAR)            

            if self.transform:
                img_numpy = self.transform(img_numpy)
                        
            images[i, ...] = img_numpy

        return images, label


    def _get_db(self, image_set):        
        csv_path = 'split/' + image_set + '.csv'
        db = []
        with open(csv_path, 'r', encoding='UTF8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                db.append(line)

        # [ img1, img2, img3, label ]
        return db


    def get_distribution(self):
        distribution = [0, 0, 0, 0, 0, 0, 0, 0]
        for one_data in self.db:
            label = int(one_data.split(',')[-1])
            distribution[label] += 1
        
        return distribution
    
    def _check_exists(self):
        exist_list = []
        
        count = 0
        for one_line in self.db:
            one_data = os.path.join(self.root, one_line.split(',')[0])
            one_data = one_data.replace('\\', '/')
            if not os.path.exists(one_data):
                count += 1
                #print(one_data)
            else:
                exist_list.append(one_line)
        #print("wrong path : {}".format(count))

        return exist_list


# final directory
class NIADataset_mp4(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', fps=3, transform=None):
        self.db = self._get_db(image_set)
        self.root = root
        self.fps = fps
        self.transform = transform
        #self.db = self._check_exists()
                
        # print(self.db)
    def __len__(self):
        return len(self.db)    
    
    
    def __getitem__(self, idx):
        images = torch.zeros((self.fps, 3, 224, 224), dtype=torch.float32)
        label = int(self.db[idx].split(',')[-1])

        for i, img_path in enumerate(self.db[idx].split(',')[:self.fps]):            
            try:
                img_path = img_path.replace('\\', '/')
                img_numpy = getOneImageFromMP4(os.path.join(self.root, img_path))
                img_numpy = cv2.resize(img_numpy, (224, 224), interpolation=cv2.INTER_LINEAR)            
            except:
                #print(img_path)
                return images, label
            #img_numpy = cv2.imread(os.path.join(self.root, img_path), cv2.IMREAD_COLOR)

            if self.transform:
                img_numpy = self.transform(img_numpy)
                        
            images[i, ...] = img_numpy

        return images, label, img_path.split('/')[-1]


    def _get_db(self, image_set):        
        csv_path = 'split/' + image_set + '.csv'
        db = []
        with open(csv_path, 'r', encoding='cp949') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                db.append(line)

        # [ img1, img2, img3, label 
        #print(db)
        return db


    def get_distribution(self):
        distribution = [0, 0, 0, 0, 0, 0, 0, 0]
        for one_data in self.db:
            label = int(one_data.split(',')[-1])
            distribution[label] += 1
        
        return distribution
    
    def _check_exists(self):
        exist_list = []
        
        count = 0
        for one_line in self.db:
            one_data = os.path.join(self.root, one_line.split(',')[0])
            one_data = one_data.replace('\\', '/')
            #print("{} : {}".format(one_data, os.path.exists(one_data)))
            if not os.path.exists(one_data):
                count += 1
                #print(one_data)
            else:
                exist_list.append(one_line)
        #print("wrong path : {}".format(count))

        return exist_list


if __name__ == "__main__":

    data_path = '/mnt/d/save_dir'
    
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    # db = NIADataset(
        # root = data_path,
        # image_set = 'val',
        # transform = transforms.Compose([
            # transforms.ToTensor(),
            # normalize
        # ])        
    # )

    db = NIADataset_mp4(
        root = data_path,
        image_set = 'test',
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])        
    )

    img, label = db[0]
    
    print(len(db))
    #db._check_exists()
    '''
    print(len(db))
    
    
    for idx in range(len(db)):
        images, label = db[idx]
        print("images = {}".format(images.shape))
        print("label : {}".format(label))

    data_distribution = db.get_distribution()
    print(data_distribution)
    '''