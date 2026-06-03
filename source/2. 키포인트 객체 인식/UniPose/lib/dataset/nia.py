from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict
import copy
import cv2
import torch, torchvision
import random

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset
from utils.read_data_from_nia import get_all_gt         # by MSP
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)

# by MSP
class NIADataset_markany(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 17
        self.flip_pair = [[0, 5], [1, 4], [2, 3], [11, 16], [12, 15], [13, 14]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 9, 12, 13, 8, 8, 14, 15]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_dis = (0, 1, 2, 3, 4, 5, 6)

        self.orgImage_width = 1920.
        self.orgImage_height = 1080.
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):        
        # create train/val split
        # file_name = os.path.join(
        #     self.root, 'annot', self.image_set+'.json'
        # )
        file_name = os.path.join(
            'split', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)



        gt_db = []
        for a in anno:
            image_name = a['image']
            image_name = image_name.replace('\\', os.sep)

            c = np.array(a['center'], dtype=np.float64)
            s = np.array([a['scale'], a['scale']], dtype=np.float64)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float64)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            gt_db.append(
                {
                    'image': os.path.join(self.root, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db
                                        

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0
        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        # if 'test' in cfg.DATASET.TEST_SET:
        #     return {'Null': 0.0}, 0.0

        # SC_BIAS = 0.6
        # threshold = 0.5

        # gt_file = os.path.join(cfg.DATASET.ROOT,
        #                        'annot',
        #                        'gt_{}.mat'.format(cfg.DATASET.TEST_SET))

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join('split',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))


        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headbboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        '''
        joint_list = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 
                    'pelvis', 'naval', 'chest', 'neck', 'head',
                    'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']
        '''
        
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        
        print("data\t path\t PCKh@0.5")
        for i in range(len(headsizes)):
            print("[{}]/[{}]\t {}\t {}".format(
                i+1, len(headsizes), self.db[i]['image'].split('/')[-1], less_than_threshold[:, i].sum() / self.num_joints
            ))
        
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 17))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            one = less_than_threshold[:, 0]
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']


    # read test files
    def read_json(self, root, find_ext=(".json")):
        json_path = []
        
        for (p, d, f) in os.walk(root):
            for filename in f:
                ext = os.path.splitext(filename)[-1]
                if ext in find_ext:
                    json_path.append(os.path.join(p, filename))

        return json_path        
        

# by MSP
class NIADataset_markany_mp4(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 17
        self.flip_pair = [[0, 5], [1, 4], [2, 3], [11, 16], [12, 15], [13, 14]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 9, 12, 13, 8, 8, 14, 15]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_dis = (0, 1, 2, 3, 4, 5, 6)

        self.orgImage_width = 1920.
        self.orgImage_height = 1080.
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):        
        # create train/val split
        # file_name = os.path.join(
        #     self.root, 'annot', self.image_set+'.json'
        # )
        file_name = os.path.join(
            'split', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']
            image_name = image_name.replace('\\', os.sep)

            c = np.array(a['center'], dtype=np.float64)
            s = np.array([a['scale'], a['scale']], dtype=np.float64)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float64)

            joints = np.array(a['joints'])
            joints[:, 0:2] = joints[:, 0:2] - 1
            joints_vis = np.array(a['joints_vis'])
            assert len(joints) == self.num_joints, \
                'joint num diff: {} vs {}'.format(len(joints),
                                                    self.num_joints)

            joints_3d[:, 0:2] = joints[:, 0:2]
            joints_3d_vis[:, 0] = joints_vis[:]
            joints_3d_vis[:, 1] = joints_vis[:]

            gt_db.append(
                {
                    'image': os.path.join(self.root, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )
        return gt_db
                                        

    def evaluate(self, cfg, preds, output_dir, list_time, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        # if 'test' in cfg.DATASET.TEST_SET:
        #     return {'Null': 0.0}, 0.0

        # SC_BIAS = 0.6
        # threshold = 0.5

        # gt_file = os.path.join(cfg.DATASET.ROOT,
        #                        'annot',
        #                        'gt_{}.mat'.format(cfg.DATASET.TEST_SET))

        SC_BIAS = 0.6
        threshold = 0.5
        
        
        gt_file = 'split/gt_test.mat'
        # gt_file = os.path.join(cfg.DATASET.ROOT,
        #                        'annot',
        #                        'gt_{}.mat'.format(cfg.DATASET.TEST_SET))


        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headbboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        '''
        joint_list = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 
                    'pelvis', 'naval', 'chest', 'neck', 'head',
                    'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']
        '''
        
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
        
        save_list = []
        
        print(list_time)
        print("data\t path\t time\t PCKh@0.5")
        save_list.append("data\t path\t time\t PCKh@0.5")
        for i in range(len(headsizes)):
            str_print = "[{}]/[{}]\t {}\t {}\t {}".format(
                i+1, len(headsizes), self.db[i]['image'].split('/')[-1], list_time[i], less_than_threshold[:, i].sum() / self.num_joints
            )
            # print("[{}]/[{}]\t {}\t {}\t {}".format(
                # i+1, len(headsizes), self.db[i]['image'].split('/')[-1], list_time[i], less_than_threshold[:, i].sum() / self.num_joints
            # ))
            print(str_print)
            save_list.append(str_print)
            
                
                
        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 17))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        str_result = [
            "| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |",
            "|---|---|---|---|---|---|---|---|---|---|",
            "| unipose | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(round(PCKh[head], 4), round(0.5 * (PCKh[lsho] + PCKh[rsho]), 4), round(0.5 * (PCKh[lelb] + PCKh[relb]), 4), round(0.5 * (PCKh[lwri] + PCKh[rwri]), 4),
            round(0.5 * (PCKh[lhip] + PCKh[rhip]), 4), round(0.5 * (PCKh[lkne] + PCKh[rkne]), 4), round(0.5 * (PCKh[lank] + PCKh[rank]), 4), round(np.sum(PCKh * jnt_ratio), 4), round(np.sum(pckAll[11, :] * jnt_ratio), 4) )
        ]
        save_list.extend(str_result)
        name_value = OrderedDict(name_value)


        with open('result.txt', 'w', encoding='utf-8') as f:
        
            for i in save_list:
                f.writelines(i)
                f.writelines('\n')
            # for i in str_result:
                # f.writelines(i)
                # f.writelines('\n')

        return name_value, name_value['Mean']


    # read test files
    def read_json(self, root, find_ext=(".json")):
        json_path = []
        
        for (p, d, f) in os.walk(root):
            for filename in f:
                ext = os.path.splitext(filename)[-1]
                if ext in find_ext:
                    json_path.append(os.path.join(p, filename))

        return json_path        
        

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        '''
        img path (json) : 3 / 10 / mp4 / folder / folder_frame.png
        video path : 3 / 10 / mp4 / folder.mp4
        '''
                
        # image_file = db_rec['image']
        # filename = db_rec['filename'] if 'filename' in db_rec else ''
        # imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        # video_file = image_file.split('/')[:-1] + '.mp4'
        # frame_num = int(image_file[ image_file.rfind('_') : -4 ])
        # reader = torchvision.io.VideoReader(video_file, "video")
        # for frame in reader.seek(frame_num):
        #     data_numpy = frame['data']
        

        # read one image from video
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        
        video_file = image_file[:image_file.rfind('/')] + '.mp4'
        # print("===",video_file)
        if not os.path.exists(video_file):
            print(video_file)
        
        #return 0, 0, os.path.exists(video_file), video_file
        
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

        
        if self.color_rgb:
            try:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                image_file = np.fromfile(image_file, np.uint8)
                data_numpy = cv2.imdecode(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
                
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0


        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                
        joints_heatmap = joints.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

        target, target_weight = self.generate_target(joints_heatmap, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
        
        
        
# check file path
if __name__ == "__main__":
    pass
    
    