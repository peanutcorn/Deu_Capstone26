from scipy.io import loadmat, savemat



# gt_file = os.path.join(cfg.DATASET.ROOT,
#                         'annot',
#                         'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
# gt_dict = loadmat(gt_file)
# dataset_joints = gt_dict['dataset_joints']
# jnt_missing = gt_dict['jnt_missing']
# pos_gt_src = gt_dict['pos_gt_src']
# headboxes_src = gt_dict['headboxes_src']



gt_path = 'gt_valid.mat'


gt_dict = loadmat(gt_path)

from pprint import pprint 

pprint(gt_dict)

dataset_joints = gt_dict['dataset_joints']

headbboxes_src = gt_dict['headboxes_src']
pos_gt_src = gt_dict['pos_gt_src']
pos_pred_src = gt_dict['pos_pred_src']
jnt_missing = gt_dict['jnt_missing']

print("len(dataset_joints) = {}".format(len(dataset_joints)))
print("len(headbboxes_src) = {}".format(len(headbboxes_src)))



print(headbboxes_src.shape)
print(headbboxes_src[:, :, 0])
print("pos_gt_src.shape = {}".format(pos_gt_src.shape))
print("pos_pred_src.shape = {}".format(pos_pred_src.shape))

print(gt_dict.keys())



import json

json_path = 'valid.json'

with open(json_path, 'r') as json_file:
    json_data = json.load(json_file)

print(len(json_data))



print("==========" * 5)
pprint(json_data[0])
pprint(pos_gt_src[:, :, 0])
pprint(headbboxes_src[:, :, 0])




import cv2

img_paths = [
    '005808361.jpg',
    '052475643.jpg',
    '051423444.jpg'
]

#img_path = '005808361.jpg'
#img_path = '052475643.jpg'
#img_path = '051423444.jpg'

print(dataset_joints)
print(dataset_joints.shape)

import numpy as np
head = np.where(dataset_joints == 'head')[1][0]
pprint(head)


b = np.where(dataset_joints=='rhip')
print(dataset_joints.shape)
print(b)
print(b[1])
print(b[1][0])

joint_list = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 
                'pelvis', 'naval', 'chest', 'neck', 'head',
                'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']
dataset_joint = np.array(joint_list, dtype='<U4')
dataset_joint = dataset_joint[np.newaxis, :]
c = np.where(dataset_joint=='rhip')
print(c)


# for i in dataset_joints[0]:
#     print(i)
#     print(i.dtype)

'''
for i, img_path in enumerate(img_paths):

    img_numpy = cv2.imread(img_path)

    joint = pos_gt_src[:, :, i]
    bbox = headbboxes_src[:, :, i]
    joint_vis = 1 - jnt_missing[:, i]

    img_numpy = cv2.rectangle(img_numpy, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), (0, 255, 0), 2)

    for j in range(len(joint)):
        if joint_vis[j] == 0:
            color = (127, 127, 127)
        else:
            color = (255, 0, 0)
        new_joints = (int(joint[j][0]), int(joint[j][1]))
        img_numpy = cv2.circle(img_numpy, new_joints, 4, color, -1, 1)


    img_save = img_path.split('.')[0] + '_draw.jpg'
    cv2.imwrite(img_save, img_numpy)
'''
