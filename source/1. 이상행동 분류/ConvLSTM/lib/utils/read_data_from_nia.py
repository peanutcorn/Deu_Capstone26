import xml.etree.ElementTree as ET

import os
import numpy as np
import copy             # for deep copy

from pprint import pprint

joints_name_dict = {
    "Right foot" : 0,
    "Right knee" : 1,
    "Right  hip" : 2,
    "Left hip" : 3,
    "Left knee" : 4,
    "Left foot" : 5,
    "Pelvis" : 6,
    "Spine naval" : 7,
    "Spine chest" : 8,
    "Neck base" : 9,
    "Center head" : 10,
    "Right hand" : 11,
    "Right elbow" : 12,
    "Right shoulder" : 13,
    "Left shoulder" : 14,
    "Left elbow" : 15,
    "Left hand" : 16
}

abnormal_event_dict = {
    'start' : [
        'fall_start',
        'broken_start',
        'fire_start',
        'smoke_start',
        'abandon_start',
        'theft_start',
        'fight_start'        
    ],
    'end' : [
        'fall_end',
        'broken_end',
        'fire_end',
        'smoke_end',
        'abandon_end',
        'theft_end',
        'fight_end'        
    ]
}


def create_joints_dict(num_frames):
    joint_dict = {}
    
    for i in range(1, num_frames+1):
        joint_dict[str(i)] = [-1]*17

    return joint_dict



def create_frame_joints_dict(num_frames):
    frame_joint_dict = {}
    
    for i in range(num_frames):
        frame_joint_dict[i] = {}

        for joint_name in joints_name_dict.keys():
            frame_joint_dict[i][joint_name] = {}
            frame_joint_dict[i][joint_name]['joints'] = []
            frame_joint_dict[i][joint_name]['joints_vis'] = []

    return frame_joint_dict



def read_one_xml(xml_path):
    '''
    Param : 
        xml_path : (str) xml file path
    Returns : 
        frame_joint_dict : (dict) joints dictionary according to frame
        event_dict : (dict) event dictionary (event, event_start, event_end)
    '''
    # read data
    
    tree = ET.parse(xml_path)

    # return root for tree
    root = tree.getroot()

    # create dictionary 
    event_dict = {
        'event' : None,
        'start_frame' : -1,
        'end_frame' : -1
    }
    frame_joint_dict = create_frame_joints_dict(180)
    for child in root.iter("track"):
        # event
        if child.attrib['label'] in abnormal_event_dict['start']:
            event_dict['event'] = child.attrib['label'].split('_')[0]
            frame = child.find("box").attrib["frame"]
            event_dict['start_frame'] = int(frame)
        elif child.attrib['label'] in abnormal_event_dict['end']:
            event_dict['event'] = child.attrib['label'].split('_')[0]
            try:
                frame = child.find("box").attrib["frame"]
            except:
                return {}, {}
            event_dict['end_frame'] = int(frame)
        # keypoint
        elif child.attrib['label'] in joints_name_dict.keys():
            for child_sub in child.iter("points"):
                # remove outside=1
                if int(child_sub.attrib['outside']) == 1:
                    continue
                frame = child_sub.attrib["frame"]
                if int(frame) >= 180:
                    continue
                points = list(map(float, child_sub.attrib["points"].split(',')))
                occluded = abs(int(child_sub.attrib["occluded"]) - 1)
                
                try:
                    frame_joint_dict[int(frame)][child.attrib['label']]['joints'].append(points)
                except:
                    print("frame = {}".format(frame))
                    print("points = {}".format(points))
                        
                frame_joint_dict[int(frame)][child.attrib['label']]['joints_vis'].append(occluded)
                
    return frame_joint_dict, event_dict



def get_one_gt(video_path, xml_path):
    '''
    Param : 
        video_path : (str) video folder
        xml_path : (str) xml folder
    Returns : 
        joints
        event
    '''

    # print(video_path)
    # print(xml_path)

    # get joint_dict, event_dict
    frame_joint_dict, event_dict = read_one_xml(xml_path)

    joint_list = []
    for frame, joint_dict in frame_joint_dict.items():
        # labeling : (start_frame <= frame <= end_frame)
        if (frame < event_dict['start_frame']):
            continue
        elif (frame > event_dict['end_frame']):
            break

        num_person = len(joint_dict['Left foot']['joints'])         # hard coding;
        if num_person == 0:
            continue
        for i in range(num_person):
            save_dict = {}
            joints, joints_vis = [-1] * 17, [0] * 17
            # get joints, joints_vis
            for joint_name in joint_dict.keys():
                try:                    
                    joints[joints_name_dict[joint_name]] = joint_dict[joint_name]['joints'][i]
                    joints_vis[joints_name_dict[joint_name]] = joint_dict[joint_name]['joints_vis'][i]
                except IndexError:
                    print(joint_dict[joint_name]['joints'])
                    print(joint_dict[joint_name]['joints_vis'])
                    # miss labeling (no all keypoints (ex) must be 17 points, but miss 3 points)
                    return [], {}
            
            save_dict['joints_vis'] = joints_vis
            save_dict['joints'] = joints
            
            '''
            if ('c_3_7_1_bu_dya_07-31_15-15-28_cb_rgb_df2_m1.xml' in xml_path):
                print("Right hand = {}".format(joint_dict["Right hand"]['joints']))
            '''
            
            # get center, std, image
            joints, joints_vis = np.array(joints), np.array(joints_vis)
            max_x, min_x = np.max(joints[:, 0]), np.min(joints[:, 0])
            max_y, min_y = np.max(joints[:, 1]), np.min(joints[:, 1])
            w, h = (max_x-min_x), (max_y-min_y)
            center_x, center_y = min_x + w/2, min_y + h/2
            std = max(w, h) / 200 * 1.1
            save_dict['center'] = [center_x, center_y]
            save_dict['scale'] = std            
            video_name = video_path.split(os.sep)[-1]
            img_path = os.path.join(video_path,
                                    video_name + "_" + (str(frame+1).zfill(6) + ".png"))
            save_dict['image'] = img_path            
            
            # if 'c_3_7_1' in xml_path:
            #     pprint(save_dict)
            
            joint_list.append(save_dict)
            
    return joint_list, event_dict
            
            

def get_all_gt(data_dir, event_list):    
    # event
    event_folders = os.listdir(data_dir)
    event_folders = [x for x in event_folders if int(x[:2]) in event_list]
    joints_list = []
    events_list = []
    
    for event_folder in event_folders:
        # get xml/video folder path
        data_path = os.path.join(data_dir, event_folder)
        data_folders = os.listdir(data_path)
        for data_folder in data_folders:
            if 'xml' in data_folder:
                xml_path = os.path.join(data_path, data_folder)
            elif 'mp4' in data_folder:
                video_path = os.path.join(data_path, data_folder)
        all_event_xml = os.listdir(xml_path)
        for one_xml in all_event_xml:
            video_folders = [x for x in os.listdir(video_path) if '.mp4' not in x or '.xml' not in x]            
            for video_folder in video_folders:
                if video_folder.lower() == one_xml.split('.')[0].lower():
                    video_sub_path = os.path.join(video_path, video_folder)
                    break

            print(video_sub_path)
            print(os.path.join(xml_path, one_xml))
            print()
            
            joint_list, event_dict = get_one_gt(video_sub_path, os.path.join(xml_path, one_xml))
            joints_list.extend(joint_list)
            events_list.append(event_dict)
                        
    return joints_list, events_list
    
                    
def init_mat_file():
    mat_dict = dict()
    
    # __globals__
    mat_dict['__globals__'] = []
                    
    # __header__
    mat_dict['__header__'] = 'created by MarkAny (2023-02-02)'

    # __version__
    mat_dict['__version__'] = '1.0'
    
    # dataset_joints
    joint_list = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 
                  'pelvis', 'naval', 'chest', 'neck', 'head',
                  'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']
    dataset_joint = np.array(joint_list, dtype='<U4')
    dataset_joint = dataset_joint[np.newaxis, :]
    mat_dict['dataset_joints'] = dataset_joint

    return mat_dict

'''
    "Right foot" : 0,
    "Right knee" : 1,
    "Right  hip" : 2,
    "Left hip" : 3,
    "Left knee" : 4,
    "Left foot" : 5,
    "Pelvis" : 6,
    "Spine naval" : 7,
    "Spine chest" : 8,
    "Neck base" : 9,
    "Center head" : 10,
    "Right hand" : 11,
    "Right elbow" : 12,
    "Right shoulder" : 13,
    "Left shoulder" : 14,
    "Left elbow" : 15,
    "Left hand" : 16
'''



def read_one_xml_onlyEvent(xml_path):
    '''
    Param : 
        xml_path : (str) xml file path
    Returns : 
        frame_joint_dict : (dict) joints dictionary according to frame
        event_dict : (dict) event dictionary (event, event_start, event_end)
    '''
    # read data
    
    tree = ET.parse(xml_path)

    # return root for tree
    root = tree.getroot()

    # create dictionary 
    event_dict = {
        'event' : None,
        'start_frame' : -1,
        'end_frame' : -1
    }
    for child in root.iter("track"):
        # event
        if child.attrib['label'] in abnormal_event_dict['start']:
            event_dict['event'] = child.attrib['label'].split('_')[0]
            frame = child.find("box").attrib["frame"]
            event_dict['start_frame'] = int(frame)
        elif child.attrib['label'] in abnormal_event_dict['end']:
            event_dict['event'] = child.attrib['label'].split('_')[0]
            try:
                frame = child.find("box").attrib["frame"]
            except:
                return {}
            event_dict['end_frame'] = int(frame)
                
    return event_dict




def get_one_gt_onlyEvent(xml_path):
    '''
    Param : 
        video_path : (str) video folder
        xml_path : (str) xml folder
    Returns : 
        joints
        event
    '''

    # print(video_path)
    # print(xml_path)

    # get event_dict
    event_dict = read_one_xml_onlyEvent(xml_path)
    
    # remove 
    if (event_dict['end_frame'] == -1) or (event_dict['start_frame'] == -1) or (event_dict['event']=='None') or not event_dict:
        print("error json : {}".format(xml_path))
        return None

    return event_dict
    
    
def get_all_gt_onlyEvent(data_dir, event_list):    
    # event
    event_folders = os.listdir(data_dir)
    event_folders = [x for x in event_folders if int(x[:2]) in event_list]
    events_list = []
    event_type = {
        'fall' : 1,
        'broken' : 2,
        'fire' : 3,
        'smoke' : 4,
        'abandon' : 5,
        'theft' : 6,
        'fight' : 7
    }
    frames_list = []

    for event_folder in event_folders:
        # get xml/video folder path
        data_path = os.path.join(data_dir, event_folder)
        data_folders = os.listdir(data_path)
        for data_folder in data_folders:
            if 'xml' in data_folder:            
                xml_path = os.path.join(data_path, data_folder)
            elif 'mp4' in data_folder:
                video_path = os.path.join(data_path, data_folder)
        all_event_xml = os.listdir(xml_path)
        for one_xml in all_event_xml:
            video_folders = [x for x in os.listdir(video_path) if '.mp4' not in x or '.xml' not in x]            
            for video_folder in video_folders:
                if video_folder.lower() == one_xml.split('.')[0].lower():
                    video_sub_path = os.path.join(video_path, video_folder)
                    break

            event_dict = get_one_gt_onlyEvent(os.path.join(xml_path, one_xml))

            from pprint import pprint
            if event_dict is not None:
                pprint(event_dict)
                events_list.append(event_dict)

                # split 3 frames
                current_frame = 0
                while (current_frame < event_dict['start_frame']):
                    tmp = []
                    tmp.append([
                        os.path.join(video_sub_path, video_folder + '_' + str(current_frame+1).zfill(6) + '.png'),
                        os.path.join(video_sub_path, video_folder + '_' + str(current_frame+2).zfill(6) + '.png'),
                        os.path.join(video_sub_path, video_folder + '_' + str(current_frame+3).zfill(6) + '.png')
                    ])
                    if (current_frame+3) >= event_dict['start_frame']:
                        tmp.append(event_type[event_dict['event']])
                    else:
                        tmp.append(0)
                    current_frame += 3
                    frames_list.append(tmp)

                # add additional
                tmp = []
                tmp.append([
                    os.path.join(video_sub_path, video_folder + '_' + str(current_frame+1).zfill(6) + '.png'),
                    os.path.join(video_sub_path, video_folder + '_' + str(current_frame+2).zfill(6) + '.png'),
                    os.path.join(video_sub_path, video_folder + '_' + str(current_frame+3).zfill(6) + '.png')
                ])
                tmp.append(event_type[event_dict['event']])
                frames_list.append(tmp)

            print(video_sub_path)
            print(os.path.join(xml_path, one_xml))
            print()
            #pprint(frames_list)
                        
    return events_list, frames_list
    
    
                    
def init_mat_file():
    mat_dict = dict()
    
    # __globals__
    mat_dict['__globals__'] = []
                    
    # __header__
    mat_dict['__header__'] = 'created by MarkAny (2023-02-02)'

    # __version__
    mat_dict['__version__'] = '1.0'
    
    # dataset_joints
    joint_list = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 
                  'pelvis', 'naval', 'chest', 'neck', 'head',
                  'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']
    dataset_joint = np.array(joint_list, dtype='<U4')
    dataset_joint = dataset_joint[np.newaxis, :]
    mat_dict['dataset_joints'] = dataset_joint

    return mat_dict    
    
    
                    
if __name__ == "__main__":
    """
    data_dir = '../../data/nia/3 이상행동'
    save_dir = 'sample'
    
    db = get_all_gt(data_dir)
    from pprint import pprint
    
    #pprint(db)
    
    import cv2 
    
    pprint(db[358])
    pprint(db[393])
    """
    
    
    # get db
    data_dir = '3 이상행동'
    #choose_event = ['7 전도', '8 파손', '9 방화', '10 흡연', '11 유기', '12 절도', '13 폭행']
    choose_event = [7, 8, 9, 19, 11, 12, 13]
    choose_event = [7]
    
    #joints_list, events_list = get_all_gt(data_dir, choose_event)
    events_list, frames_list = get_all_gt_onlyEvent(data_dir, choose_event)
    print(len(events_list))
    print(len(frames_list))
    
    from pprint import pprint 
    for one_event in events_list:
        pprint(one_event)

    pprint(frames_list)    
    
    
    # write json file
    """
    import json

    data_dir = '3 이상행동'
    
    mat_dict = init_mat_file()
    pprint(mat_dict)
        
    # ======================================== #
    # read joint
    # ======================================== #
    joint_path = 'train_08.json'
    with open(joint_path, 'r') as joint_file:
        joint_data = json.load(joint_file)    


    # ======================================== #
    # read bbox
    # ======================================== #
    bbox_path = 'bbox.json'
    with open(bbox_path, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)
    
    print("len(joint_data) = {}".format(len(joint_data)))
    print("len(bbox_data) = {}".format(len(bbox_data)))    
    
    
    
    total_data = copy.deepcopy(joint_data)

    val_mat_dict = {}
    val_mat_bboxes = []
    val_json_list = []
    for bbox in bbox_data:
        for joint in total_data:
            # print(bbox['image'].lower())
            # print(joint['image'].split(os.sep)[-1].lower())
            # print()
            if bbox['image'].lower() in joint['image'].split(os.sep)[-1].lower():
                print(bbox['image'].lower())
                print(joint['image'].lower())
                print()
                
                joint['headbboxes_src'] = bbox['headbboxes_src']
                break
            else:
                joint['headbboxes_src'] = []
        
        
    for data in total_data:
        print(not data['headbboxes_src'])
        print(len(data['headbboxes_src']))
        print()
    
    #pprint(total_data)
    """ 