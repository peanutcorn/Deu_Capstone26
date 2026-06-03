import xml.etree.ElementTree as ET

import os
import numpy as np

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
    
    for i in range(1, num_frames+1):
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
            frame = child.find("box").attrib["frame"]
            event_dict['end_frame'] = int(frame)
        # keypoint
        elif child.attrib['label'] in joints_name_dict.keys():
            for child_sub in child.iter("points"):
                # remove outside=1
                if int(child_sub.attrib['outside']) == 1:
                    continue
                frame = child_sub.attrib["frame"]
                points = list(map(float, child_sub.attrib["points"].split(',')))
                occluded = abs(int(child_sub.attrib["occluded"]) - 1)
                
                frame_joint_dict[int(frame)][child.attrib['label']]['joints'].append(points)
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

    print(video_path)
    print(xml_path)

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
                    # miss labeling (no all keypoints (ex) must be 17 points, but miss 3 points)
                    return joint_list
            
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
            
            

def get_all_gt(data_dir):    
    # event
    event_folders = os.listdir(data_dir)
    joints_list = []
    event_list = []
    
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

            joint_list, event_dict = get_one_gt(video_sub_path, os.path.join(xml_path, one_xml))
            joints_list.extend(joint_list)
                        
    return joints_list
    
                    
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
    
    from pprint import pprint
    import cv2

    data_dir = '../../data/nia/3 이상행동'
    save_dir = 'sample'
    
    db = get_all_gt(data_dir)
    #print("len(db) = {}".format(len(db)))
    
    
    for i in range(len(db)):
        img_path = db[i]['image']
        
        print(img_path)
        img_numpy = cv2.imread(img_path)
        joints = db[i]['joints']
        joints_vis = db[i]['joints_vis']

        for j in range(len(joints)):
            if joints_vis[j] == 1:
                color = (0, 0, 255)
            else:
                color = (127, 127, 127)
            new_joints = (int(joints[j][0]), int(joints[j][1]))
            img_numpy = cv2.circle(img_numpy, new_joints, 4, color, -1, 1)
        
        save_name = img_path.split(os.sep)[-1].split('.')[0] + '_' + str(i) + ".jpg"
        cv2.imwrite(os.path.join(save_dir, save_name), img_numpy)


    '''
    for i in range(len(db)):
        img_path = db[i]['image']
        joints = db[i]['joints']
        joints_vis = db[i]['joints_vis']
        
        img_numpy = cv2.imread(img_path)        
        
        print("img_path = {}".format(img_path))
        print(joints[11])
        # print(db[i]['joints'][11], db[i]['joints_vis'][11])

        import sys
        print(joints)
        print(len(joints))
    
        for j in range(len(joints)):
            if joints_vis[j] == 1:
                color = (0, 0, 255)
            else:
                color = (127, 127, 127)
            new_joints = (int(joints[j][0]), int(joints[j][1]))
            if j == 11:
                print(new_joints)
            img_numpy = cv2.circle(img_numpy, new_joints, 4, color, -1, 1)

        save_name = img_path.split(os.sep)[-1].split('.')[0] + '_' + str(i) + ".jpg"
        cv2.imwrite(os.path.join(save_dir, save_name), img_numpy)
    '''
    
    
    '''
    count = 0
    for i in range(len(db)):
    # for i in range(3):
        img_path = db[i]['image']
        img_numpy = cv2.imread(img_path)
        
        if 'C_3_7_1_BU_DYA_07-31_15-15-28_CB_RGB_DF2_M1' not in img_path:
            continue
        
        keypoints = db[i]['joints']
        keypoints_vis = db[i]['joints_vis']
                
        for j in range(len(keypoints)):
            if keypoints_vis[j] == 1:
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
                
        
            #cv2.circle(img_numpy, (int(keypoints[j][0]), int(keypoints[j][1])), 2, color, -1)        
            cv2.circle(img_numpy, (int(keypoints[j][0]), int(keypoints[j][1])), 4, color, -1)        
        
        Left_foot = "(" + str(int(keypoints[11][0])) + ", " + str(int(keypoints[11][1])) + ")"
        cv2.putText(img_numpy, Left_foot, (100, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        print(i, keypoints)
        save_name = img_path.split(os.sep)[-1].split('.')[0] + '_' + str(i) + ".jpg"
        print(save_name)
        cv2.imwrite(os.path.join(save_dir, save_name), img_numpy)
    '''
        
        
        
    
    
    '''
    video_path = 'data/nia/3 이상행동/7 전도/_S3 최종데이터 mp4_전도807/C_3_7_1_BU_DYA_07-31_15-15-25_CA_RGB_DF2_M1'
    xml_path = 'data/nia/3 이상행동/7 전도/_S3 최종데이터 xml_전도807/c_3_7_1_bu_dya_07-31_15-15-25_ca_rgb_df2_m1.xml'
    
    joint_list, event_dict = get_one_gt(video_path, xml_path)

    from pprint import pprint
    pprint(joint_list)
    print(len(joint_list))
    '''
    
    
    
    
    '''
    gt_list = []
    folder_list = os.listdir(data_dir)
    for folder in folder_list:
        frame_joint_dict = read_one_xml(os.path.join(data_dir, folder, folder.upper()+'.xml'))
        gt = get_one_gt(folder, frame_joint_dict)
        gt_list.extend(gt)
    
    from pprint import pprint
    #pprint(gt_list)
    print(len(gt_list))
    '''

'''
import xml.etree.ElementTree as ET
tree = ET.parse(xml_path)



'''