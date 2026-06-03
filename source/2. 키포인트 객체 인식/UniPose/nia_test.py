import os
import json
import argparse
import shutil

import numpy as np

def get_db():

    # create train/val split
    # file_name = os.path.join(
    #     self.root, 'annot', self.image_set+'.json'
    # )
    file_name = 'split/test.json'
    
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

        joints_3d = np.zeros((17, 3), dtype=np.float64)
        joints_3d_vis = np.zeros((17,  3), dtype=np.float64)

        joints = np.array(a['joints'])
        joints[:, 0:2] = joints[:, 0:2] - 1
        joints_vis = np.array(a['joints_vis'])

        joints_3d[:, 0:2] = joints[:, 0:2]
        joints_3d_vis[:, 0] = joints_vis[:]
        joints_3d_vis[:, 1] = joints_vis[:]

        gt_db.append(
            {
                'image': os.path.join('/mnt/f/NIA', image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            }
        )
    
    return gt_db
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-dir', type=str, default='/mnt/f/NIA/3 이상행동')
    parser.add_argument('--save-dir', type=str, default='/mnt/f/save_dir/3 이상행동')
    
    args = parser.parse_args()
    
    db = get_db()

    print(len(db))
    
    video_set = set()
    for data in db:
        image_file = data['image']
        save_file = image_file.replace(args.data_dir, args.save_dir)
        filename = data['filename'] if 'filename' in data else ''
        imgnum = data['imgnum'] if 'imgnum' in data else ''
        video_file = image_file[:image_file.rfind('/')] + '.mp4'
        
        xml_file = video_file.replace('mp4', 'xml')
        
        video_save_file = video_file.replace(args.data_dir, args.save_dir)
        xml_save_file = xml_file.replace(args.data_dir, args.save_dir)

        idx_split = xml_file.rfind('/')
        xml_folder = xml_file[:idx_split]
        xml_save_folder = xml_folder.replace(args.data_dir, args.save_dir)
        video_save_folder = xml_save_folder.replace('xml', 'mp4')
        print(xml_save_folder)
        # if not os.path.exists(xml_save_folder):
            # os.makedirs(xml_save_folder, exist_ok=True)
        # if not os.path.exists(video_save_folder):
            # os.makedirs(video_save_folder, exist_ok=True)
        
        if not os.path.exists(video_save_file):
            shutil.copyfile(video_file, video_save_file)
            print("=====> save video file =====")
            print(video_file)
            print("↓")
            print(video_save_file)
        if not os.path.exists(xml_save_file):
            shutil.copyfile(xml_file, xml_save_file)
            print("=====> save xml file =====")
            print(xml_file)
            print("↓")
            print(xml_save_file)
        

        

        # if not os.path.exists(xml_file):
            # print(xml_file)
            # print()
        
        video_set.add(video_file)
        
        
    '''
    from pprint import pprint
    pprint(video_set)
    print(len(video_set))
    
    print(args.save_dir)

    video_set = list(video_set)   
       
    print(video_set[0])
    print()
    print(video_set[1])
    '''