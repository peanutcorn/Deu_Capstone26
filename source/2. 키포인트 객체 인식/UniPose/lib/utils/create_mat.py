from read_data_from_nia import get_all_gt





if __name__ == "__main__":
    from pprint import pprint
    import cv2
    import os

    root_dir = '2023 NIA 머리'

    data_dir = os.listdir(root_dir)
    data_dir = [x for x in data_dir if not x.endswith('zip')]

    print(data_dir)

    '''
    data_dir = '../../data/nia/3 이상행동'
    save_dir = 'sample'
    
    db = get_all_gt(data_dir)
    #print("len(db) = {}".format(len(db)))
    
    pprint(db)
    
    
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
        
        