import os
import io
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

# from IPython import embed


# data1 = h5py.File("/home/gaofeng/data/cobot_data/pick_can/episode_0.hdf5", "r")
# data2 = h5py.File("/home/gaofeng/arm_ws/EmbodiedAgent/RDT/data/datasets/agilex/test/rdt_data/close_glasses_box/episode_0.hdf5", "r")

src_dir = "/home/gaofeng/arm_ws/EmbodiedAgent/embodied_agent/third_party/vla/rdt/data/datasets/agilex/cobot_data/new_open_drawer_gf"
tgt_dir = "/home/gaofeng/arm_ws/EmbodiedAgent/embodied_agent/third_party/vla/rdt/data/datasets/agilex/cobot_data/new_open_drawer_processed"

data_dict = {
    # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
    '/observations/qpos': [],
    '/observations/qvel': [],
    '/observations/effort': [],
    '/observations/ee_pose': [],
    '/observations/images/cam_high': [],
    '/observations/images/cam_high': [],
    '/observations/images/cam_high': [],
    '/action': [],
    '/base_action': [],
}

for dir_root, _, files in os.walk(src_dir, followlinks=True):
    for filename in sorted(fnmatch.filter(files, '*.hdf5')):
        src_path = os.path.join(dir_root, filename)
        f = h5py.File(src_path, 'r')

        data_size = f['observations/qpos'].shape[0]
        print(f"Processing {src_path} with {data_size} samples")

        tgt_path = src_path.replace(src_dir, tgt_dir)
        tgt_base = os.path.basename(tgt_path)
        if not os.path.exists(tgt_base):
            os.makedirs(tgt_base, exist_ok=True)
        root = h5py.File(tgt_path, 'w') #, rdcc_nbytes=1024**2*2)
    
        root.attrs['sim'] = False
        root.attrs['compress'] = True

        obs = root.create_group('observations')
        image = obs.create_group('images')

        for cam_name in f['observations/images'].keys():
            max_length = 0
            img_list = []
            for img in f['observations/images/' + cam_name]:
                # encoded_image = cv2.imencode('.jpeg', img)[1]
                encoded_image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                encoded_image = cv2.imencode('.jpeg', encoded_image)[1]
                img_bytes = encoded_image.tobytes()
                img_list.append(img_bytes)
                max_length = max(max_length, len(img_bytes))

            # img_list = np.array(img_list, dtype=np.uint8)
            fixed_length_dtype = f'|S{max_length}'
            _ = image.create_dataset(cam_name, shape=len(img_list), dtype=fixed_length_dtype, chunk=1)
            # Write images into the dataset
            for i, img_bytes in enumerate(img_list):
                root['observations/images/' + cam_name][i] = img_bytes
            # image.create_dataset(cam_name, shape=len(img_list), data=img_list, dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
            # import pdb; pdb.set_trace()
            # root['observations/images/' + cam_name][...] = img_list

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = obs.create_dataset('ee_pose', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        root['observations/qpos'][...] = f['observations/qpos'][...]
        root['observations/qvel'][...] = f['observations/qvel'][...]
        root['observations/effort'][...] = f['observations/effort'][...]
        root['observations/ee_pose'][...] = f['observations/ee_pose'][...]
        root['action'][...] = f['action'][...]
        root['base_action'][...] = f['base_action'][...]

        f.close()
        root.close()

        print(f"Processed {src_path} -> {tgt_path}")
        # exit(0)
