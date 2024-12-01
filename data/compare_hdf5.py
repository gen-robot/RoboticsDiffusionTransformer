import os
import io
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from IPython import embed


# data1 = h5py.File("/home/gaofeng/data/cobot_data/pick_can/episode_0.hdf5", "r")
# data2 = h5py.File("/home/gaofeng/arm_ws/EmbodiedAgent/RDT/data/datasets/agilex/test/rdt_data/close_glasses_box/episode_0.hdf5", "r")

src_dir = "/home/gaofeng/data/cobot_data/pick_can"
tgt_dir = "/home/gaofeng/data/cobot_data_processed/pick_can"

data_dict = {
    # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
    '/observations/qpos': [],
    '/observations/qvel': [],
    '/observations/effort': [],
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
        root = h5py.File(tgt_path, 'w', rdcc_nbytes=1024**2*2)
    
        root.attrs['sim'] = False
        root.attrs['compress'] = True

        obs = root.create_group('observations')
        image = obs.create_group('images')

        for cam_name in f['observations/images'].keys():
            dt = h5py.vlen_dtype(np.bytes_)
            # import pdb; pdb.set_trace()
            _ = image.create_dataset(cam_name, (data_size,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            img_list = []
            for img in f['observations/images/' + cam_name]:
                encoded_image = cv2.imencode('.jpeg', img)[1];
                # image_buffer = io.BytesIO(encoded_image) # convert array to bytes
                # image_bytes = image_buffer.getvalue() # retrieve bytes string
                # image_np = np.asarray(image_bytes)
                img_list.append(np.frombuffer(encoded_image.tobytes(), dtype='uint8'))
                # img_list.append(encoded_image.tobytes(), dtype=np.bytes_))
                recovered_image = cv2.imdecode(np.frombuffer(img_list[-1], np.uint8), cv2.IMREAD_COLOR)
                # # save raw image, encoded image and recovered image
                cv2.imwrite('raw.jpg', img)
                # cv2.imwrite('encoded.jpg', encoded_image)
                cv2.imwrite('recovered.jpg', recovered_image)
            # import pdb; pdb.set_trace()
            root['observations/images/' + cam_name][...] = img_list

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        root['observations/qpos'][...] = f['observations/qpos'][...]
        root['observations/qvel'][...] = f['observations/qvel'][...]
        root['observations/effort'][...] = f['observations/effort'][...]
        root['action'][...] = f['action'][...]
        root['base_action'][...] = f['base_action'][...]

        f.close()
        root.close()

        print(f"Processed {src_path} -> {tgt_path}")