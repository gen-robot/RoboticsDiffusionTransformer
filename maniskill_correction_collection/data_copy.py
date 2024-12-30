import os
import h5py
import numpy as np

def modify_and_save_data(original_path, new_path, file_count):

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for count in range(file_count):
        original_file = f"{original_path}/episode_{count + 1}.hdf5"
        new_file = f"{new_path}/episode_{count + 1}.hdf5"
        
        with h5py.File(original_file, 'r') as original_hdf5:
            with h5py.File(new_file, 'w', rdcc_nbytes=1024**2*2) as new_hdf5:
                # Copy attributes
                for attr_name, attr_value in original_hdf5.attrs.items():
                    new_hdf5.attrs[attr_name] = attr_value

                # Copy and rename groups and datasets
                obs = new_hdf5.create_group('observations')
                image = obs.create_group('images')
                
                # Rename and copy the dataset for images
                original_images = original_hdf5['/observations/images/front']
                _ = image.create_dataset('cam_high', data=original_images[...], dtype=original_images.dtype)

                # Copy other datasets
                for dataset_name in ['qpos', '/action']:
                    if f'/observations/{dataset_name}' in original_hdf5:
                        original_data = original_hdf5[f'/observations/{dataset_name}']
                        _ = obs.create_dataset(dataset_name, data=original_data[...], dtype=original_data.dtype)
                    elif dataset_name in original_hdf5:
                        original_data = original_hdf5[dataset_name]
                        _ = new_hdf5.create_dataset(dataset_name, data=original_data[...], dtype=original_data.dtype)
        
        print(f"Processed file {count}: {original_file} → {new_file}")

modify_and_save_data('/nvme0n1/rdt/datas/StackCube-v1/correction_process', 
                     '/nvme0n1/rdt/datas/training_data/stack_cube_only_correction', 
                     400)
