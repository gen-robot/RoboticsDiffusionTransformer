import os
import pickle
import numpy as np
import cv2
import yaml
import h5py
import copy
import torch

file_path = '/nvme_data/embodied_agent/robotwin_data/blocks_stack_easy_part_repick_black_hdf5/language_ins.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    print(data.keys(), data['general'])