import os
import pickle
import numpy as np
import cv2
import yaml
import h5py
import copy
import torch

task = "empty_cup_place_part_repick"
file_path = f'/nvme_data/embodied_agent/robotwin_data/{task}_hdf5/language_ins.pkl'

languages = {
    "blocks_stack_easy_part_repick_black": [
        "Stop and grab the black cube now.",
        "Forget what you're doing, focus on the black cube and pick it up.",
        "Immediately stop and pick up the black cube.",
        "Pause your current task and grab the black cube.",
        "Stop what you're working on and retrieve the black cube.",
        "Change your focus to the black cube and lift it.",
        "Stop and lift the black cube right now.",
        "Interrupt your task and go grab the black cube.",
        "Pause everything and pick up the black cube.",
        "Stop what you're doing and proceed with picking up the black cube.",
        "Forget the current task, pick up the black cube instead.",
        "Disregard your previous action and grab the black cube.",
        "Cancel your task and pick up the black cube now.",
        "Discontinue the current process and pick up the black cube.",
        "Stop all actions and focus on lifting the black cube.",
        "Stop and focus on picking up the black cube.",
        "Right now, pick up the black cube.",
        "Immediately grab the black cube and forget the other task.",
        "Put everything aside and pick up the black cube.",
        "Stop everything and handle the black cube first."
    ],
    "blocks_stack_easy_part_repick_red": [
        "Stop and grab the red cube now.",
        "Forget what you're doing, focus on the red cube and pick it up.",
        "Immediately stop and pick up the red cube.",
        "Pause your current task and grab the red cube.",
        "Stop what you're working on and retrieve the red cube.",
        "Change your focus to the red cube and lift it.",
        "Stop and lift the red cube right now.",
        "Interrupt your task and go grab the red cube.",
        "Pause everything and pick up the red cube.",
        "Stop what you're doing and proceed with picking up the red cube.",
        "Forget the current task, pick up the red cube instead.",
        "Disregard your previous action and grab the red cube.",
        "Cancel your task and pick up the red cube now.",
        "Discontinue the current process and pick up the red cube.",
        "Stop all actions and focus on lifting the red cube.",
        "Stop and focus on picking up the red cube.",
        "Right now, pick up the red cube.",
        "Immediately grab the red cube and forget the other task.",
        "Put everything aside and pick up the red cube.",
        "Stop everything and handle the red cube first."
    ],
    "empty_cup_place_part_repick": [
        "Stop what you're doing and pick up the cup now.",
        "Forget your current task, focus on the cup and grab it.",
        "Pause your action and lift the cup.",
        "Cancel your current task and pick up the cup immediately.",
        "Stop and grab the empty cup again.",
        "Disregard what you’re doing and focus on grabbing the cup.",
        "Interrupt your task and pick up the cup now.",
        "Stop everything and lift the cup.",
        "Change your focus to the cup and pick it up.",
        "Stop and go grab the empty cup.",
        "Immediately stop and pick up the cup.",
        "Put your current task aside and grab the cup.",
        "Discontinue your task and lift the cup.",
        "Focus on grabbing the cup and stop everything else.",
        "Right now, pick up the empty cup.",
        "Stop your current process and lift the cup.",
        "Forget the other task and focus on picking up the cup.",
        "Pause and pick up the cup as soon as possible.",
        "Put down what you're doing and pick up the cup now.",
        "Forget your current focus, just grab the cup."
    ]
}

with open(file_path, "wb") as f:
    now_language = {
        "general" : languages[task],
    }
    pickle.dump(now_language, f)