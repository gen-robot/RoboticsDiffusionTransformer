# Please first clone the repository and install dependencies
# Then switch to the root directory of the repository by "cd RoboticsDiffusionTransformer"

# Import a create function from the code base
import torch
from PIL import Image
from scripts.agilex_model import create_model
from typing import List, Optional
import cv2
import numpy as np
from collections import deque

########################## preliminary work ##########################

observation_window = None

# Get the observation from the ROS topic
def get_observation(args):
    result = ... # TODO
    (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
        puppet_arm_left, puppet_arm_right, robot_base) = result
    # print(f"sync success when get_ros_observation")
    return (img_front, img_left, img_right,
        puppet_arm_left, puppet_arm_right)


# Update the observation window buffer
def update_observation_window(args, config):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
            }
        )
        
    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_observation(args)
    
    img_front = jpeg_mapping(img_front)
    img_left = jpeg_mapping(img_left)
    img_right = jpeg_mapping(img_right)
    
    qpos = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
    qpos = torch.from_numpy(qpos).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                    config["camera_names"][2]: img_left,
                },
        }
    )

########################## formal work ##########################

# Names of cameras used for visual input
CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
config = {
    'episode_len': 1000,  # Max length of one episode
    'state_dim': 14,      # Dimension of the robot's state
    'chunk_size': 64,     # Number of actions to predict in one step
    'camera_names': CAMERA_NAMES,
}
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384" 
# Create the model with the specified configuration
model = create_model(
    args=config,
    dtype=torch.bfloat16, 
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
    pretrained='robotics-diffusion-transformer/rdt-1b',
    control_frequency=25,
)

# Start inference process
# Load the pre-computed language embeddings
# Refer to scripts/encode_lang.py for how to encode the language instruction
lang_embeddings_path = 'your/language/embedding/path'
text_embedding = torch.load(lang_embeddings_path)['embeddings']

# TODO
update_observation_window(None,None)

image_arrs = [
    observation_window[-2]['images'][config['camera_names'][0]],
    observation_window[-2]['images'][config['camera_names'][1]],
    observation_window[-2]['images'][config['camera_names'][2]],
    
    observation_window[-1]['images'][config['camera_names'][0]],
    observation_window[-1]['images'][config['camera_names'][1]],
    observation_window[-1]['images'][config['camera_names'][2]]
]



# The images from last 2 frames
images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]  
# The current robot state # get last qpos in shape [14, ]
proprio = observation_window[-1]['qpos'] 
# unsqueeze to [1, 14]
proprio = proprio.unsqueeze(0)
# Perform inference to predict the next `chunk_size` actions

actions = model.step(  # policy.step()
    proprio=proprio,
    images=images,
    text_embeds=text_embedding 
)
