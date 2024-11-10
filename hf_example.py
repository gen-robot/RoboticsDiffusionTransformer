# Please first clone the repository and install dependencies
# Then switch to the root directory of the repository by "cd RoboticsDiffusionTransformer"

# Import a create function from the code base
from scripts.agilex_model import create_model
from PIL import Image as PImage
import torch
import numpy as np
import random

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
    # pretrained='robotics-diffusion-transformer/rdt-1b',
    pretrained='google/rdt-1b',
    control_frequency=25,
)

# Start inference process
# Load the pre-computed language embeddings
# Refer to scripts/encode_lang.py for how to encode the language instruction
lang_embeddings_path = 'your/language/embedding/path'
text_embedding = torch.load(lang_embeddings_path)['embeddings']  

def random_rgb():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

images = [PImage.new('RGB', (320, 240), color=random_rgb()) for _ in range(6)]

# The current robot state
# get last qpos in shape [14, ]
proprio = torch.randn(14,)
# unsqueeze to [1, 14]
proprio = proprio.unsqueeze(0)

# Perform inference to predict the next `chunk_size` actions
actions = model.step(
    proprio=proprio,
    images=images,
    text_embeds=text_embedding 
)

print(actions)