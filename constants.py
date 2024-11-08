import os 


RDT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
RDT_CONFIG_DIR = os.path.join(RDT_ROOT_DIR, "configs")
RDT_DEFAULT_CONFIG = os.path.join(RDT_CONFIG_DIR, "base.yaml")

# Pretrained Models from huggingface
SIGLIP_PATH = os.path.join(RDT_ROOT_DIR, "google/siglip-so400m-patch14-384")
T5_PATH = os.path.join(RDT_ROOT_DIR, "google/t5-v1_1-xxl")
RDT1B_PATH = os.path.join(RDT_ROOT_DIR, "google/rdt-1b")
