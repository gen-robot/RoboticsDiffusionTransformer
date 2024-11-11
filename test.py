

import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.hdf5_vla_dataset import HDF5VLADataset
from train.image_corrupt import image_corrupt
from train.dataset import VLAConsumerDataset,DataCollatorForVLAConsumerDataset,get_clean_item,save_dirty_bit,read_dirty_bit


import copy
import logging
import math
import os
from pathlib import Path
import argparse
import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from safetensors.torch import load_model

from models.ema_model import EMAModel
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res

from constants import RDT_ROOT_DIR


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Main script for training RDT.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base.yaml",
        help="Path to the configuration file. Default is `configs/base.yaml`.",
    )

    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default=None,
        help="Pretrained text encoder name or path if not the same as model_name",
    )
    parser.add_argument(
        "--pretrained_vision_encoder_name_or_path",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Pretrained vision encoder name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--load_from_hdf5",
        action="store_true",
        default=True,
        help=(
            "Whether to load the dataset directly from HDF5 files. "
            "If False, the dataset will be loaded using producer-consumer pattern, "
            "where the producer reads TFRecords and saves them to buffer, and the consumer reads from buffer."
        )
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=8, help="Batch size (per device) for the sampling dataloader."
    )
    parser.add_argument(
        "--num_sample_batches", type=int, default=2, help="Number of batches to sample from the dataset."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_period",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_period`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Path or name of a pretrained checkpoint to load the model from.\n",
            "   This can be either:\n"
            "   - a string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co, e.g., `robotics-diffusion-transformer/rdt-1b`,\n"
            "   - a path to a *directory* containing model weights saved using [`~RDTRunner.save_pretrained`] method, e.g., `./my_model_directory/`.\n"
            "   - a path to model checkpoint (*.pt), .e.g, `my_model_directory/checkpoint-10000/pytorch_model/mp_rank_00_model_states.pt`"
            "   - `None` if you are randomly initializing model using configuration at `config_path`."
        )
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cond_mask_prob",
        type=float,
        default=0.1,
        help=(
            "The probability to randomly mask the conditions (except states) during training. "
            "If set to 0, the conditions are not masked."
        ),
    )
    parser.add_argument(
        "--cam_ext_mask_prob",
        type=float,
        default=-1.0,
        help=(
            "The probability to randomly mask the external camera image during training. "
            "If set to < 0, the external camera image is masked with the probability of `cond_mask_prob`."
        ),
    )
    parser.add_argument(
        "--state_noise_snr",
        type=float,
        default=40,
        help=(
            "The signal-to-noise ratio (SNR, unit: dB) for adding noise to the states. "
            "Default is None, which means no noise is added."
        ),
    )
    parser.add_argument(
        "--image_aug",
        action="store_true",
        default=True,
        help="Whether or not to apply image augmentation (ColorJitter, blur, noise, etc) to the input images.",
    )
    parser.add_argument(
        "--precomp_lang_embed",
        action="store_true",
        default=True,
        help="Whether or not to use precomputed language embeddings.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    parser.add_argument('--dataset_type', 
        type=str, 
        default="finetune",
        required=False,
        help="Whether to load the pretrain dataset or finetune dataset."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def get_raw_data(args):
    data_list=[]
    
    # args.config_path = "configs/base.yaml"
    # args.pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    # args.dataset_type = "finetune"
    # args.image_aug = True
    # args.cond_mask_prob = 0.1
    # args.cam_ext_mask_prob = -1.0
    # args.state_noise_snr= 40
    # args.load_from_hdf5 = True
    # args.precomp_lang_embed = True
    
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
        
    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor
    
    # Dataset and DataLoaders creation:                                                         
    train_dataset = MyVLAConsumerDataset(
        config=config["dataset"],
        tokenizer=None,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=args.train_batch_size,
        shuffle=False,
        # collate_fn=data_collator,
        # num_workers=args.dataloader_num_workers,
        # pin_memory=True,
        # persistent_workers=True
    )
    
    for step_id in range(0,10):
        item = train_dataset.__getitem__(0, step_id)
        data_list.append(item)
        print("appending data: ",item["step_id"],"/",item["total_timesteps"])       
        step_id += 1
    return data_list
    
    
class MyVLAConsumerDataset(VLAConsumerDataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """
    
    def __init__(
        self, 
        config,
        image_processor,
        num_cameras,
        img_history_size,
        tokenizer=None,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type='finetune',
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        use_precomp_lang_embed=False,
    ):
        tokenizer = None  
        all_kwargs = locals()
        all_kwargs.pop("self")
        all_kwargs.pop("__class__")
        super().__init__(**all_kwargs)


    def __getitem__(self, index, step_id):
        # For robustness, we will try to load the data until we succeed
        '''
            return
                data_dict={
                    dataset_name
                    step_id
                    total_timesteps
                    instruction
                    dataset_idx
                    ctrl_freq
                    states
                    actions
                    state_elem_mask
                    state_norm
                    images
                }
        '''
        while True:
            data_dict = None
            try:
                if self.use_hdf5:
                    # index indicates the index of the episode, step_id indicates the id of steps of this episode.
                    res = self.hdf5_dataset.get_item(step_id=step_id, index=0) # not use state_only
                    content = res['meta']
                    states = res['state']
                    actions = res['actions']
                    state_elem_mask = res['state_indicator']
                    image_metas = [
                        res['cam_high'], res['cam_high_mask'],
                        res['cam_right_wrist'], res['cam_right_wrist_mask'],
                        res['cam_left_wrist'], res['cam_left_wrist_mask'],
                    ]
                    state_std = res['state_std']
                    state_mean = res['state_mean']
                    state_norm = res['state_norm']
                else:
                    (content, _, states, _, actions, _, 
                    state_elem_mask, *image_metas, 
                    state_std, state_mean, state_norm) = self._safe_load(index)
                
                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name'] # this have 
                data_dict['step_id'] = content['step_id']
                data_dict['total_timesteps'] = content['#steps']
                data_dict['instruction'] = content['instruction']
                data_dict['dataset_idx'] = self.dataset_name2id[data_dict['dataset_name']]
                data_dict['ctrl_freq'] = self.control_freq[data_dict['dataset_name']] \
                    if random.random() > self.cond_mask_prob else 0
                
                if self.state_noise_snr is not None:
                    states += np.random.normal(
                        0.0, state_std / np.sqrt(10 ** (self.state_noise_snr / 10)), 
                        states.shape)
                ds_state_mean = np.array(self.dataset_stat[data_dict['dataset_name']]['state_mean'])
                ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                # Randomly mask the states by the mean state
                data_dict["states"] = states \
                    if random.random() > self.cond_mask_prob else ds_state_mean
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)
                
                # Stat for the episode that the step belongs to 
                data_dict["state_norm"] = state_norm
                
                # We replace the invalid images with the background image
                # and also randomly mask images by the background image
                background_color = np.array([
                    int(x*255) for x in self.image_processor.image_mean
                ], dtype=np.uint8).reshape(1, 1, 3)
                background_image = np.ones((
                    self.image_processor.size["height"], 
                    self.image_processor.size["width"], 3), dtype=np.uint8
                ) * background_color
                
                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        if valid and (math.prod(image.shape) > 0) and \
                            (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))
                
                preprocessed_images = []
                processor = self.image_processor
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image) # (1008, 336)
                    # assert image.height == 336, "We haven't prepare for training with images of different resolutions."
                    
                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
                    # Only apply image augmentation to 50% of the images
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)
                    
                    if self.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    preprocessed_images.append(image)
                data_dict["images"] = preprocessed_images

                # if self.use_precomp_lang_embed:
                #     if content["instruction"][-1] == ".":
                #         content["instruction"] = content["instruction"][:-1]
                #     data_dict["lang_embed"] = torch.load(content["instruction"]) \
                #         if random.random() > self.cond_mask_prob else self.empty_lang_embed
                
                # for k, v in data_dict.items():
                #     if isinstance(v, np.ndarray):
                #         data_dict[k] = torch.from_numpy(v)

                # for k, v in data_dict.items():
                #     assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
                        # data_dict[k] = torch.from_numpy(v)
        
                return data_dict
            except BaseException as e:
                # Print the error info
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # Try incresing the index
                index = (index + 1) % len(self)


if __name__ == "__main__":
    args = parse_args()
    data_list = get_raw_data(args)
    
    # directory = os.path.join(RDT_ROOT_DIR,"data/data_processed")
    # filename = "output.json"

    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # t0 = time.time()
    # filepath = os.path.join(directory, filename)
    # with open(filepath, "w") as f:
    #     json.dump(data_list, f, indent=4)
    # print("write json file time: ",time.time()-t0,"s")
    # print(f"数据已保存到 {filepath}")