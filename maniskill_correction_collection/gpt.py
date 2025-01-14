import os
import sys

if __name__=="__main__":
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_path)

import base64
import io
from textwrap import dedent
import json
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

from maniskill_correction_collection.system_prompt import (
    judge_system_prompt,
    stack_cube_system_prompt_improve
)

class Judgement(BaseModel):
    KeyPointID: int
    Success: bool
    Reason: str

class Judgements(BaseModel):
    judgements: list[Judgement]

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def encoder_image_array_to_base64(image):
    try:
        pil_image = Image.fromarray(image)
        
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        base64_string = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"Error encoding image array: {e}")
        return None

def get_reply(client, system_prompt, user_prompt, image_base64, output_format):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": dedent(system_prompt)},
                # {"role": "system", "content": dedent(stack_cube_system_prompt_improve)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dedent(user_prompt)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ],
            response_format=output_format,
            # temperature=0,  # Set to 0 for deterministic responses
        )
        parsed_output = completion.choices[0].message.parsed.dict()
        return parsed_output
    except Exception as e:
        
        print("Input: ", user_prompt)
        print("Error: ", e)
        return None

def process_item(client, item):
    task_description = item['task_description']
    key_points = item['key_points']
    if 'image_path' in item:
        image_path = item['image_path']
        image_base64 = encode_image_to_base64(image_path)
    elif 'raw_image' in item:
        image_base64 = encoder_image_array_to_base64(item['raw_image'])
    else:
        print("No image provided")
        return None
    key_points_input = {idx: key_point for idx, key_point in enumerate(key_points)}
    user_prompt = f"""
    Task Description: {task_description}
    Key Points: {key_points_input}
    """
    
    original_report = get_reply(
        client, judge_system_prompt, user_prompt, image_base64, Judgements
    )

    if original_report is not None:
        judgements = original_report["judgements"]
    else:
        judgements = None
    
    return judgements

task_descriptions = {
    "StackCube-v1": "Guide a Franka Panda robot to grasp the red cube and stack it on the green cube."
}
key_points = {
    "StackCube-v1": ["The red cube is being grasped.", "The red cube is stacking on the green cube."]
}

class GPTAgent(object):
    def __init__(self, task_name, save_judgement=True):
        self.task_name = task_name
        self.task_description = task_descriptions[task_name]
        self.key_point = key_points[task_name]
        self.save_id = 0

        self.gpt = OpenAI()
        self.save_judgement = save_judgement
    
    def save_detection(self, image, result):
        save_dir = "./outs/gpt_result/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"{self.save_id}.png")
        image = image[:, :, [2, 1, 0]]
        cv2.imwrite(save_path, image)

        save_path = os.path.join(save_dir, f"{self.save_id}.json")
        with open(save_path, 'w') as f:
            json.dump(result, f)
        
        self.save_id += 1

    def request(self, image):
        item = {}
        item['task_description'] = self.task_description
        item['key_points'] = self.key_point
        item['raw_image'] = image

        max_query_times = 5

        for i in range(max_query_times):
            result = process_item(self.gpt, item)
            if result is not None:
                break

        # IF GPT failed, the result should be not correct
        ret_dict = {
            "is_grasped": True,
            "is_stacked": True
        }

        if result is not None:
            for sub_result in result:
                if sub_result['KeyPointID'] == 0:
                    ret_dict['is_grasped'] = sub_result['Success']
                elif sub_result['KeyPointID'] == 1:
                    ret_dict['is_stacked'] = sub_result['Success']
            
            print("####################### GPT Detection #######################")
            print("Grasped: ", ret_dict['is_grasped'])
            print("Stacked: ", ret_dict['is_stacked'])
            print("Raw Result: ", result)
            print("#############################################################")

            if self.save_judgement:
                self.save_detection(image, result)
    
        return ret_dict

    def request_by_path(self, image_path):
        item = {}
        item['task_description'] = self.task_description
        item['key_points'] = self.key_point
        item['image_path'] = image_path

        max_query_times = 5

        for i in range(max_query_times):
            result = process_item(self.gpt, item)
            if result is not None:
                break

        # IF GPT failed, the result should be not correct
        ret_dict = {
            "is_grasped": True,
            "is_stacked": True
        }

        if result is not None:
            for sub_result in result:
                if sub_result['KeyPointID'] == 0:
                    ret_dict['is_grasped'] = sub_result['Success']
                elif sub_result['KeyPointID'] == 1:
                    ret_dict['is_stacked'] = sub_result['Success']
            
            print("####################### GPT Detection #######################")
            print("Grasped: ", ret_dict['is_grasped'])
            print("Stacked: ", ret_dict['is_stacked'])
            print("Raw Result: ", result)
            print("#############################################################")

            if self.save_judgement:
                self.save_detection(image, result)
    
        return ret_dict


if __name__=="__main__":
    gpt = GPTAgent("StackCube-v1", save_judgement=False)
    gpt.request_by_path("./outs/gpt_result/33.png")