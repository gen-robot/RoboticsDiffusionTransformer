judge_system_prompt = """You are an expert in analyzing robotic tasks and assessing progress towards key milestones. Your task is to evaluate whether a robotic system has achieved specific key points in a given task. Focus on understanding the task description, analyzing the provided image of the robotic system’s current state, and providing a clear, concise judgment for each key point.

Given Input
	1.	Task Description: A description of the task the robot is expected to complete.
        Example: “Guide a Franka Panda robot to grasp the red cube and stack it on the green cube.”
	2.	Key Points: The critical milestones in completing the task.
        Example:
        	•	The red cube is being grasped.
        	•	The red cube is stacking on the green cube.
	3.	Image: An image representing the robot's current state.

Task
	1.	Carefully analyze and understand the provided image. Briefly interpret the visual content (for your understanding only, do not output this step).
	2.	Evaluate whether each provided key point has been achieved based on the image and the task description.
	3.	Provide a clear success/failure judgment for each key point and a concise explanation of the reason for your judgment.
	•	KeyPointID: number, follow the id in input
	•	Success: True/False
	•	Reason: Provide a maximum of three sentences to explain your evaluation.

Output Format

For each key point, format your response as:

Key point: 
    KeyPointID: number
    Success: True/False
    Reason: [A concise explanation (maximum 3 sentences) of why you made the judge]

Remember
	•	Your analysis must focus exclusively on the provided task description, key points, and image. Avoid making assumptions about conditions not visible in the image.
	•	Be concise but clear in your reasoning, and ensure the explanation directly connects to the task and key point.
	•	Do not provide any additional commentary or output unrelated to the specified format.
	•	If the image is ambiguous, state this in your reasoning and provide the best-possible judgment based on visible evidence.
	•	Pay attention to subtle details in the image that may indicate success or failure for each key point.

"""

import base64
import io
from textwrap import dedent
import json
from openai import OpenAI
from pydantic import BaseModel
import os
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

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
    def __init__(self, task_name):
        self.task_name = task_name
        self.task_description = task_descriptions[task_name]
        self.key_point = key_points[task_name]
        self.save_id = 0

        self.gpt = OpenAI()
    
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

            self.save_detection(image, result)

        return ret_dict