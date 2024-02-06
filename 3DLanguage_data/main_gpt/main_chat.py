import os
import yaml
from tqdm import tqdm
import json
import numpy as np
import argparse
import glob
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_incrementing,
)  # for exponential backoff



def prepare_chatgpt_message(task_prompt, questions, answers, question):
    messages = [{"role": "system", "content": task_prompt}]
    
    assert len(questions) == len(answers)
    for q, a in zip(questions, answers):
        messages.append({'role': 'user', 'content': q})
        messages.append({'role': 'assistant', 'content': a})
    messages.append({"role": "user", "content": question})
    
    return messages


def prepare_scene_desp(room_json_path):
    bboxes = json.load(open(room_json_path, "r"))
    scene_desp = []
    obj_num = {}
    max_num_bbox = 50  # 避免超出gpt max token
    for instance in bboxes[: min(len(bboxes), max_num_bbox)]:
        obj_name = instance['class_name']
        if obj_name not in obj_num:
            obj_id = 0
            obj_num[obj_name] = 1
        else:
            obj_id = obj_num[obj_name]
            obj_num[obj_name] += 1

        instance['bbox'] = [[instance['bbox'][0], instance['bbox'][1], instance['bbox'][2]], [instance['bbox'][3], instance['bbox'][4], instance['bbox'][5]]]
        obj_center = np.round(np.array(instance['bbox']).mean(0), 3).tolist()

        scene_desp.append(f'<{obj_name}>({obj_id}): {obj_center}')
    scene_desp = "\n".join(scene_desp)
    # print(f'scene_desp:\n{scene_desp}')
    return scene_desp


# def prepare_prompt(room_dir):
#     task_prompt = f"You are a conversation generator in a room. All object instances in this room are given, along with their center point position.  The center points are represent by a 3D coordinate (x, y, z) with units of meters. You need to generate 4~10 round convervation between a human and a robot assistant. "\
#     f"\n\nThe human know all information in this room, including all objects described above and all small things that are not visible now. The human will ask the robot to do a high-level task. The robot will tell its observation and its state (e.g., location) to the human and will ask for help when it is ambiguous about the task. Remenber, the high-level task should be done in this room. "\

def prepare_prompt(room_dir):
    task_prompt = f"You are a conversation generator in a room. All object instances in this room are given, along with their center point position.  The center points are represent by a 3D coordinate (x, y, z) with units of meters. You need to generate 4~10 round convervation between a human and a robot assistant. "\
    f"\n\nThe human know all information in this room, including all objects described above that might be invisible the the robot. The human will ask the robot to do a high-level task. The robot will tell its observation and its state (e.g., location) to the human and will ask for help when it is ambiguous about the task. Remenber, the high-level task should be done in this room. Do not include objects that do not exist."\
    
        # "You are an AI visual assistant that can analyze a 3D scene. All object instances in this 3D scene are given, along with their center point position.  The center points are represent by a 3D coordinate (x, y, z) with units of meters. Using the provided object instance information, design a high-level task that can be performed in this 3D scene. Besides, decomposing this high-level task into a sequence of action steps that can be performed using the instances in this 3D scene. \n\nRemenber, the high-level task and action steps must be able to be performed in the 3D scene using the given object instances."
    # print(task_prompt)

# You are an AI visual assistant, and you are seeing a single image. What you see are provided with five sentences, describing the same image you are looking at. Answer all questions as you are seeing the image.

# Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question.
# Ask diverse questions and give corresponding answers.

# Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:
# (1) one can see the content in the image that the question asks about and can answer confidently;
# (2) one can determine confidently from the image that it is not in the image.
# Do not ask any question that cannot be answered confidently.

# Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.
# Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized.  You can include multiple paragraphs if necessary.





    # scene_desp2 = prepare_scene_desp("./per_room_bboxes2/00141-iigzG1rtanx_13.json")
    # answer2 = f'Human: Bring me a toothbrush\n' \
    #         f"Robot: Sure! Where is it?\n"\
    #         f'Human: It maybe in the <sink>(0)\n' \
    #         f"Robot: I have arrived <sink>(0) but I can not find any toothbrush. I found an closed <bathroom cabinet>(0), is it possible to contains what you need?\n"\
    #         f'Human: Yes. Check it.\n' \
    #         f"Robot: I have opened the <bathroom cabinet>(0) and found two toothbrush, a yellow one and a blue one. Which one do you need?\n"\
    #         f'Human: The blue one.\n' \
    #         f"Robot: Ok. I will bring the blue one to you.\n"\
    scene_desp1 = prepare_scene_desp("example.json")
    answer1 = f'Human: Bring me my guitar(0).\n' \
            f"Robot: Sure! Where is it?\n"\
            f'Human: I may have left it near the tv(0). \n' \
            f"Robot: I have arrived at the tv(0) but I can not find a guitar. Is there anywhere else it might be?\n"\
            f'Human: Check the bed(0).\n' \
            f"Robot: OK. I'm now facing the bed(9) and there's a guitar(0) on the left. Anything else I can do for you?\n"\
            f'Human: No, thanks.\n' \

    # scene_desp2 = prepare_scene_desp("example.json")
    # answer2 = f'Human: Throw the box away for me\n' \
    #         f"Robot: Sure! I have arrived at the trash can but it seems too small for the box.\n"\
    #         f'Human: Which trash can are you at?\n' \
    #         f"Robot: The one between the bed and the desk.\n"\
    #         f'Human: That\'s a small one. There\'s a large trash can in the kitchen beside the door.\n' \
    #         f"Robot: OK. I have thrown away the box in the large trash can. Is there anything else you need me to do?\n"\
    #         f'Human: No, thanks.\n'

    prompt_questions = [scene_desp1]    # use one example is enough
    prompt_answers = [answer1]

    return task_prompt, prompt_questions, prompt_answers

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# @retry(wait=wait_incrementing(start=1, increment=3), stop=stop_after_attempt(10))
def call_chatgpt(chatgpt_messages, max_tokens=400, model="gpt-3.5-turbo", temperature = 0.5):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temperature, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def parse_task(ans):
    ans = ans.split("\n\n")[0]
    # ans = ans.split(":")[-1]
    ans = ans[ans.index(":")+2: ]
    return ans


def parse_action(ans):
    action = ans.split("\n\n")[1]
    action = action[action.index(":\n")+2: ]
    return action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--room_dir', type=str, default='./final_bbox_label')
    parser.add_argument('--save_root', type=str, default='./results_chat_final')
    parser.add_argument('--openai_api', type=str, default="")

    args = parser.parse_args()
    return args

import random
if __name__ == "__main__":
    args = parse_args()
    openai.api_key = args.openai_api
    # openai.proxy = {"http": "192.168.1.234:7890", "https": "192.168.1.234:7890"}

    task_prompt, prompt_questions, prompt_answers = prepare_prompt(args.room_dir)
    room_list = glob.glob(f"{args.room_dir}/*.json")
    random.shuffle(room_list)

    for room_path in tqdm(room_list):
        try:
            room_name = os.path.basename(room_path).replace(".json", "")
            save_path = os.path.join(args.save_root, f"{room_name}_new.json")
            if os.path.exists(save_path):
                continue

            scene_desp = prepare_scene_desp(room_path)
            msg = prepare_chatgpt_message(task_prompt, prompt_questions, prompt_answers, scene_desp)
            # print(msg)
            ans, n_tokens = call_chatgpt(msg, temperature = 0.7)
            
            result = {
                "room_name": [room_name],
                "chat": ans,
                "scene_desp": scene_desp,
            }
            print(result['chat'])
            json.dump(result, open(save_path, "w"))
        except:
            pass
