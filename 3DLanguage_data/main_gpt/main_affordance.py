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
    max_num_bbox = 100  # 避免超出gpt max token
    for instance in bboxes[: min(len(bboxes), max_num_bbox)]:
        obj_name = instance['class_name']
        if obj_name not in obj_num:
            obj_id = 0
            obj_num[obj_name] = 1
        else:
            obj_id = obj_num[obj_name]
            obj_num[obj_name] += 1
        if "scene" in room_json_path:
            instance['bbox'] = [[instance['bbox'][0], instance['bbox'][1], instance['bbox'][2]], [instance['bbox'][3], instance['bbox'][4], instance['bbox'][5]]]
        obj_center = np.round(np.array(instance['bbox']).mean(0), 3).tolist()

        scene_desp.append(f'<{obj_name}>({obj_id}): {obj_center}')
    scene_desp = "\n".join(scene_desp)
    # print(f'scene_desp:\n{scene_desp}')
    return scene_desp


def prepare_prompt(room_dir):
    # task_prompt = f"You are a captioner in a room. All object instances in this room are given, along with their center point position.  The center points are represent by a 3D coordinate (x, y, z) with units of meters. For each object in the room, you need to describe it by referring to it using relationships to other objects. If several objects of the same class exist in the room, you need to write a description that can distinguish it from other objects of the same class."\
    task_prompt = f"You are a captioner in a room. All object instances in this room are given, along with their center point position.  The center points are represent by a 3D coordinate (x, y, z) with units of meters. For each object in the room, you need to describe it using their affordances or functionality (how you can use it). You should write your sentence like \"Where can I <do something>\" or \"Find me something that I can <do something> with\" or \"Where should I go for <something>\""\

    scene_desp1 = prepare_scene_desp("./final_bbox_label/scene0000_00.json")
    answer1 = '{\"<cabinet>(2)\": \"Where can I store my things\"}{\"<toaster oven>(0)\": \"Find me something that I can make my breakfast with\"}{\"<trash can>(0)\": \"I want to throw away my trash. Where should I go? \"}{\"<guitar>(0)\": \"I want to make some music. Where should I go?\"}{\"<stool>(1)\": \"Find me somewhere I can sit.\"}'

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

    parser.add_argument('--save_root', type=str, default='./results_grounding_final')
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

    for i in range(5):
        for room_path in tqdm(room_list):
            try:
                room_name = os.path.basename(room_path).replace(".json", "")
                save_path = os.path.join(args.save_root, f"{room_name}_hm3d_affordance_{i+5}.json")
                if os.path.exists(save_path):
                    continue

                scene_desp = prepare_scene_desp(room_path)
                msg = prepare_chatgpt_message(task_prompt, prompt_questions, prompt_answers, scene_desp)
                # print(msg)
                ans, n_tokens = call_chatgpt(msg, temperature = 0.2 + 0.1 * i)
                
                result = {
                    "room_name": [room_name],
                    "chat": ans,
                    "scene_desp": scene_desp,
                }
                print(result['chat'])
                json.dump(result, open(save_path, "w"))
            except: 
                pass
