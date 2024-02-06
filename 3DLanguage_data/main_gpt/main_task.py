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
    max_num_bbox = 120  # 避免超出gpt max token
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
        # scene_desp.append(f'<{obj_name}>({obj_id}): {obj_center}')
        scene_desp.append(f'{obj_name}: {obj_center}')
    scene_desp = "\n".join(scene_desp)
    # print(f'scene_desp:\n{scene_desp}')
    return scene_desp

def prepare_prompt(room_dir):
    task_prompt = f"You are an AI visual assistant that can analyze a 3D scene. All object instances in this 3D scene are given, along with their center point position.  The center points are represent by a 3D coordinate (x, y, z) with units of meters. Using the provided object instance information, design a high-level task that can be performed in this 3D scene. Besides, decomposing this high-level task into a sequence of action steps that can be performed using the instances in this 3D scene. The number of action steps should be LESS THAN TEN (< 10). \n\nRemember, the high-level task and action steps must be able to be performed in the 3D scene using the given object instances. Do not include objects that do not exist. Do not generate similar action steps. Do not include specific locations, numbers in the action steps."

    scene_desp1 = prepare_scene_desp("./per_room_bboxes2/00669-DNWbUAJYsPy_1.json")
    answer1 = f'High-Level Task: make up\n\n' \
                f'Low-Level Action Steps:\n1. go to the cabinet\n2. take out cosmetics from the cabinet\n3. bring the cosmetics to the table with mirror\n4. open the lamp\n5. make up'

    scene_desp2 = prepare_scene_desp("./per_room_bboxes2/00669-DNWbUAJYsPy_2.json")
    answer2 = f'High-Level Task: get ready for work\n\n' \
                f"Low-Level Action Steps:\n1. go to the toilet\n2. use the toilet paper\n3. wash hands in the sink cabinet\n4. go to the mirror\n5. turn on the lamp\n6. brush teeth and wash face\n7. turn off the lamp\n8. leave the room through the door"

    prompt_questions = [scene_desp2]    # use one example is enough
    prompt_answers = [answer2]

    return task_prompt, prompt_questions, prompt_answers

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# @retry(wait=wait_incrementing(start=1, increment=3), stop=stop_after_attempt(10))
def call_chatgpt(chatgpt_messages, max_tokens=400, model="gpt-3.5-turbo", temperature=0.5):
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
    parser.add_argument('--save_root', type=str, default='./results_task_final')
    parser.add_argument('--openai_api', type=str, default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    openai.api_key = args.openai_api
    # openai.proxy = {"http": "192.168.1.234:7890", "https": "192.168.1.234:7890"}

    task_prompt, prompt_questions, prompt_answers = prepare_prompt(args.room_dir)
    print (task_prompt, prompt_questions, prompt_answers)
    for i in range(4):
        for room_path in tqdm(glob.glob(f"{args.room_dir}/*.json")):
            try:
                room_name = os.path.basename(room_path).replace(".json", "")
                save_path = os.path.join(args.save_root, f"{room_name}_new_{i}.json")
                if os.path.exists(save_path):
                    continue

                scene_desp = prepare_scene_desp(room_path)
                msg = prepare_chatgpt_message(task_prompt, prompt_questions, prompt_answers, scene_desp)
                # print(msg)
                ans, n_tokens = call_chatgpt(msg, temperature = 0.5+0.1*i)
                
                result = {
                    "room_name": [room_name],
                    "task": parse_task(ans),
                    "action": parse_action(ans),
                    "scene_desp": scene_desp,
                }
                print(result["task"], result["action"])
                json.dump(result, open(save_path, "w"))
            except:
                pass

