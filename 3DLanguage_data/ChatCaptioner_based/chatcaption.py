import argparse
import os
import yaml
import torch
import json
from tqdm import tqdm
import openai
from blip2 import Blip2
from quetion_model import load_model
from caption import caption_images_from_dir


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-13b-v1.1",
        help="The path to the vicuna weights",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument("--load-8bit", action="store_true", default=False, help="Use 8-bit quantization.")
    parser.add_argument("--clip_select", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--question_model", type=str, choices=["chatgpt", "vicuna", "gpt3"], default="vicuna")
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--scene_path", type=str, default="./objaverse_render/output")
    parser.add_argument("--result_save_path", type=str, default="output/")
    parser.add_argument("--specific_scene", type=str)
    parser.add_argument("--views", type=str, nargs="+", default=["outside"])
    args = parser.parse_args()

    return args


def main():
    args = argparser()

    # specify caption scene (use objaverse id)
    if args.specific_scene:
        scene_list = [args.specific_scene]
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)

    # inittialize BLIP2 and vicuna models
    blip2s = {"FlanT5 XXL": Blip2("FlanT5 XXL", device_id=1, bit8=False)}
    print("Successfully Load BLIP2 Model")

    if args.question_model == "vicuna":
        vicuna_model, vicuna_tokenizer = load_model(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            debug=args.debug,
        )
    else:
        vicuna_model, vicuna_tokenizer = None, None

    for i, scene in enumerate(tqdm(scene_list)):
        result_dict = {}

        result_dict.update({"scene_id": scene})

        print("Processing : {}".format(scene))
        if os.path.isfile(f"{args.result_save_path}/{scene}.json"):
            print("Already processed : {}".format(scene))
            continue

        one_scene = os.path.join(args.scene_path, scene)
        print(one_scene)

        caption_result = caption_images_from_dir(
            blip2s,
            args=args,
            image_dir=one_scene,
            save_path=args.result_save_path,
            n_rounds=args.n_rounds,
            n_blip2_context=0,
            model=args.question_model,
            print_mode="yes",
            vicuna_model=vicuna_model,
            vicuna_tokenizer=vicuna_tokenizer,
            clip_select=args.clip_select,
            views=args.views,
        )

        result_dict.update({"caption_result": caption_result})

        save_path = args.result_save_path + "/" + scene + ".json"
        with open(save_path, "w", encoding="utf8") as fp:
            json.dump(result_dict, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
