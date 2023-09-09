import os
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import tyro
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange
from typing_extensions import Literal

import glob
import argparse
from tqdm import tqdm


@dataclass
class ProgramArgs:
    # Torch device to run computation on (E.g., "cpu")
    device: str = "cuda"

    # SAM checkpoint and model params
    checkpoint_path: Union[str, Path] = (
        Path.home() / "code" / "gradslam-foundation" / "examples" / "checkpoints" / "sam_vit_h_4b8939.pth"
    )
    model_type = "vit_h"
    # Ignore masks that have valid pixels less than this fraction (of the image area)
    bbox_area_thresh: float = 0.0005
    # Number of query points (grid size) to be sampled by SAM
    points_per_side: int = 32

    # gradslam mode ("incremental" vs "batch")
    mode: Literal["incremental", "batch"] = "incremental"

    # Path to the data config (.yaml) file
    dataconfig_path: str = "dataconfigs/icl.yaml"
    # Path to the dataset directory
    data_dir: Union[str, Path] = Path.home() / "data" / "icl"
    # Sequence from the dataset to load
    sequence: str = "living_room_traj1_frei_png"
    # Start frame index
    start_idx: int = 0
    # End frame index
    end_idx: int = -1
    # Stride (number of frames to skip between successive fusion steps)
    stride: int = 20
    # Desired image width and height
    desired_height: int = 120
    desired_width: int = 160

    # CLIP model config
    open_clip_model = "ViT-H-14"
    open_clip_pretrained_dataset = "laion2b_s32b_b79k"

    # Directory to save extracted features
    save_dir: str = "saved-feat"


def main():
    torch.autograd.set_grad_enabled(False)

    sam = sam_model_registry["vit_h"](checkpoint=Path("./data/sam_vit_h_4b8939.pth"))
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    save_dir = "./data/objaverse_masks/"

    os.makedirs(save_dir, exist_ok=True)

    print("Extracting SAM masks...")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model.cuda()
    model.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument("--all_jobs", default=160, type=int)
    parser.add_argument("--job", default=0, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_dir = "./data/objaverse_frame/"
    scene_list = os.listdir(dataset_dir)
    length = len(scene_list)
    jobs = args.all_jobs
    ids = length // jobs + 1

    print("Total length:", length)

    for room in tqdm(scene_list[args.job * ids : min(length, args.job * ids + ids)]):
        dataset_path = dataset_dir + room + "/*view*png"
        data_list = glob.glob(dataset_path)

        room_mask_path = os.path.join(save_dir, room)
        os.makedirs(room_mask_path, exist_ok=True)
        if len(os.listdir(room_mask_path)) >= len(data_list):
            continue

        for img_name in data_list:
            try:
                savefile = os.path.join(
                    room_mask_path,
                    os.path.basename(img_name).replace(".png", ".pt"),
                )
                if os.path.exists(savefile):
                    continue

                imgfile = img_name
                img = cv2.imread(imgfile)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                masks = mask_generator.generate(img)
                cur_mask = masks[0]["segmentation"]
                _savefile = os.path.join(
                    save_dir,
                    room,
                    os.path.splitext(os.path.basename(imgfile))[0] + ".pt",
                )

                mask_list = []
                for mask_item in masks:
                    mask_list.append(mask_item["segmentation"])

                mask_np = np.asarray(mask_list)
                mask_torch = torch.from_numpy(mask_np)
                torch.save(mask_torch, _savefile)
            except:
                pass


if __name__ == "__main__":
    main()
