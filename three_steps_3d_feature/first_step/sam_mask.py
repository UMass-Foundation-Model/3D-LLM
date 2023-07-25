import os
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange

import glob
import argparse
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument('--scene_dir_path', default="./masked_rdp_data/", type=str)
    parser.add_argument('--save_dir_path', default="./sam_masks/", type=str)
    args = parser.parse_args()

    torch.autograd.set_grad_enabled(False)

    sam = sam_model_registry["vit_h"](checkpoint=Path("sam_vit_h_4b8939.pth"))
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    save_dir = args.save_dir_path

    os.makedirs(save_dir, exist_ok=True)

    print("Extracting SAM masks...")
    room_list = os.listdir(args.scene_dir_path)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    model.cuda()
    model.eval()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    dataset_dir = args.scene_dir_path

    for room in tqdm(os.listdir(dataset_dir)):
        os.makedirs(save_dir + room, exist_ok=True)
        dataset_path = dataset_dir + room + "/*png"
        data_list = glob.glob(dataset_path)


        for img_name in data_list:
            img_base_name = os.path.basename(img_name)

            try:
                savefile = os.path.join(
                    save_dir,
                    room,
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
