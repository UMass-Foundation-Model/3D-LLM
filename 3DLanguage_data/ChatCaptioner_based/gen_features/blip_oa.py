import time
import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from torch import nn
from lavis.processors import load_processor
from lavis.models.eva_vit import create_eva_vit_g
import argparse

LOAD_IMG_HEIGHT = 240
LOAD_IMG_WIDTH = 320


def get_new_pallete(num_colors: int) -> torch.Tensor:
    """Create a color pallete given the number of distinct colors to generate.

    Args:
        num_colors (int): Number of colors to include in the pallete

    Returns:
        torch.Tensor: Generated color pallete of shape (num_colors, 3)
    """
    pallete = []
    # The first color is always black, so generate one additional color
    # (we will drop the black color out)
    for j in range(num_colors + 1):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        if j > 0:
            pallete.append([r, g, b])
    return torch.Tensor(pallete).float() / 255.0


def get_bbox_around_mask(mask):
    # mask: (img_height, img_width)
    # compute bbox around mask
    bbox = None
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (
            topleft[0].item(),
            topleft[1].item(),
            botright[0].item(),
            botright[1].item(),
        )  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default=0, type=int)
    parser.add_argument("--all_jobs", default=160, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    frame_path = "./data/objaverse_frame"
    mask_path = "./data/objaverse_masks"
    out_path = "./data/objaverse_2dfeat"
    voxel_feature_path = "./data/objaverse_feat/"
    out_folder = "nps_blip"

    scene_list = os.listdir(frame_path)

    N = len(scene_list)
    jobs = args.all_jobs
    ids = N // jobs + 1
    visual_encoder = create_eva_vit_g(LOAD_IMG_HEIGHT).to(device).eval()
    vis_processor = load_processor("blip_image_eval").build(image_size=LOAD_IMG_HEIGHT)

    for scene in tqdm(scene_list[args.job * ids : min(N, args.job * ids + ids)]):
        if os.path.exists(os.path.join(voxel_feature_path, "features", f"{scene}_outside.pt")) and os.path.exists(
            os.path.join(voxel_feature_path, "points", f"{scene}_outside.npy")
        ):
            print("voxelized feature exists")
            continue
        if len(os.listdir(os.path.join(frame_path, scene))) < 25:
            continue
        scene_mask_path = os.path.join(mask_path, scene)
        if not os.path.exists(scene_mask_path) or len(os.listdir(scene_mask_path)) < 8:
            continue
        try:
            os.makedirs(os.path.join(out_path, scene, out_folder), exist_ok=True)
            img_list = [
                img for img in os.listdir(os.path.join(frame_path, scene)) if img.endswith(".png") and "norm" not in img
            ]
            if len(os.listdir(os.path.join(out_path, scene, out_folder))) == len(img_list):
                print("done")
                continue

            for file in os.listdir(scene_mask_path):
                INPUT_IMAGE_PATH = os.path.join(frame_path, scene, file.replace(".pt", ".png"))
                SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(out_path, scene, out_folder, file)
                if os.path.exists(SEMIGLOBAL_FEAT_SAVE_FILE):
                    continue

                raw_image = cv2.imread(INPUT_IMAGE_PATH)
                raw_image = cv2.resize(raw_image, (LOAD_IMG_WIDTH, LOAD_IMG_HEIGHT))
                raw_image2 = Image.fromarray(raw_image[:LOAD_IMG_HEIGHT, :LOAD_IMG_HEIGHT])
                image = vis_processor(raw_image2).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = visual_encoder(image)

                global_feat = output.clone().detach()
                global_feat = global_feat.half().cuda()
                global_feat = global_feat[:, :-1, :].resize(1, 17, 17, 1408).permute((0, 3, 1, 2))
                m = nn.AdaptiveAvgPool2d((1, 1))
                global_feat = m(global_feat)
                global_feat = global_feat.squeeze(-1).squeeze(-1)

                global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
                FEAT_DIM = global_feat.shape[-1]

                cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

                MASK_LOAD_FILE = os.path.join(mask_path, scene, file)
                outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_HEIGHT, FEAT_DIM, dtype=torch.half)

                # print(f"Loading instance masks {MASK_LOAD_FILE}...")
                mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W
                mask = mask[:, :, :LOAD_IMG_HEIGHT, :LOAD_IMG_HEIGHT]

                num_masks = mask.shape[-3]
                pallete = get_new_pallete(num_masks)

                rois = []
                roi_similarities_with_global_vec = []
                roi_sim_per_unit_area = []
                feat_per_roi = []
                roi_nonzero_inds = []

                for _i in range(num_masks):
                    curmask = mask[0, _i]
                    bbox, nonzero_inds = get_bbox_around_mask(curmask)
                    x0, y0, x1, y1 = bbox

                    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                    img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
                    iou = bbox_area / img_area

                    if iou < 0.005:
                        continue
                    roi = np.ones((LOAD_IMG_HEIGHT, LOAD_IMG_HEIGHT, 3)).astype(int)
                    img_roi = raw_image[:LOAD_IMG_HEIGHT, :LOAD_IMG_HEIGHT][x0:x1, y0:y1]
                    roi[x0:x1, y0:y1] = img_roi
                    roi = Image.fromarray(roi.astype(np.uint8))
                    img_roi = vis_processor(roi).unsqueeze(0).to(device)

                    with torch.no_grad():
                        roifeat = visual_encoder(img_roi)
                    roifeat = roifeat.clone().detach()
                    roifeat = roifeat.half().cuda()
                    roifeat = roifeat[:, :-1, :].resize(1, 17, 17, 1408).permute((0, 3, 1, 2))
                    m = nn.AdaptiveAvgPool2d((1, 1))
                    roifeat = m(roifeat)
                    roifeat = roifeat.squeeze(-1).squeeze(-1)

                    roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
                    feat_per_roi.append(roifeat)
                    roi_nonzero_inds.append(nonzero_inds)

                    _sim = cosine_similarity(global_feat, roifeat)

                    rois.append(torch.tensor(list(bbox)))
                    roi_similarities_with_global_vec.append(_sim)
                    roi_sim_per_unit_area.append(_sim)  # / iou)

                rois = torch.stack(rois)
                scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
                # nms not implemented for Long tensors
                # nms on CUDA is not stable sorted; but the CPU version is
                retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
                feat_per_roi = torch.cat(feat_per_roi, dim=0)  # N, 1024

                # print(f"retained {len(retained)} masks of {rois.shape[0]} total")
                retained_rois = rois[retained]
                retained_scores = scores[retained]
                retained_feat = feat_per_roi[retained]
                retained_nonzero_inds = []
                for _roiidx in range(retained.shape[0]):
                    retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])

                mask_sim_mat = torch.nn.functional.cosine_similarity(
                    retained_feat[:, :, None], retained_feat.t()[None, :, :]
                )
                mask_sim_mat.fill_diagonal_(0.0)
                mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
                softmax_scores = retained_scores.cuda() - mask_sim_mat
                softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0)

                for _roiidx in range(retained.shape[0]):
                    _weighted_feat = (
                        softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
                    )
                    _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                    outfeat[
                        retained_nonzero_inds[_roiidx][:, 0],
                        retained_nonzero_inds[_roiidx][:, 1],
                    ] += (
                        _weighted_feat[0].detach().cpu().half()
                    )

                    outfeat[
                        retained_nonzero_inds[_roiidx][:, 0],
                        retained_nonzero_inds[_roiidx][:, 1],
                    ] = torch.nn.functional.normalize(
                        outfeat[
                            retained_nonzero_inds[_roiidx][:, 0],
                            retained_nonzero_inds[_roiidx][:, 1],
                        ].float(),
                        dim=-1,
                    ).half()

                outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
                outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
                outfeat = torch.nn.functional.interpolate(outfeat, [LOAD_IMG_HEIGHT, LOAD_IMG_HEIGHT], mode="nearest")
                outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
                outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
                outfeat = outfeat[0].half()  # --> H, W, feat_dim

                os.makedirs(os.path.dirname(SEMIGLOBAL_FEAT_SAVE_FILE), exist_ok=True)
                torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)
        except:
            print(scene, "fail")
            time.sleep(0.05)
