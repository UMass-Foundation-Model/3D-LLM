import torch
import torchvision
import cv2
import numpy as np
from tqdm import tqdm
import os
from torch import nn
import argparse
import clip
import open_clip
LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512
from PIL import Image


def get_bbox_around_mask(mask):
    bbox = None
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    OPENCLIP_MODEL = "ViT-L-14"  # "ViT-bigG-14"
    OPENCLIP_DATA = "laion2b_s32b_b82k"  # "laion2b_s39b_b160k"
    print("Initializing model...")
    model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL, OPENCLIP_DATA)
    model.visual.output_tokens = True
    model.cuda()
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

    dataset_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/hm3d_shuhong/new_scenes_temp/"

    scene_lists = sorted(os.listdir(dataset_dir))
    for scene in tqdm(scene_lists):
        if True:
            try:
                os.mkdir(os.path.join(dataset_dir, scene, "nps_1024_hiddAve_ViTL"))
            except:
                pass

            for file in os.listdir(os.path.join(dataset_dir, scene, "masks")):
              
                try:
                    INPUT_IMAGE_PATH = os.path.join(dataset_dir, scene, file.replace(".pt", ".png"))
                    SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(dataset_dir, scene, "nps_1024_hiddAve_ViTL", file)
                    if os.path.isfile(SEMIGLOBAL_FEAT_SAVE_FILE):
                        continue

                    raw_image = cv2.imread(INPUT_IMAGE_PATH)
                    raw_image = cv2.resize(raw_image, (512, 512))
                    image = torch.tensor(raw_image).to(device)
            
                    """
                    Extract and save global feat vec
                    """
                    global_feat = None
                    with torch.cuda.amp.autocast():
                        _img = preprocess(Image.open(INPUT_IMAGE_PATH)).unsqueeze(0)    # [1, 3, 224, 224]
                        imgfeat = model.visual(_img.cuda())[1]  # All image token feat [1, 256, 1024]
                        imgfeat = torch.mean(imgfeat, dim=1)

                    global_feat = imgfeat.half().cuda()
                    

                    global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
                    FEAT_DIM = global_feat.shape[-1]
    
                    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

                    MASK_LOAD_FILE = os.path.join(dataset_dir, scene, "masks", file)
                    outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, FEAT_DIM, dtype=torch.half)

                    mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W
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
                        with torch.no_grad():
                            img_roi = image[x0:x1, y0:y1]
                            img_roi = Image.fromarray(img_roi.detach().cpu().numpy())
                            img_roi = preprocess(img_roi).unsqueeze(0).cuda()
                            roifeat = model.visual(img_roi)[1]  # All image token feat [1, 256, 1024]
                            roifeat = torch.mean(roifeat, dim=1)

                            feat_per_roi.append(roifeat)
                            roi_nonzero_inds.append(nonzero_inds)
                            _sim = cosine_similarity(global_feat, roifeat)

                            rois.append(torch.tensor(list(bbox)))
                            roi_similarities_with_global_vec.append(_sim)
                            roi_sim_per_unit_area.append(_sim)
                    
                    rois = torch.stack(rois)
                    scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
                    retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
                    feat_per_roi = torch.cat(feat_per_roi, dim=0)  # N, 1024
        
                    retained_rois = rois[retained]
                    retained_scores = scores[retained]
                    retained_feat = feat_per_roi[retained]
                    retained_nonzero_inds = []
                    for _roiidx in range(retained.shape[0]):
                        retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])
        

                    mask_sim_mat = torch.nn.functional.cosine_similarity(
                        retained_feat[:, :, None], retained_feat.t()[None, :, :]
            )
                    mask_sim_mat.fill_diagonal_(0.)
                    mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
                    softmax_scores = retained_scores.cuda() - mask_sim_mat
                    softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0)
                    for _roiidx in range(retained.shape[0]):
                        _weighted_feat = softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
                        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                        outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
                        outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] = torch.nn.functional.normalize(
                        outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(), dim=-1
            ).half()

                    outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
                    outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
                    outfeat = torch.nn.functional.interpolate(outfeat, [512, 512], mode="nearest")
                    outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
                    outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
                    outfeat = outfeat[0].half() # --> H, W, feat_dim

                    
                    torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)
                except:
                    print(SEMIGLOBAL_FEAT_SAVE_FILE, "fail")
        
