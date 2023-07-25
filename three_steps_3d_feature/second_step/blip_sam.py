import torch
import torchvision
import cv2
import numpy as np
from tqdm import tqdm
import os
from torch import nn
from lavis.models.eva_vit import create_eva_vit_g
import argparse
LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512

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
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument('--scene_dir_path', default="./masked_rdp_data/", type=str)
    parser.add_argument('--mask_dir_path', default="./sam_masks/", type=str)
    parser.add_argument('--save_dir_path', default="./nps_sam_blip/", type=str)
    args = parser.parse_args()
    
    for scene in tqdm(sorted(os.listdir(args.scene_dir_path))):
        try:
            os.makedirs(os.path.join(args.save_dir_path, scene), exist_ok=True)
    
            for file in os.listdir(os.path.join(args.mask_dir_path, scene)):
                INPUT_IMAGE_PATH = os.path.join(args.scene_dir_path, scene, file.replace(".pt", ".png"))
                SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(args.save_dir_path, scene, file)
    
                if os.path.isfile(SEMIGLOBAL_FEAT_SAVE_FILE):
                    continue
    
                raw_image = cv2.imread(INPUT_IMAGE_PATH)
                raw_image = cv2.resize(raw_image, (512, 512))
                image = torch.tensor(raw_image[:512, :512]).permute(2, 0, 1).unsqueeze(0).to(device)
    
                visual_encoder = create_eva_vit_g(512).to(device)
                output = visual_encoder(image)
        
                global_feat = torch.tensor(output)
                global_feat = global_feat.half().cuda()
                global_feat = global_feat[:,:-1,:].resize(1,36,36,1408).permute((0,3,1,2))
                m = nn.AdaptiveAvgPool2d((1, 1))
                global_feat = m(global_feat)
                global_feat = global_feat.squeeze(-1).squeeze(-1)
    
                global_feat = torch.nn.functional.normalize(global_feat, dim=-1) 
                FEAT_DIM = global_feat.shape[-1]
    
                cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    
                MASK_LOAD_FILE = os.path.join(args.mask_dir_path, scene, file)
                outfeat = torch.zeros(512, 512, FEAT_DIM, dtype=torch.half)
    
                print(f"Loading instance masks {MASK_LOAD_FILE}...")
                mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W
                
                mask = mask[:, :, :512, :512]
                num_masks = mask.shape[-3]
    
                rois = []
                roi_similarities_with_global_vec = []
                roi_sim_per_unit_area = []
                feat_per_roi = []
                roi_nonzero_inds = []
    
                for _i in range(num_masks):
                    curmask = mask[0, _i].long()
                    bbox, nonzero_inds = get_bbox_around_mask(curmask)
                    x0, y0, x1, y1 = bbox
    
                    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                    img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
                    iou = bbox_area / img_area
    
                    if iou < 0.005:
                        continue
                    roi = torch.ones((512,512,3))
                    img_roi = torch.tensor(raw_image[:512, :512])[x0:x1, y0:y1]
                    roi[x0:x1, y0:y1] = img_roi
                    img_roi = roi.permute(2, 0, 1).unsqueeze(0).to(device)
                    roifeat = visual_encoder(img_roi)
                    roifeat = torch.tensor(roifeat)
                    roifeat = roifeat.half().cuda()
                    roifeat = roifeat[:,:-1,:].resize(1,36,36,1408).permute((0,3,1,2))
                    m = nn.AdaptiveAvgPool2d((1, 1))
                    roifeat = m(roifeat)
                    roifeat = roifeat.squeeze(-1).squeeze(-1)
    
                    roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
                    feat_per_roi.append(roifeat)
                    roi_nonzero_inds.append(nonzero_inds)
    
                    _sim = cosine_similarity(global_feat, roifeat)
    
                    rois.append(torch.tensor(list(bbox)))
                    roi_similarities_with_global_vec.append(_sim)
                    roi_sim_per_unit_area.append(_sim)
    
    
                rois = torch.stack(rois)
                scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
                retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
                feat_per_roi = torch.cat(feat_per_roi, dim=0)
    
                print(f"retained {len(retained)} masks of {rois.shape[0]} total")
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
    
                print(outfeat.shape)
                torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)
        except:
           pass
