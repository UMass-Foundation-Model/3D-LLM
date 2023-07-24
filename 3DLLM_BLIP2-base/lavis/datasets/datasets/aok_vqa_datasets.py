"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answer": "; ".join(ann["answers"]),
                "image": sample["image"],
                "pc_feat": sample["pc_feat"],
                "pc": sample["pc"],
            }
        )


class AOKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        new_annotation = []
        for ann in self.annotation:
            try:
                img_id = ann["scene_id"]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1
                new_annotation.append(ann)
            except:
                pass
        self.annotation = new_annotation
        self.pc_feat_root = "examples/voxelized_features_sam_nonzero"  # 2
        self.voxel_root = "examples/voxelized_voxels_sam_nonzero"  # flatten
        self.annotation = [
            ann for ann in self.annotation if os.path.exists(os.path.join(self.pc_feat_root, ann["scene_id"] + ".pt"))
        ]

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        image = self.vis_processor(image)

        caption = self.text_processor(ann["question"])
        scene_id = ann["scene_id"]
        pc_feat = torch.load(os.path.join(self.pc_feat_root, f"{scene_id}.pt"), map_location="cpu")
        pc = np.load(os.path.join(self.voxel_root, f"{scene_id}.npy"))
        pc = torch.tensor(pc).float().cpu()
        # sample 10000 points: [N, 1408] -> [10000, 1408]

        if pc_feat.shape[0] > 10000:
            idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:10000])[1]
            pc_feat = pc_feat[idxes]
            pc = pc[idxes]
        else:
            pc_feat = torch.cat([pc_feat, torch.zeros(10000 - pc_feat.shape[0], 1408)], dim=0)

            pc = torch.cat([pc, torch.zeros(10000 - pc.shape[0], 3)], dim=0)

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "pc_feat": pc_feat,
            "pc": pc,
            "text_input": caption,
            "answer": answers,
            "weight": weights,
            "image_id": self.img_ids[ann["scene_id"]],
            "question_id": index,
        }

    def __len__(self):
        return len(self.annotation)


class AOKVQAEvalDataset(VQAEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        new_annotation = []
        for ann in self.annotation:
            try:
                img_id = ann["scene_id"]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1
                new_annotation.append(ann)
            except:
                pass
        self.annotation = new_annotation
        self.pc_feat_root = "examples/voxelized_features_sam_nonzero"
        self.voxel_root = "examples/voxelized_voxels_sam_nonzero"
        self.annotation = [
            ann for ann in self.annotation if os.path.exists(os.path.join(self.pc_feat_root, ann["scene_id"] + ".pt"))
        ]

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        image = self.vis_processor(image)
        caption = self.text_processor(ann["question"])
        scene_id = ann["scene_id"]
        pc_feat = torch.load(os.path.join(self.pc_feat_root, f"{scene_id}.pt"), map_location="cpu")  # [N, 1408]
        pc = np.load(os.path.join(self.voxel_root, f"{scene_id}.npy"))
        pc = torch.tensor(pc).float().cpu()
        # sample 10000 points: [N, 1408] -> [10000, 1408]

        if pc_feat.shape[0] > 10000:
            idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:10000])[1]
            pc_feat = pc_feat[idxes]
            pc = pc[idxes]

        else:
            pc_feat = torch.cat([pc_feat, torch.zeros(10000 - pc_feat.shape[0], 1408)], dim=0)
            pc = torch.cat([pc, torch.zeros(10000 - pc.shape[0], 3)], dim=0)

        return {
            "image": image,
            "pc_feat": pc_feat,
            "pc": pc,
            "text_input": caption,
            "image_id": self.img_ids[scene_id],
            "instance_id": scene_id,
            "question_id": index,
        }


class NoCapsEvalDataset(VQAEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        image = self.vis_processor(image)
        scene_id = ann["scene_id"]
        pc_feat = torch.load(os.path.join(self.pc_feat_root, f"{scene_id}.pt"), map_location="cpu")
        # sample 10000 points: [N, 1408] -> [10000, 1408]
        if pc_feat.shape[0] > 10000:
            pc_feat = pc_feat[torch.randperm(pc_feat.shape[0])[:10000]]
        else:
            pc_feat = torch.cat([pc_feat, torch.zeros(10000 - pc_feat.shape[0], 1408)], dim=0)
        caption = self.text_processor(ann["question"])
        return {
            "image": image,
            "pc_feat": pc_feat,
            "text_input": question,
            "image_id": self.img_ids[scene_id],
            "instance_id": scene_id,
        }
