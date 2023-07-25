import torch

from lavis.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        pc_list, points_list, question_list, answer_list, weight_list = [], [], [], [], []

        num_answers = []

        for sample in samples:
            pc_list.append(sample["pc_feat"])
            points_list.append(sample["pc"])
            question_list.append(sample["text_input"])
            weight_list.extend(sample["weight"])
            answers = sample["answer"]
            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "pc_feat": torch.stack(pc_list, dim=0),
            "pc": torch.stack(points_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
