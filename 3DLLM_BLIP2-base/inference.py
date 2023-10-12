import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
import json
import argparse

from lavis.common.registry import registry

# ======== Step 0: Configurations >>>>>>>>
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="object", type=str, choices=["object", "room"])
parser.add_argument("-v", "--visualize", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "pretrain_blip2_sam_flant5xl_v1.pth"
assert os.path.exists(ckpt_path), "Please specify the checkpoint path."

obj_id_path = "assets/objaverse_subset_ids_100.json"
obj_feat_path = "data/objaverse_feat"

# ======== Step 1: Load model from checkpoint >>>>>>>>
print("Loading model from checkpoint...")
model_cfg = {
    "arch": "blip2_t5",
    "model_type": "pretrain_flant5xl",
    "use_grad_checkpoint": False,
}
model_cfg = OmegaConf.create(model_cfg)
model = registry.get_model_class(model_cfg.arch).from_pretrained(model_type=model_cfg.model_type)
checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()
model.to(DEVICE)

processor_cfg = {"name": "blip_question", "prompt": ""}
processor_cfg = OmegaConf.create(processor_cfg)
text_processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)

# ======== Step 2: Prepare input >>>>>>>>
print("Preparing input...")
if args.mode == "object":
    prompt = [
        "Describe the 3D scene.",
        "Describe the 3D scene in one sentence.",
        "What's depicted in the scene?",
        "What's the color of the object?",
        "What's the material of the object?",
    ]
    with open(obj_id_path, "r") as f:
        obj_ids = json.load(f)
    obj_id = np.random.choice(obj_ids)
    print("obj_id: ", obj_id)
    feature_path = os.path.join(obj_feat_path, "features", f"{obj_id}_outside.pt")
    points_path = os.path.join(obj_feat_path, "points", f"{obj_id}_outside.npy")
else:
    prompt = [
        "Agent 1: What's the room like??",
        "Agent 1: I really like the layout of this room! It has a table where I can read books.",
        "I want to clean up this room. What should I do next?",
        "I want to clean up this room. I have done these things: 1. cleaned up the table; 2. clean up the chair. What should I do next?",
        "Agent 1: What do you think about this room?",
        "Human: Could you bring my pillow for me?",
        "Human: Could you bring my pillow for me? Robot: Sure! Where is the pillow? Human: It's on the sofa.",
        "Where is the sofa located?",
        "Agent 1: Check this out! This room is what I dream of! Agent 2: Yes! I see a television here!"
    ]
    room_id = "scene0480_00"
    print("room_id: ", room_id)
    feature_path = os.path.join("assets", f"{room_id}.pt")
    points_path = os.path.join("assets", f"{room_id}.npy")

prompt = np.random.choice(prompt)
prompt = text_processor(prompt)
pc_feature = torch.load(feature_path)  # (N, 1408)
if isinstance(pc_feature, np.ndarray):
    pc_feature = torch.from_numpy(pc_feature)
pc_feature = pc_feature.to(DEVICE).unsqueeze(0)  # (1, N, 1408)
pc_points = torch.from_numpy(np.load(points_path)).long().to(DEVICE).unsqueeze(0)  # (1, N, 3)

print("text_input: ", prompt)
model_inputs = {"text_input": prompt, "pc_feat": pc_feature, "pc": pc_points}

# ======== Step 3: Inference >>>>>>>>
model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}
model_inputs["text_input"] = prompt
model_outputs = model.predict_answers(
    samples=model_inputs,
    max_len=50,
    length_penalty=1.2,
    repetition_penalty=1.5,
)
model_outputs = model_outputs[0]
print(model_outputs)

# ======= Stage 4: Visualization >>>>>>>>
if args.visualize:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ax = plt.figure().add_subplot(111, projection="3d")
    tsne = TSNE(n_components=3, random_state=0, learning_rate=200, init="random")
    idx = np.random.choice(pc_feature.shape[1], 10000)
    pc_feature = pc_feature[:, idx, :]
    pc_points = pc_points[:, idx, :]
    pc_feature = pc_feature.squeeze(0).cpu().numpy()  # (N, 1408)
    pc_feature = tsne.fit_transform(pc_feature)  # (N, 3)
    pc_feature = (pc_feature - pc_feature.min()) / (pc_feature.max() - pc_feature.min() + 1e-6)
    pc_points = pc_points.squeeze(0).cpu().numpy()  # (N, 3)
    ax.scatter(pc_points[:, 0], pc_points[:, 1], pc_points[:, 2], c=pc_feature, s=1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    os.makedirs("tmp", exist_ok=True)
    plt.savefig("tmp/pc.png")
