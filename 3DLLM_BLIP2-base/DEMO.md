# 3D-LLM DEMO

## Generate text

The following code shows how to generate text from a 3D scene.
You can also run the command `python inference.py [-v]` to generate text conditioned on a 3D scene and text prompt.

### Step 0: Configuration

Download the sub-set of the features from [here](https://drive.google.com/file/d/1mJZONfWREfIUAPYXP65D65uS2EoplAfR/view?usp=sharing) and put it in `data/objaverse_feat_new_blip`.

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "CKPT_PATH"

obj_id_path = "assets/objaverse_subset_ids_100.json"
obj_feat_path = "data/objaverse_feat"
```

### Step 1: Load model from checkpoint

```python
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
```

### Step 2: Prepare input

```python
prompt = "Describe the 3D scene."
with open(obj_id_path, "r") as f:
    obj_ids = json.load(f)
obj_id = np.random.choice(obj_ids)
feature_path = os.path.join(obj_feat_path, "features", f"{obj_id}_outside.pt")
points_path = os.path.join(obj_feat_path, "points", f"{obj_id}_outside.npy")

prompt = text_processor(prompt)
pc_feature = torch.load(feature_path)  # (N, 1408)
if isinstance(pc_feature, np.ndarray):
    pc_feature = torch.from_numpy(pc_feature)
pc_feature = pc_feature.to(DEVICE).unsqueeze(0)  # (1, N, 1408)
pc_points = torch.from_numpy(np.load(points_path)).long().to(DEVICE).unsqueeze(0)  # (1, N, 3)

model_inputs = {"text_input": prompt, "pc_feat": pc_feature, "pc": pc_points}
```

### Step 3: Inference

```python
model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}
model_inputs["text_input"] = prompt
model_outputs = model.predict_answers(
    samples=model_inputs,
    max_len=50,
    length_penalty=1.2,
    repetition_penalty=1.5,
)
model_outputs = model_outputs[0]
```