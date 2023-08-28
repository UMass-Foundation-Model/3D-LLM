# Generate Features for 3D-LLM

## Preparation

Data:

```bash
$ mkdir data
$ ln -s {path/to/objaverse_frame_cap3d} ./data/objaverse_frame_cap3d
$ ln -s {path/to/sam_vit_h_4b8939.pth} ./data/sam_vit_h_4b8939.pth
```

Environment:

```bash
# https://github.com/UMass-Foundation-Model/3D-LLM/tree/main#installation
$ conda activate lavis
```

## Extract Mask Via SAM

```bash
$ python sam_mask.py --all_jobs 1
```

## Extract 2D Features

We use BLIP and CLIP to extract 2D features.

ATTENTION: please check if the height and width of the images are the same as the variables `LOAD_IMG_HEIGHT` and `LOAD_IMG_WIDTH` in `*_oa_cap3d.py` before running the following commands.

### BLIP

```bash
$ python blip_oa_cap3d.py --all_jobs 1
```

### CLIP

```bash
$ python clip_oa_cap3d.py --all_jobs 1
```

## Visualize 2D Features

```bash
$ python vis/visualize.py --model blip --scene {scene}
$ python vis/visualize.py --model clip --scene {scene}
```

## Voxelized 3D Features

ATTENTION: the 2D feature occupies a significant amount of space, so it will be automatically removed after generating the voxelized feature.

## BLIP

```bash
$ python gen_scene_feat_blip.py --all_jobs 1
```

## CLIP

```bash
$ python gen_scene_feat.py --all_jobs 1
```