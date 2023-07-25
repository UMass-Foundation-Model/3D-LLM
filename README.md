<br />
<p align="center">
  <h1 align="center">3D-LLM: Injecting the 3D World into Large Language Models</h1>
  <p align="center">
    <a href="https://evelinehong.github.io">Yining Hong</a>,
    <a href="https://haoyuzhen.com">Haoyu Zhen</a>,
    <a href="https://peihaochen.github.io">Peihao Chen</a>,
    <a href="https://zsh2000.github.io">Shuhong Zheng</a>,
    <a href="https://yilundu.github.io">Yilun Du</a>,
    <a href="https://zfchenunique.github.io">Zhenfang Chen</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2307.12981'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://vis-www.cs.umass.edu/3dllm/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
  <p align="center">
    <img src="figs/pipeline.png" alt="Logo" width="80%">
  </p>
</p>

## Three-step 3D Feature Extraction
### First step
Installation: 

Please follow [Mask2Former](https://github.com/facebookresearch/Mask2Former) to install the environment and download the [pretrained weight](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl) to the current directory
if extracting the masks with [Mask2Former](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf).

Please follow [Segment Anything](https://github.com/facebookresearch/segment-anything) to install the environment and download the [pretrained weight](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to the current directory if extracting the masks with [SAM](https://arxiv.org/abs/2304.02643).

Extract masks with Mask2Former:

```shell
$ cd ./three_steps_3d_feature/first_step

$ python maskformer_mask.py --scene_dir_path DATA_DIR_WITH_RGB_IMAGES --save_dir_path DIR_YOU_WANT_TO_SAVE_THE_MASKS
```

Extract masks with Segment Anything:

```shell
$ cd ./three_steps_3d_feature/first_step

$ python sam_mask.py --scene_dir_path DATA_DIR_WITH_RGB_IMAGES --save_dir_path DIR_YOU_WANT_TO_SAVE_THE_MASKS
```

After the first step, we are expected to obtain a directory of masks (specified by ``--save_dir_path``) that contains extracted masks for
multi-view images of the scenes.

### Second step
Installation: The same as the following ``3D-LLM_BLIP2-based`` section to install [salesforce-lavis](https://github.com/salesforce/LAVIS).

There are four options: (1) Extract CLIP feature with Mask2Former masks; (2) Extract CLIP feature with SAM masks;
(3) Extract BLIP feature with Mask2Former masks; (4) Extract BLIP feature with SAM masks.

Extract 2D CLIP features with Mask2Former masks:
```shell
$ cd ./three_steps_3d_feature/third_step/

$ python clip_maskformer.py --scene_dir_path DATA_DIR_WITH_RGB_IMAGES --mask_dir_path MASK_DIR_FROM_1ST_STEP --save_dir_path DIR_YOU_WANT_TO_SAVE_THE_FEAT
```

For the other options, the scripts are in similar format.

After the second step, we are expected to obtain a directory of features (specified by ``--save_dir_path``) that contains 2D features for
multi-view images of the scenes.

### Third step
Installation:

Please install the [Habitat environment](https://github.com/facebookresearch/habitat-lab/tree/challenge-2022).

Reconstruct 3D feature from multi-view 2D features:

```shell
$ cd ./three_steps_3d_feature/third_step/

$ python sam_mask.py --data_dir_path DATA_DIR_WITH_RGB_IMAGES --depth_dir_path DATA_DIR_WITH_DEPTH_IMAGES --feat_dir_path FEATURE_DIR_FROM_2ND_STEP
```

After the third step, we are expected to obtain two files (``pcd_pos.pt`` and ``pcd_feat.pt``) for each room inside the corresponding RGB directory.
``pcd_pos.pt`` contains the point positions of the 3D point cloud (shape: ``N * 3``). ``pcd_feat.pt`` contains the point features of the 3D point cloud (shape: ``N * n_dim``).
``N`` is the number of sampled points in the point cloud (default: 300000) and ``n_dim`` is the feature dimension (1024 for CLIP feature, 1408 for BLIP feature).

## 3D-LLM_BLIP2-based
### Installation

Install [salesforce-lavis](https://github.com/salesforce/LAVIS)

```shell
$ conda create -n lavis python=3.8
$ conda activate lavis

$ git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS
$ cd SalesForce-LAVIS
$ pip install -e .

$ pip install positional_encodings
```

### Training

```shell
$ cd 3DLLM_BLIP2-base

$ conda activate lavis
# use facebook/opt-2.7b:
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/3dvqa_ft.yaml
# use flant5
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/3dvqa_flant5_ft.yaml
```
## 3D-LLM_flamingo-based
TODO.

## Citation

If you find our work useful, please consider citing:

```
@article{3dllm,
 author = {Hong, Yining and Zhen, Haoyu and Chen, Peihao and Zheng, Shuhong and Du, Yilun and Chen, Zhenfang and Gan, Chuang},
 title = {3D-LLM: Injecting the 3D World into Large Language Models},
 journal = {arXiv},
 year = {2023},
} 
```

### Acknowledgements

https://github.com/salesforce/LAVIS

https://github.com/mlfoundations/open_flamingo
