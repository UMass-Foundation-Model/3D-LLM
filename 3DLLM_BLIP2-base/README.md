
# 3D-LLM BLIP2

## Preparation

Install [salesforce-lavis](https://github.com/salesforce/LAVIS)

```shell
$ conda create -n lavis python=3.8
$ conda activate lavis

$ git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS
$ cd SalesForce-LAVIS
$ pip install -e .

$ pip install positional_encodings
```

Prepare the dataset

```shell
# preparation
$ cd ../LAIVS
$ ln -s {path/to/examples} .
```

## Training

```shell
$ conda activate lavis
# use facebook/opt-2.7b:
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/3dvqa_ft.yaml
# use flant5
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/3dvqa_flant5_ft.yaml
```
