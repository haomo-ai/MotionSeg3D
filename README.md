# Efficient Spatial-Temporal Information Fusion for LiDAR-Based 3D Moving Object Segmentation

### [Paper](https://arxiv.org/abs/2207.02201) | [Project page](https://npucvr.github.io/MotionSeg3D) | [Overview Video](https://youtu.be/kao26zX1Hdo) | [Supp. Video](https://youtu.be/carIdfwLX_s) <br>

This repo contains the code for our paper: 

> Efficient Spatial-Temporal Information Fusion for LiDAR-Based 3D Moving Object Segmentation.  
> [Jiadai Sun](https://github.com/MaxChanger), [Yuchao Dai](https://scholar.google.com/citations?user=fddAbqsAAAAJ&hl=en&oi=ao), [Xianjing Zhang](https://github.com/zoojing), Jintao Xu, Rui Ai, Weihao Gu, and [Xieyuanli Chen](https://github.com/Chen-Xieyuanli)  
> *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2022*

<!-- ![method-framework](assets/framework.png) -->
<!-- ![method-framework](assets/visualization.jpg) -->
<p align='center'>  
<a href="https://youtu.be/kao26zX1Hdo">
  <img src='assets/visualization.jpg' width='500' alt='demo video'/>
</a>
</p>

<br/>

## News
- [2022/12/09] Fix the 'nan' bug caused by softpool, [#11](https://github.com/haomo-ai/MotionSeg3D/issues/11),[#9](https://github.com/haomo-ai/MotionSeg3D/issues/9),[#6](https://github.com/haomo-ai/MotionSeg3D/issues/6)
- [2022/09/26] Update the KITTI-Road labels [here](https://drive.google.com/file/d/131tKKhJiNeSiJpnlrXS43bHgZJHh9tug/view?usp=sharing), remap the label to 9 and 251
## How to Use


### Installation
```bash
# Ubuntu 18.04 and above is recommended.
conda env create -f environment.yaml
conda activate mos3d

# Install SoftPool follow https://github.com/alexandrosstergiou/SoftPool
git clone https://github.com/alexandrosstergiou/SoftPool.git
cd SoftPool
git checkout 2d2ec6d # rollback to 2d2ec6dca10b7683ffd41061a27910d67816bfa5
cd pytorch

make install
--- (optional) ---
make test

# Install TorchSparse follow https://github.com/mit-han-lab/torchsparse
sudo apt install libsparsehash-dev 
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

```

### Pretrained Model on KITTI-MOS
Download the [toy-dataset](https://drive.google.com/file/d/1t8OuDgFzUspWtYVHSfiGkXtGrBsuvtWL/view) and [pretrained weights](https://drive.google.com/file/d/199hRJBs-3MVgqrd4Tb08Eo5pjBG74cSX/view), and unzip them to project_path. You can also use [gdown](https://github.com/wkentaro/gdown) to download them in command line.
<details>
  <summary>[Download command for toy-dataset and checkpoints (click to expand)]</summary>
  
```bash
gdown --id 1t8OuDgFzUspWtYVHSfiGkXtGrBsuvtWL # for toy-data
unzip toydata.zip

mkdir log && cd log
gdown --id 199hRJBs-3MVgqrd4Tb08Eo5pjBG74cSX # for checkpoints
unzip ckpt_motionseg3d_pointrefine.zip

```
</details>

Then you could use the follow command to inference and visualize the predictions. If you use toy dataset, please modify the `seq_id` corresponding to `valid` in `model_path/data_cfg.yaml`.
```bash
# To inference the predictions.
python infer.py -d ./toydata -m ./log/motionseg3d_pointrefine -l ./pred/oursv1 -s valid
python infer.py -d ./toydata -m ./log/motionseg3d_pointrefine -l ./pred/oursv2 -s valid --pointrefine

# Visualize the predictions.
python utils/visualize_mos.py -d ./toydata -p ./pred/oursv2 --offset 0 -s 38
```
  

### Data Preperation for SemanticKITTI-MOS and KITTI-Road-MOS (newly annotated by us)
1. Download KITTI Odometry Benchmark Velodyne point clouds (80 GB) from [here](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip).
2. Download KITTI Odometry Benchmark calibration data (1 MB) from [here](http://www.cvlibs.net/download.php?file=data_odometry_calib.zip).
3. Download SemanticKITTI label data (179 MB) (alternatively the data in Files corresponds to the same data) from [here](http://www.semantic-kitti.org/assets/data_odometry_labels.zip).
4. Download KITTI-Road Velodyne point clouds from [original website](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road), more details can be found in [config/kitti_road_mos.md](config/kitti_road_mos.md)
5. Download the KITTI-Road-MOS label data annotated by us, the pose and calib files from [here](https://drive.google.com/file/d/131tKKhJiNeSiJpnlrXS43bHgZJHh9tug/view?usp=sharing) (6.4 MB) [Remap the label to 9 and 251, consistent with the SemanticKITTI-MOS benchmark]. ~~[old version here](https://drive.google.com/file/d/1pdpcGReJHOJp01pbgXUbcGROWOBd_2kj/view?usp=sharing) (6.1 MB)~~.
6. Extract everything into the same folder, as follow:
<details>
  <summary>[Expected directory structure of SemanticKITTI (click to expand)]</summary>
  
```
DATAROOT
├── sequences
│   └── 08
│       ├── calib.txt                       # calibration file provided by KITTI
│       ├── poses.txt                       # ground truth poses file provided by KITTI
│       ├── velodyne                        # velodyne 64 LiDAR scans provided by KITTI
│       │   ├── 000000.bin
│       │   ├── 000001.bin
│       │   └── ...
│       ├── labels                          # ground truth labels provided by SemantiKITTI
│       │   ├── 000000.label
│       │   ├── 000001.label
│       │   └── ...
│       └── residual_images_1               # the proposed residual images
│           ├── 000000.npy
│           ├── 000001.npy
│           └── ...
```
</details>


7. Next run the data preparation script (based on [LMNet](https://github.com/PRBonn/LiDAR-MOS)) to generate the residual images. More parameters about the data preparation can be found in the yaml file [config/data_preparing.yaml](./config/data_preparing.yaml).


```shell
python utils/auto_gen_residual_images.py 
```

### Inference on SemanticKITTI-MOS
> The newly labeled KITTI-Road-MOS data is divided into train/valid set.  
> The useage of data can be controlled by specifying `--data_config` in training.
> During inference, if you use toy dataset or do not download the KITTI-Road-MOS, please modify the `seq_id` corresponding to `valid` in `model_path/data_cfg.yaml`.

```bash
# validation split
python infer.py -d DATAROOT -m ./log/model_path/logs/TIMESTAMP/ -l ./predictions/ -s valid 

# test split
python infer.py -d DATAROOT -m ./log/model_path/logs/TIMESTAMP/ -l ./predictions/ -s test
```
The predictions/labels will be saved to `./predictions/`.


### Evaluation on SemanticKITTI-MOS validation split
```bash
# Only on seq08
python utils/evaluate_mos.py -d DATAROOT -p ./predictions/ --datacfg config/labels/semantic-kitti-mos.raw.yaml

# On seq08 + road-validation-split
python utils/evaluate_mos.py -d DATAROOT -p ./predictions/ --datacfg config/labels/semantic-kitti-mos.yaml
```


### Training on SemanticKITTI-MOS
<!-- Our program is a two-stage training process,  -->
The training is seperated into two phases, and switching between phases is currently manually controlled.
`--data_config` determines whether to use new label data KITTI-Road-MOS, such as `-dc config/labels/semantic-kitti-mos.yaml` or `-dc config/labels/semantic-kitti-mos.raw.yaml`

- Phase 1 (multi-gpu): Only the range image is used for input and supervision. The training log and checkpoint will be stored in `./log/ours_motionseg3d/logs/TIMESTAMP/`. 

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py -d DATAROOT -ac ./train_yaml/mos_coarse_stage.yml -l log/ours_motionseg3d
```

- Phase 2 (single gpu): After the first phase of training, use the following command to start the second phase of training for the PointRefine module.

```bash
export CUDA_VISIBLE_DEVICES=0
python train_2stage.py -d DATAROOT -ac ./train_yaml/mos_pointrefine_stage.yml -l log/ours_motionseg3d_pointrefine -p "./log/ours_motionseg3d/logs/TIMESTAMP/"
```


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{sun2022mos3d,
  title={Efficient Spatial-Temporal Information Fusion for LiDAR-Based 3D Moving Object Segmentation},
  author={Sun, Jiadai and Dai, Yuchao and Zhang, Xianjing and Xu, Jintao and Ai, Rui and Gu, Weihao and Chen, Xieyuanli},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022},
  organization={IEEE}
}
```

## Acknowledgment
We would like to thank Yufei Wang and Mochu Xiang for their insightful and effective discussions.  
Some of the code in this repo is borrowed from [SoftPool](https://github.com/alexandrosstergiou/SoftPool), [RangeDet](https://github.com/tusen-ai/RangeDet), [LMNet](https://github.com/PRBonn/LiDAR-MOS) and [spvnas](https://github.com/mit-han-lab/spvnas), thanks all authors for their awesome projects.

## Copyright
Copyright 2022, Jiadai Sun, Xieyuanli Chen, Xianjing Zhang, HAOMO.AI Technology Co., Ltd., China.

This project is free software made available under the GPL v3.0 License. For details see the LICENSE file.
