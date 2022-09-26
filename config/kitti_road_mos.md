# KITTI-Road-MOS 

To enrich the dataset in the moving object segmentation (MOS) task and to reduce the gap of different data distributions between the validation and test sets in the existing SemanticKITTI-MOS dataset, we automatically annotated and manually corrected the [KITTI-Road](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road) dataset.

More specifically, we first use auto-mos labeling method ([link](https://arxiv.org/pdf/2201.04501.pdf)) to automatically generate the MOS labels for KITTI-Road data. We then use a point labeler ([link](https://github.com/jbehley/point_labeler)) to manually refined the labels.

We follow semantic SLAM [SuMa++](https://github.com/PRBonn/semantic_suma) to rename the sequences of KITTI-Road data as follows.

```
raw_id -> seq_id
2011_09_26_drive_0015 -> 30
2011_09_26_drive_0027 -> 31
2011_09_26_drive_0028 -> 32
2011_09_26_drive_0029 -> 33
2011_09_26_drive_0032 -> 34
2011_09_26_drive_0052 -> 35
2011_09_26_drive_0070 -> 36
2011_09_26_drive_0101 -> 37
2011_09_29_drive_0004 -> 38
2011_09_30_drive_0016 -> 39
2011_10_03_drive_0042 -> 40
2011_10_03_drive_0047 -> 41
```
We provide a simple download and conversion script [utils/download_kitti_road.sh](../utils/download_kitti_road.sh), please modify the `DATA_ROOT` path and manually move the result folder `sequences` to the target folder.
And you need to download the KITTI-Road-MOS label data annotated by us, the pose and calib files from [here](https://drive.google.com/file/d/131tKKhJiNeSiJpnlrXS43bHgZJHh9tug/view?usp=sharing) (6.4 MB) [Remap the label to 9 and 251, consistent with the SemanticKITTI-MOS benchmark]. ~~[old version here](https://drive.google.com/file/d/1pdpcGReJHOJp01pbgXUbcGROWOBd_2kj/view?usp=sharing) (6.1 MB)~~.

We organize our proposed KITTI-Road-MOS using the same setup and data structure used in SemanticKITTI-MOS:

```
DATAROOT
├── sequences
│   └── 30
│       ├── calib.txt                       # calibration file provided by KITTI
│       ├── poses.txt                       # ground truth poses file provided by KITTI
│       ├── velodyne                        # velodyne 64 LiDAR scans provided by KITTI
│       │   ├── 000000.bin
│       │   ├── 000001.bin
│       │   └── ...
│       ├── labels                          # ground truth labels from us
│       │   ├── 000000.label
│       │   ├── 000001.label
│       │   └── ...
│       └── residual_images_1               # the proposed residual images
│           ├── 000000.npy
│           ├── 000001.npy
│           └── ...
```
