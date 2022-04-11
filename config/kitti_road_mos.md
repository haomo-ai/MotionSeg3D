# KITTI-Road-MOS 

To enrich the dataset in the moving object segmentation (MOS) task and to reduce the gap of different data distributions between the validation and test sets in the existing SemanticKITTI-MOS dataset, we automatically annotated and manually corrected the [KITTI-Road](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road) dataset.

More specifically, we first use our auto-mos labeling method ([link](https://arxiv.org/pdf/2201.04501.pdf)) to automatically generate the MOS labels for KITTI-Road data. We then use a point labeller([link](https://github.com/jbehley/point_labeler)) to manually refined the labels.

We follow our semantic SLAM [SuMa++](https://github.com/PRBonn/semantic_suma) to rename the sequences of KITTI-Road data as follows:

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
