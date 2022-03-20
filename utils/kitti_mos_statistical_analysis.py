#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by Jiadai Sun

import os
import sys
import glob
import yaml
import numpy as np

from tqdm import tqdm
from glob import glob
from icecream import ic


def get_frame_data(pc_path, label_path):
    pc_data = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
    label = np.fromfile(label_path, dtype=np.uint32).reshape((-1))
    sem_label = label & 0xFFFF
    ins_label = label >> 16

    return pc_data, sem_label, ins_label


class SeqKITTI():
    def __init__(self, dataset_path, split, data_cfg_path):
        self.dataset_path = dataset_path
        self.split = split  # valid train test
        self.data_yaml = yaml.load(open(data_cfg_path))

        print(split, self.data_yaml["split"][split])
        self.seqs = []
        for seq in self.data_yaml["split"][split]:
            seq = '{0:02d}'.format(int(seq))
            # print(split, seq)
            self.seqs.append(seq)

    def get_file_list(self, seq_id):
        velodyne_seq_path = os.path.join(self.dataset_path, "sequences", seq_id, "velodyne")
        velodyne_seq_files = sorted(glob.glob(os.path.join(velodyne_seq_path, "*.bin")))

        # load gt semantic segmentation files
        gtsemantic_seq_path = os.path.join(self.dataset_path, "sequences", seq_id, "labels")
        gtsemantic_seq_files = sorted(glob.glob(os.path.join(gtsemantic_seq_path, "*.label")))

        assert len(velodyne_seq_files) == len(gtsemantic_seq_files)

        return velodyne_seq_files, gtsemantic_seq_files

    def count_dynamic_frames(self, write_to_txt=False):

        if write_to_txt:
            fo = open("KITTI_train_split_dynamic_pointnumber.txt", "w")

        self.moving_threshold_num_points = 100

        for seq in self.seqs:

            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(
                seq_id=seq)
            num_moving_frames = 0

            for frame_idx in range(len(velodyne_seq_files)):

                f1_xyzi, f1_semlabel, f1_inslabel = \
                    get_frame_data(
                        pc_path=velodyne_seq_files[frame_idx], label_path=gtsemantic_seq_files[frame_idx])

                f1_moving_label_mask = (f1_semlabel > 250)

                if f1_moving_label_mask.sum() > self.moving_threshold_num_points:
                    num_moving_frames += 1
                
                if write_to_txt:
                    linestr = f"{seq} " + "%06d"%frame_idx + f" {f1_moving_label_mask.sum()}\n"
                    fo.write(linestr)

            print(f"Seq {seq} | Moving frames / all == {num_moving_frames}/{len(velodyne_seq_files)} = {num_moving_frames / len(velodyne_seq_files)}")

        pass

    def count_seqs_points(self,):

        for seq in self.seqs:

            length_min = 1000000
            length_max = -1

            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(seq_id=seq)
            # assert len(velodyne_seq_files) == len(gtsemantic_seq_files)

            for frame_idx in tqdm(range(len(velodyne_seq_files))):
                f1_xyzi = np.fromfile(velodyne_seq_files[frame_idx], dtype=np.float32).reshape((-1, 4))

                if f1_xyzi.shape[0] < length_min:
                    length_min = f1_xyzi.shape[0]
                if f1_xyzi.shape[0] > length_max:
                    length_max = f1_xyzi.shape[0]

            print(f"Seq {seq} | min: {length_min} / max: {length_max}")

    def count_moving_points_in_seqs(self,):

        for seq in self.seqs:

            length_min = 1000000
            length_max = -1
            # load point cloud files
            velodyne_seq_path = os.path.join(dataset_path, "sequences", seq, "velodyne")
            velodyne_seq_files = sorted(glob.glob(os.path.join(velodyne_seq_path, "*.bin")))

            velodyne_seq_files, gtsemantic_seq_files = self.get_file_list(seq_id=seq)
            # assert len(velodyne_seq_files) == len(gtsemantic_seq_files)

            num_moving_frames = 0
            for frame_idx in tqdm(range(len(velodyne_seq_files))):

                f1_xyzi, f1_semlabel, f1_inslabel = \
                    get_frame_data(pc_path=velodyne_seq_files[frame_idx], label_path=gtsemantic_seq_files[frame_idx])

                # mapping rae semantic labels to LiDAR-MOS labels
                f1_moving_label_mask = (f1_semlabel > 250)
                f1_semlabel[f1_moving_label_mask] = 251
                f1_semlabel[~f1_moving_label_mask] = 9

                a, b = np.unique(f1_semlabel, return_counts=True)
                print(a, b)

            print(f"Seq {seq} | min: {length_min} / max: {length_max}")


if __name__ == '__main__':

    dataset_path = '/home1/datasets/semantic_kitti/dataset'
    split = 'train'  # 'valid'
    data_cfg_path = 'config/labels/semantic-kitti-mos.yaml'
    seqKITTI = SeqKITTI(dataset_path, split, data_cfg_path)

    seqKITTI.count_seqs_points()
    # seqKITTI.count_dynamic_frames()
    # seqKITTI.count_dynamic_frames(write_to_txt=True)
    # seqKITTI.count_moving_points_in_seqs()

