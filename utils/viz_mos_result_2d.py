#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# @author: Jiadai Sun

import sys
import cv2
import yaml
import numpy as np

from kitti_utils import load_vertex, load_labels
from auxiliary.laserscan import LaserScan, SemLaserScan
from matplotlib import pyplot as plt
from utils import load_yaml, check_and_makedirs

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def remap(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]


if __name__ == '__main__':

    seq = "08"
    frame_id = [98, 218, 222, 1630, 1641, 50]

    data_path = "/path_to_dataset/"
    save_path = "plot2d_result"
    check_and_makedirs(save_path)

    path = {
        "gtlabel": data_path,
        "method1": "/the prediction result of method_1",
        "method2": "/the prediction result of method_2",
        "ours": "/the prediction result of ours"
    }

    config_path = "config/labels/semantic-kitti-mos.yaml"
    CFG = load_yaml(config_path)

    color_dict = CFG["color_map"]
    nclasses = len(color_dict)

    scan = SemLaserScan(nclasses, color_dict, project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0)

    for f_id in frame_id:
        str_fid = "%06d"%(f_id)
        print(str_fid)

        scan_path = f'{data_path}/sequences/{seq}/velodyne/{str_fid}.bin'
        scan.open_scan(scan_path)

        for key, value in path.items():
            if key == 'gtlabel':
                label_path = f'{value}/sequences/{seq}/labels/{str_fid}.label'
            else:
                label_path = f'{value}/sequences/{seq}/predictions/{str_fid}.label'

            print(key)

            scan.open_label(label_path)
            scan.sem_label = remap(scan.sem_label, CFG["learning_map"])
            scan.sem_label = remap(scan.sem_label, CFG["learning_map_inv"])
            # print(scan.sem_label.max())
            scan.colorize()
            scan.do_label_projection()

            power = 16
            data = np.copy(scan.proj_range)

            data[data > 0] = data[data > 0]**(1 / power)
            data[data < 0] = data[data > 0].min()

            data = (data - data[data > 0].min()) / \
                (data.max() - data[data > 0].min()) * 255

            out_img = cv2.applyColorMap(data.astype(np.uint8), get_mpl_colormap('viridis'))
            imgpath = f'{save_path}/Range2D_{key}_seq{seq}_fid{f_id}.png'
            cv2.imwrite(imgpath, out_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            seg_vis2D = scan.proj_sem_color * 255 # (64, 1024, 3)
            imgpath = f'{save_path}/Pred2d_{key}_seq{seq}_fid{f_id}.png'
            cv2.imwrite(imgpath, seg_vis2D, [cv2.IMWRITE_PNG_COMPRESSION, 0])
