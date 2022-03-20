#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of the project OverlapNet: https://github.com/PRBonn/OverlapNet
# Brief: a script to generate depth data

import os
import cv2
import numpy as np
import scipy.linalg as linalg

from kitti_utils import load_files, range_projection
from matplotlib import pyplot as plt


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def rotate_mat( axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix
    # print(type(rot_matrix))


def gen_depth_data(scan_folder, dst_folder, normalize=False):
    """ Generate projected range data in the shape of (64, 900, 1).
        The input raw data are in the shape of (Num_points, 3).
    """
    # specify the goal folder
    dst_folder = os.path.join(dst_folder, 'depth')
    try:
        os.stat(dst_folder)
        print('generating depth data in: ', dst_folder)
    except:
        print('creating new depth folder: ', dst_folder)
        os.makedirs(dst_folder)

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)

    depths = []
    axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]

    # iterate over all scan files
    for idx in range(len(scan_paths)):
        # load a point cloud
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        current_vertex = current_vertex.reshape((-1, 4))

        # proj_vertex = range_projection(current_vertex, fov_up=14.8940, fov_down=-16.1760, proj_H=32, proj_W=900, max_range=50)
        proj_vertex = range_projection(current_vertex, fov_up=3, fov_down=-25, proj_H=64, proj_W=2048, max_range=50)

        proj_range = proj_vertex[:, :, -1]
        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range) * 255

        # generate the destination path
        dst_path = os.path.join(dst_folder, str(idx).zfill(6))

        # np.save(dst_path, proj_range)
        filename = dst_path + ".png"
        color_img = cv2.applyColorMap(proj_range.astype(np.uint8),  get_mpl_colormap('viridis'))#cv2.COLORMAP_RAINBOW)

        cv2.imwrite(filename, color_img)
        print('finished generating depth data at: ', filename)

    return depths


if __name__ == '__main__':

    scan_folder = "data/sequences/05/velodyne"
    save_folder = 'viz_depth_result'

    depth_data = gen_depth_data(scan_folder, save_folder)
