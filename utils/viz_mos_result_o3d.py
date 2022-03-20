#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# @author: Jiadai Sun

import numpy as np
import open3d as o3d
from kitti_utils import load_vertex, load_labels

if __name__ == '__main__':

    seq = "08"
    frame_id = [98, 218, 222, 1630, 1641, 4016]

    data_path = "/path_to_dataset/"
    path = {
        "gtlabel": data_path,
        "method1": "/the prediction result of method_1",
        "method2": "/the prediction result of method_2",
        "ours": "/the prediction result of ours"
    }

    for f_id in frame_id:
        str_fid = "%06d" % (f_id)
        print(str_fid)

        scan_path = f'{data_path}/sequences/{seq}/velodyne/{str_fid}.bin'
        scan = load_vertex(scan_path)

        for key, value in path.items():
            if key == 'gtlabel':
                label_path = f'{value}/sequences/{seq}/labels/{str_fid}.label'
            else:
                label_path = f'{value}/sequences/{seq}/predictions/{str_fid}.label'

            print(key)
            label, _ = load_labels(label_path)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
            pcd.paint_uniform_color([0.25, 0.25, 0.25])
            colors = np.array(pcd.colors)
            colors[label > 200] = [1.0, 0.0, 0.0]

            pcd.colors = o3d.utility.Vector3dVector(colors)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f'{key}_seq{seq}_frame{f_id}', width=1000, height=1000)
            vis.add_geometry(pcd)
            # parameters = o3d.io.read_pinhole_camera_parameters("/home/user/Repo/LiDAR-MOS/ScreenCamera_2022-02-20-21-03-42.json")
            # ctr = vis.get_view_control()
            # ctr.convert_from_pinhole_camera_parameters(parameters)
            vis.run()
            vis.destroy_window()
