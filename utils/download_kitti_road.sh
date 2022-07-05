#!/bin/bash

# raw_id -> seq_id
# 2011_09_26_drive_0015 -> 30
# 2011_09_26_drive_0027 -> 31
# 2011_09_26_drive_0028 -> 32
# 2011_09_26_drive_0029 -> 33
# 2011_09_26_drive_0032 -> 34
# 2011_09_26_drive_0052 -> 35
# 2011_09_26_drive_0070 -> 36
# 2011_09_26_drive_0101 -> 37
# 2011_09_29_drive_0004 -> 38
# 2011_09_30_drive_0016 -> 39
# 2011_10_03_drive_0042 -> 40
# 2011_10_03_drive_0047 -> 41

road_raw_id_list=(2011_09_26_drive_0015 2011_09_26_drive_0027 2011_09_26_drive_0028 2011_09_26_drive_0029 
                  2011_09_26_drive_0032 2011_09_26_drive_0052 2011_09_26_drive_0070 2011_09_26_drive_0101 
                  2011_09_29_drive_0004 2011_09_30_drive_0016 2011_10_03_drive_0042 2011_10_03_drive_0047)
sub_id=(30 31 32 33 34 35 36 37 38 39 40 41)

# Please modify it to the local path
DATA_ROOT=DEBUG_kitti_road

mkdir $DATA_ROOT
cd $DATA_ROOT

# wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip

for i in $(seq 0 `expr ${#road_raw_id_list[@]} - 1`); do
    raw_id=${road_raw_id_list[i]}
    sub_id=${sub_id[i]}

    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"$raw_id"/"$raw_id"_sync.zip 
    unzip "$raw_id"_sync.zip 
    mv "${raw_id:0:10}"/"$raw_id"_sync "$sub_id"
    
    cd $sub_id
    mv velodyne_points/data velodyne
    for vbin in velodyne/*;
        do
            if [ ${#vbin} != 23 ]
            then
                echo "[ != ] error, please check the folder"
                break
            fi
            mv "$vbin" "${vbin:0:9}${vbin:13}" # 0000000077.bin --> 000077.bin         
        done
    cd ../
    mkdir -p sequences/$sub_id
    mv $sub_id/velodyne $sub_id/image_02 sequences/$sub_id
    rm -rf $sub_id

done

# rm -rf 2011_09_26 2011_09_29 2011_09_30 2011_10_03

# Move the subfolder(30~41) under the `sequences` folder to the directory corresponding to SemanticKITTI-MOS