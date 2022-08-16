from numba import njit
import numpy 

import torch
from typing import Any, List, Tuple, Union

@njit(parallel=True)
def dynamic_voxel_generator(lidar: numpy , voxel_size: numpy, point_cloud_range: numpy, max_pt_per_vox: int, max_voxels: int = 20000 ):
    

    # max_points: int = 35,
    # max_voxels: int = 20000,

    # maxiumum number of points per voxel

    # voxel size in meters voxel_size=(0.2, 0.2, 4)
    # vox_depth = 0.4          # 0.4
    # vox_height = 0.2
    # vox_width = 0.2

    vox_depth = voxel_size[2]         # 0.4
    vox_height =voxel_size[1]
    vox_width = voxel_size[0]


    # points cloud range  point_cloud_range=(0, -40, -3, 70.4, 40, 1),
    range_x = (point_cloud_range[0], point_cloud_range[3])
    range_y = (point_cloud_range[1], point_cloud_range[4])
    range_z = (point_cloud_range[2], point_cloud_range[5])

    # range_x = (0, 70.4)
    # range_y = (-40, 40)
    # range_z = (-3, 1)

    # numpy.random.shuffle(lidar)
    voxel_coords = ((lidar[:, :3] - numpy.array([range_x[0], range_y[0], range_z[0]])) /
                     (vox_width, vox_height, vox_depth)).astype(numpy.int32)
    # convert to  (D, H, W)
    voxel_coords = voxel_coords[:,[2,1,0]]
    # unique voxel coordinates, index in original array of each element in unique array
    voxel_coords, inv_ind, voxel_counts = numpy.unique(voxel_coords, axis=0, \
                                              return_inverse=True, return_counts=True)
    voxel_points = list()

    num_points_per_voxel_out = list()

    for i in range(len(voxel_coords)):
        #all pts belong to that voxel    
        pts = lidar[inv_ind == i]
        voxel_points.append(pts)
        voxel_counts[i] = pts.shape[0]
        num_points_per_voxel_out.append(pts.shape[0])

    # voxel_points = numpy.array(voxel_points)
    return voxel_points, voxel_coords, num_points_per_voxel_out  #voxels_out, coors_out, num_points_per_voxel_out