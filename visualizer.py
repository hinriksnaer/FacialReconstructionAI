import numpy as np
import open3d as o3d

def draw_pointclouds(source, target, color_1=None, color_2=None):
    
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)

    if color_1 is not None:
        pcd_source.colors = o3d.utility.Vector3dVector(color_1)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)

    if color_2 is not None:
        pcd_target.colors = o3d.utility.Vector3dVector(color_2)

    o3d.visualization.draw_geometries([pcd_source, pcd_target])

def draw_pointcloud(source, colors = None):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    if colors is not None:
        pcd_source.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_source])