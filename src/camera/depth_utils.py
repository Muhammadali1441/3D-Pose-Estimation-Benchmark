import numpy as np
import pyrealsense2 as rs

def depth_to_pointcloud(depth_frame, intrinsics):
    """Convert depth frame to 3D point cloud in world coordinates."""
    points = []
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    
    for y in range(height):
        for x in range(width):
            depth = depth_frame.get_distance(x, y)
            if depth > 0.01 and depth < 2.0:  # Filter invalid depths
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                points.append(point)
    return np.array(points)

def get_3d_point(depth_frame, intrinsics, x, y):
    """Convert single pixel to 3D world coordinate."""
    depth = depth_frame.get_distance(x, y)
    if depth <= 0.01 or depth > 2.0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)