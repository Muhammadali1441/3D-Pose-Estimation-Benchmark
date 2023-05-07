import pyrealsense2 as rs
import numpy as np
from .base_camera import BaseCamera
from .depth_utils import depth_to_pointcloud

class RealSenseCamera(BaseCamera):
    """Intel RealSense D400 series camera implementation with depth utilities."""
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.intrinsics = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        print(f"RealSense camera started ({self.intrinsics.width}x{self.intrinsics.height})")
        return self.intrinsics

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image, depth_frame

    def get_pointcloud(self):
        """Get full point cloud in world coordinates."""
        _, _, depth_frame = self.get_frames()
        if depth_frame is None:
            return None
        return depth_to_pointcloud(depth_frame, self.intrinsics)

    def stop(self):
        self.pipeline.stop()
        print("RealSense camera stream stopped.")