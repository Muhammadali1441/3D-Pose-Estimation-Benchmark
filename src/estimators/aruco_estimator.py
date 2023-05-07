import cv2.aruco as aruco
from .base_estimator import BasePoseEstimator
from src.utils.visualization import draw_axes

class ArUcoEstimator(BasePoseEstimator):
    """Estimates pose using ArUco markers with robust error handling."""
    def __init__(self, config):
        super().__init__(config)
        self.marker_size = config['aruco']['marker_size']
        aruco_dict_name = config['aruco']['dictionary']
        self.aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_name))
        self.parameters = aruco.DetectorParameters_create()

    def estimate(self, color_image, depth_frame=None):
        """Estimate pose using ArUco markers."""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.mtx, self.dist
            )
            # Draw axes for first detected marker
            color_image = draw_axes(color_image, self.mtx, self.dist, rvecs[0], tvecs[0], self.marker_size * 0.5)
        return color_image