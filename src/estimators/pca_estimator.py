import numpy as np
from sklearn.decomposition import PCA
from .base_estimator import BasePoseEstimator
from src.utils.visualization import draw_3d_axes_from_pca
from src.camera.depth_utils import get_3d_point

class PcaEstimator(BasePoseEstimator):
    """
    Estimates object orientation using PCA on its 3D point cloud.
    
    This implementation properly converts pixel+depth to real-world 3D coordinates
    and visualizes axes using camera projection.
    """
    def __init__(self, config):
        super().__init__(config)
        self.hsv_lower = np.array(config['segmentation']['blue_hsv_lower'])
        self.hsv_upper = np.array(config['segmentation']['blue_hsv_upper'])
        self.min_points = 100
        self.max_points = 5000
        self.pca = PCA(n_components=3)

    def estimate(self, color_image, depth_frame):
        """Estimate pose using PCA on segmented object's 3D points."""
        # Segment object
        mask = self._segment_object(color_image)
        if mask is None:
            return color_image
            
        # Get 3D points in world coordinates
        points_3d = self._get_world_points(depth_frame, mask)
        if len(points_3d) < self.min_points:
            return color_image
            
        # Run PCA
        self.pca.fit(points_3d)
        center = self.pca.mean_
        axes = self.pca.components_
        
        # Draw results
        return draw_3d_axes_from_pca(
            color_image, 
            self.mtx, 
            self.dist, 
            center, 
            axes
        )

    def _segment_object(self, image):
        """Segment blue object using HSV color space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask

    def _get_world_points(self, depth_frame, mask):
        """Convert masked pixels to 3D world coordinates."""
        points = []
        height, width = mask.shape
        
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    point = get_3d_point(depth_frame, self.intrinsics, x, y)
                    if point is not None:
                        points.append(point)
                        if len(points) >= self.max_points:
                            break
            if len(points) >= self.max_points:
                break
                
        return np.array(points)