import cv2
import numpy as np
from .base_estimator import BasePoseEstimator
from src.utils.visualization import draw_axes
from src.utils.geometry import sort_4_points_clockwise

class SolvePnpEstimator(BasePoseEstimator):
    """
    Estimates pose using SolvePnP with contour-based corner detection.
    
    This implementation:
    - Uses contour approximation for reliable corner detection
    - Sorts corners in consistent order
    - Scales object points based on real-world size
    """
    def __init__(self, config):
        super().__init__(config)
        self.hsv_lower = np.array(config['segmentation']['blue_hsv_lower'])
        self.hsv_upper = np.array(config['segmentation']['blue_hsv_upper'])
        self.object_size = config.get('solvepnp', {}).get('object_size', 0.1)
        self.object_points = self._scale_object_points(config['solvepnp']['object_points'])

    def _scale_object_points(self, points):
        """Scale 3D model points to real-world size."""
        return np.array(points) * self.object_size

    def estimate(self, color_image, depth_frame=None):
        """Estimate pose using SolvePnP with contour-based corner detection."""
        # Segment object
        mask = self._segment_object(color_image)
        if mask is None:
            return color_image
            
        # Find contours and approximate corners
        corners = self._find_object_corners(mask)
        if corners is None:
            return color_image
            
        # SolvePnP
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners,
            self.mtx,
            self.dist
        )
        
        if success:
            # Draw results
            color_image = draw_axes(color_image, self.mtx, self.dist, rvec, tvec, self.object_size * 0.5)
            for point in corners:
                x, y = int(point[0]), int(point[1])
                cv2.circle(color_image, (x, y), 4, (0, 255, 0), -1)
                
        return color_image

    def _segment_object(self, image):
        """Segment blue object using HSV color space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask

    def _find_object_corners(self, mask):
        """Find and sort object corners using contour approximation."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Check if we have exactly 4 corners
        if len(approx) != 4:
            return None
            
        # Reshape and sort corners
        corners = approx.reshape(4, 2)
        return sort_4_points_clockwise(corners).astype(np.float32)