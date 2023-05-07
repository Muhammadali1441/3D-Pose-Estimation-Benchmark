import cv2
import numpy as np

def draw_axes(image: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
              rvec: np.ndarray, tvec: np.ndarray, length: float = 0.1) -> np.ndarray:
    """Draw 3D coordinate axes on image with proper scaling."""
    axis_points = np.float32([
        [0, 0, 0], 
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ]).reshape(-1, 3)
    
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
    img_points = img_points.squeeze().astype(int)
    
    # Draw axes with different colors for each axis
    cv2.line(image, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # X-axis (red)
    cv2.line(image, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # Y-axis (green)
    cv2.line(image, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # Z-axis (blue)
    return image

def draw_3d_axes_from_pca(image: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                         center: np.ndarray, axes: np.ndarray, length: float = 0.1) -> np.ndarray:
    """Draw PCA-derived axes on image using proper projection."""
    # Create axis points in world coordinates
    axis_points = [
        center,
        center + axes[0] * length,
        center + axes[1] * length,
        center + axes[2] * length
    ]
    
    # Project to image
    img_points, _ = cv2.projectPoints(
        np.array(axis_points), 
        np.zeros(3), 
        np.zeros(3), 
        mtx, 
        dist
    )
    img_points = img_points.squeeze().astype(int)
    
    # Draw axes
    cv2.line(image, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # X
    cv2.line(image, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # Y
    cv2.line(image, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # Z
    return image