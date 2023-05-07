import numpy as np

def sort_4_points_clockwise(points: np.ndarray) -> np.ndarray:
    """
    Sort 4 points in clockwise order (top-left, top-right, bottom-right, bottom-left).
    
    Args:
        points: Nx2 array of points
        
    Returns:
        Sorted 4x2 array of points
    """
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Calculate angles from centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort by angle (counter-clockwise order)
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # Adjust to start from top-left
    if sorted_points[0][0] > sorted_points[1][0]:
        sorted_points = np.roll(sorted_points, -1, axis=0)
    
    return sorted_points

def project_points(points_3d: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                  rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D image coordinates."""
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)
    return points_2d.squeeze()