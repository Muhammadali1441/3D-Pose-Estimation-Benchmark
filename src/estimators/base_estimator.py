from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Any

class BasePoseEstimator(ABC):
    """Abstract base class for pose estimation algorithms.
    
    This class defines the common interface for all pose estimation methods.
    Concrete subclasses (like ArUcoEstimator, PcaEstimator, SolvePnpEstimator)
    must implement the estimate() method and inherit from this base class.
    """
    
    def __init__(self, config: dict):
        """
        Initialize estimator with configuration.
        
        Args:
            config: Dictionary containing camera parameters and estimator-specific settings.
                    Must contain 'camera_matrix' and 'distortion_coeffs' keys.
        """
        self.config = config
        self.mtx = np.array(config['camera_matrix'])
        self.dist = np.array(config['distortion_coeffs'])
        
    @abstractmethod
    def estimate(self, color_image: np.ndarray, depth_frame: Optional[Any] = None) -> np.ndarray:
        """
        Estimate pose from input frames and visualize results.
        
        Args:
            color_image: BGR color image as numpy array (HxWx3)
            depth_frame: Raw depth frame data (type depends on camera implementation)
                         Optional - may be None for estimators that don't need depth data
        
        Returns:
            Processed image with pose visualization (BGR format, HxWx3)
            
        Raises:
            RuntimeError: If pose estimation fails (handled by concrete implementations)
        """
        pass