from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple

class BaseCamera(ABC):
    """Abstract base class for camera interfaces.
    
    This class defines the contract that all camera implementations must follow.
    Concrete subclasses (like RealSenseCamera) must implement all abstract methods.
    """
    
    @abstractmethod
    def start(self) -> None:
        """Start camera stream and initialize resources.
        
        Raises:
            RuntimeError: If camera initialization fails
        """
        pass
        
    @abstractmethod
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray, Optional[object]]:
        """Get synchronized color and depth frames.
        
        Returns:
            Tuple containing:
            - color_image: BGR color image as numpy array (HxWx3)
            - depth_image: Depth image as numpy array (HxW)
            - depth_frame: Raw depth frame object (camera-specific)
            
            Returns (None, None, None) if frame capture fails
        """
        pass
        
    @abstractmethod
    def get_intrinsics(self) -> Optional[object]:
        """Get camera intrinsic parameters.
        
        Returns:
            Camera intrinsics object (type depends on camera implementation)
            or None if intrinsics are not available
        """
        pass
        
    @abstractmethod
    def stop(self) -> None:
        """Stop camera stream and release resources.
        
        Cleans up all resources allocated during start()
        """
        pass