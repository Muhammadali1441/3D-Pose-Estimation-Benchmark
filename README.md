# 3D Pose Estimation Toolkit

This project is a collection of tools and algorithms for estimating the 3D position and orientation (pose) of objects from 2D images, with a focus on using Intel RealSense depth cameras. It serves as an educational framework to explore, compare, and understand various techniques, from fiducial markers to point cloud analysis.

## Features

- **Modular Estimators**: Easily switch between different pose estimation algorithms.
  - **ArUco Marker Estimation**: Robust and accurate tracking using ArUco markers.
  - **PCA Orientation Estimation**: Accurate orientation estimation from 3D point clouds with proper world coordinate conversion.
  - **SolvePnP Corner Estimation**: Reliable pose estimation using contour-based corner detection with consistent ordering.
- **RealSense Camera Integration**: Clean interface with full depth processing capabilities.
- **Configuration Driven**: All parameters managed in a central `config.yaml` file.
- **Advanced Visualization**: Proper 3D axis projection for accurate pose visualization.

## Setup

1. **Install Dependencies**: Ensure you have the Intel RealSense SDK 2.0 installed. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Parameters**: Open `config.yaml` and adjust:
   - Camera matrix (`camera_matrix`) and distortion coefficients (`distortion_coeffs`) from your calibration
   - HSV color ranges for object segmentation (`segmentation`)
   - Object size for SolvePnP (`solvepnp.object_size`)

## How to Run

Use the main entry point `main.py` to run a live demo. You can select the estimator you want to use via the command line.

**Run ArUco detection:**
```bash
python main.py --estimator aruco
```

**Run PCA-based orientation detection:**
```bash
python main.py --estimator pca
```

**Run SolvePnP-based pose detection:**
```bash
python main.py --estimator solvepnp
```

## Technical Notes

- **PCA Estimator**: Properly converts pixel coordinates to real-world 3D coordinates using camera intrinsics. 
- **SolvePnP Estimator**: Uses contour approximation and corner sorting for reliable 2D-3D correspondence.
- **Visualization**: All estimators now use proper camera projection for axis drawing (not image-space approximations).
- **Error Handling**: Robust checks for invalid depth values, insufficient points, and contour detection failures.
