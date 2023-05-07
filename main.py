import argparse
import yaml
import cv2
from src.camera.realsense_camera import RealSenseCamera
from src.estimators.aruco_estimator import ArUcoEstimator
from src.estimators.pca_estimator import PcaEstimator
from src.estimators.solvepnp_estimator import SolvePnpEstimator

ESTIMATORS = {
    "aruco": ArUcoEstimator,
    "pca": PcaEstimator,
    "solvepnp": SolvePnpEstimator
}

def main(args):
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize camera
    camera = RealSenseCamera()
    intrinsics = camera.start()
    print(f"Camera intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")

    # Initialize selected estimator
    estimator_class = ESTIMATORS.get(args.estimator)
    if not estimator_class:
        raise ValueError(f"Unknown estimator '{args.estimator}'. Available: {list(ESTIMATORS.keys())}")
    
    estimator = estimator_class(config)
    print(f"Running with {estimator.__class__.__name__} estimator...")

    try:
        while True:
            color_image, _, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            # Process based on estimator type
            if args.estimator == 'pca':
                processed_image = estimator.estimate(color_image, depth_frame)
            else:
                processed_image = estimator.estimate(color_image, None)

            cv2.imshow('Pose Estimation', processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 3D Pose Estimation.")
    parser.add_argument('--estimator', type=str, required=True, choices=ESTIMATORS.keys(),
                        help='The pose estimation algorithm to use.')
    args = parser.parse_args()
    main(args)