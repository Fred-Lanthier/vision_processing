import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
from scipy.spatial.transform import Rotation as R
from Segment_robot import SegmentRobot
import yaml
import json
import os
import glob
import re
from GroundingDINO_with_Segment_Anything_Utils import *

def compute_bounding_box_simple(T, pcd, camera):
    """
    Simple bounding box computation and filtering.
    """
    bbox_limits = {
        'x_min': -0.1,
        'x_max': 0.9,
        'y_min': -0.415,
        'y_max': 0.415,
        'z_min': -0.1,
        'z_max': 0.9
    }
    
    # Transform to robot frame
    point_clouds_camera_augment = np.hstack([pcd, np.ones((len(pcd), 1))])
    point_clouds_robot_augment = (T @ point_clouds_camera_augment.T).T
    point_clouds_robot = point_clouds_robot_augment[:, :3]
    
    # Filter in robot frame
    if camera == 'static':
        bbox_limits['z_min'] = 0.01  # Adjust z_min for static camera if needed
    elif camera == 'ee':
        point_clouds_robot = point_clouds_robot[::5]

    mask = ((bbox_limits['x_min'] <= point_clouds_robot[:, 0]) & (point_clouds_robot[:, 0] <= bbox_limits['x_max']) &
            (bbox_limits['y_min'] <= point_clouds_robot[:, 1]) & (point_clouds_robot[:, 1] <= bbox_limits['y_max']) &
            (bbox_limits['z_min'] <= point_clouds_robot[:, 2]) & (point_clouds_robot[:, 2] <= bbox_limits['z_max']))
    
    valid_point_cloud = point_clouds_robot[mask]
    
    print(f"Total points: {len(point_clouds_robot)}")
    print(f"Filtered point cloud: {len(valid_point_cloud)} points inside bounding box")
    print(f"Points outside: {len(point_clouds_robot) - len(valid_point_cloud)}")
    
    return valid_point_cloud

def Transform_points_camera_to_robot(T, file, camera, filter_percentile=3, robot_mask=None, inverse_mask=False):
    """
    Transform depth points to robot frame.
    
    Args:
        T: Transformation matrix
        file: Path to depth .npy file
        camera: 'static' or 'ee'
        filter_percentile: Percentage to filter outliers
        robot_mask: Binary mask for filtering
        inverse_mask: If False, keeps robot pixels. If True, filters out robot pixels (keeps environment)
    """
    depth = np.load(file)

    if camera == 'static':
        fx = 607.18261719  # Focal length x
        fy = 606.91986084  # Focal length y
        cx = 320.85250854    # Principal point x (image center)
        cy = 243.40284729    # Principal point y (image center)
    elif camera == 'ee':
        fx = 616.1005249
        fy = 615.82617188
        cx = 318.38803101  # (ppx - principal point x)
        cy = 249.23504639  # (ppy - principal point y)

    # Get image dimensions
    h, w = depth.shape
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Filter valid points (non-zero depth values)
    valid = depth > 0
    
    # Apply robot mask if provided
    if robot_mask is not None:
        # Ensure mask has same shape as depth image
        if robot_mask.shape != depth.shape:
            print(f"Warning: Robot mask shape {robot_mask.shape} doesn't match depth shape {depth.shape}")
        else:
            if inverse_mask:
                # Filter OUT robot pixels (keep environment)
                valid = valid & (~robot_mask.astype(bool))
                print(f"Robot mask applied (inverse): filtering out {robot_mask.sum()} robot pixels, keeping environment")
            else:
                # KEEP ONLY robot pixels
                valid = valid & robot_mask.astype(bool)
                print(f"Robot mask applied: keeping only {robot_mask.sum()} robot pixels")
    
    depth_valid = depth[valid]

    if len(depth_valid) == 0:
        print("Warning: No valid points after mask filtering!")
        return np.array([]), np.array([]), np.array([])

    # Calculate percentile thresholds
    lower_threshold = np.percentile(depth_valid, filter_percentile)
    upper_threshold = np.percentile(depth_valid, 100 - filter_percentile)

    print(f"Filtering: keeping depths between {lower_threshold:.1f} and {upper_threshold:.1f} mm")
    print(f"Original valid points: {valid.sum()}")

    # Apply outlier filtering
    valid_filtered = valid & (depth >= lower_threshold) & (depth <= upper_threshold)

    print(f"After filtering: {valid_filtered.sum()} points ({valid_filtered.sum()/max(valid.sum(), 1)*100:.1f}% remaining)")

    u_valid = u[valid_filtered]
    v_valid = v[valid_filtered]
    z_valid = depth[valid_filtered] / 1000.0  # Convert mm to meters
    if camera == "static":
        z_valid += 0.029
    # Convert from pixel coordinates to 3D camera coordinates
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    pcd = np.column_stack([x_valid, y_valid, z_valid])
    
    # Transform to robot frame
    valid_point_cloud = compute_bounding_box_simple(T, pcd, camera)
    x_valid = valid_point_cloud[:, 0]
    y_valid = valid_point_cloud[:, 1]
    z_valid = valid_point_cloud[:, 2]

    return x_valid, y_valid, z_valid

def compute_T_c_s_ee(state_pos: np.ndarray, state_rot: np.ndarray) -> np.ndarray:
    """
    Compute transformation from static camera to robot base.
    
    Transformation chain explanation:
    ---------------------------------
    T_ee_s : End-effector → Robot base (from robot state)
    T_g_ee : Gripper → End-effector (45° rotation + offset)
    T_c2_g : Camera2 → Gripper (from camera holder)
    T_a_c2 : ArUco → Camera2 (from scan pose)

    Final chain: T_c2_s = T_c2_g @ T_g_s
    where : T_g_s = T_g_ee @ T_ee_s
    """
    # End-effector to robot base transformation
    R_ee = R.from_quat(state_rot).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_ee
    T[:3, 3] = state_pos
    T_ee_s = T
    print("T_ee_s (end-effector to robot base):", T_ee_s)
    # Gripper to end-effector (45° rotation around Z-axis)
    angle_degrees = 0
    T_g_ee = np.array([[ np.cos(np.radians(angle_degrees)),  -np.sin(np.radians(angle_degrees)),   0,       0],
                        [np.sin(np.radians(angle_degrees)),    np.cos(np.radians(angle_degrees)),   0,       0],
                        [       0,          0,   1,  -0.0],
                        [       0,          0,   0,       1]])
    T_ee_g = np.linalg.inv(T_g_ee)
    
    # Camera2 to gripper transformation
    T_c2_g = np.array([[ 0,  1, 0, -0.052],
                        [-1,  0, 0,  0.035],
                        [ 0,  0, 1, -0.045],
                        [ 0,  0, 0,      1]])
    
    T_c2_s = T_ee_s @ T_ee_g @ T_c2_g
    return T_c2_s

def quick_3d_view_robot_only(depth_image_path_static, step, json_file, filter_percentile=3, robot_mask=None):
    """
    Visualize robot point cloud (red) and environment point cloud (blue) from static camera using the robot mask.
    
    Args:
        depth_image_path_static: Path to .npy depth file from static camera
        step: Step number for visualization title
        json_file: Path to JSON file with transformation data
        filter_percentile: Percentage to filter from both ends (default 3%)
        robot_mask: Binary mask to separate robot from environment
    """
    with open(json_file, "r") as file:
        data = json.load(file)
    T_c1_s = np.array(data["T_static_s"])

    # Load and transform depth data for ROBOT pixels (keeping only robot)
    print("\n=== Extracting Robot Points ===")
    x_valid_robot, y_valid_robot, z_valid_robot = Transform_points_camera_to_robot(
        T_c1_s, depth_image_path_static, camera='static', 
        filter_percentile=filter_percentile, robot_mask=robot_mask, inverse_mask=False
    )

    # Load and transform depth data for ENVIRONMENT pixels (filtering out robot)
    print("\n=== Extracting Environment Points ===")
    x_valid_env, y_valid_env, z_valid_env = Transform_points_camera_to_robot(
        T_c1_s, depth_image_path_static, camera='static', 
        filter_percentile=filter_percentile, robot_mask=robot_mask, inverse_mask=True
    )

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot environment points in blue
    if len(x_valid_env) > 0:
        n_env = 10  # Subsample environment points
        ax.scatter(x_valid_env[::n_env], y_valid_env[::n_env], z_valid_env[::n_env], 
                  c='blue', s=1, label='Environment', alpha=0.5)
        print(f"Environment points plotted: {len(x_valid_env)}")
    
    # Plot robot points in red
    if len(x_valid_robot) > 0:
        n_robot = 1  # Show all robot points (or use n=5 to subsample)
        ax.scatter(x_valid_robot[::n_robot], y_valid_robot[::n_robot], z_valid_robot[::n_robot], 
                  c='red', s=1, label='Robot', alpha=0.8)
        print(f"Robot points plotted: {len(x_valid_robot)}")
    else:
        print("❌ No robot points to visualize!")

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Robot (Red) and Environment (Blue) Point Clouds')
    ax.legend()
    plt.show()
    
    # Return robot point cloud for saving if needed
    if len(x_valid_robot) > 0:
        robot_pcd = np.column_stack([x_valid_robot, y_valid_robot, z_valid_robot])
        print(f"✅ Robot point cloud extracted: {len(robot_pcd)} points")
        return robot_pcd
    else:
        return None


# --- Load config file ---
def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    image_path = os.path.join(package_path, 'scripts', 'images')
    static_rgb = os.path.join(image_path, 'static_rgb_step_000010.png')
    static_depth = os.path.join(image_path, 'static_depth_step_000010.npy')
    json_file = os.path.join(package_path, 'scripts', 'images', 'trajectory_6.json')

    # Load and display static RGB image
    static_rgb_image = load_image(static_rgb)
    cv2.imshow("Static RGB Image", np.array(static_rgb_image))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Segment robot in static camera image
    model_path = os.path.join(package_path, 'scripts', 'sam_vit_l.pth')
    segment_robot = SegmentRobot(static_rgb, model_name=model_path)

    # Detect robot using GroundingDINO
    detections = segment_robot.grounding_dino_detect(static_rgb, segment_robot.detection_list)
    best_detection = [max(detections, key=lambda x: x.score)]

    # Visualize GroundingDINO detections
    save_path = os.path.join(package_path, 'scripts', 'grounding_dino_detections_robot_only.png')
    segment_robot.visualize_grounding_dino_results(static_rgb, best_detection, save_path)

    # Get robot mask using SAM
    robot_detection = segment_robot.detect_robot(static_rgb_image, best_detection[0])
    robot_mask = robot_detection['mask']
    
    print(f"Robot mask shape: {robot_mask.shape}")
    print(f"Robot pixels detected: {robot_mask.sum()}")

    # Visualize the robot mask
    result_image = np.array(static_rgb_image).copy()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    result_image[robot_mask > 0] = [0, 255, 0]  # Green for robot
    cv2.imwrite(os.path.join(package_path, 'scripts', 'robot_mask_visualization_only.png'), result_image)
    print("✅ Robot mask visualization saved")

    # Generate 3D point cloud keeping ONLY robot points
    step = 10  # Step number for visualization title
    robot_pcd = quick_3d_view_robot_only(static_depth, step, json_file, filter_percentile=3, robot_mask=robot_mask)
    
    # Optionally save the robot point cloud
    if robot_pcd is not None and len(robot_pcd) > 0:
        output_path = os.path.join(package_path, 'scripts', 'Robot_point_cloud_from_mask.npy')
        np.save(output_path, robot_pcd)
        print(f"✅ Robot point cloud saved to: {output_path}")

if __name__ == "__main__":
    main()
