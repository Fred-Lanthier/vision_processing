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

def Transform_points_camera_to_robot(T, file, camera, filter_percentile=3, robot_mask=None):
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

    # Filter valid points ee (non-zero depth values)
    valid = depth > 0
    
    # Apply robot mask if provided (filter out robot pixels)
    if robot_mask is not None:
        # Ensure mask has same shape as depth image
        if robot_mask.shape != depth.shape:
            print(f"Warning: Robot mask shape {robot_mask.shape} doesn't match depth shape {depth.shape}")
        else:
            # Filter out pixels where robot is detected (mask > 0)
            valid = valid & (~robot_mask.astype(bool))
            print(f"Robot mask applied: filtering out {robot_mask.sum()} robot pixels")
    
    depth_valid = depth[valid]

    # Calculate percentile thresholds
    lower_threshold = np.percentile(depth_valid, filter_percentile)
    upper_threshold = np.percentile(depth_valid, 100 - filter_percentile)

    print(f"Filtering: keeping depths between {lower_threshold:.1f} and {upper_threshold:.1f} mm")
    print(f"Original valid points: {valid.sum()}")

    # Apply outlier filtering
    valid_filtered = valid & (depth >= lower_threshold) & (depth <= upper_threshold)

    print(f"After filtering: {valid_filtered.sum()} points ({valid_filtered.sum()/valid.sum()*100:.1f}% remaining)")

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

def quick_3d_view(depth_image_path_ee, depth_image_path_static, step, json_file, filter_percentile=3, robot_mask=None, robot_pcd_path=None):
    """
    Merge and visualize point clouds from two depth images in robot frame.
    Args:
        depth_image_path_ee: Path to .npy depth file
        depth_image_path_static: Path to .npy depth file
        filter_percentile: Percentage to filter from both ends (default 5%)
        robot_mask: Binary mask to filter out robot pixels (from static camera)
        robot_pcd_path: Path to robot point cloud .npy file (optional)
    """
    with open(json_file, "r") as file:
        data = json.load(file)
    T_c1_s = np.array(data["T_static_s"])
    state_datas = data["states"][0]
    state_pos = np.array(state_datas["end_effector_position"])
    state_rot = np.array(state_datas["end_effector_orientation"])

    T_c2_s = compute_T_c_s_ee(state_pos, state_rot)
    print("T_c2_s (camera EE to robot base):", T_c2_s)
    # Load and transform depth data from EE camera
    x_valid_ee, y_valid_ee, z_valid_ee = Transform_points_camera_to_robot(T_c2_s, depth_image_path_ee, camera='ee', filter_percentile=filter_percentile)
    # Load and transform depth data from Static camera (with robot mask filtering)
    x_valid_static, y_valid_static, z_valid_static = Transform_points_camera_to_robot(T_c1_s, depth_image_path_static, camera='static', filter_percentile=filter_percentile, robot_mask=robot_mask)

    # Combine point clouds
    x_valid = np.concatenate([x_valid_ee, x_valid_static])
    y_valid = np.concatenate([y_valid_ee, y_valid_static])
    z_valid = np.concatenate([z_valid_ee, z_valid_static])

    # Load robot point cloud if provided
    robot_pcd = None
    if robot_pcd_path is not None and os.path.exists(robot_pcd_path):
        print(f"Loading robot point cloud from: {robot_pcd_path}")
        robot_pcd = np.load(robot_pcd_path)
        print(f"Robot point cloud shape: {robot_pcd.shape}")
        print(f"Robot points: {len(robot_pcd)}")

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot scene points
    n = 10
    scatter = ax.scatter(x_valid[::n], y_valid[::n], z_valid[::n], 
                        c=z_valid[::n], cmap='viridis', s=1, label='Scene')

    # Plot robot points if available (in different color)
    if robot_pcd is not None:
        ax.scatter(robot_pcd[::n, 0], robot_pcd[::n, 1], robot_pcd[::n, 2] + 0.043,
                  c='red', s=1, label='Robot', alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D View (filtered {filter_percentile}% outliers from each end)')
    ax.legend()
    plt.colorbar(scatter, label='Z (m)')
    plt.show()
    
    # return valid_point_cloud  # Return filtered point cloud


# --- Load config file ---
def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    image_path = os.path.join(package_path, 'scripts', 'images')
    static_rgb = os.path.join(image_path, 'static_rgb_step_000010.png')
    static_depth = os.path.join(image_path, 'static_depth_step_000010.npy')
    ee_rgb = os.path.join(image_path, 'ee_rgb_step_000001.png')
    ee_depth = os.path.join(image_path, 'ee_depth_step_000001.npy')
    json_file = os.path.join(package_path, 'scripts', 'images', 'trajectory_6.json')
    robot_file = os.path.join(image_path, 'Robot_point_cloud_0001.npy')

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
    save_path = os.path.join(package_path, 'scripts', 'grounding_dino_detections.png')
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
    cv2.imwrite(os.path.join(package_path, 'scripts', 'robot_mask_visualization.png'), result_image)
    print("✅ Robot mask visualization saved")

    # Generate 3D point cloud with robot filtered out
    step = 10  # Step number for visualization title
    quick_3d_view(ee_depth, static_depth, step, json_file, filter_percentile=3, robot_mask=robot_mask, robot_pcd_path=robot_file)

if __name__ == "__main__":
    main()
