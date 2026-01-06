import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
import os
import json

def quick_3d_view(depth_image_path, robot_path, filter_percentile=0):
    """
    Quick 3D visualization with outlier filtering
    Args:
        depth_image_path: Path to .npy depth file
        filter_percentile: Percentage to filter from both ends (default 5%)
    """
    
    # Load depth data
    pcd_fork = np.load(depth_image_path)
    pcd_fork = pcd_fork / 1000

    # Transformation Parameters
    roll = 0
    pitch = -(30 + 90) / 360 * 2 * np.pi
    yaw = -(45 + 5) / 360 * 2 * np.pi
    translation = np.array([-0.009, 0.009, 0.233])

    # Rotation Matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
                    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
                    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined Rotation (assuming Fixed Frame RPY -> Rz * Ry * Rx)
    R = R_z @ R_y @ R_x

    # Apply Transformation: P_new = R * P + T
    # pcd shape is (N, 3), so we compute (pcd @ R.T) + T
    # pcd_fork = pcd_fork @ R.T + translation

    robot_pcd = np.load(robot_path)
    pcd = np.concatenate((pcd_fork, robot_pcd), axis=0)
    idx_valid = np.where((pcd[:, 0] > 0.2) & (pcd[:, 0] < 0.8))[0]
    pcd = pcd[idx_valid]
    # pcd = pcd_fork
    # pcd[:,0] = pcd[:,0] - np.min(pcd[:,0])
    # pcd[:,1] = pcd[:,1] - np.min(pcd[:,1])
    # pcd[:,2] = pcd[:,2] - np.min(pcd[:,2])
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1)
    # scatter = ax.scatter(x_valid, y_valid, z_valid, 
    #                     c=z_valid, cmap='viridis', s=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.1, 0.8])
    ax.set_ylim([-0.45, 0.45])
    ax.set_zlim([0, 0.8])
    ax.view_init(elev=0, azim=90) # Set view angle (elevation, azimuth)

    ax.set_title(f'3D View (filtered {filter_percentile}% outliers from each end)')
    plt.colorbar(scatter, label='Z (m)')
    plt.savefig("test_fork.png")

# Usage
def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')

    # --- Build both file paths dynamically ---
    fork_path = os.path.join(package_path, 'src', "vision_processing", "diffusion_model_train", "fork.npy")
    robot_path = os.path.join(package_path, 'datas', "Trajectories_record", "Trajectory_25", "images_Trajectory_25", "Robot_point_cloud_0074.npy")
    # json_path = f"{base_path}/Trajectories_preprocess/Trajectory_9/trajectory_9.json"
    
    quick_3d_view(fork_path, robot_path)

if __name__ == "__main__":
    main()