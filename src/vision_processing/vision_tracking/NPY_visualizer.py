import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
import os
import json

def quick_3d_view(depth_image_path, json_path, filter_percentile=0):
    """
    Quick 3D visualization with outlier filtering
    Args:
        depth_image_path: Path to .npy depth file
        filter_percentile: Percentage to filter from both ends (default 5%)
    """
    # Camera intrinsics for RealSense D435i at 640x480 (ADJUST THESE FOR YOUR CAMERA!)
    # fx = 607.18261719  # Focal length x
    # fy = 606.91986084  # Focal length y
    # cx = 320.85250854    # Principal point x (image center)
    # cy = 243.40284729    # Principal point y (image center)

    fx = 616.1005249
    fy = 615.82617188
    cx = 318.38803101
    cy = 249.23504639
    
    # Load depth data
    with open(json_path, "r") as file:
        data = json.load(file)
    T_c1_s = np.array(data["T_static_s"])
    pcd = np.load(depth_image_path)
    pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
    pcd = (T_c1_s @ pcd.T).T        
    x_valid = pcd[:, 0]
    y_valid = pcd[:, 1]
    z_valid = pcd[:, 2]

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(y_valid[::], -x_valid[::], z_valid[::], 
                        c=z_valid[::], cmap='viridis', s=1)
    # scatter = ax.scatter(x_valid, y_valid, z_valid, 
    #                     c=z_valid, cmap='viridis', s=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.7, 0.1])
    ax.set_zlim([0, 0.8])
    ax.view_init(elev=35, azim=-30) # Set view angle (elevation, azimuth)

    ax.set_title(f'3D View (filtered {filter_percentile}% outliers from each end)')
    plt.colorbar(scatter, label='Z (m)')
    plt.savefig("test.png")

# Usage
def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')

    # --- Build both file paths dynamically ---
    base_path = os.path.join(package_path, 'scripts')
    depth_image_path = f"{base_path}/Robot_pcd_filtered/robot_cloud_latest.npy"
    json_path = f"{base_path}/Robot_depth_trajectory/trajectory_1.json"
    
    quick_3d_view(depth_image_path, json_path)

if __name__ == "__main__":
    main()