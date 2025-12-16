import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
import os
import json

def quick_3d_view(depth_image_path, filter_percentile=0):
    """
    Quick 3D visualization with outlier filtering
    Args:
        depth_image_path: Path to .npy depth file
        filter_percentile: Percentage to filter from both ends (default 5%)
    """
    
    # Load depth data
    pcd = np.load(depth_image_path)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], 
                        c=pcd[:, 2], cmap='viridis', s=1)
    # scatter = ax.scatter(x_valid, y_valid, z_valid, 
    #                     c=z_valid, cmap='viridis', s=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.1, 0.6])
    ax.set_ylim([-0.4, 0.4])
    ax.set_zlim([0, 0.8])
    ax.view_init(elev=30, azim=45) # Set view angle (elevation, azimuth)

    ax.set_title(f'3D View (filtered {filter_percentile}% outliers from each end)')
    plt.colorbar(scatter, label='Z (m)')
    plt.savefig("test.png")

# Usage
def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')

    # --- Build both file paths dynamically ---
    base_path = os.path.join(package_path, 'datas')
    depth_image_path = f"{base_path}/Trajectories_preprocess/Trajectory_17/Merged_pcd_Trajectory_17/Merged_0059.npy"
    # json_path = f"{base_path}/Trajectories_preprocess/Trajectory_9/trajectory_9.json"
    
    quick_3d_view(depth_image_path)

if __name__ == "__main__":
    main()