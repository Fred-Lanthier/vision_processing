import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
import yaml
import os
def quick_3d_view(depth_image_path, filter_percentile=3):
    """
    Quick 3D visualization with outlier filtering
    Args:
        depth_image_path: Path to .npy depth file
        filter_percentile: Percentage to filter from both ends (default 5%)
    """
    # Camera intrinsics for RealSense D435i at 640x480 (ADJUST THESE FOR YOUR CAMERA!)
    fx = 616.1005249  # Focal length x
    fy = 615.82617188  # Focal length y
    cx = 318.38803101    # Principal point x (image center)
    cy = 249.23504639    # Principal point y (image center)
    
    # Load depth data
    depth = np.load(depth_image_path)
    print(f"Loaded depth image: {depth.shape}, range: {depth.min()}-{depth.max()}")
    
    # Get image dimensions
    h, w = depth.shape
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Filter valid points (non-zero depth values)
    valid = depth > 0
    depth_valid = depth[valid]
    
    # Calculate percentile thresholds
    lower_threshold = np.percentile(depth_valid, filter_percentile)
    upper_threshold = np.percentile(depth_valid, 100 - filter_percentile)
    
    print(f"Filtering: keeping depths between {lower_threshold:.1f} and {upper_threshold:.1f} mm")
    print(f"Original valid points: {valid.sum()}")
    
    # Apply outlier filtering
    valid_filtered = valid & (depth >= lower_threshold) & (depth <= upper_threshold)
    
    print(f"After filtering: {valid_filtered.sum()} points ({valid_filtered.sum()/valid.sum()*100:.1f}% remaining)")
    
    # Get valid pixel coordinates and depth
    u_valid = u[valid_filtered]
    v_valid = v[valid_filtered]
    z_valid = depth[valid_filtered] / 1000.0  # Convert mm to meters
    
    # Convert from pixel coordinates to 3D camera coordinates
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(x_valid[::10], y_valid[::10], z_valid[::10], 
                        c=z_valid[::10], cmap='viridis', s=1)
    # scatter = ax.scatter(x_valid, y_valid, z_valid, 
    #                     c=z_valid, cmap='viridis', s=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    ax.set_zlim([0.3, 0.5])
    ax.set_title(f'3D View (filtered {filter_percentile}% outliers from each end)')
    plt.colorbar(scatter, label='Z (m)')
    plt.show()

# Usage
def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    depth_image_path = os.path.join(package_path, "src", "vision_processing", "SDF", "Images_Test", "resultat_depth_mask.npy")
    
    quick_3d_view(depth_image_path)

if __name__ == "__main__":
    main()