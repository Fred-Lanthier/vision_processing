"""
Visualize grasp detection on raw trajectories.

This script does NOT go through the DataLoader (which flattens everything 
into local-frame sliding windows). Instead, it loads trajectories directly 
to show:
  1. The fork-tip to food distance curve with the detected grasp index
  2. An animated 3D view of the point clouds showing fork + food dynamics

Usage:
    python Visualize_Grasp.py
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import rospkg


# ============================================================
# UTILS (same as your preprocessing)
# ============================================================

def pose_to_matrix(pos, quat):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def apply_transform(points, T):
    if points.size == 0:
        return points
    if points.ndim == 1:
        points = points.reshape(1, -1)
    ones = np.ones((points.shape[0], 1))
    pts_hom = np.hstack((points, ones))
    return (T @ pts_hom.T).T[:, :3]

def get_t_ee_fork():
    T = np.eye(4)
    rot = R.from_euler('xyz', [0, -207.5, 0.0], degrees=True)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [-0.0055, 0.0, 0.233 - 0.1034]
    return T

def load_matrix_from_json(json_data, key, step_index=0):
    if key not in json_data:
        raise ValueError(f"Key {key} not found in JSON.")
    arr = np.array(json_data[key])
    if arr.shape == (4, 4):
        return arr
    elif arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr[min(step_index, arr.shape[0] - 1)]
    elif arr.shape == (16,):
        return arr.reshape(4, 4)
    elif arr.ndim == 2 and arr.shape[1] == 16:
        return arr[min(step_index, arr.shape[0] - 1)].reshape(4, 4)
    raise ValueError(f"Unknown shape for {key}: {arr.shape}")


# ============================================================
# GRASP DETECTION (same as preprocessing)
# ============================================================

def detect_grasp_index(json_data, food_centroid_world, T_ee_fork, sigma=2):
    states = json_data['states']
    distances = []
    fork_positions = []

    for state in states:
        T_world_ee = pose_to_matrix(
            state['end_effector_position'],
            state['end_effector_orientation']
        )
        T_world_fork = T_world_ee @ T_ee_fork
        fork_tip = T_world_fork[:3, 3]
        fork_positions.append(fork_tip)
        distances.append(np.linalg.norm(fork_tip - food_centroid_world))

    distances = np.array(distances)
    distances_smooth = gaussian_filter1d(distances, sigma=sigma)
    grasp_idx = int(np.argmin(distances_smooth))
    
    return grasp_idx, distances, distances_smooth, np.array(fork_positions)


# ============================================================
# LOAD ONE TRAJECTORY
# ============================================================

def load_trajectory(traj_folder, T_ee_fork):
    """
    Loads everything needed to visualize one trajectory:
      - fork tip positions in world
      - food centroid in world
      - merged point clouds (fork-local frame, as saved by Step 7)
      - grasp index
    """
    folder_name = os.path.basename(traj_folder)
    traj_id = folder_name.split('_')[-1]
    
    # JSON
    json_path = os.path.join(traj_folder, f'trajectory_{traj_id}.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(traj_folder, f'trajectory{traj_id}.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    states = json_data['states']
    
    # Food
    food_pcd_path = os.path.join(traj_folder, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
    food_centroid = None
    if os.path.exists(food_pcd_path):
        food_cam = np.load(food_pcd_path)
        if food_cam.size > 0:
            T_ee_s_step0 = load_matrix_from_json(json_data, "T_ee_s", step_index=0)
            food_world = apply_transform(food_cam, T_ee_s_step0)
            food_centroid = food_world.mean(axis=0)
    
    # Grasp detection
    if food_centroid is not None:
        grasp_idx, dists_raw, dists_smooth, fork_positions = detect_grasp_index(
            json_data, food_centroid, T_ee_fork
        )
    else:
        grasp_idx = None
        dists_raw = dists_smooth = None
        fork_positions = []
        for s in states:
            T_w_ee = pose_to_matrix(s['end_effector_position'], s['end_effector_orientation'])
            T_w_fork = T_w_ee @ T_ee_fork
            fork_positions.append(T_w_fork[:3, 3])
        fork_positions = np.array(fork_positions)
    
    # Merged PCDs (from Step 7 = fork local frame)
    pcd_dir = os.path.join(traj_folder, f'Merged_Fork_{folder_name}')
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, 'Merged_Fork_*.npy')))
    
    return {
        'folder_name': folder_name,
        'json_data': json_data,
        'states': states,
        'fork_positions': fork_positions,
        'food_centroid': food_centroid,
        'grasp_idx': grasp_idx,
        'dists_raw': dists_raw,
        'dists_smooth': dists_smooth,
        'pcd_files': pcd_files,
    }


# ============================================================
# PLOT 1: Distance curve with grasp marker
# ============================================================

def plot_distance_curve(traj_data, save_path=None):
    """
    Shows fork-tip to food distance over time.
    The grasp should be a clear V-shaped minimum.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    steps = np.arange(len(traj_data['dists_raw']))
    
    ax.plot(steps, traj_data['dists_raw'], alpha=0.3, color='blue', label='Raw distance')
    ax.plot(steps, traj_data['dists_smooth'], color='blue', linewidth=2, label='Smoothed distance')
    
    grasp_idx = traj_data['grasp_idx']
    ax.axvline(x=grasp_idx, color='red', linestyle='--', linewidth=2, label=f'Grasp (step {grasp_idx})')
    ax.scatter([grasp_idx], [traj_data['dists_smooth'][grasp_idx]], 
               color='red', s=100, zorder=5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Distance fork tip → food centroid (m)')
    ax.set_title(f"{traj_data['folder_name']} — Grasp Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# PLOT 2: 3D trajectory with grasp coloring
# ============================================================

def plot_3d_trajectory(traj_data, save_path=None):
    """
    Shows fork tip path in 3D, colored by phase:
      - Blue = before grasp (approaching)
      - Red  = after grasp (lifting/moving to user)
    Also shows food centroid position.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    fork_pos = traj_data['fork_positions']
    grasp_idx = traj_data['grasp_idx']
    
    # Before grasp
    ax.plot(fork_pos[:grasp_idx+1, 0], fork_pos[:grasp_idx+1, 1], fork_pos[:grasp_idx+1, 2],
            'b-', linewidth=1.5, alpha=0.7, label='Before grasp')
    # After grasp
    ax.plot(fork_pos[grasp_idx:, 0], fork_pos[grasp_idx:, 1], fork_pos[grasp_idx:, 2],
            'r-', linewidth=1.5, alpha=0.7, label='After grasp')
    
    # Mark grasp point
    gp = fork_pos[grasp_idx]
    ax.scatter(*gp, s=200, c='red', marker='*', zorder=5, label=f'Grasp (step {grasp_idx})')
    
    # Mark start and end
    ax.scatter(*fork_pos[0], s=100, c='green', marker='o', zorder=5, label='Start')
    ax.scatter(*fork_pos[-1], s=100, c='black', marker='s', zorder=5, label='End')
    
    # Food centroid
    if traj_data['food_centroid'] is not None:
        fc = traj_data['food_centroid']
        ax.scatter(*fc, s=150, c='orange', marker='D', zorder=5, label='Food centroid')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{traj_data['folder_name']} — Fork Tip Trajectory")
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# PLOT 3: Animated PCD (fork local frame, as your model sees it)
# ============================================================

def animate_pcd_local_frame(traj_data, step_skip=5, save_path=None):
    """
    Animated GIF of the merged point clouds in fork-local frame.
    This is what your Flow Matching model actually sees as input.
    
    Before grasp: food should be moving (fork approaches food)
    After grasp:  food should be roughly static relative to fork
    """
    pcd_files = traj_data['pcd_files']
    grasp_idx = traj_data['grasp_idx']
    
    # Subsample frames for speed
    frame_indices = list(range(0, len(pcd_files), step_skip))
    if not frame_indices:
        print("  ⚠️ No PCD files found.")
        return
    
    # Load first to get bounds
    first_pcd = np.load(pcd_files[0])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_num):
        ax.clear()
        file_idx = frame_indices[frame_num]
        pcd = np.load(pcd_files[file_idx])
        
        # Extract step number from filename
        step_str = os.path.basename(pcd_files[file_idx]).split('.')[0].split('_')[-1]
        step_idx = int(step_str) - 1
        
        # Color by phase
        phase = "AFTER GRASP (food attached)" if step_idx >= grasp_idx else "BEFORE GRASP (approaching)"
        color = 'red' if step_idx >= grasp_idx else 'blue'
        
        # Plot points (subsample for speed)
        ax.scatter(pcd[::2, 0], pcd[::2, 1], pcd[::2, 2], s=1, c=color, alpha=0.5)
        
        # Fork origin in local frame is always (0,0,0)
        ax.scatter(0, 0, 0, s=100, c='green', marker='o', label='Fork tip (origin)')
        
        # Fixed view bounds (fork-local frame is centered, so bounds are symmetric)
        lim = 0.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{traj_data['folder_name']} — Step {step_idx} | {phase}")
        ax.legend(loc='upper left', fontsize=8)
        return []
    
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=200, blit=False)
    
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=5))
        print(f"  🎬 Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    base_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    
    T_ee_fork = get_t_ee_fork()
    
    # Pick trajectories to visualize (first 3, or change as needed)
    traj_folders = sorted(
        glob.glob(os.path.join(base_path, 'Trajectory*')),
        key=lambda x: int(x.split('_')[-1])
    )
    
    # How many to visualize
    n_viz = min(3, len(traj_folders))
    
    output_dir = os.path.join(pkg_path, 'debug_grasp_viz')
    os.makedirs(output_dir, exist_ok=True)
    
    for traj_folder in traj_folders[:n_viz]:
        folder_name = os.path.basename(traj_folder)
        print(f"\n{'='*60}")
        print(f"  Visualizing: {folder_name}")
        print(f"{'='*60}")
        
        traj_data = load_trajectory(traj_folder, T_ee_fork)
        
        if traj_data['grasp_idx'] is None:
            print("  ⚠️ No food found, skipping.")
            continue
        
        print(f"  Grasp detected at step {traj_data['grasp_idx']}")
        print(f"  Total steps: {len(traj_data['states'])}")
        print(f"  Min distance: {traj_data['dists_smooth'][traj_data['grasp_idx']]:.4f}m")
        
        # Plot 1: Distance curve
        plot_distance_curve(
            traj_data, 
            save_path=os.path.join(output_dir, f'{folder_name}_distance.png')
        )
        
        # Plot 2: 3D trajectory
        plot_3d_trajectory(
            traj_data, 
            save_path=os.path.join(output_dir, f'{folder_name}_trajectory_3d.png')
        )
        
        # Plot 3: Animated PCD (only if files exist)
        if traj_data['pcd_files']:
            animate_pcd_local_frame(
                traj_data,
                step_skip=10,
                save_path=os.path.join(output_dir, f'{folder_name}_pcd_anim.gif')
            )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()