#!/usr/bin/env python3
"""
Master's Thesis Visualization Generator

Creates all figures needed for presenting your Diffusion Policy model:
1. 3D trajectory plots (generated vs demo overlay)
2. Joint profiles over time
3. Phase analysis
4. Inference timing
5. TCP vs Fork comparison

Usage:
    # Generate plots from recorded data
    python3 thesis_plots.py \
        --tcp_data tcp_trajectory.json \
        --fork_data fork_trajectory.json \
        --demo_data demonstration.json \
        --timing_data timing.json \
        --output ./thesis_figures

Data Recording (run these separately while executing your policy):
    1. rosrun vision_processing trajectory_recorder.py   # Records poses
    2. rosrun vision_processing inference_timer.py       # Records timing
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import json
import argparse
import os
from scipy import interpolate

# ============================================================
# STYLE CONFIGURATION (Publication Quality)
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'demo': '#27ae60',       # Green
    'tcp': '#3498db',        # Blue
    'fork': '#e74c3c',       # Red
    'phase1': '#9b59b6',     # Purple
    'phase2': '#f39c12',     # Orange
    'phase3': '#1abc9c',     # Teal
}


# ============================================================
# DATA LOADING
# ============================================================
def load_json(filepath):
    """Load JSON file"""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def extract_positions(data, key='ee_positions'):
    """Extract positions from data"""
    if data is None:
        return None
    
    if key in data:
        return np.array(data[key])
    elif 'tcp_poses' in data:
        poses = np.array(data['tcp_poses'])
        return poses[:, 1:4] if poses.shape[1] > 3 else poses
    elif 'fork_poses' in data:
        poses = np.array(data['fork_poses'])
        return poses[:, 1:4] if poses.shape[1] > 3 else poses
    
    return None


def extract_joints(data):
    """Extract joint positions"""
    if data is None:
        return None
    
    if 'joint_positions' in data:
        joints = np.array(data['joint_positions'])
        # Remove timestamp if present
        if joints.shape[1] == 8:
            return joints[:, 1:8]
        return joints[:, :7]
    
    return None


def extract_timestamps(data):
    """Extract timestamps"""
    if data is None:
        return None
    
    if 'timestamps' in data:
        t = np.array(data['timestamps'])
        return t - t[0]  # Start from 0
    elif 'tcp_poses' in data:
        poses = np.array(data['tcp_poses'])
        if poses.shape[1] > 3:
            t = poses[:, 0]
            return t - t[0]
    
    return None


# ============================================================
# FIGURE 1: 3D TRAJECTORY COMPARISON
# ============================================================
def plot_3d_trajectory_overlay(tcp_data, fork_data, demo_data, output_dir):
    """
    Create 3D trajectory plot with demo overlay.
    Shows generated trajectories on top of demonstration.
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Left: TCP Model
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Demo trajectory
    if demo_data:
        demo_pos = extract_positions(demo_data)
        if demo_pos is not None:
            ax1.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                    color=COLORS['demo'], linewidth=3, alpha=0.8,
                    label='Demonstration', linestyle='--')
    
    # TCP generated
    if tcp_data:
        tcp_pos = extract_positions(tcp_data)
        if tcp_pos is not None:
            ax1.plot(tcp_pos[:, 0], tcp_pos[:, 1], tcp_pos[:, 2],
                    color=COLORS['tcp'], linewidth=2, alpha=0.9,
                    label='Generated (TCP)')
            # Start/end markers
            ax1.scatter(*tcp_pos[0], color='green', s=100, marker='o', 
                       edgecolor='black', zorder=5, label='Start')
            ax1.scatter(*tcp_pos[-1], color='red', s=100, marker='s',
                       edgecolor='black', zorder=5, label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('TCP Control Mode', fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Right: Fork Model
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Demo trajectory
    if demo_data:
        demo_pos = extract_positions(demo_data)
        if demo_pos is not None:
            ax2.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                    color=COLORS['demo'], linewidth=3, alpha=0.8,
                    label='Demonstration', linestyle='--')
    
    # Fork generated
    if fork_data:
        fork_pos = extract_positions(fork_data)
        if fork_pos is not None:
            ax2.plot(fork_pos[:, 0], fork_pos[:, 1], fork_pos[:, 2],
                    color=COLORS['fork'], linewidth=2, alpha=0.9,
                    label='Generated (Fork)')
            ax2.scatter(*fork_pos[0], color='green', s=100, marker='o',
                       edgecolor='black', zorder=5)
            ax2.scatter(*fork_pos[-1], color='red', s=100, marker='s',
                       edgecolor='black', zorder=5)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Fork Tip Control Mode', fontweight='bold')
    ax2.legend(loc='upper left')
    
    plt.suptitle('3D End-Effector Trajectories: Generated vs Demonstration', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'fig1_3d_trajectory_comparison.png')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    print(f"✅ Saved: {save_path}")
    
    return fig


def plot_3d_single_with_overlay(gen_data, demo_data, output_dir, mode='fork'):
    """
    Single 3D plot with multiple generated trajectories overlaid on demo.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Demo trajectory (thick dashed line)
    if demo_data:
        demo_pos = extract_positions(demo_data)
        if demo_pos is not None:
            ax.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                   color=COLORS['demo'], linewidth=4, alpha=0.9,
                   label='Demonstration', linestyle='--')
    
    # Generated trajectory
    if gen_data:
        gen_pos = extract_positions(gen_data)
        if gen_pos is not None:
            color = COLORS['fork'] if mode == 'fork' else COLORS['tcp']
            ax.plot(gen_pos[:, 0], gen_pos[:, 1], gen_pos[:, 2],
                   color=color, linewidth=2, alpha=0.8,
                   label=f'Generated ({mode.upper()})')
            
            # Markers
            ax.scatter(*gen_pos[0], color='green', s=150, marker='o',
                      edgecolor='black', linewidth=2, zorder=5, label='Start')
            ax.scatter(*gen_pos[-1], color='red', s=150, marker='X',
                      edgecolor='black', linewidth=2, zorder=5, label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Fork Tip Trajectory: Generated vs Demonstration', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'fig_3d_overlay_{mode}.png')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    print(f"✅ Saved: {save_path}")
    
    return fig


# ============================================================
# FIGURE 2: JOINT PROFILES
# ============================================================
def plot_joint_profiles(tcp_data, fork_data, output_dir):
    """
    Plot joint angles over time for both control modes.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    joint_names = ['Joint 1\n(Shoulder)', 'Joint 2\n(Shoulder)', 
                   'Joint 3\n(Elbow)', 'Joint 4\n(Elbow)',
                   'Joint 5\n(Wrist)', 'Joint 6\n(Wrist)', 
                   'Joint 7\n(Flange)']
    
    # TCP joints
    tcp_joints = extract_joints(tcp_data)
    tcp_t = extract_timestamps(tcp_data)
    print(f"DEBUG TCP: joints shape {tcp_joints.shape}, timestamps shape {len(tcp_t)}")
    
    # Fork joints
    fork_joints = extract_joints(fork_data)
    fork_t = extract_timestamps(fork_data)
    print(f"DEBUG Fork: joints shape {fork_joints.shape}, timestamps shape {len(fork_t)}")
    
    if tcp_joints is not None and tcp_t is not None:
        min_len = min(len(tcp_t), tcp_joints.shape[0])
        tcp_t = tcp_t[:min_len]
        tcp_joints = tcp_joints[:min_len, :]

    if fork_joints is not None and fork_t is not None:
        min_len_f = min(len(fork_t), fork_joints.shape[0])
        fork_t = fork_t[:min_len_f]
        fork_joints = fork_joints[:min_len_f, :]
    
    for i in range(7):
        ax = axes[i]
        
        if tcp_joints is not None and tcp_t is not None:
            ax.plot(tcp_t, np.rad2deg(tcp_joints[:, i]),
                   color=COLORS['tcp'], linewidth=1.5, 
                   label='TCP Model', alpha=0.8)
        
        if fork_joints is not None and fork_t is not None:
            ax.plot(fork_t, np.rad2deg(fork_joints[:, i]),
                   color=COLORS['fork'], linewidth=1.5,
                   label='Fork Model', alpha=0.8)
        
        ax.set_ylabel('Angle (°)')
        ax.set_xlabel('Time (s)')
        ax.set_title(joint_names[i], fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide 8th subplot, add legend
    axes[7].axis('off')
    
    # Create custom legend in empty subplot
    legend_elements = [
        Line2D([0], [0], color=COLORS['tcp'], linewidth=2, label='TCP Control'),
        Line2D([0], [0], color=COLORS['fork'], linewidth=2, label='Fork Control'),
    ]
    axes[7].legend(handles=legend_elements, loc='center', fontsize=12)
    axes[7].set_title('Legend', fontsize=12)
    
    plt.suptitle('Joint Angle Profiles Over Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'fig2_joint_profiles.png')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    print(f"✅ Saved: {save_path}")
    
    return fig


def plot_single_joint_profile(data, output_dir, mode='fork'):
    """Single model joint profile"""
    joints = extract_joints(data)
    t = extract_timestamps(data)
    
    if joints is not None and t is not None:
        min_len = min(len(t), joints.shape[0])
        t = t[:min_len]
        joints = joints[:min_len, :]
    
    if joints is None or t is None:
        print(f"⚠️ No joint data for {mode}")
        return
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4',
                   'Joint 5', 'Joint 6', 'Joint 7 (Wrist)']
    
    color = COLORS['fork'] if mode == 'fork' else COLORS['tcp']
    
    for i in range(7):
        ax = axes[i]
        ax.plot(t, np.rad2deg(joints[:, i]), color=color, linewidth=1.5)
        ax.set_ylabel('Angle (°)')
        ax.set_xlabel('Time (s)')
        ax.set_title(joint_names[i])
        ax.grid(True, alpha=0.3)
        
        # Highlight wrist joint (important for flip phase)
        if i == 6:
            ax.set_facecolor('#fff9e6')
    
    axes[7].axis('off')
    
    plt.suptitle(f'Joint Profiles - {mode.upper()} Control', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'fig_joints_{mode}.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")


# ============================================================
# FIGURE 3: POSITION OVER TIME (PHASE ANALYSIS)
# ============================================================
def plot_xyz_with_phases(data, demo_data, output_dir, mode='fork'):
    """
    Plot X, Y, Z positions over time with phase annotations.
    """
    gen_pos = extract_positions(data)
    gen_t = extract_timestamps(data)
    
    if gen_pos is None:
        print(f"⚠️ No position data for {mode}")
        return
    
    if gen_t is None:
        gen_t = np.arange(len(gen_pos)) * 0.1
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    labels = ['X Position', 'Y Position (Lateral)', 'Z Position (Vertical)']
    color = COLORS['fork'] if mode == 'fork' else COLORS['tcp']
    
    # Demo data
    demo_pos = extract_positions(demo_data) if demo_data else None
    demo_t = None
    if demo_pos is not None:
        demo_t = np.linspace(0, gen_t[-1], len(demo_pos))
    
    for i, ax in enumerate(axes):
        # Demo
        if demo_pos is not None and demo_t is not None:
            ax.plot(demo_t, demo_pos[:, i], color=COLORS['demo'],
                   linewidth=2, alpha=0.7, linestyle='--', label='Demo')
        
        # Generated
        ax.plot(gen_t, gen_pos[:, i], color=color, linewidth=2, label='Generated')
        
        ax.set_ylabel(f'{labels[i]} (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    
    # Add phase annotations
    total_time = gen_t[-1]
    phase_boundaries = [0, total_time*0.3, total_time*0.6, total_time]
    phase_labels = ['Phase 1:\nApproach Food', 'Phase 2:\nWrist Flip', 'Phase 3:\nApproach User']
    phase_colors = [COLORS['phase1'], COLORS['phase2'], COLORS['phase3']]
    
    for ax in axes:
        for j in range(3):
            ax.axvspan(phase_boundaries[j], phase_boundaries[j+1],
                      alpha=0.1, color=phase_colors[j])
    
    # Phase labels at top
    for j in range(3):
        mid = (phase_boundaries[j] + phase_boundaries[j+1]) / 2
        axes[0].text(mid, axes[0].get_ylim()[1], phase_labels[j],
                    ha='center', va='bottom', fontsize=9, color=phase_colors[j])
    
    plt.suptitle(f'Position Components Over Time - {mode.upper()} Control',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'fig3_xyz_phases_{mode}.png')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    print(f"✅ Saved: {save_path}")
    
    return fig


# ============================================================
# FIGURE 4: INFERENCE TIMING
# ============================================================
def plot_inference_timing(timing_data, output_dir):
    """
    Plot inference timing statistics.
    """
    if timing_data is None:
        print("⚠️ No timing data provided")
        return
    
    # Extract timing
    if 'inference_times' in timing_data:
        times = np.array(timing_data['inference_times'])
        if times.ndim == 2:
            times = times[:, 1]  # Second column is the interval
    else:
        times = np.array(timing_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(times, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(times), color='red', linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(times):.1f} ms')
    ax1.axvline(np.median(times), color='green', linewidth=2, linestyle=':',
               label=f'Median: {np.median(times):.1f} ms')
    ax1.axvline(100, color='orange', linewidth=2, linestyle='-',
               label='10 Hz Target (100 ms)')
    
    ax1.set_xlabel('Inference Interval (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Inference Time Distribution')
    ax1.legend(loc='upper right')
    
    # Statistics box
    ax2 = axes[1]
    ax2.axis('off')
    
    stats_text = f"""
    Inference Timing Statistics
    {'='*40}
    
    Samples:           {len(times)}
    Mean:              {np.mean(times):.2f} ms
    Std Dev:           {np.std(times):.2f} ms
    Median:            {np.median(times):.2f} ms
    Min:               {np.min(times):.2f} ms
    Max:               {np.max(times):.2f} ms
    
    95th Percentile:   {np.percentile(times, 95):.2f} ms
    99th Percentile:   {np.percentile(times, 99):.2f} ms
    
    {'='*40}
    Average Rate:      {1000/np.mean(times):.1f} Hz
    Meets 10Hz:        {np.mean(times < 100)*100:.1f}%
    Meets 20Hz:        {np.mean(times < 50)*100:.1f}%
    """
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
            fontsize=11, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Real-Time Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'fig4_inference_timing.png')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    print(f"✅ Saved: {save_path}")
    
    return fig


# ============================================================
# FIGURE 5: COMPREHENSIVE COMPARISON
# ============================================================
def plot_comprehensive_comparison(tcp_data, fork_data, demo_data, output_dir):
    """
    Create comprehensive comparison figure.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 3D TCP
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    tcp_pos = extract_positions(tcp_data)
    demo_pos = extract_positions(demo_data)
    
    if demo_pos is not None:
        ax1.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                color=COLORS['demo'], linewidth=2, linestyle='--', alpha=0.7)
    if tcp_pos is not None:
        ax1.plot(tcp_pos[:, 0], tcp_pos[:, 1], tcp_pos[:, 2],
                color=COLORS['tcp'], linewidth=2)
    ax1.set_title('TCP - 3D Path')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # 3D Fork
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    fork_pos = extract_positions(fork_data)
    
    if tcp_pos is not None:
        min_len = min(len(tcp_pos), tcp_pos.shape[0])
        tcp_pos = tcp_pos[:min_len, :]

    if fork_pos is not None:
        fork_pos = fork_pos[:min_len, :]
    
    if demo_pos is not None:
        ax2.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2],
                color=COLORS['demo'], linewidth=2, linestyle='--', alpha=0.7)
    if fork_pos is not None:
        ax2.plot(fork_pos[:, 0], fork_pos[:, 1], fork_pos[:, 2],
                color=COLORS['fork'], linewidth=2)
    ax2.set_title('Fork - 3D Path')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # XY Projection
    ax3 = fig.add_subplot(gs[0, 2])
    if demo_pos is not None:
        ax3.plot(demo_pos[:, 0], demo_pos[:, 1], color=COLORS['demo'],
                linestyle='--', linewidth=2, label='Demo')
    if tcp_pos is not None:
        ax3.plot(tcp_pos[:, 0], tcp_pos[:, 1], color=COLORS['tcp'],
                linewidth=1.5, label='TCP')
    if fork_pos is not None:
        ax3.plot(fork_pos[:, 0], fork_pos[:, 1], color=COLORS['fork'],
                linewidth=1.5, label='Fork')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    ax3.set_title('XY Projection')
    ax3.legend()
    ax3.axis('equal')
    
    # Y position over time
    ax4 = fig.add_subplot(gs[1, 0])
    tcp_t = extract_timestamps(tcp_data)
    fork_t = extract_timestamps(fork_data)
    
    if tcp_t is not None:
        min_len = min(len(tcp_t), tcp_t.shape[0])
        tcp_t = tcp_t[:min_len]

    if fork_t is not None:
        fork_t = fork_t[:min_len]
    
    if tcp_pos is not None and tcp_t is not None:
        ax4.plot(tcp_t, tcp_pos[:, 1], color=COLORS['tcp'], label='TCP')
    if fork_pos is not None and fork_t is not None:
        ax4.plot(fork_t, fork_pos[:, 1], color=COLORS['fork'], label='Fork')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Y (m)')
    ax4.set_title('Y Position (Lateral)')
    ax4.legend()
    
    # Z position over time
    ax5 = fig.add_subplot(gs[1, 1])
    if tcp_pos is not None and tcp_t is not None:
        ax5.plot(tcp_t, tcp_pos[:, 2], color=COLORS['tcp'], label='TCP')
    if fork_pos is not None and fork_t is not None:
        ax5.plot(fork_t, fork_pos[:, 2], color=COLORS['fork'], label='Fork')
    ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Z (m)')
    ax5.set_title('Z Position (Vertical)')
    ax5.legend()
    
    # Joint 7 (Wrist)
    ax6 = fig.add_subplot(gs[1, 2])
    tcp_joints = extract_joints(tcp_data)
    fork_joints = extract_joints(fork_data)
    
    if tcp_joints is not None and tcp_t is not None:
        ax6.plot(tcp_t, np.rad2deg(tcp_joints[:, 6]), color=COLORS['tcp'], label='TCP')
    if fork_joints is not None and fork_t is not None:
        ax6.plot(fork_t, np.rad2deg(fork_joints[:, 6]), color=COLORS['fork'], label='Fork')
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Angle (°)')
    ax6.set_title('Joint 7 (Wrist Rotation)')
    ax6.legend()
    
    # Metrics comparison
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Compute metrics
    metrics_text = compute_comparison_metrics(tcp_data, fork_data, demo_data)
    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('TCP vs Fork Control Mode Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    save_path = os.path.join(output_dir, 'fig5_comprehensive_comparison.png')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    print(f"✅ Saved: {save_path}")
    
    return fig


def compute_comparison_metrics(tcp_data, fork_data, demo_data):
    """Compute comparison metrics between TCP and Fork"""
    
    metrics = "QUANTITATIVE COMPARISON\n" + "="*60 + "\n\n"
    metrics += f"{'Metric':<30} {'TCP Model':>15} {'Fork Model':>15}\n"
    metrics += "-"*60 + "\n"
    
    tcp_pos = extract_positions(tcp_data)
    fork_pos = extract_positions(fork_data)
    demo_pos = extract_positions(demo_data)
    
    # Trajectory length
    if tcp_pos is not None:
        tcp_length = np.sum(np.linalg.norm(np.diff(tcp_pos, axis=0), axis=1))
    else:
        tcp_length = 0
    
    if fork_pos is not None:
        fork_length = np.sum(np.linalg.norm(np.diff(fork_pos, axis=0), axis=1))
    else:
        fork_length = 0
    
    metrics += f"{'Trajectory Length (m)':<30} {tcp_length:>15.3f} {fork_length:>15.3f}\n"
    
    # Smoothness (jerk)
    def compute_jerk(pos):
        if pos is None or len(pos) < 4:
            return 0
        vel = np.gradient(pos, axis=0)
        acc = np.gradient(vel, axis=0)
        jerk = np.gradient(acc, axis=0)
        return np.mean(np.linalg.norm(jerk, axis=1))
    
    tcp_jerk = compute_jerk(tcp_pos)
    fork_jerk = compute_jerk(fork_pos)
    metrics += f"{'Mean Jerk (m/s³)':<30} {tcp_jerk:>15.4f} {fork_jerk:>15.4f}\n"
    
    # Error vs demo (if available)
    if demo_pos is not None:
        def compute_error(gen_pos):
            if gen_pos is None:
                return 0, 0
            # Resample to same length
            t_gen = np.linspace(0, 1, len(gen_pos))
            t_demo = np.linspace(0, 1, len(demo_pos))
            
            interp_x = interpolate.interp1d(t_gen, gen_pos[:, 0], fill_value='extrapolate')
            interp_y = interpolate.interp1d(t_gen, gen_pos[:, 1], fill_value='extrapolate')
            interp_z = interpolate.interp1d(t_gen, gen_pos[:, 2], fill_value='extrapolate')
            
            gen_resampled = np.column_stack([interp_x(t_demo), interp_y(t_demo), interp_z(t_demo)])
            errors = np.linalg.norm(gen_resampled - demo_pos, axis=1)
            return np.mean(errors), np.max(errors)
        
        tcp_mean_err, tcp_max_err = compute_error(tcp_pos)
        fork_mean_err, fork_max_err = compute_error(fork_pos)
        
        metrics += f"{'Mean Position Error (m)':<30} {tcp_mean_err:>15.4f} {fork_mean_err:>15.4f}\n"
        metrics += f"{'Max Position Error (m)':<30} {tcp_max_err:>15.4f} {fork_max_err:>15.4f}\n"
    
    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Generate Thesis Figures')
    parser.add_argument('--tcp_data', type=str, help='TCP model trajectory JSON')
    parser.add_argument('--fork_data', type=str, help='Fork model trajectory JSON')
    parser.add_argument('--demo_data', type=str, help='Demonstration trajectory JSON')
    parser.add_argument('--timing_data', type=str, help='Inference timing JSON')
    parser.add_argument('--output', type=str, default='./thesis_figures',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    tcp_data = load_json(args.tcp_data)
    fork_data = load_json(args.fork_data)
    demo_data = load_json(args.demo_data)
    timing_data = load_json(args.timing_data)
    
    # Generate figures
    print("\n🎨 Generating figures...\n")
    # Figure 1: 3D trajectories
    if tcp_data or fork_data:
        plot_3d_trajectory_overlay(tcp_data, fork_data, demo_data, args.output)
        
        if fork_data:
            plot_3d_single_with_overlay(fork_data, demo_data, args.output, 'fork')
        if tcp_data:
            plot_3d_single_with_overlay(tcp_data, demo_data, args.output, 'tcp')
    
    # Figure 2: Joint profiles
    if tcp_data or fork_data:
        plot_joint_profiles(tcp_data, fork_data, args.output)
        
        if fork_data:
            plot_single_joint_profile(fork_data, args.output, 'fork')
        if tcp_data:
            plot_single_joint_profile(tcp_data, args.output, 'tcp')
    
    # Figure 3: XYZ with phases
    if fork_data:
        plot_xyz_with_phases(fork_data, demo_data, args.output, 'fork')
    if tcp_data:
        plot_xyz_with_phases(tcp_data, demo_data, args.output, 'tcp')
    
    # Figure 4: Inference timing
    if timing_data:
        plot_inference_timing(timing_data, args.output)
    
    # Figure 5: Comprehensive comparison
    if tcp_data and fork_data:
        plot_comprehensive_comparison(tcp_data, fork_data, demo_data, args.output)
    
    print(f"\n✅ All figures saved to: {args.output}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output)):
        print(f"   {f}")


if __name__ == '__main__':
    main()
