import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import pandas as pd
from tqdm import tqdm
from dtw import dtw
import rospkg
import time

# Import Flow Matching model and data loader
from Train_Fork_FM import FlowMatchingAgent, Normalizer, compute_dataset_stats, custom_collate_fn
from Data_Loader_Fork_FM import Robot3DDataset

# ==============================================================================
# 0. UNIFIED STYLE CONFIG
# ==============================================================================

STYLE = {
    # Colors
    'gt_color':       '#2ecc71',   # Green for ground truth
    'pred_color':     '#e74c3c',   # Red for predictions
    'hist_color':     '#3498db',   # Blue for history/past
    'env_color':      '#95a5a6',   # Gray for point cloud
    'accent_color':   '#f39c12',   # Orange for thresholds/highlights
    'bg_color':       '#ffffff',   # White background

    # Fonts
    'title_size':     16,
    'label_size':     13,
    'tick_size':      11,
    'legend_size':    11,
    'font_family':    'sans-serif',

    # Lines
    'gt_lw':          2.5,
    'pred_lw':        1.5,
    'hist_lw':        2.0,
    'grid_alpha':     0.3,
    'pred_alpha':     0.15,

    # Output
    'dpi':            200,
}

def apply_global_style():
    """
    Sets matplotlib rcParams once so every plot inherits the same base look.
    No need to repeat fontsize/grid/background in each function.
    """
    plt.rcParams.update({
        'font.family':          STYLE['font_family'],
        'font.size':            STYLE['tick_size'],
        'axes.titlesize':       STYLE['title_size'],
        'axes.labelsize':       STYLE['label_size'],
        'axes.titleweight':     'bold',
        'axes.edgecolor':       '#333333',
        'axes.linewidth':       0.8,
        'xtick.labelsize':      STYLE['tick_size'],
        'ytick.labelsize':      STYLE['tick_size'],
        'legend.fontsize':      STYLE['legend_size'],
        'legend.framealpha':    0.9,
        'legend.edgecolor':     '#cccccc',
        'figure.facecolor':     STYLE['bg_color'],
        'axes.facecolor':       STYLE['bg_color'],
        'savefig.facecolor':    STYLE['bg_color'],
        'savefig.dpi':          STYLE['dpi'],
        'savefig.bbox':         'tight',
        'grid.alpha':           STYLE['grid_alpha'],
        'grid.linestyle':       '--',
    })

apply_global_style()


# ==============================================================================
# 1. UTILITIES
# ==============================================================================

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ortho6d_to_rotation_matrix(d6):
    """
    Convert 6D rotation representation -> rotation matrix via Gram-Schmidt.
    Works on arbitrary batch shapes: (..., 6) -> (..., 3, 3).
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to unit quaternion (w, x, y, z).
    Works cleanly on batched trajectories of shape (H, 3, 3).
    """
    R = np.asarray(R)
    shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    q_list = []
    
    for r in R_flat:
        tr = np.trace(r)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (r[2, 1] - r[1, 2]) / S
            qy = (r[0, 2] - r[2, 0]) / S
            qz = (r[1, 0] - r[0, 1]) / S
        elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
            S = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            qw = (r[2, 1] - r[1, 2]) / S
            qx = 0.25 * S
            qy = (r[0, 1] + r[1, 0]) / S
            qz = (r[0, 2] + r[2, 0]) / S
        elif r[1, 1] > r[2, 2]:
            S = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            qw = (r[0, 2] - r[2, 0]) / S
            qx = (r[0, 1] + r[1, 0]) / S
            qy = 0.25 * S
            qz = (r[1, 2] + r[2, 1]) / S
        else:
            S = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            qw = (r[1, 0] - r[0, 1]) / S
            qx = (r[0, 2] + r[2, 0]) / S
            qy = (r[1, 2] + r[2, 1]) / S
            qz = 0.25 * S
        q_list.append([qw, qx, qy, qz])
        
    q = np.array(q_list).reshape(shape + (4,))
    # Enforce canonical form (positive scalar part) to avoid antipodal doubling artifacts
    flip = q[..., 0] < 0
    q[flip] *= -1.0
    return q


def geodesic_angle_deg(R1, R2):
    """
    Geodesic (rotation) angle in degrees between two rotation matrices.
    Batched over arbitrary leading dimensions: (..., 3, 3) -> (...).
    """
    r_diff = np.matmul(R1, np.swapaxes(R2, -1, -2))
    trace = np.trace(r_diff, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_ade_position_error(pred_pos, gt_pos):
    """
    Standard point-to-point ADE between two 3D trajectories.
    """
    min_len = min(len(pred_pos), len(gt_pos))
    if min_len == 0:
        return np.nan

    return np.mean(
        np.linalg.norm(pred_pos[:min_len] - gt_pos[:min_len], axis=1)
    )


def compute_dtw_position_error(pred_pos, gt_pos, normalize=True):
    """
    DTW distance between predicted and ground-truth 3D position trajectories.
    """
    pred_pos = np.asarray(pred_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)

    if pred_pos.ndim != 2 or gt_pos.ndim != 2:
        raise ValueError("pred_pos and gt_pos must be 2D arrays: (T, D).")
    if pred_pos.shape[1] != gt_pos.shape[1]:
        raise ValueError(
            f"Trajectory dimensions must match, got {pred_pos.shape[1]} "
            f"and {gt_pos.shape[1]}."
        )
    if len(pred_pos) == 0 or len(gt_pos) == 0:
        return np.nan

    alignment = dtw(
        pred_pos,
        gt_pos,
        dist_method='euclidean',
        keep_internals=False
    )

    if not normalize:
        return alignment.distance

    path_len = max(len(alignment.index1), 1)
    return alignment.distance / path_len


# ==============================================================================
# 2. INFERENCE
# ==============================================================================

def infer_single(model, sample, DEVICE, num_steps=10, method='euler'):
    """
    Single FM inference via ODE integration from t=0 to t=1.
    Returns: predicted action, final position error, final angle error, wall time (s).
    """
    model.eval()

    obs_dict = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE),
        'agent_pos':   sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE),
    }

    start_time = time.perf_counter()

    with torch.no_grad():
        result = model.predict_action(obs_dict, num_steps=num_steps, method=method)
        pred_action = result['action_pred'].cpu().numpy()[0]

    inference_time = time.perf_counter() - start_time

    gt_action = sample['action'].numpy()

    # Final Distance Error
    final_dist = np.linalg.norm(pred_action[-1, :3] - gt_action[-1, :3])

    # Final Rotation Error (geodesic on SO(3))
    gt_rot_mat  = ortho6d_to_rotation_matrix(gt_action[-1, 3:][None, None, :])[0, 0]
    pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:][None, None, :])[0, 0]
    r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
    cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
    final_angle = np.arccos(cos_theta) * (180 / np.pi)

    return pred_action, final_dist, final_angle, inference_time


def infer_with_intermediate_steps(model, sample, DEVICE, num_steps=10):
    """
    ODE integration capturing every intermediate state — used for the GIF.
    Each frame is one Euler step along the learned velocity field.
    """
    model.eval()

    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE)
    agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        obs = {
            'point_cloud': pcd,
            'agent_pos':   model.normalizer.normalize(agent_pos),
        }

        B = 1
        x = torch.randn(B, model.pred_horizon, model.action_dim, device=DEVICE)
        dt = 1.0 / num_steps

        frames = [model.normalizer.unnormalize(x).cpu().numpy()[0]]

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(B, device=DEVICE) * t_val
            v = model.forward(obs, x, t)
            x = x + v * dt
            frames.append(model.normalizer.unnormalize(x).cpu().numpy()[0])

    return frames


# ==============================================================================
# 3. PLOT: TRAINING HISTORY
# ==============================================================================

def plot_training_history(history):
    if history is None:
        print("⚠️ No history found in checkpoint.")
        return

    print("📊 Generating training history plot...")
    fig, ax = plt.subplots(figsize=(10, 5))

    if 'train_loss' in history:
        ax.plot(history['train_loss'],
                color=STYLE['pred_color'], linewidth=2, label='Train Loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'],
                color=STYLE['hist_color'], linewidth=2, label='Validation Loss')

    ax.set_yscale('log')
    ax.set_title('Flow Matching — Training Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.legend()
    ax.grid(True, which='both')

    fig.savefig('01_fm_train_history.png')
    plt.close(fig)


# ==============================================================================
# 4. PLOT: 3D TRAJECTORIES (LOCAL FRAME)
# ==============================================================================

def plot_3d_multimodality_with_orientation(model, sample, DEVICE, idx,
                                           n_samples=20, num_steps=10):
    print(f"🍝 Generating 3D plot for seq {idx}...")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # A. Point cloud
    pcd = sample['obs']['point_cloud'].numpy()
    if len(pcd) > 512:
        pcd_vis = pcd[np.random.choice(len(pcd), 512, replace=False)]
    else:
        pcd_vis = pcd
    ax.scatter(pcd_vis[:, 0], pcd_vis[:, 1], pcd_vis[:, 2],
               c=STYLE['env_color'], s=4, alpha=0.15, label='Point Cloud')

    # B. GT + history
    gt  = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
            color=STYLE['gt_color'], linestyle='--', linewidth=STYLE['gt_lw'],
            label='Ground Truth')
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2],
            color=STYLE['hist_color'], marker='o', markersize=4,
            linewidth=STYLE['hist_lw'], label='History')

    # C. Stochastic predictions
    final_preds = []
    for _ in range(n_samples):
        pred_action, _, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
        final_preds.append(pred_action)
        ax.plot(pred_action[:, 0], pred_action[:, 1], pred_action[:, 2],
                color=STYLE['pred_color'], alpha=STYLE['pred_alpha'], linewidth=1)

    # D. Orientation quivers at final timestep
    scale = 0.05
    gt_last = gt[-1]
    gt_rot = ortho6d_to_rotation_matrix(gt_last[3:][None, None, :])[0, 0]
    ax.quiver(gt_last[0], gt_last[1], gt_last[2],
              gt_rot[0, 0], gt_rot[1, 0], gt_rot[2, 0],
              color=STYLE['gt_color'], linewidth=2, length=scale)

    pred_last = final_preds[0][-1]
    pred_rot = ortho6d_to_rotation_matrix(pred_last[3:][None, None, :])[0, 0]
    ax.quiver(pred_last[0], pred_last[1], pred_last[2],
              pred_rot[0, 0], pred_rot[1, 0], pred_rot[2, 0],
              color=STYLE['pred_color'], linewidth=2, length=scale, label='Pred Orientation')

    # Auto-zoom
    center = np.mean(gt[:, :3], axis=0)
    r = 0.2
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

    ax.set_title(f'Seq {idx} — 3D Trajectories (Local Frame)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left', fontsize=STYLE['legend_size'] - 1)

    fig.savefig(f'02_fm_3d_spaghetti_seq{idx}.png')
    plt.close(fig)


# ==============================================================================
# 5. PLOT: 2D SPAGHETTI (XY DISPERSION)
# ==============================================================================

def plot_multimodality_spaghetti_2d(model, sample, DEVICE, idx,
                                    n_samples=50, num_steps=10):
    print(f"🍝 Generating 2D spaghetti plot for seq {idx}...")
    fig, ax = plt.subplots(figsize=(9, 9))

    gt  = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()

    for i in range(n_samples):
        pred, _, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
        ax.plot(pred[:, 0], pred[:, 1],
                color=STYLE['pred_color'], alpha=0.1, linewidth=STYLE['pred_lw'],
                label='Predictions' if i == 0 else '')

    ax.plot(obs[:, 0], obs[:, 1],
            color=STYLE['hist_color'], marker='o', markersize=4,
            linewidth=STYLE['hist_lw'], label='History')
    ax.plot(gt[:, 0], gt[:, 1],
            color=STYLE['gt_color'], linestyle='--', linewidth=STYLE['gt_lw'],
            label='Ground Truth')

    ax.set_title(f'Seq {idx} — 2D Dispersion (Local Frame)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    fig.savefig(f'02_fm_spaghetti_2d_seq{idx}.png')
    plt.close(fig)


# ==============================================================================
# 6. PLOT: ROTATION ERROR OVER TIME
# ==============================================================================

def plot_rotation_error_over_time(gt_action, pred_action, idx):
    """Geodesic angular error at each timestep along the trajectory."""
    gt_rot  = ortho6d_to_rotation_matrix(gt_action[None, :, 3:])[0]
    pred_rot = ortho6d_to_rotation_matrix(pred_action[None, :, 3:])[0]

    errors = []
    for t in range(len(gt_action)):
        r_diff = np.dot(gt_rot[t], pred_rot[t].T)
        cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(np.arccos(cos_theta) * (180.0 / np.pi))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(errors, color=STYLE['pred_color'], marker='o', markersize=5,
            linewidth=STYLE['pred_lw'], label='Angular Error')
    ax.axhline(y=15, color=STYLE['accent_color'], linestyle='--',
               linewidth=1.5, label='Threshold (15°)')

    ax.set_title(f'Seq {idx} — Rotation Error Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Error (°)')
    ax.legend()
    ax.grid(True)

    fig.savefig(f'05_fm_rotation_error_seq{idx}.png')
    plt.close(fig)


# ==============================================================================
# 7. GIF: FLOW MATCHING ODE INTEGRATION
# ==============================================================================

def make_flow_matching_gif(model, sample, DEVICE, idx, num_steps=10):
    """
    Animated GIF of the ODE integration: from random noise to final trajectory.
    """
    print(f"🎬 Generating FM integration GIF for seq {idx}...")

    frames = infer_with_intermediate_steps(model, sample, DEVICE, num_steps=num_steps)

    # Hold final frame
    for _ in range(15):
        frames.append(frames[-1])

    gt = sample['action'].numpy()
    center = np.mean(gt[:, :3], axis=0)
    pc_np = sample['obs']['point_cloud'].numpy()[::5]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(f):
        ax.clear()
        path = frames[f]
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=STYLE['pred_color'], linewidth=2.5, label='Flow Matching')
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
                color=STYLE['gt_color'], alpha=0.35, linewidth=1.5, label='Ground Truth')
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                   s=1, c=STYLE['env_color'], alpha=0.1)

        n_real = len(frames) - 15
        step_text = f'ODE Step {f}/{num_steps}' if f < n_real else 'Final Prediction'
        ax.set_title(f'Flow Matching Integration\n{step_text}',
                     fontsize=STYLE['title_size'], fontweight='bold')
        ax.set_xlim(center[0] - 0.2, center[0] + 0.2)
        ax.set_ylim(center[1] - 0.2, center[1] + 0.2)
        ax.set_zlim(center[2] - 0.2, center[2] + 0.2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save(f'03_fm_integration_seq{idx}.gif', writer='pillow')
    
    # Clean up reference hook to avoid memory leaks
    del ani
    plt.close(fig)


# ==============================================================================
# 8. QUANTITATIVE ANALYSIS
# ==============================================================================

def run_quantitative_analysis(model, val_dataset, DEVICE,
                              n_samples=20, num_steps=10,
                              use_dtw_for_success=True):
    """
    Quantitative evaluation of the Flow Matching policy.
    Calculates exact evaluation-wide micro-averaged performance metrics.
    """
    model.eval()
    results = []

    THRESHOLD_POS = 0.02   # 2 cm, applied to normalized DTW by default
    THRESHOLD_ROT = 15.0   # 15 deg

    num_eval = min(30, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, num_eval, dtype=int)

    metric_name = 'DTW' if use_dtw_for_success else 'ADE'
    print(
        f"📊 Evaluating {num_eval} sequences "
        f"(success = {metric_name} < {THRESHOLD_POS * 100:.1f} cm "
        f"and RotErr < {THRESHOLD_ROT:.1f}°)..."
    )

    total_trials = 0
    total_successes = 0

    for idx in tqdm(indices):
        sample = val_dataset[idx]
        gt_pos = sample['action'].numpy()[:, :3]

        seq_dtws, seq_ades, seq_angles, seq_preds = [], [], [], []

        for _ in range(n_samples):
            pred, _, final_angle, _ = infer_single(
                model, sample, DEVICE, num_steps=num_steps)

            pred_pos = pred[:, :3]

            current_dtw = compute_dtw_position_error(
                pred_pos, gt_pos, normalize=True)
            current_ade = compute_ade_position_error(pred_pos, gt_pos)

            seq_dtws.append(current_dtw)
            seq_ades.append(current_ade)
            seq_angles.append(final_angle)
            seq_preds.append(pred)

        seq_preds = np.array(seq_preds)

        # Dispersion between stochastic predictions, in meters.
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        path_variance = np.mean(
            np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))

        # Track tracking success rates micro-architecturally across trial runs
        position_errors = seq_dtws if use_dtw_for_success else seq_ades
        success_count = sum(
            pos_err < THRESHOLD_POS and rot < THRESHOLD_ROT
            for pos_err, rot in zip(position_errors, seq_angles))
        
        total_trials += n_samples
        total_successes += success_count
        success_rate = (success_count / n_samples) * 100

        results.append({
            'seq_idx':     idx,
            'DTW':         np.mean(seq_dtws) * 100,       # cm
            'ADE':         np.mean(seq_ades) * 100,       # cm
            'RotErr':      np.mean(seq_angles),           # deg
            'SuccessRate': success_rate,                  # %
            'Dispersion':  path_variance * 100,           # cm
        })

    df = pd.DataFrame(results)
    global_micro_success = (total_successes / total_trials) * 100

    print("\n" + "=" * 60)
    print(f"🚀 RESULTS (Success = {metric_name} < {THRESHOLD_POS * 100:.1f} cm)")
    print("=" * 60)
    print(f"  Micro Success Rate   : {global_micro_success:.2f} %")
    print(f"  Mean Position DTW    : {df['DTW'].mean():.2f} cm")
    print(f"  Mean Position ADE    : {df['ADE'].mean():.2f} cm")
    print(f"  Mean Rotation Error  : {df['RotErr'].mean():.2f} deg")
    print(f"  Mean Dispersion      : {df['Dispersion'].mean():.2f} cm")
    print("=" * 60)

    df.to_csv('fm_quantitative_results_dtw.csv', index=False)
    print("💾 Saved quantitative results to fm_quantitative_results_dtw.csv")

    return df


# ==============================================================================
# 9. PLOT: ERROR DISTRIBUTIONS
# ==============================================================================

def plot_error_distributions(df):
    print("📊 Generating error distribution plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: success rate histogram
    ax1.hist(df['SuccessRate'], bins=10, color=STYLE['gt_color'],
             edgecolor='white', linewidth=0.8, alpha=0.85)
    ax1.set_title('Success Rate Distribution')
    ax1.set_xlabel('Success Rate (%)')
    ax1.set_ylabel('Count')
    ax1.grid(True, axis='y')

    # Right: DTW vs RotErr scatter
    sc = ax2.scatter(df['DTW'], df['RotErr'],
                     c=df['SuccessRate'], cmap='RdYlGn',
                     edgecolors='#333333', linewidths=0.4,
                     s=60, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label('Success Rate (%)', fontsize=STYLE['label_size'])
    ax2.set_title('DTW Position Error vs Rotation Error')
    ax2.set_xlabel('Normalized DTW (cm)')
    ax2.set_ylabel('Rotation Error (°)')
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig('04_fm_metrics_distribution_dtw.png')
    plt.close(fig)


def plot_dtw_vs_ade(df):
    """
    Diagnostic plot: separates timing errors from spatial errors.
    """
    print("📊 Generating DTW vs ADE diagnostic plot...")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(df['ADE'], df['DTW'],
                    c=df['SuccessRate'], cmap='RdYlGn',
                    edgecolors='#333333', linewidths=0.4,
                    s=60, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=STYLE['label_size'])

    max_err = max(df['ADE'].max(), df['DTW'].max())
    ax.plot([0, max_err], [0, max_err],
            color=STYLE['accent_color'], linestyle='--', linewidth=1.5,
            label='ADE = DTW')

    ax.set_title('DTW vs ADE Diagnostic')
    ax.set_xlabel('ADE point-to-point (cm)')
    ax.set_ylabel('Normalized DTW (cm)')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('04b_fm_dtw_vs_ade.png')
    plt.close(fig)


# ==============================================================================
# 10. INFERENCE SPEED BENCHMARK
# ==============================================================================

def run_inference_speed_benchmark(model, val_dataset, DEVICE,
                                  num_steps=10, n_warmup=5, n_runs=100):
    """
    Measures inference latency over many runs on random samples.
    """
    model.eval()
    print(f"⏱️  Inference speed benchmark  (warmup={n_warmup}, runs={n_runs})")

    indices = np.random.choice(len(val_dataset), size=n_runs + n_warmup, replace=True)
    all_times_ms = []

    for i, idx in enumerate(tqdm(indices, desc="Benchmarking")):
        sample = val_dataset[idx]
        _, _, _, dt = infer_single(model, sample, DEVICE, num_steps=num_steps)

        if i >= n_warmup:
            all_times_ms.append(dt * 1000.0)

    times = np.array(all_times_ms)

    mean_ms = np.mean(times)
    std_ms  = np.std(times)
    med_ms  = np.median(times)
    p95_ms  = np.percentile(times, 95)
    p99_ms  = np.percentile(times, 99)
    min_ms  = np.min(times)
    max_ms  = np.max(times)
    freq_hz = 1000.0 / mean_ms

    print("\n" + "=" * 60)
    print("⏱️  INFERENCE SPEED RESULTS")
    print("=" * 60)
    print(f"  Mean latency     : {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  Median latency   : {med_ms:.2f} ms")
    print(f"  Min / Max        : {min_ms:.2f} / {max_ms:.2f} ms")
    print(f"  95th percentile  : {p95_ms:.2f} ms")
    print(f"  99th percentile  : {p99_ms:.2f} ms")
    print(f"  Throughput       : {freq_hz:.1f} Hz")
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.hist(times, bins=30, color=STYLE['hist_color'],
            edgecolor='white', linewidth=0.8, alpha=0.85)
    ax.axvline(mean_ms, color=STYLE['pred_color'], linestyle='-',
               linewidth=1.8, label=f'Mean  {mean_ms:.1f} ms')
    ax.axvline(med_ms, color=STYLE['gt_color'], linestyle='--',
               linewidth=1.8, label=f'Median  {med_ms:.1f} ms')
    ax.axvline(p95_ms, color=STYLE['accent_color'], linestyle=':',
               linewidth=1.8, label=f'P95  {p95_ms:.1f} ms')
    ax.set_title('Inference Latency Distribution')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, axis='y')

    fig.tight_layout()
    fig.savefig('06_fm_inference_speed.png')
    plt.close(fig)

    return times


# ==============================================================================
# 10b. FRÉCHET DISTANCE (FID-STYLE GENERATIVE METRIC)
# ==============================================================================

from scipy.linalg import sqrtm

def _compute_fid_score(mu1, sigma1, mu2, sigma2):
    """
    Standard mathematical calculation for Fréchet Distance using SciPy.
    """
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _fid_from_features(ref, other):
    """
    Handles data conditioning and computes FID using robust covariance tracking.
    """
    ref = np.asarray(ref)
    other = np.asarray(other)

    # Automatically drop completely static dimensions (variance < 1e-6)
    active_dims = ref.std(axis=0) > 1e-6
    ref, other = ref[:, active_dims], other[:, active_dims]

    # Standardize data relative to the Reference (GT) distribution
    mu_ref, std_ref = ref.mean(axis=0), ref.std(axis=0)
    std_ref = np.where(std_ref < 1e-6, 1.0, std_ref) # Safe boundary protection

    ref_scaled = (ref - mu_ref) / std_ref
    other_scaled = (other - mu_ref) / std_ref

    # Extract means and covariances cleanly
    mu_r, sig_r = ref_scaled.mean(axis=0), np.cov(ref_scaled, rowvar=False)
    mu_o, sig_o = other_scaled.mean(axis=0), np.cov(other_scaled, rowvar=False)

    # Regularize slightly to guarantee numerical stability and fix baseline explosions
    sig_r += np.eye(sig_r.shape[0]) * 1e-5
    sig_o += np.eye(sig_o.shape[0]) * 1e-5

    return _compute_fid_score(mu_r, sig_r, mu_o, sig_o)


def compute_trajectory_fid(model, dataset, DEVICE, num_steps=10, n_per_side=500):
    """
    FID-style Fréchet distance using downsampled unit quaternions keyframes 
    to maintain geometry-accurate manifold tracking without numerical collinearity explosions.
    """
    model.eval()

    n = min(n_per_side, len(dataset) // 2)
    real_indices = np.linspace(0, len(dataset) - 1, 2 * n, dtype=int)
    gen_indices = real_indices[:n]   
    print(f"📏 Computing trajectory FID  ({n} samples/side, one chunk per seq)...")

    def feats(action):
        pos = action[:, :3]
        rot_mats = ortho6d_to_rotation_matrix(action[None, :, 3:])[0]  # (H, 3, 3)
        quats = rotation_matrix_to_quaternion(rot_mats) # (H, 4)
        
        # Global canonical sign alignment to fix double-cover artifacts
        if quats[0, 0] < 0:
            quats *= -1.0
            
        # Downsample to strategic keyframes to prevent ill-conditioned covariances
        indices = np.linspace(0, len(quats) - 1, 4, dtype=int)
        downsampled_quats = quats[indices]
        downsampled_pos = pos[indices]
        
        full_feat = np.concatenate([downsampled_pos.reshape(-1), downsampled_quats.reshape(-1)])
        return full_feat, downsampled_pos.reshape(-1), downsampled_quats.reshape(-1)

    real_full, real_pos, real_rot = [], [], []
    for idx in tqdm(real_indices, desc="FID GT features"):
        f, p, r = feats(dataset[int(idx)]['action'].numpy())
        real_full.append(f); real_pos.append(p); real_rot.append(r)

    gen_full, gen_pos, gen_rot = [], [], []
    for idx in tqdm(gen_indices, desc="FID sampling"):
        sample = dataset[int(idx)]
        pred, _, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
        f, p, r = feats(pred)
        gen_full.append(f); gen_pos.append(p); gen_rot.append(r)

    results, baseline = {}, {}
    for name, real_list, gen_list in [
        ('FID_full', real_full, gen_full),
        ('FID_pos',  real_pos,  gen_pos),
        ('FID_rot',  real_rot,  gen_rot),
    ]:
        real = np.asarray(real_list)
        half_a, half_b = real[:n], real[n:]            
        results[name] = _fid_from_features(half_a, np.asarray(gen_list))
        baseline[name] = _fid_from_features(half_a, half_b)

    print("\n" + "=" * 60)
    print("📏 TRAJECTORY FID (Fréchet distance, lower = better)")
    print("=" * 60)
    for k in results:
        print(f"  {k:10s} : {results[k]:.4f}   (baseline {baseline[k]:.4f})")
    print("=" * 60)

    out = {'fid': results, 'baseline': baseline}
    with open('fm_fid_results.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print("💾 Saved FID results to fm_fid_results.json")
    return out


# ==============================================================================
# 11. TIER-1 DIAGNOSTICS — IS RIEMANNIAN FLOW MATCHING NECESSARY?
# ==============================================================================

def characterize_rotation_range(dataset, max_seqs=None):
    """
    (a) Quantify the rotation regime of the dataset.
    """
    n = len(dataset) if max_seqs is None else min(max_seqs, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    print(f"📐 Characterizing rotation range over {n} action chunks...")

    per_step, per_chunk = [], []
    for idx in tqdm(indices, desc="Scanning rotations"):
        action = dataset[int(idx)]['action'].numpy()      
        rot = ortho6d_to_rotation_matrix(action[None, :, 3:])[0]  

        step_ang = geodesic_angle_deg(rot[1:], rot[:-1])
        per_step.extend(step_ang.tolist())
        per_chunk.append(float(geodesic_angle_deg(rot[-1], rot[0])))

    per_step = np.asarray(per_step)
    per_chunk = np.asarray(per_chunk)

    stats = {
        'step_mean':       float(per_step.mean()),
        'step_p95':        float(np.percentile(per_step, 95)),
        'step_max':        float(per_step.max()),
        'chunk_spread_mean': float(per_chunk.mean()),
        'chunk_spread_p95':  float(np.percentile(per_chunk, 95)),
        'chunk_spread_max':  float(per_chunk.max()),
    }

    print("\n" + "=" * 60)
    print("📐 ROTATION RANGE (per action chunk)")
    print("=" * 60)
    print(f"  Per-step angle    : mean {stats['step_mean']:.2f}°  "
          f"p95 {stats['step_p95']:.2f}°  max {stats['step_max']:.2f}°")
    print(f"  Per-chunk spread  : mean {stats['chunk_spread_mean']:.2f}°  "
          f"p95 {stats['chunk_spread_p95']:.2f}°  max {stats['chunk_spread_max']:.2f}°")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.hist(per_step, bins=40, color=STYLE['hist_color'],
             edgecolor='white', linewidth=0.6, alpha=0.85)
    ax1.axvline(stats['step_p95'], color=STYLE['accent_color'], linestyle='--',
                linewidth=1.8, label=f"P95 {stats['step_p95']:.1f}°")
    ax1.set_title('Per-Step Rotation Angle')
    ax1.set_xlabel('Geodesic angle between consecutive waypoints (°)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, axis='y')

    ax2.hist(per_chunk, bins=40, color=STYLE['gt_color'],
             edgecolor='white', linewidth=0.6, alpha=0.85)
    ax2.axvline(stats['chunk_spread_p95'], color=STYLE['accent_color'], linestyle='--',
                linewidth=1.8, label=f"P95 {stats['chunk_spread_p95']:.1f}°")
    ax2.set_title('Per-Chunk Rotation Spread')
    ax2.set_xlabel('Geodesic angle, first→last waypoint (°)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, axis='y')

    fig.tight_layout()
    fig.savefig('07_fm_rotation_range.png')
    plt.close(fig)
    return stats


def plot_chord_vs_geodesic_error(operating_angle_deg=None):
    """
    (b) Analytic distortion of linear interpolation vs the true geodesic (slerp).
    """
    print("📈 Computing chord-vs-geodesic (lerp vs slerp) distortion curve...")

    thetas = np.linspace(1.0, 179.0, 179)
    ts = np.linspace(0.0, 1.0, 51)
    max_err = []

    for th in thetas:
        omega = np.radians(th)
        p = np.array([1.0, 0.0])
        q = np.array([np.cos(omega), np.sin(omega)])

        errs = []
        for t in ts:
            slerp = (np.sin((1 - t) * omega) * p + np.sin(t * omega) * q) / np.sin(omega)
            lin = (1 - t) * p + t * q
            nlerp = lin / (np.linalg.norm(lin) + 1e-12)
            cos_a = np.clip(np.dot(slerp, nlerp), -1.0, 1.0)
            errs.append(np.degrees(np.arccos(cos_a)))
        max_err.append(max(errs))

    max_err = np.asarray(max_err)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thetas, max_err, color=STYLE['pred_color'], linewidth=2.2,
            label='Max interpolation error (lerp vs slerp)')

    if operating_angle_deg is not None:
        op_err = np.interp(operating_angle_deg, thetas, max_err)
        ax.axvline(operating_angle_deg, color=STYLE['gt_color'], linestyle='--',
                   linewidth=1.8,
                   label=f'Operating point  {operating_angle_deg:.1f}°  →  {op_err:.2f}° error')
        ax.scatter([operating_angle_deg], [op_err], color=STYLE['gt_color'],
                   s=60, zorder=5)

    ax.axvspan(150, 180, color=STYLE['accent_color'], alpha=0.12,
               label='Degenerate regime (→180°)')
    ax.set_title('Geometric Distortion of Euclidean Interpolation on SO(3)')
    ax.set_xlabel('Angle between orientations θ (°)')
    ax.set_ylabel('Worst-case angular error (°)')
    ax.set_xlim(0, 180)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('08_fm_chord_vs_geodesic.png')
    plt.close(fig)


def analyze_orthonormality(model, dataset, DEVICE, num_steps=10,
                           n_seqs=60, n_samples=5):
    """
    (c) Check orthonormality of raw 6D output before Gram-Schmidt.
    """
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(n_seqs, len(dataset)), dtype=int)
    print(f"🔎 Orthonormality of raw 6D output over {len(indices)} seqs "
          f"× {n_samples} samples...")

    def ortho_stats(d6):
        a = d6[..., 0:3]
        b = d6[..., 3:6]
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        norm_err = np.concatenate([np.abs(na - 1.0).ravel(),
                                   np.abs(nb - 1.0).ravel()])
        cos_ab = np.sum(a * b, axis=-1) / (na * nb + 1e-12)
        angle = np.degrees(np.arccos(np.clip(cos_ab, -1.0, 1.0)))
        non_ortho = np.abs(90.0 - angle).ravel()
        return norm_err, non_ortho

    pred_norm, pred_ortho, gt_norm, gt_ortho = [], [], [], []
    for idx in tqdm(indices, desc="Sampling outputs"):
        sample = dataset[int(idx)]
        gn, go = ortho_stats(sample['action'].numpy()[:, 3:])
        gt_norm.extend(gn.tolist())
        gt_ortho.extend(go.tolist())
        for _ in range(n_samples):
            pred, _, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
            pn, po = ortho_stats(pred[:, 3:])
            pred_norm.extend(pn.tolist())
            pred_ortho.extend(po.tolist())

    pred_norm = np.asarray(pred_norm); pred_ortho = np.asarray(pred_ortho)
    gt_norm = np.asarray(gt_norm);     gt_ortho = np.asarray(gt_ortho)

    print("\n" + "=" * 60)
    print("🔎 RAW 6D OUTPUT — DISTANCE TO ROTATION MANIFOLD")
    print("=" * 60)
    print(f"  Unit-norm error   | pred mean {pred_norm.mean():.4f}  "
          f"p95 {np.percentile(pred_norm, 95):.4f}  | gt mean {gt_norm.mean():.4f}")
    print(f"  Non-orthogonality | pred mean {pred_ortho.mean():.2f}°  "
          f"p95 {np.percentile(pred_ortho, 95):.2f}°  | gt mean {gt_ortho.mean():.2f}°")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.hist(pred_norm, bins=40, color=STYLE['pred_color'], alpha=0.7,
             edgecolor='white', linewidth=0.5, label='Prediction', density=True)
    ax1.hist(gt_norm, bins=40, color=STYLE['gt_color'], alpha=0.5,
             edgecolor='white', linewidth=0.5, label='Ground Truth', density=True)
    ax1.set_title('Column Unit-Norm Deviation')
    ax1.set_xlabel('|‖column‖ - 1|')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, axis='y')

    ax2.hist(pred_ortho, bins=40, color=STYLE['pred_color'], alpha=0.7,
             edgecolor='white', linewidth=0.5, label='Prediction', density=True)
    ax2.hist(gt_ortho, bins=40, color=STYLE['gt_color'], alpha=0.5,
             edgecolor='white', linewidth=0.5, label='Ground Truth', density=True)
    ax2.set_title('Column Non-Orthogonality')
    ax2.set_xlabel('|90° - angle(col1, col2)|  (°)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, axis='y')

    fig.tight_layout()
    fig.savefig('09_fm_orthonormality.png')
    plt.close(fig)


def analyze_inference_steps(model, dataset, DEVICE,
                            steps_list=(1, 2, 3, 5, 8, 10, 15, 20, 30),
                            n_seqs=40, n_samples=3):
    """
    (d) Convergence of position vs rotation error as the number of ODE steps grows.
    """
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(n_seqs, len(dataset)), dtype=int)
    print(f"🪜 Inference-steps sensitivity over {len(indices)} seqs "
          f"for steps {list(steps_list)}...")

    pos_ade, rot_ade = [], []
    for steps in steps_list:
        seq_pos, seq_rot = [], []
        for idx in tqdm(indices, desc=f"steps={steps}", leave=False):
            sample = dataset[int(idx)]
            gt = sample['action'].numpy()
            gt_rot = ortho6d_to_rotation_matrix(gt[None, :, 3:])[0]
            for _ in range(n_samples):
                pred, _, _, _ = infer_single(model, sample, DEVICE, num_steps=int(steps))
                seq_pos.append(compute_ade_position_error(pred[:, :3], gt[:, :3]))
                pred_rot = ortho6d_to_rotation_matrix(pred[None, :, 3:])[0]
                seq_rot.append(geodesic_angle_deg(pred_rot, gt_rot).mean())
        pos_ade.append(np.mean(seq_pos) * 100.0)   
        rot_ade.append(np.mean(seq_rot))           

    steps_arr = np.asarray(steps_list)

    print("\n" + "=" * 60)
    print("🪜 ERROR vs ODE STEPS")
    print("=" * 60)
    for s, p, r in zip(steps_arr, pos_ade, rot_ade):
        print(f"  steps={s:>3}  |  pos ADE {p:6.2f} cm  |  rot ADE {r:6.2f}°")
    print("=" * 60)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(steps_arr, pos_ade, color=STYLE['hist_color'], marker='o',
             linewidth=2.0, label='Position ADE (cm)')
    ax1.set_xlabel('Number of ODE integration steps')
    ax1.set_ylabel('Position ADE (cm)', color=STYLE['hist_color'])
    ax1.tick_params(axis='y', labelcolor=STYLE['hist_color'])
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(steps_arr, rot_ade, color=STYLE['pred_color'], marker='s',
             linewidth=2.0, label='Rotation ADE (°)')
    ax2.set_ylabel('Rotation ADE (°)', color=STYLE['pred_color'])
    ax2.tick_params(axis='y', labelcolor=STYLE['pred_color'])

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.set_title('Convergence vs ODE Steps — Position vs Rotation')

    fig.tight_layout()
    fig.savefig('10_fm_steps_sensitivity.png')
    plt.close(fig)


# ==============================================================================
# 12. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {DEVICE}")

    NUM_STEPS = 5  

    # 1. Paths
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, 'models', 'best_fm_model_high_dim_CFM_relative_dropout_1024_H.ckpt')

    # 2. Dataset
    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=1024, obs_horizon=2, pred_horizon=16)
    print(f"✅ Validation set: {len(val_dataset)} sequences")

    # 3. Model
    model = FlowMatchingAgent(
        action_dim=9,
        obs_horizon=2,
        pred_horizon=16,
        encoder_output_dim=64,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8).to(DEVICE)

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if 'history' in checkpoint:
        plot_training_history(checkpoint['history'])

    weights = checkpoint.get('model_state_dict',
                             checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(weights, strict=False)
    model.eval()
    print("✅ Weights loaded.")

    # 4. Visual inspection on one sequence
    sample_idx = 25
    if len(val_dataset) > sample_idx:
        sample = val_dataset[sample_idx]
        plot_3d_multimodality_with_orientation(
            model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        plot_multimodality_spaghetti_2d(
            model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        make_flow_matching_gif(
            model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)

    # 5. Full benchmark
    df = run_quantitative_analysis(
        model, val_dataset, DEVICE, n_samples=20, num_steps=NUM_STEPS,
        use_dtw_for_success=True)
    plot_error_distributions(df)
    plot_dtw_vs_ade(df)

    # 5b. Generative distribution metric (FID-style Fréchet distance)
    compute_trajectory_fid(
        model, val_dataset, DEVICE, num_steps=NUM_STEPS, n_per_side=500)

    # 6. Inference speed benchmark
    run_inference_speed_benchmark(
        model, val_dataset, DEVICE, num_steps=NUM_STEPS,
        n_warmup=5, n_runs=100)

    # 6b. Tier-1 diagnostics: is Riemannian flow matching necessary?
    rot_stats = characterize_rotation_range(val_dataset)            
    plot_chord_vs_geodesic_error(
        operating_angle_deg=rot_stats['chunk_spread_p95'])         
    analyze_orthonormality(model, val_dataset, DEVICE,
                           num_steps=NUM_STEPS)                     
    analyze_inference_steps(model, val_dataset, DEVICE)            

    # 7. Worst-case analysis
    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    idx_worst = int(worst_seq['seq_idx'])
    print(f"\n🚨 Worst sequence (idx {idx_worst}): "
          f"Success {worst_seq['SuccessRate']:.1f}% | "
          f"DTW {worst_seq['DTW']:.2f} cm | "
          f"ADE {worst_seq['ADE']:.2f} cm | "
          f"RotErr {worst_seq['RotErr']:.2f}°")

    worst_sample = val_dataset[idx_worst]
    plot_3d_multimodality_with_orientation(
        model, worst_sample, DEVICE, idx_worst, num_steps=NUM_STEPS)

    pred_worst, _, _, _ = infer_single(
        model, worst_sample, DEVICE, num_steps=NUM_STEPS)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)

    print("\n✅ Analysis complete. All figures saved.")


if __name__ == "__main__":
    main()