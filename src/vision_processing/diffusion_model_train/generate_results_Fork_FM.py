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


def compute_ade_position_error(pred_pos, gt_pos):
    """
    Standard point-to-point ADE between two 3D trajectories.

    If the trajectories do not have the same length, both are truncated to the
    shortest length so this metric remains usable. DTW should be preferred when
    timing or trajectory length differs.
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

    pred_pos: array-like, shape (T_pred, 3), in meters
    gt_pos:   array-like, shape (T_gt, 3), in meters

    If normalize=True, returns the average DTW cost per alignment-pair, which is
    easier to interpret like an ADE after temporal alignment. The result is still
    in meters if the input positions are in meters.
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

    # dtw-python supports multivariate trajectories directly. With 3D inputs,
    # dist_method='euclidean' computes the Euclidean distance between positions.
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
    Each frame = one Euler step along the learned velocity field.
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
    plt.close(fig)


# ==============================================================================
# 8. QUANTITATIVE ANALYSIS
# ==============================================================================

def run_quantitative_analysis(model, val_dataset, DEVICE,
                              n_samples=20, num_steps=10,
                              use_dtw_for_success=True):
    """
    Quantitative evaluation of the Flow Matching policy.

    Main position metric:
      - DTW: trajectory distance after temporal alignment, in cm.

    Extra diagnostic metric:
      - ADE: point-to-point trajectory distance, in cm.

    Success uses normalized DTW by default:
      normalized DTW < 2 cm AND final rotation error < 15 deg.
    """
    model.eval()
    results = []

    THRESHOLD_POS = 0.02   # 2 cm, applied to normalized DTW by default
    THRESHOLD_ROT = 15.0   # 15 deg

    num_eval = min(len(val_dataset) // 3, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, num_eval, dtype=int)

    metric_name = 'DTW' if use_dtw_for_success else 'ADE'
    print(
        f"📊 Evaluating {num_eval} sequences "
        f"(success = {metric_name} < {THRESHOLD_POS * 100:.1f} cm "
        f"and RotErr < {THRESHOLD_ROT:.1f}°)..."
    )

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

        # Success based on DTW by default. ADE is still saved for diagnostics.
        position_errors = seq_dtws if use_dtw_for_success else seq_ades
        success_count = sum(
            pos_err < THRESHOLD_POS and rot < THRESHOLD_ROT
            for pos_err, rot in zip(position_errors, seq_angles))
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

    print("\n" + "=" * 60)
    print(f"🚀 RESULTS (Success = {metric_name} < {THRESHOLD_POS * 100:.1f} cm)")
    print("=" * 60)
    print(f"  Mean Success Rate    : {df['SuccessRate'].mean():.2f} %")
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

    - Low DTW + high ADE: good path, bad timing.
    - High DTW + high ADE: bad path.
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

    Warmup: the first few CUDA calls are slow (lazy kernel compilation
    and memory allocation). We discard them so the histogram reflects
    the steady-state latency your robot actually sees.
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
# 11. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {DEVICE}")

    NUM_STEPS = 5  # ODE integration steps

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

    # 6. Inference speed benchmark
    run_inference_speed_benchmark(
        model, val_dataset, DEVICE, num_steps=NUM_STEPS,
        n_warmup=5, n_runs=100)

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