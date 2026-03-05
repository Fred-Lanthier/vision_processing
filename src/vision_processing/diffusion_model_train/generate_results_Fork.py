import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import pandas as pd
from tqdm import tqdm
import rospkg
import time
import seaborn as sns
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from Train_Fork import DP3AgentRobust, Normalizer
from Data_Loader_Fork import Robot3DDataset, seed_everything

# ==============================================================================
# 0. UNIFIED STYLE CONFIG
# ==============================================================================
# One place to control the look of every single plot.
# Changing a value here propagates everywhere.

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
    'pred_alpha':     0.15,   # For spaghetti-style overlays

    # Output
    'dpi':            200,
    'fig_format':     'png',
}

def apply_global_style():
    """
    Sets matplotlib rcParams so every plot inherits the same base look
    without having to repeat styling code in each function.
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

# Apply once at import time
apply_global_style()


# ==============================================================================
# 1. MATH UTILITIES
# ==============================================================================

def ortho6d_to_rotation_matrix(d6):
    """
    Converts 6D rotation representation (B, T, 6) -> Rotation Matrix (B, T, 3, 3).
    Uses Gram-Schmidt orthogonalization on the first two columns.
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


# ==============================================================================
# 2. PLOT: TRAINING HISTORY
# ==============================================================================

def plot_training_history(history):
    """Plots training/validation loss curves from checkpoint history."""
    if history is None:
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
    ax.set_title('Training Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.legend()
    ax.grid(True, which='both')

    fig.savefig('01_train_history.png')
    plt.close(fig)


# ==============================================================================
# 3. INFERENCE + TIMING
# ==============================================================================

def infer_single(model, sample, scheduler, DEVICE):
    """
    Runs one full reverse-diffusion inference pass.
    Returns: predicted action array, final position error, final angle error, wall time.
    """
    model.eval()

    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)

    start_time = time.perf_counter()

    with torch.no_grad():
        # Normalize proprioception
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')

        # Encode vision + proprio into a single conditioning vector
        global_cond = torch.cat([
            model.point_encoder(pcd),
            model.robot_mlp(norm_agent_pos.reshape(1, -1))
        ], dim=-1)

        # Reverse diffusion loop: start from pure noise
        noisy_action = torch.randn((1, 16, 9), device=DEVICE)
        for t in scheduler.timesteps:
            noise_pred = model.noise_pred_net(
                noisy_action, torch.tensor([t], device=DEVICE).long(), global_cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample

        # Denormalize back to real units
        pred_action = model.normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0]

    inference_time = time.perf_counter() - start_time

    # --- Metrics ---
    gt_action = sample['action'].numpy()

    # Final Distance Error (FDE) on position
    final_dist = np.linalg.norm(pred_action[-1, :3] - gt_action[-1, :3])

    # Final Rotation Error (geodesic angle between SO(3) matrices)
    gt_rot_mat = ortho6d_to_rotation_matrix(gt_action[-1, 3:][None, None, :])[0, 0]
    pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:][None, None, :])[0, 0]
    r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
    cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
    final_angle = np.arccos(cos_theta) * (180 / np.pi)

    return pred_action, final_dist, final_angle, inference_time


# ==============================================================================
# 4. PLOT: 3D TRAJECTORY (LOCAL FRAME)
# ==============================================================================

def plot_3d_local_frame(model, sample, scheduler, DEVICE, idx, n_samples=10):
    """
    3D plot showing the point cloud, GT trajectory, observation history,
    and multiple stochastic predictions — all in the local (fork-centered) frame.
    """
    print(f"🍝 Generating 3D local-frame plot for seq {idx}...")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # A. Point cloud (subsampled for readability)
    pcd = sample['obs']['point_cloud'].numpy()
    if len(pcd) > 512:
        pcd_vis = pcd[np.random.choice(len(pcd), 512, replace=False)]
    else:
        pcd_vis = pcd
    ax.scatter(pcd_vis[:, 0], pcd_vis[:, 1], pcd_vis[:, 2],
               c=STYLE['env_color'], s=4, alpha=0.15, label='Point Cloud')

    # B. Ground truth + history
    gt = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
            color=STYLE['gt_color'], linestyle='--', linewidth=STYLE['gt_lw'], label='Ground Truth')
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2],
            color=STYLE['hist_color'], marker='o', markersize=4, linewidth=STYLE['hist_lw'], label='History')

    # C. Stochastic predictions
    final_preds = []
    for i in range(n_samples):
        pred_action, _, _, _ = infer_single(model, sample, scheduler, DEVICE)
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

    # Auto-zoom around GT center
    center = np.mean(gt[:, :3], axis=0)
    r = 0.15
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

    ax.set_title(f'Seq {idx} — 3D Trajectories (Local Frame)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left', fontsize=STYLE['legend_size'] - 1)

    fig.savefig(f'02_3d_local_seq{idx}.png')
    plt.close(fig)


# ==============================================================================
# 5. PLOT: 2D SPAGHETTI (XY DISPERSION)
# ==============================================================================

def plot_multimodality_spaghetti_2d(model, sample, scheduler, DEVICE, idx, n_samples=50):
    """Top-down XY view showing prediction spread around the GT."""
    print(f"🍝 Generating 2D spaghetti plot for seq {idx}...")
    fig, ax = plt.subplots(figsize=(9, 9))

    gt = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()

    for i in range(n_samples):
        pred, _, _, _ = infer_single(model, sample, scheduler, DEVICE)
        ax.plot(pred[:, 0], pred[:, 1],
                color=STYLE['pred_color'], alpha=0.1, linewidth=STYLE['pred_lw'],
                label='Predictions' if i == 0 else '')

    ax.plot(obs[:, 0], obs[:, 1],
            color=STYLE['hist_color'], marker='o', markersize=4,
            linewidth=STYLE['hist_lw'], label='History')
    ax.plot(gt[:, 0], gt[:, 1],
            color=STYLE['gt_color'], linestyle='--', linewidth=STYLE['gt_lw'], label='Ground Truth')

    ax.set_title(f'Seq {idx} — 2D Dispersion (Local Frame)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    fig.savefig(f'02_spaghetti_2d_seq{idx}.png')
    plt.close(fig)


# ==============================================================================
# 6. PLOT: ROTATION ERROR OVER TIME
# ==============================================================================

def plot_rotation_error_over_time(gt_action, pred_action, idx):
    """
    Shows the geodesic angular error between GT and predicted orientation
    at each timestep along the trajectory.
    """
    gt_rot = ortho6d_to_rotation_matrix(
        torch.from_numpy(gt_action).unsqueeze(0)[..., 3:])[0]
    pred_rot = ortho6d_to_rotation_matrix(
        torch.from_numpy(pred_action).unsqueeze(0)[..., 3:])[0]

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

    fig.savefig(f'05_rotation_error_seq{idx}.png')
    plt.close(fig)


# ==============================================================================
# 7. GIF: DIFFUSION PROCESS
# ==============================================================================

def make_diffusion_gif(model, sample, scheduler, DEVICE, idx):
    """
    Animated GIF showing the reverse diffusion: from noise to final trajectory.
    """
    print(f"🎬 Generating diffusion GIF for seq {idx}...")
    model.eval()

    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)
    frames = []

    with torch.no_grad():
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        global_cond = torch.cat([
            model.point_encoder(pcd),
            model.robot_mlp(norm_agent_pos.reshape(1, -1))
        ], dim=-1)

        noisy_action = torch.randn((1, 16, 9), device=DEVICE)
        frames.append(model.normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0])

        for t in scheduler.timesteps:
            noise_pred = model.noise_pred_net(
                noisy_action, torch.tensor([t], device=DEVICE).long(), global_cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample
            frames.append(model.normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0])

    # Hold final frame
    for _ in range(10):
        frames.append(frames[-1])

    gt = sample['action'].numpy()
    pc_np = sample['obs']['point_cloud'].numpy()[::5]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(f):
        ax.clear()
        path = frames[f]
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=STYLE['pred_color'], linewidth=2.5, label='Diffusion')
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
                color=STYLE['gt_color'], alpha=0.35, linewidth=1.5, label='Ground Truth')
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                   s=1, c=STYLE['env_color'], alpha=0.1)

        n_real = len(frames) - 10
        step_text = f'Step {f}/{n_real}' if f < n_real else 'Final Prediction'
        ax.set_title(f'Reverse Diffusion Process\n{step_text}',
                     fontsize=STYLE['title_size'], fontweight='bold')
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_zlim(-0.15, 0.15)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save(f'03_diffusion_process_seq{idx}.gif', writer='pillow')
    plt.close(fig)


# ==============================================================================
# 8. QUANTITATIVE ANALYSIS
# ==============================================================================

def run_quantitative_analysis(model, val_dataset, scheduler, DEVICE, n_samples=20):
    model.eval()
    results = []
    times = []

    THRESHOLD_POS = 0.02  # 2 cm (applied to ADE)
    THRESHOLD_ROT = 15.0  # 15 deg

    num_eval = min(50, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, num_eval, dtype=int)

    print(f"📊 Evaluating {num_eval} sequences (success = ADE < 2 cm)...")

    for idx in tqdm(indices):
        sample = val_dataset[idx]
        gt_pos = sample['action'].numpy()[:, :3]

        seq_ades, seq_angles, seq_preds = [], [], []

        for _ in range(n_samples):
            pred, final_dist, final_angle, dt = infer_single(model, sample, scheduler, DEVICE)
            pred_pos = pred[:, :3]
            current_ade = np.mean(np.linalg.norm(pred_pos - gt_pos, axis=1))

            seq_ades.append(current_ade)
            seq_angles.append(final_angle)
            seq_preds.append(pred)
            times.append(dt)

        seq_preds = np.array(seq_preds)
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        path_variance = np.mean(np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))

        success_count = sum(
            ade < THRESHOLD_POS and rot < THRESHOLD_ROT
            for ade, rot in zip(seq_ades, seq_angles))
        success_rate = (success_count / n_samples) * 100

        results.append({
            'seq_idx':     idx,
            'ADE':         np.mean(seq_ades) * 100,     # cm
            'RotErr':      np.mean(seq_angles),          # deg
            'SuccessRate': success_rate,                  # %
            'Dispersion':  path_variance * 100,           # cm
        })

    df = pd.DataFrame(results)
    avg_time = np.mean(times)
    freq = 1.0 / avg_time if avg_time > 0 else 0

    print("\n" + "=" * 60)
    print("🚀 RESULTS (Success = ADE < 2 cm)")
    print("=" * 60)
    print(f"  Mean Success Rate    : {df['SuccessRate'].mean():.2f} %")
    print(f"  Mean Position (ADE)  : {df['ADE'].mean():.2f} cm")
    print(f"  Mean Rotation (FDE)  : {df['RotErr'].mean():.2f} deg")
    print(f"  Inference Frequency  : {freq:.1f} Hz  ({avg_time*1000:.1f} ms/sample)")
    print("=" * 60)

    return df


# ==============================================================================
# 9. PLOT: ERROR DISTRIBUTIONS
# ==============================================================================

def plot_error_distributions(df):
    """Histogram of success rates + scatter of ADE vs rotation error."""
    print("📊 Generating error distribution plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Success rate histogram
    ax1.hist(df['SuccessRate'], bins=10, color=STYLE['gt_color'],
             edgecolor='white', linewidth=0.8, alpha=0.85)
    ax1.set_title('Success Rate Distribution')
    ax1.set_xlabel('Success Rate (%)')
    ax1.set_ylabel('Count')
    ax1.grid(True, axis='y')

    # Right: ADE vs RotErr scatter, colored by success rate
    sc = ax2.scatter(df['ADE'], df['RotErr'],
                     c=df['SuccessRate'], cmap='RdYlGn',
                     edgecolors='#333333', linewidths=0.4,
                     s=60, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label('Success Rate (%)', fontsize=STYLE['label_size'])
    ax2.set_title('Position vs Rotation Error')
    ax2.set_xlabel('ADE (cm)')
    ax2.set_ylabel('Rotation Error (°)')
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig('04_metrics_distribution.png')
    plt.close(fig)


# ==============================================================================
# 10. INFERENCE SPEED BENCHMARK
# ==============================================================================

def run_inference_speed_benchmark(model, val_dataset, scheduler, DEVICE,
                                  n_warmup=5, n_runs=100):
    """
    Measures inference latency over many runs and produces a summary plot.

    Why warmup?  The first few CUDA calls are slower because the GPU lazily
    allocates memory and compiles kernels.  We throw those away so the
    histogram reflects steady-state performance — the numbers your robot
    will actually see in production.

    The plot has two panels:
      Left  — histogram of per-call latency (ms).
      Right — cumulative distribution (CDF).  Read it as:
              "X% of calls finish in less than Y ms."
    """
    model.eval()
    print(f"⏱️  Inference speed benchmark  (warmup={n_warmup}, runs={n_runs})")

    # Pick random samples so we don't benchmark on a single easy/hard case
    indices = np.random.choice(len(val_dataset), size=n_runs + n_warmup, replace=True)

    all_times_ms = []

    for i, idx in enumerate(tqdm(indices, desc="Benchmarking")):
        sample = val_dataset[idx]
        _, _, _, dt = infer_single(model, sample, scheduler, DEVICE)

        if i >= n_warmup:                       # skip warmup
            all_times_ms.append(dt * 1000.0)    # seconds -> ms

    times = np.array(all_times_ms)

    # --- Console summary ---
    mean_ms  = np.mean(times)
    std_ms   = np.std(times)
    med_ms   = np.median(times)
    p95_ms   = np.percentile(times, 95)
    p99_ms   = np.percentile(times, 99)
    min_ms   = np.min(times)
    max_ms   = np.max(times)
    freq_hz  = 1000.0 / mean_ms

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
    fig.savefig('06_inference_speed.png')
    plt.close(fig)

    return times


# ==============================================================================
# 11. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {DEVICE}")

    # 1. Paths
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, 'models', 'Last_Fork_256_points_Relative.ckpt')

    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path}, trying Last_Fork...")
        ckpt_path = os.path.join(pkg_path, 'models', 'Last_Fork.ckpt')

    # 2. Dataset
    print(f"📂 Loading dataset: {data_path}")
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42)
    print(f"✅ Validation set: {len(val_dataset)} sequences")

    # 3. Model
    print(f"🧠 Loading model: {ckpt_path}")
    model = DP3AgentRobust(action_dim=9, robot_state_dim=9).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if 'history' in checkpoint:
        plot_training_history(checkpoint['history'])

    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # 4. Scheduler (DDIM fast inference)
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        prediction_type='sample',
        clip_sample=True)
    scheduler.set_timesteps(10)

    # 5. Visual inspection
    if len(val_dataset) > 0:
        idx = 25
        plot_3d_local_frame(model, val_dataset[idx], scheduler, DEVICE, idx)
        make_diffusion_gif(model, val_dataset[idx], scheduler, DEVICE, idx)
        plot_multimodality_spaghetti_2d(model, val_dataset[idx], scheduler, DEVICE, idx)

    # 6. Full benchmark
    df = run_quantitative_analysis(model, val_dataset, scheduler, DEVICE, n_samples=10)
    plot_error_distributions(df)

    # 7. Inference speed benchmark
    run_inference_speed_benchmark(model, val_dataset, scheduler, DEVICE,
                                  n_warmup=5, n_runs=100)

    # 8. Worst-case analysis
    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    idx_worst = int(worst_seq['seq_idx'])
    print(f"\n🚨 Worst sequence (idx {idx_worst}): Success {worst_seq['SuccessRate']:.1f}%")

    worst_sample = val_dataset[idx_worst]
    plot_3d_local_frame(model, worst_sample, scheduler, DEVICE, idx_worst)

    pred_worst, _, _, _ = infer_single(model, worst_sample, scheduler, DEVICE)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)

    print("\n✅ Analysis complete. All figures saved.")


if __name__ == "__main__":
    main()