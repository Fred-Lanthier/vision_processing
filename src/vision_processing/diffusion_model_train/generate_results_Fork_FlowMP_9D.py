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

# Import Flow Matching model and data loader
from Train_Fork_FlowMP_9D import FlowMatchingAgent, Normalizer, compute_dataset_stats, custom_collate_fn
from Data_Loader_Fork_FlowMP_9D import Robot3DDataset

# ==============================================================================
# 0. UNIFIED STYLE CONFIG
# ==============================================================================

STYLE = {
    'gt_color':       '#25b34b',   # Green for ground truth
    'pred_color':     '#e74c3c',   # Red for predictions
    'hist_color':     '#3498db',   # Blue for history/past
    'env_color':      '#95a5a6',   # Gray for point cloud
    'accent_color':   '#f39c12',   # Orange for thresholds/highlights
    'bg_color':       '#ffffff',   # White background

    'title_size':     16,
    'label_size':     13,
    'tick_size':      11,
    'legend_size':    11,
    'font_family':    'sans-serif',

    'gt_lw':          2.5,
    'pred_lw':        1.5,
    'hist_lw':        2.0,
    'grid_alpha':     0.3,
    'pred_alpha':     0.15,

    'dpi':            200,
}

def apply_global_style():
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
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


# ==============================================================================
# 2. INFERENCE
# ==============================================================================

def infer_single(model, sample, DEVICE, num_steps=10, method='euler'):
    model.eval()

    obs_dict = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE),
        'agent_pos':   sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE),
    }

    start_time = time.perf_counter()

    with torch.no_grad():
        result = model.predict_action(obs_dict, num_steps=num_steps, method=method)
        pred_action = result['action_pred'].cpu().numpy()[0] # Full 9D prediction

    inference_time = time.perf_counter() - start_time

    gt_action = sample['action'].numpy() # Full 9D Ground Truth

    # Final Distance Error (Positions only: dims 0:3)
    final_dist = np.linalg.norm(pred_action[-1, :3] - gt_action[-1, :3])

    # Final Rotation Error (Rotations only: dims 3:9)
    gt_rot_mat  = ortho6d_to_rotation_matrix(gt_action[-1, 3:9][None, None, :])[0, 0]
    pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:9][None, None, :])[0, 0]
    r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
    cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
    final_angle = np.arccos(cos_theta) * (180 / np.pi)

    return pred_action, final_dist, final_angle, inference_time


def infer_with_intermediate_steps(model, sample, DEVICE, num_steps=10):
    model.eval()

    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE)
    agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        obs = {
            'point_cloud': pcd,
            'agent_pos':   model.normalizer.normalize_obs(agent_pos),
        }

        B = 1
        # Noise is generated in the 9D space
        x = torch.randn(B, model.pred_horizon, model.action_dim, device=DEVICE)
        dt = 1.0 / num_steps

        frames = [model.normalizer.unnormalize_act(x).cpu().numpy()[0]]

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(B, device=DEVICE) * t_val
            v = model.forward(obs, x, t)
            x = x + v * dt
            frames.append(model.normalizer.unnormalize_act(x).cpu().numpy()[0])

    return frames


# ==============================================================================
# 3. PLOTS
# ==============================================================================

def plot_training_history(history):
    if history is None:
        return
    print("📊 Generating training history plot...")
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'train_loss' in history:
        ax.plot(history['train_loss'], color=STYLE['pred_color'], linewidth=2, label='Train Loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], color=STYLE['hist_color'], linewidth=2, label='Validation Loss')

    ax.set_yscale('log')
    ax.set_title('Flow Matching — Training Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.legend()
    ax.grid(True, which='both')
    fig.savefig('01_fm_train_history.png')
    plt.close(fig)


def plot_3d_multimodality_with_orientation(model, sample, DEVICE, idx, n_samples=20, num_steps=10):
    print(f"🍝 Generating 3D plot for seq {idx}...")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    pcd = sample['obs']['point_cloud'].numpy()
    if len(pcd) > 512:
        pcd_vis = pcd[np.random.choice(len(pcd), 512, replace=False)]
    else:
        pcd_vis = pcd
    ax.scatter(pcd_vis[:, 0], pcd_vis[:, 1], pcd_vis[:, 2], c=STYLE['env_color'], s=4, alpha=0.15)

    gt  = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=STYLE['gt_color'], linestyle='--', linewidth=STYLE['gt_lw'], label='Ground Truth')
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], color=STYLE['hist_color'], marker='o', markersize=4, linewidth=STYLE['hist_lw'], label='History')

    final_preds = []
    for _ in range(n_samples):
        pred_action, _, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
        final_preds.append(pred_action)
        ax.plot(pred_action[:, 0], pred_action[:, 1], pred_action[:, 2], color=STYLE['pred_color'], alpha=STYLE['pred_alpha'], linewidth=1)

    scale = 0.05
    gt_last = gt[-1]
    gt_rot = ortho6d_to_rotation_matrix(gt_last[3:9][None, None, :])[0, 0]
    ax.quiver(gt_last[0], gt_last[1], gt_last[2], gt_rot[0, 0], gt_rot[1, 0], gt_rot[2, 0], color=STYLE['gt_color'], linewidth=2, length=scale)

    pred_last = final_preds[0][-1]
    pred_rot = ortho6d_to_rotation_matrix(pred_last[3:9][None, None, :])[0, 0]
    ax.quiver(pred_last[0], pred_last[1], pred_last[2], pred_rot[0, 0], pred_rot[1, 0], pred_rot[2, 0], color=STYLE['pred_color'], linewidth=2, length=scale, label='Pred Orientation')

    center = np.mean(gt[:, :3], axis=0)
    r = 0.2
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

    ax.set_title(f'Seq {idx} — 3D Trajectories (Local Frame)')
    ax.legend(loc='upper left', fontsize=STYLE['legend_size'] - 1)
    fig.savefig(f'02_fm_3d_spaghetti_seq{idx}.png')
    plt.close(fig)


def plot_multimodality_spaghetti_2d(model, sample, DEVICE, idx, n_samples=50, num_steps=10):
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

    ax.set_title(f'2D Dispersion of the Final Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    fig.savefig(f'02_fm_spaghetti_2d_seq{idx}.png')
    plt.close(fig)


def plot_rotation_error_over_time(gt_action, pred_action, idx):
    gt_rot  = ortho6d_to_rotation_matrix(gt_action[None, :, 3:9])[0]
    pred_rot = ortho6d_to_rotation_matrix(pred_action[None, :, 3:9])[0]

    errors = []
    for t in range(len(gt_action)):
        r_diff = np.dot(gt_rot[t], pred_rot[t].T)
        cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(np.arccos(cos_theta) * (180.0 / np.pi))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(errors, color=STYLE['pred_color'], marker='o', markersize=5, linewidth=STYLE['pred_lw'], label='Angular Error')
    ax.axhline(y=15, color=STYLE['accent_color'], linestyle='--', linewidth=1.5, label='Threshold (15°)')

    ax.set_title(f'Seq {idx} — Rotation Error Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Error (°)')
    ax.legend()
    ax.grid(True)
    fig.savefig(f'05_fm_rotation_error_seq{idx}.png')
    plt.close(fig)





def make_flow_matching_gif(model, sample, DEVICE, idx, num_steps=10):
    print(f"🎬 Generating FM integration GIF for seq {idx}...")
    frames = infer_with_intermediate_steps(model, sample, DEVICE, num_steps=num_steps)

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
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=STYLE['pred_color'], linewidth=2.5, label='Flow Matching')
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=STYLE['gt_color'], alpha=0.35, linewidth=1.5, label='Ground Truth')
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=1, c=STYLE['env_color'], alpha=0.1)

        n_real = len(frames) - 15
        step_text = f'ODE Step {f}/{num_steps}' if f < n_real else 'Final Prediction'
        ax.set_title(f'Flow Matching Integration\n{step_text}', fontsize=STYLE['title_size'], fontweight='bold')
        ax.set_xlim(center[0] - 0.2, center[0] + 0.2)
        ax.set_ylim(center[1] - 0.2, center[1] + 0.2)
        ax.set_zlim(center[2] - 0.2, center[2] + 0.2)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save(f'03_fm_integration_seq{idx}.gif', writer='pillow')
    plt.close(fig)


# ==============================================================================
# 4. QUANTITATIVE & SPEED
# ==============================================================================

def run_quantitative_analysis(model, val_dataset, DEVICE, n_samples=20, num_steps=10):
    model.eval()
    results = []

    THRESHOLD_POS = 0.02   # 2 cm
    THRESHOLD_ROT = 15.0   # 15 deg

    num_eval = min(len(val_dataset) // 3, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, num_eval, dtype=int)

    print(f"📊 Evaluating {num_eval} sequences...")

    for idx in tqdm(indices):
        sample = val_dataset[idx]
        gt_pos = sample['action'].numpy()[:, :3]

        seq_ades, seq_angles, seq_preds = [], [], []

        for _ in range(n_samples):
            pred, _, final_angle, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
            pred_pos = pred[:, :3]
            current_ade = np.mean(np.linalg.norm(pred_pos - gt_pos, axis=1))

            seq_ades.append(current_ade)
            seq_angles.append(final_angle)
            seq_preds.append(pred)

        seq_preds = np.array(seq_preds)
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        path_variance = np.mean(np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))

        success_count = sum(ade < THRESHOLD_POS and rot < THRESHOLD_ROT for ade, rot in zip(seq_ades, seq_angles))
        success_rate = (success_count / n_samples) * 100

        results.append({
            'seq_idx':     idx,
            'ADE':         np.mean(seq_ades) * 100,
            'RotErr':      np.mean(seq_angles),
            'SuccessRate': success_rate,
            'Dispersion':  path_variance * 100,
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("🚀 RESULTS (Success = ADE < 2 cm & Rot < 15°)")
    print("=" * 60)
    print(f"  Mean Success Rate    : {df['SuccessRate'].mean():.2f} %")
    print(f"  Mean Position (ADE)  : {df['ADE'].mean():.2f} cm")
    print(f"  Mean Rotation (FDE)  : {df['RotErr'].mean():.2f} deg")
    print("=" * 60)
    return df


def plot_error_distributions(df):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))

    sc = ax.scatter(df['ADE'], df['RotErr'], c=df['SuccessRate'], cmap='RdYlGn', edgecolors='#333333', linewidths=0.4, s=60, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=STYLE['label_size'])
    ax.set_title('Position vs Rotation Error')
    ax.set_xlabel('ADE (cm)')
    ax.set_ylabel('Rotation Error (°)')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('04_fm_metrics_distribution.png')
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.hist(times, bins=30, color=STYLE['gt_color'],
            edgecolor='white', linewidth=0.8, alpha=0.85)
    ax.axvline(mean_ms, color=STYLE['pred_color'], linestyle='-',
               linewidth=1.8, label=f'Mean  {mean_ms:.1f} ms')
    ax.axvline(med_ms, color=STYLE['hist_color'], linestyle='--',
               linewidth=1.8, label=f'Median  {med_ms:.1f} ms')
    ax.axvline(p95_ms, color=STYLE['accent_color'], linestyle=':',
               linewidth=1.8, label=f'P95  {p95_ms:.1f} ms')
    ax.set_title('Inference Latency Distribution')
    ax.set_xlim(57, 62)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, axis='y')

    fig.tight_layout()
    fig.savefig('06_fm_inference_speed.png')
    plt.close(fig)

    return times


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {DEVICE}")

    NUM_STEPS = 5 
    OBS_DIM = 9
    ACTION_DIM = 9

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, 'models', 'best_fm_model_9D_dynamics_1024.ckpt')

    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=1024, obs_horizon=2, pred_horizon=16, data_source='all')
    print(f"✅ Validation set: {len(val_dataset)} sequences")

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    STATS = checkpoint.get('stats', None)

    model = FlowMatchingAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=2,
        pred_horizon=16,
        encoder_output_dim=64,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        stats=STATS
    ).to(DEVICE)

    if 'history' in checkpoint:
        plot_training_history(checkpoint['history'])

    weights = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(weights, strict=False)
    model.eval()
    print("✅ Weights loaded.")

    sample_idx = 25
    if len(val_dataset) > sample_idx:
        sample = val_dataset[sample_idx]
        plot_3d_multimodality_with_orientation(model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        plot_multimodality_spaghetti_2d(model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        make_flow_matching_gif(model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        
        pred_action, _, _, _ = infer_single(model, sample, DEVICE, num_steps=NUM_STEPS)
        

    df = run_quantitative_analysis(model, val_dataset, DEVICE, n_samples=20, num_steps=NUM_STEPS)
    plot_error_distributions(df)
    run_inference_speed_benchmark(model, val_dataset, DEVICE, num_steps=NUM_STEPS, n_warmup=5, n_runs=100)

    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    idx_worst = int(worst_seq['seq_idx'])
    print(f"\n🚨 Worst sequence (idx {idx_worst}): Success {worst_seq['SuccessRate']:.1f}%")

    worst_sample = val_dataset[idx_worst]
    plot_3d_multimodality_with_orientation(model, worst_sample, DEVICE, idx_worst, num_steps=NUM_STEPS)
    plot_multimodality_spaghetti_2d(model, worst_sample, DEVICE, idx_worst, num_steps=NUM_STEPS)

    pred_worst, _, _, _ = infer_single(model, worst_sample, DEVICE, num_steps=NUM_STEPS)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)
    

    print("\n✅ Analysis complete. All figures saved.")

if __name__ == "__main__":
    main()