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
import seaborn as sns
import time

# Import Flow Matching model and data loader
from Train_Fork_FM import FlowMatchingAgent, Normalizer, compute_dataset_stats, custom_collate_fn
from Data_Loader_Fork_FM import Robot3DDataset

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
    """Convert 6D representation (B, T, 6) to rotation matrix (B, T, 3, 3)"""
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
    """
    Run a single FM inference using ODE integration.
    
    Unlike diffusion which uses a scheduler (DDIM/DDPM),
    flow matching integrates the velocity field from t=0 to t=1.
    """
    model.eval()
    
    # FM data loader returns nested dict: sample['obs']['point_cloud'], etc.
    obs_dict = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE),
        'agent_pos': sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)
    }
    with torch.no_grad():
        result = model.predict_action(obs_dict, num_steps=num_steps, method=method)
        pred_action = result['action_pred'].cpu().numpy()[0]  # (pred_horizon, 9)
    
    gt_action = sample['action'].numpy()  # (pred_horizon, 9)
    
    # Final position error
    final_dist = np.linalg.norm(pred_action[-1, :3] - gt_action[-1, :3])
    
    # Final rotation error
    gt_rot_mat = ortho6d_to_rotation_matrix(gt_action[-1, 3:][None, None, :])[0, 0]
    pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:][None, None, :])[0, 0]
    r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
    cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
    final_angle = np.arccos(cos_theta) * (180 / np.pi)
    
    return pred_action, final_dist, final_angle


def infer_with_intermediate_steps(model, sample, DEVICE, num_steps=10):
    """
    Run FM inference and capture intermediate ODE states for visualization.
    Returns list of trajectories at each integration step.
    """
    model.eval()
    device = DEVICE
    
    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(device)
    agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Normalize obs
        obs = {
            'point_cloud': pcd,
            'agent_pos': model.normalizer.normalize(agent_pos)
        }
        
        B = 1
        x = torch.randn(B, model.pred_horizon, model.action_dim, device=device)
        dt = 1.0 / num_steps
        
        frames = []
        # Save initial noise state
        frames.append(model.normalizer.unnormalize(x).cpu().numpy()[0])
        
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(B, device=device) * t_val
            
            v = model.forward(obs, x, t)
            x = x + v * dt
            
            # Save intermediate state (unnormalized)
            frames.append(model.normalizer.unnormalize(x).cpu().numpy()[0])
    
    return frames

# ==============================================================================
# 3. VISUALIZATIONS
# ==============================================================================

def plot_training_history(history):
    if history is None:
        print("⚠️ No history found in checkpoint.")
        return
    
    print("📊 Generating training history plot...")
    plt.figure(figsize=(10, 5))
    
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    
    plt.yscale('log')
    plt.title("Flow Matching Training Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('01_fm_train_history.svg', format='svg')
    plt.close()


def plot_3d_multimodality_with_orientation(model, sample, DEVICE, idx, n_samples=20, num_steps=10):
    print(f"🍝 Generating 3D spaghetti plot for Seq {idx}...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Point cloud
    pcd = sample['obs']['point_cloud'].numpy()
    if len(pcd) > 512:
        indices = np.random.choice(len(pcd), 512, replace=False)
        pcd_vis = pcd[indices]
    else:
        pcd_vis = pcd
    ax.scatter(pcd_vis[:,0], pcd_vis[:,1], pcd_vis[:,2], 
               c=pcd_vis[:,2], cmap='Greys', s=5, alpha=0.2, label='Env (PCD)')
    
    # Ground truth
    gt = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g--', linewidth=3, label='Ground Truth')
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], 'b-o', linewidth=2, label='History')
    
    # Multiple inferences
    final_preds = []
    for _ in range(n_samples):
        pred_action, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
        final_preds.append(pred_action)
        ax.plot(pred_action[:, 0], pred_action[:, 1], pred_action[:, 2], 
                color='red', alpha=0.15, linewidth=1)
    
    # Orientation at final point
    last_pred = final_preds[0][-1]
    gt_last = gt[-1]
    scale = 0.05
    
    gt_rot = ortho6d_to_rotation_matrix(gt_last[3:][None, None, :])[0, 0]
    origin = gt_last[:3]
    ax.quiver(origin[0], origin[1], origin[2], 
              gt_rot[0,0], gt_rot[1,0], gt_rot[2,0], color='green', lw=2, length=scale)
    
    pred_rot = ortho6d_to_rotation_matrix(last_pred[3:][None, None, :])[0, 0]
    origin_p = last_pred[:3]
    ax.quiver(origin_p[0], origin_p[1], origin_p[2], 
              pred_rot[0,0], pred_rot[1,0], pred_rot[2,0], 
              color='red', lw=2, length=scale, label='Orient. Pred')
    
    ax.set_title(f"Seq {idx} : 3D Analysis (Red=Pred, Green=GT)")
    center = np.mean(gt, axis=0)[:3]
    radius = 0.2
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    plt.legend()
    plt.savefig(f'02_fm_3d_spaghetti_seq{idx}.png', dpi=200)
    plt.close()


def plot_multimodality_spaghetti_2d(model, sample, DEVICE, idx, n_samples=50, num_steps=10):
    print(f"🍝 Generating 2D spaghetti plot for Seq {idx}...")
    plt.figure(figsize=(10, 10))
    gt = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()
    
    for i in range(n_samples):
        pred, _, _ = infer_single(model, sample, DEVICE, num_steps=num_steps)
        plt.plot(pred[:, 0], pred[:, 1], color='red', alpha=0.1, linewidth=1.5, 
                 label='Predictions' if i == 0 else "")
    
    plt.plot(obs[:,0], obs[:,1], 'b-o', label='History', linewidth=2.5)
    plt.plot(gt[:, 0], gt[:, 1], 'g--', label='Ground Truth', linewidth=3)
    plt.title(f"2D Dispersion - Seq {idx} (N={n_samples})")
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig(f'02_fm_spaghetti_2d_seq{idx}.png', dpi=300)
    plt.close()


def plot_rotation_error_over_time(gt_action, pred_action, idx):
    gt_rot = ortho6d_to_rotation_matrix(gt_action[None, :, 3:])[0]
    pred_rot = ortho6d_to_rotation_matrix(pred_action[None, :, 3:])[0]
    
    errors = []
    for t in range(len(gt_action)):
        r_diff = np.dot(gt_rot[t], pred_rot[t].T)
        val = (np.trace(r_diff) - 1.0) / 2.0
        cos_theta = np.clip(val, -1.0, 1.0)
        err_deg = np.arccos(cos_theta) * (180.0 / np.pi)
        errors.append(err_deg)
    
    plt.figure(figsize=(8, 4))
    plt.plot(errors, 'r-o', label='Angular Error')
    plt.axhline(y=15, color='orange', linestyle='--', label='Threshold (15°)')
    plt.xlabel("Timestep")
    plt.ylabel("Error (°)")
    plt.title(f"Angular Drift - Seq {idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'05_fm_rotation_error_seq{idx}.png')
    plt.close()


def make_flow_matching_gif(model, sample, DEVICE, idx, num_steps=10):
    """
    Visualize the ODE integration process.
    Unlike diffusion (which denoises from noise), flow matching
    integrates a velocity field: each frame is one Euler step.
    """
    print(f"🎬 Generating FM integration GIF for Seq {idx}...")
    
    frames = infer_with_intermediate_steps(model, sample, DEVICE, num_steps=num_steps)
    
    # Add pause at the end
    for _ in range(15):
        frames.append(frames[-1])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    gt = sample['action'].numpy()
    center = np.mean(gt[:, :3], axis=0)
    
    def update(f):
        ax.clear()
        path = frames[f]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=3, label='Flow Matching')
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color='green', alpha=0.3, linewidth=1, label='Target')
        
        pc_np = sample['obs']['point_cloud'].numpy()[::5]
        ax.scatter(pc_np[:,0], pc_np[:,1], pc_np[:,2], s=1, c='gray', alpha=0.1)
        
        if f < (len(frames) - 15):
            step_text = f"ODE Step: {f}/{num_steps}"
        else:
            step_text = "Final Prediction"
        ax.set_title(f"Flow Matching Integration\n{step_text}")
        ax.set_xlim(center[0]-0.2, center[0]+0.2)
        ax.set_ylim(center[1]-0.2, center[1]+0.2)
        ax.set_zlim(center[2]-0.2, center[2]+0.2)
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save(f'03_fm_integration_seq{idx}.gif', writer='pillow')
    plt.close()

# ==============================================================================
# 4. QUANTITATIVE ANALYSIS
# ==============================================================================

def run_quantitative_analysis(model, val_dataset, DEVICE, n_samples=20, num_steps=10):
    model.eval()
    results = []
    
    THRESHOLD_POS = 0.02   # 2 cm
    THRESHOLD_ROT = 15.0   # 15 degrees
    
    print(f"📊 Quantitative analysis ({n_samples} samples/seq)...")
    num_eval_sequences = min(30, len(val_dataset))
    indices = np.linspace(0, len(val_dataset)-1, num_eval_sequences, dtype=int)
    
    infer_times = []
    for idx in tqdm(indices):
        sample = val_dataset[idx]
        
        seq_preds = []
        seq_final_angles = []
        seq_final_dists = []
        
        for _ in range(n_samples):
            start = time.time()
            pred, final_dist, final_angle = infer_single(
                model, sample, DEVICE, num_steps=num_steps
            )
            infer_times.append(time.time() - start)
            seq_preds.append(pred)
            seq_final_dists.append(final_dist)
            seq_final_angles.append(final_angle)
        
        seq_preds = np.array(seq_preds)  # (N, 16, 9)
        
        mean_fde = np.mean(seq_final_dists)
        mean_rot = np.mean(seq_final_angles)
        
        success_count = sum([
            (d < THRESHOLD_POS and a < THRESHOLD_ROT) 
            for d, a in zip(seq_final_dists, seq_final_angles)
        ])
        success_rate = (success_count / n_samples) * 100
        
        gt_action = sample['action'].numpy()
        gt_pos_all = gt_action[:, :3]
        pred_pos_all = seq_preds[:, :, :3]
        ade_dist = np.mean(np.linalg.norm(pred_pos_all - gt_pos_all, axis=2), axis=1)
        mean_ade = np.mean(ade_dist)
        
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        path_variance = np.mean(np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))
        
        results.append({
            'seq_idx': idx,
            'ADE': mean_ade * 100,
            'FDE': mean_fde * 100,
            'RotErr': mean_rot,
            'SuccessRate': success_rate,
            'Dispersion': path_variance * 100,
            'InferTime': np.mean(infer_times),
            'InferTimeStd': np.std(infer_times)
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY (Validation Subset)")
    print("="*50)
    print(f"Success Rate      : {df['SuccessRate'].mean():.2f} %")
    print(f"Position Error ADE: {df['ADE'].mean():.2f} cm")
    print(f"Position Error FDE: {df['FDE'].mean():.2f} cm")
    print(f"Rotation Error    : {df['RotErr'].mean():.2f} deg")
    print(f"Dispersion        : {df['Dispersion'].mean():.2f} cm")
    print(f"Infer Time        : {df['InferTime'].mean():.4f} +- {df['InferTimeStd'].mean():.4f} s")
    print("="*50)
    
    return df


def plot_error_distributions(df):
    print("📊 Generating error distribution plots...")
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df['SuccessRate'], bins=10, kde=False, ax=ax1, color="green")
    ax1.set_title("Success Rate Distribution per Sequence")
    ax1.set_xlabel("Success (%)")
    
    sns.scatterplot(data=df, x='ADE', y='RotErr', ax=ax2, 
                    hue='SuccessRate', palette='viridis', alpha=0.8)
    ax2.set_title("Position Error vs Rotation Error")
    ax2.set_xlabel("Position Error (cm)")
    ax2.set_ylabel("Rotation Error (deg)")
    
    plt.tight_layout()
    plt.savefig('04_fm_metrics_distribution.png', dpi=300)
    plt.close()

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {DEVICE}")
    
    NUM_STEPS = 10  # ODE integration steps
    
    # Paths
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, "models", "last_fm_model_high_dim_CFM_relative.ckpt")
    
    # Dataset (uses nested dict format)
    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=256, obs_horizon=2, pred_horizon=16
    )
    print(f"✅ Validation Set: {len(val_dataset)} sequences")
    
    # Build model with same architecture as training
    model = FlowMatchingAgent(
        action_dim=9,
        obs_horizon=2,
        pred_horizon=16,
        encoder_output_dim=64,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8
    ).to(DEVICE)
    
    # Load checkpoint
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    if 'history' in checkpoint:
        plot_training_history(checkpoint['history'])
    
    weights = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(weights, strict=False)
    model.eval()
    print("✅ Weights loaded.")
    
    # Visualizations on a specific sequence
    sample_idx = 25
    if len(val_dataset) > sample_idx:
        sample = val_dataset[sample_idx]
        
        plot_3d_multimodality_with_orientation(
            model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS
        )
        plot_multimodality_spaghetti_2d(
            model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS
        )
        make_flow_matching_gif(
            model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS
        )
    
    # Global quantitative analysis
    df = run_quantitative_analysis(
        model, val_dataset, DEVICE, n_samples=20, num_steps=NUM_STEPS
    )
    plot_error_distributions(df)
    
    # Worst sequence analysis
    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    print(f"\n🚨 WORST SEQUENCE (Idx {int(worst_seq['seq_idx'])}):")
    print(f"   Success: {worst_seq['SuccessRate']:.1f}% | ADE: {worst_seq['ADE']:.2f}cm | Rot: {worst_seq['RotErr']:.1f}°")
    
    idx_worst = int(worst_seq['seq_idx'])
    worst_sample = val_dataset[idx_worst]
    
    print(f"📸 Generating visuals for outlier Seq {idx_worst}...")
    plot_3d_multimodality_with_orientation(
        model, worst_sample, DEVICE, idx_worst, num_steps=NUM_STEPS
    )
    
    pred_worst, _, _ = infer_single(model, worst_sample, DEVICE, num_steps=NUM_STEPS)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)
    
    print("\n✅ Analysis complete. Check generated PNG/SVG/GIF files.")

if __name__ == "__main__":
    main()