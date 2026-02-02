"""
generate_results_Fork_FM.py - Visualization & Validation for Flow Matching Model

This script validates the Flow Matching model by:
1. Running inference using ODE integration (Euler method)
2. Comparing predicted trajectories vs ground truth
3. Generating visualizations (3D spaghetti, 2D plots, GIF of ODE process)
4. Computing quantitative metrics (ADE, FDE, rotation error, success rate)

Key difference from diffusion:
- Diffusion: start from noise, iteratively DENOISE backwards (t: T -> 0)
- Flow Matching: start from noise, integrate ODE FORWARD (t: 0 -> 1)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json
import pandas as pd
from tqdm import tqdm
import rospkg
import time
import seaborn as sns

# Import Flow Matching model and dataset
from Train_Fork_FM import DP3AgentFlowMatching, Normalizer
from Data_Loader_Fork import Robot3DDataset

"""
Diagnostic script for Flow Matching inference issues.
This will help identify why trajectories are zigzag.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def diagnose_flow_matching(model, sample, DEVICE, num_steps=20):
    """
    Diagnostic function to understand FM inference behavior.
    """
    model.eval()
    
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    gt_action = sample['action'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Normalize
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos)
        x_1 = model.normalizer.normalize(gt_action)  # Target (normalized)
        
        # Build condition
        point_features = model.point_encoder(pcd)
        robot_features = model.robot_mlp(norm_agent_pos.reshape(1, -1))
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        # Start from noise
        x_0 = torch.randn_like(x_1)
        
        # ===== TEST 1: Check velocity at known interpolation points =====
        print("\n" + "="*60)
        print("TEST 1: Velocity prediction at known interpolation points")
        print("="*60)
        
        true_velocity = x_1 - x_0  # True target velocity
        print(f"True velocity magnitude: {true_velocity.norm().item():.4f}")
        
        for t_val in [0.0, 0.25, 0.5, 0.75, 0.99]:
            # Create x_t exactly as in training
            t = torch.tensor([t_val], device=DEVICE)
            t_expanded = t.view(1, 1, 1)
            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0  # Exact interpolation
            
            # Get network prediction
            v_pred = model.velocity_net(x_t, t, global_cond)
            
            # Compare
            error = (v_pred - true_velocity).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                v_pred.flatten(), true_velocity.flatten(), dim=0
            ).item()
            
            print(f"  t={t_val:.2f}: pred_mag={v_pred.norm().item():.3f}, "
                  f"error={error:.4f}, cos_sim={cos_sim:.4f}")
        
        # ===== TEST 2: Run actual inference and track drift =====
        print("\n" + "="*60)
        print("TEST 2: ODE Integration with drift tracking")
        print("="*60)
        
        x = x_0.clone()
        dt = 1.0 / num_steps
        
        drift_history = []
        velocity_history = []
        position_history = []
        
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=DEVICE)
            t_expanded = t.view(1, 1, 1)
            
            # Where SHOULD x be? (on the linear path)
            x_ideal = t_expanded * x_1 + (1 - t_expanded) * x_0
            
            # Where IS x? (from ODE integration)
            drift = (x - x_ideal).norm().item()
            drift_history.append(drift)
            
            # Predict velocity
            v = model.velocity_net(x, t, global_cond)
            velocity_history.append(v.norm().item())
            
            # Store current position (only first waypoint, position dims)
            pos = x[0, 0, :3].cpu().numpy()
            position_history.append(pos.copy())
            
            # Euler step
            x = x + v * dt
        
        # Final state
        final_error = (x - x_1).norm().item()
        print(f"  Final error (normalized): {final_error:.4f}")
        print(f"  Max drift during integration: {max(drift_history):.4f}")
        print(f"  Velocity magnitude range: [{min(velocity_history):.3f}, {max(velocity_history):.3f}]")
        
        # ===== TEST 3: Check for velocity oscillations =====
        print("\n" + "="*60)
        print("TEST 3: Velocity oscillation check")
        print("="*60)
        
        # Run inference and collect velocity vectors
        x = x_0.clone()
        velocities = []
        
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=DEVICE)
            v = model.velocity_net(x, t, global_cond)
            velocities.append(v.cpu().numpy())
            x = x + v * dt
        
        velocities = np.array(velocities)  # (num_steps, 1, 16, 9)
        
        # Check direction changes (sign flips) in first waypoint, first position dim
        v_first = velocities[:, 0, 0, 0]  # First dimension of first waypoint
        sign_changes = np.sum(np.diff(np.sign(v_first)) != 0)
        print(f"  Sign changes in v[0,0,0]: {sign_changes} out of {num_steps-1} steps")
        
        # Check overall velocity consistency
        v_diffs = np.diff(velocities, axis=0)
        v_diff_norms = np.linalg.norm(v_diffs.reshape(num_steps-1, -1), axis=1)
        print(f"  Velocity change per step: mean={v_diff_norms.mean():.4f}, max={v_diff_norms.max():.4f}")
        
        # ===== VISUALIZATION =====
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Drift over time
        axes[0, 0].plot(drift_history, 'b-o', markersize=3)
        axes[0, 0].set_xlabel('ODE Step')
        axes[0, 0].set_ylabel('Drift from ideal path')
        axes[0, 0].set_title('State drift during integration')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Velocity magnitude over time
        axes[0, 1].plot(velocity_history, 'r-o', markersize=3)
        axes[0, 1].axhline(y=true_velocity.norm().item(), color='g', 
                           linestyle='--', label='True velocity mag')
        axes[0, 1].set_xlabel('ODE Step')
        axes[0, 1].set_ylabel('Predicted velocity magnitude')
        axes[0, 1].set_title('Velocity magnitude over integration')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: First dimension velocity over time
        axes[1, 0].plot(v_first, 'g-o', markersize=3)
        axes[1, 0].axhline(y=true_velocity[0, 0, 0].cpu().item(), color='r',
                           linestyle='--', label='True v[0,0,0]')
        axes[1, 0].set_xlabel('ODE Step')
        axes[1, 0].set_ylabel('Velocity[0,0,0]')
        axes[1, 0].set_title('Single velocity component (check for oscillation)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Position trajectory (first waypoint, XY)
        position_history = np.array(position_history)
        axes[1, 1].plot(position_history[:, 0], position_history[:, 1], 'b-o', 
                        markersize=3, label='ODE path')
        axes[1, 1].scatter(position_history[0, 0], position_history[0, 1], 
                          c='green', s=100, marker='s', label='Start')
        axes[1, 1].scatter(position_history[-1, 0], position_history[-1, 1], 
                          c='red', s=100, marker='*', label='End')
        
        # Add ideal path
        ideal_positions = []
        for i in range(num_steps):
            t_val = i * dt
            x_ideal = t_val * x_1 + (1 - t_val) * x_0
            ideal_positions.append(x_ideal[0, 0, :3].cpu().numpy())
        ideal_positions = np.array(ideal_positions)
        axes[1, 1].plot(ideal_positions[:, 0], ideal_positions[:, 1], 'g--', 
                        alpha=0.5, label='Ideal path')
        
        axes[1, 1].set_xlabel('X (normalized)')
        axes[1, 1].set_ylabel('Y (normalized)')
        axes[1, 1].set_title('Position trajectory (first waypoint)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fm_diagnostic.png', dpi=150)
        plt.close()
        
        print("\n✅ Diagnostic plots saved to 'fm_diagnostic.png'")
        
        return {
            'drift_history': drift_history,
            'velocity_history': velocity_history,
            'final_error': final_error,
            'sign_changes': sign_changes
        }


def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {DEVICE}")
    
    # Paths
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    
    # Checkpoint path (adjust as needed)
    ckpt_path = os.path.join(pkg_path, "models", "flowmatching_last_V2.ckpt")
    
    # Load validation dataset
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42)
    print(f"✅ Validation Set: {len(val_dataset)} sequences")

    # Build model
    model = DP3AgentFlowMatching(action_dim=9, robot_state_dim=9).to(DEVICE)
    
    # Load checkpoint
    if not os.path.exists(ckpt_path):
        print(f"❌ ERROR: Checkpoint not found: {ckpt_path}")
        print("   Make sure to train the Flow Matching model first!")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    # Plot training history
    if 'history' in checkpoint:
        plot_training_history(checkpoint['history'])
    
    # Load weights
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)
    print("✅ Model weights loaded.")
    
    # Print config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"📋 Model Config: {config.get('model_type', 'Unknown')}")
        NUM_STEPS = config.get('num_inference_steps', 20)
    else:
        NUM_STEPS = 20
    
    print(f"🔧 Using {NUM_STEPS} ODE steps for inference")

    # Measure inference time
    sample = val_dataset[0]
    inference_time = infer_with_timing(model, sample, DEVICE, NUM_STEPS)
    print(f"⏱️ Inference time: {inference_time:.2f} ms per trajectory")

    # === VISUALIZATIONS ===
    sample_idx = 25
    if len(val_dataset) > sample_idx:
        sample = val_dataset[sample_idx]
        
        # 3D spaghetti plot with orientation
        plot_3d_multimodality_with_orientation(model, sample, DEVICE, sample_idx, 
                                                n_samples=20, num_steps=NUM_STEPS)
        
        # 2D spaghetti plot
        plot_multimodality_spaghetti_2d(model, sample, DEVICE, sample_idx, 
                                        n_samples=50, num_steps=NUM_STEPS)
        
        # ODE integration GIF
        make_flow_matching_gif(model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        
        # Flow field visualization
        plot_flow_field_2d(model, sample, DEVICE, sample_idx, num_steps=NUM_STEPS)
        
        # Single prediction error analysis
        pred, _, _ = infer_single_fm(model, sample, DEVICE, NUM_STEPS)
        plot_position_error_over_time(sample['action'].numpy(), pred, sample_idx)
        plot_rotation_error_over_time(sample['action'].numpy(), pred, sample_idx)

    # === QUANTITATIVE ANALYSIS ===
    df = run_quantitative_analysis(model, val_dataset, DEVICE, 
                                   n_samples=20, num_steps=NUM_STEPS)
    plot_error_distributions(df)
    
    # === ODE STEPS ANALYSIS ===
    if len(val_dataset) > 0:
        compare_num_steps(model, val_dataset[0], DEVICE)
    
    # === OUTLIER ANALYSIS ===
    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    print(f"\n🚨 WORST SEQUENCE (Idx {int(worst_seq['seq_idx'])}):")
    print(f"   Success: {worst_seq['SuccessRate']:.1f}% | "
          f"FDE: {worst_seq['FDE']:.2f}cm | Rot: {worst_seq['RotErr']:.1f}°")
    
    # Visualize worst sequence
    idx_worst = int(worst_seq['seq_idx'])
    worst_sample = val_dataset[idx_worst]
    print(f"📸 Generating visuals for outlier Seq {idx_worst}...")
    plot_3d_multimodality_with_orientation(model, worst_sample, DEVICE, idx_worst,
                                           n_samples=20, num_steps=NUM_STEPS)
    
    pred_worst, _, _ = infer_single_fm(model, worst_sample, DEVICE, NUM_STEPS)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)
    plot_position_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)
    
    print("\n✅ Analysis complete. Check the generated PNG/GIF/SVG files.")


if __name__ == "__main__":
    main()