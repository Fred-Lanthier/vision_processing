"""
Diagnostic script for Flow Matching inference issues.
FIXED: Proper normalization handling throughout.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from Train_Fork_FM import DP3AgentFlowMatching
from Data_Loader_Fork import Robot3DDataset, seed_everything
import os
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json
import pandas as pd
from tqdm import tqdm
import rospkg
import time
import seaborn as sns

from improved_fm_inference_fixed import (
        compare_integrators, 
        visualize_trajectory_smoothness,
        visualize_3d_trajectory,
        infer_fm_rk4
    )

def diagnose_flow_matching(model, sample, DEVICE, num_steps=20):
    """
    Diagnostic function to understand FM inference behavior.
    All operations are done in NORMALIZED space (matching training).
    """
    model.eval()
    
    # ===== Load raw data =====
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)  # (1, 1024, 3) - NOT normalized (same as training)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)  # (1, obs_horizon, 9)
    raw_gt_action = sample['action'].unsqueeze(0).to(DEVICE)  # (1, pred_horizon, 9)
    
    with torch.no_grad():
        # ===== NORMALIZE (critical!) =====
        # Agent pos: normalize for conditioning
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        
        # Ground truth action: normalize to get x_1 (target in normalized space)
        x_1 = model.normalizer.normalize(raw_gt_action, 'action')  # (1, 16, 9) NORMALIZED
        
        # ===== Build conditioning (same as training) =====
        point_features = model.point_encoder(pcd)  # (1, 64)
        robot_features = model.robot_mlp(norm_agent_pos.reshape(1, -1))  # (1, 64)
        global_cond = torch.cat([point_features, robot_features], dim=-1)  # (1, 128)
        
        # ===== Sample noise x_0 (in normalized space) =====
        x_0 = torch.randn_like(x_1)  # (1, 16, 9) - noise, roughly N(0,1)
        
        # ===== TEST 1: Velocity prediction at known interpolation points =====
        print("\n" + "="*60)
        print("TEST 1: Velocity prediction at known interpolation points")
        print("="*60)
        print("(All values in NORMALIZED space)")
        
        # True velocity (what the network should predict)
        true_velocity = x_1 - x_0  # (1, 16, 9)
        print(f"True velocity magnitude: {true_velocity.norm().item():.4f}")
        
        for t_val in [0.0, 0.25, 0.5, 0.75, 0.99]:
            # Create x_t EXACTLY as in training: x_t = t * x_1 + (1-t) * x_0
            t = torch.tensor([t_val], device=DEVICE)
            t_expanded = t.view(1, 1, 1)
            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0  # Exact interpolation
            
            # Get network prediction
            v_pred = model.velocity_net(x_t, t, global_cond)
            
            # Compare predicted vs true velocity
            error = (v_pred - true_velocity).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                v_pred.flatten(), true_velocity.flatten(), dim=0
            ).item()
            
            print(f"  t={t_val:.2f}: v_pred_mag={v_pred.norm().item():.3f}, "
                  f"error={error:.4f}, cos_sim={cos_sim:.4f}")
        
        # ===== TEST 2: ODE Integration with drift tracking =====
        print("\n" + "="*60)
        print("TEST 2: ODE Integration with drift tracking")
        print("="*60)
        
        x = x_0.clone()  # Start from same noise
        dt = 1.0 / num_steps
        
        drift_history = []
        velocity_mag_history = []
        
        # For visualization (in REAL space, not normalized)
        position_history_real = []
        ideal_position_history_real = []
        
        for i in range(num_steps):
            t_val = i * dt
            t = torch.tensor([t_val], device=DEVICE)
            t_expanded = t.view(1, 1, 1)
            
            # Where SHOULD x be? (on the linear path in normalized space)
            x_ideal = t_expanded * x_1 + (1 - t_expanded) * x_0
            
            # Compute drift in normalized space
            drift = (x - x_ideal).norm().item()
            drift_history.append(drift)
            
            # Store positions in REAL space for visualization
            x_real = model.normalizer.unnormalize(x, 'action')
            x_ideal_real = model.normalizer.unnormalize(x_ideal, 'action')
            position_history_real.append(x_real[0, 0, :3].cpu().numpy().copy())
            ideal_position_history_real.append(x_ideal_real[0, 0, :3].cpu().numpy().copy())
            
            # Predict velocity (in normalized space)
            v = model.velocity_net(x, t, global_cond)
            velocity_mag_history.append(v.norm().item())
            
            # Euler step (in normalized space)
            x = x + v * dt
        
        # Final state comparison
        final_error_normalized = (x - x_1).norm().item()
        
        # Unnormalize for real-space error
        x_final_real = model.normalizer.unnormalize(x, 'action')
        x_1_real = model.normalizer.unnormalize(x_1, 'action')
        final_error_real = (x_final_real - x_1_real).norm().item()
        final_pos_error_cm = (x_final_real[0, -1, :3] - x_1_real[0, -1, :3]).norm().item() * 100
        
        print(f"  Final error (normalized space): {final_error_normalized:.4f}")
        print(f"  Final error (real space): {final_error_real:.4f}")
        print(f"  Final position error: {final_pos_error_cm:.2f} cm")
        print(f"  Max drift during integration: {max(drift_history):.4f}")
        print(f"  Velocity magnitude range: [{min(velocity_mag_history):.3f}, {max(velocity_mag_history):.3f}]")
        print(f"  True velocity magnitude: {true_velocity.norm().item():.3f}")
        
        # ===== TEST 3: Velocity oscillation check =====
        print("\n" + "="*60)
        print("TEST 3: Velocity oscillation check")
        print("="*60)
        
        x = x_0.clone()
        velocities = []
        
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=DEVICE)
            v = model.velocity_net(x, t, global_cond)
            velocities.append(v.cpu().numpy())
            x = x + v * dt
        
        velocities = np.array(velocities)  # (num_steps, 1, 16, 9)
        
        # Check sign changes in velocity components
        v_first = velocities[:, 0, 0, 0]  # First dim of first waypoint
        sign_changes = np.sum(np.diff(np.sign(v_first)) != 0)
        print(f"  Sign changes in v[0,0,0]: {sign_changes} out of {num_steps-1} steps")
        
        # Check velocity consistency
        v_diffs = np.diff(velocities, axis=0)
        v_diff_norms = np.linalg.norm(v_diffs.reshape(num_steps-1, -1), axis=1)
        print(f"  Velocity change per step: mean={v_diff_norms.mean():.4f}, max={v_diff_norms.max():.4f}")
        
        # ===== VISUALIZATION =====
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Drift over time (normalized space)
        axes[0, 0].plot(drift_history, 'b-o', markersize=3)
        axes[0, 0].set_xlabel('ODE Step')
        axes[0, 0].set_ylabel('Drift from ideal path (normalized)')
        axes[0, 0].set_title('State drift during integration')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Velocity magnitude over time
        axes[0, 1].plot(velocity_mag_history, 'r-o', markersize=3, label='Predicted')
        axes[0, 1].axhline(y=true_velocity.norm().item(), color='g', 
                           linestyle='--', label='True velocity mag')
        axes[0, 1].set_xlabel('ODE Step')
        axes[0, 1].set_ylabel('Velocity magnitude (normalized)')
        axes[0, 1].set_title('Velocity magnitude over integration')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Single velocity component
        axes[1, 0].plot(v_first, 'g-o', markersize=3)
        axes[1, 0].axhline(y=true_velocity[0, 0, 0].cpu().item(), color='r',
                           linestyle='--', label='True v[0,0,0]')
        axes[1, 0].set_xlabel('ODE Step')
        axes[1, 0].set_ylabel('Velocity[0,0,0] (normalized)')
        axes[1, 0].set_title('Single velocity component (check for oscillation)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Position trajectory in REAL space (meters)
        position_history_real = np.array(position_history_real)
        ideal_position_history_real = np.array(ideal_position_history_real)
        
        axes[1, 1].plot(position_history_real[:, 0], position_history_real[:, 1], 
                        'b-o', markersize=3, label='ODE path')
        axes[1, 1].plot(ideal_position_history_real[:, 0], ideal_position_history_real[:, 1], 
                        'g--', alpha=0.7, linewidth=2, label='Ideal linear path')
        axes[1, 1].scatter(position_history_real[0, 0], position_history_real[0, 1], 
                          c='green', s=100, marker='s', label='Start', zorder=5)
        axes[1, 1].scatter(position_history_real[-1, 0], position_history_real[-1, 1], 
                          c='red', s=100, marker='*', label='End', zorder=5)
        
        # Add ground truth end point
        gt_final_pos = x_1_real[0, 0, :3].cpu().numpy()
        axes[1, 1].scatter(gt_final_pos[0], gt_final_pos[1], 
                          c='purple', s=100, marker='X', label='Target', zorder=5)
        
        axes[1, 1].set_xlabel('X (meters)')
        axes[1, 1].set_ylabel('Y (meters)')
        axes[1, 1].set_title('Position trajectory - First waypoint (REAL space)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plt.savefig('fm_diagnostic.png', dpi=150)
        plt.close()
        
        print("\n✅ Diagnostic plots saved to 'fm_diagnostic.png'")
        
        return {
            'drift_history': drift_history,
            'velocity_mag_history': velocity_mag_history,
            'final_error_normalized': final_error_normalized,
            'final_error_real': final_error_real,
            'final_pos_error_cm': final_pos_error_cm,
            'sign_changes': sign_changes,
            'true_velocity_mag': true_velocity.norm().item()
        }


if __name__ == "__main__":
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

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
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
    sample = val_dataset[25]
    results = diagnose_flow_matching(model, sample, DEVICE, num_steps=20)
    print(results)

    # Compare integration methods
    results = compare_integrators(model, sample, DEVICE)
    
    # Visualize smoothness
    visualize_trajectory_smoothness(model, sample, DEVICE)
    
    # 3D visualization
    visualize_3d_trajectory(model, sample, DEVICE)
    
    # Single inference with RK4
    pred = infer_fm_rk4(model, sample, DEVICE, num_steps=20)
    print(f"Prediction shape: {pred.shape}")  # Should be (16, 9)