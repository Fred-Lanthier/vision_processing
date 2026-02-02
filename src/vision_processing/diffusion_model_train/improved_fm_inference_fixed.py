"""
Improved Flow Matching Inference with proper normalization.
FIXED: Explicit normalization/unnormalization at correct points.

Key insight: The network operates entirely in NORMALIZED space.
- Input x_t: normalized
- Output velocity v: in normalized space
- Conditioning agent_pos: normalized
- Point cloud: NOT normalized (same as training)
"""

import torch
import numpy as np
import time


def infer_fm_euler(model, sample, DEVICE, num_steps=20):
    """
    Euler method with proper normalization.
    """
    model.eval()
    
    # Load data
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)  # (1, 1024, 3) - NOT normalized
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)  # (1, obs_horizon, 9)
    
    with torch.no_grad():
        # NORMALIZE agent position for conditioning
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        
        # Build conditioning
        point_features = model.point_encoder(pcd)
        robot_features = model.robot_mlp(norm_agent_pos.reshape(1, -1))
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        # Start from noise (in normalized space - this is correct since 
        # during training x_0 ~ N(0,I) which matches normalized data range roughly)
        x = torch.randn((1, 16, 9), device=DEVICE)
        
        dt = 1.0 / num_steps
        
        # ODE integration in NORMALIZED space
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=DEVICE)
            v = model.velocity_net(x, t, global_cond)  # v is in normalized space
            x = x + v * dt  # x stays in normalized space
        
        # UNNORMALIZE final prediction to get real-world coordinates
        pred_action = model.normalizer.unnormalize(x, 'action')
    
    return pred_action.cpu().numpy()[0]


def infer_fm_rk4(model, sample, DEVICE, num_steps=20):
    """
    RK4 integration with proper normalization - much more stable than Euler.
    
    RK4 formula:
    k1 = f(t, x)
    k2 = f(t + dt/2, x + dt/2 * k1)
    k3 = f(t + dt/2, x + dt/2 * k2)
    k4 = f(t + dt, x + dt * k3)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    model.eval()
    
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Normalize conditioning
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        
        # Build conditioning
        point_features = model.point_encoder(pcd)
        robot_features = model.robot_mlp(norm_agent_pos.reshape(1, -1))
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        # Start from noise (normalized space)
        x = torch.randn((1, 16, 9), device=DEVICE)
        dt = 1.0 / num_steps
        
        # RK4 integration in normalized space
        for i in range(num_steps):
            t = i * dt
            
            t1 = torch.tensor([t], device=DEVICE)
            k1 = model.velocity_net(x, t1, global_cond)
            
            t2 = torch.tensor([t + dt/2], device=DEVICE)
            k2 = model.velocity_net(x + dt/2 * k1, t2, global_cond)
            
            k3 = model.velocity_net(x + dt/2 * k2, t2, global_cond)
            
            t4 = torch.tensor([t + dt], device=DEVICE)
            k4 = model.velocity_net(x + dt * k3, t4, global_cond)
            
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Unnormalize to real space
        pred_action = model.normalizer.unnormalize(x, 'action')
    
    return pred_action.cpu().numpy()[0]


def infer_fm_midpoint(model, sample, DEVICE, num_steps=20):
    """
    Midpoint method (2nd order RK) with proper normalization.
    Good balance of speed and accuracy.
    """
    model.eval()
    
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        
        point_features = model.point_encoder(pcd)
        robot_features = model.robot_mlp(norm_agent_pos.reshape(1, -1))
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        x = torch.randn((1, 16, 9), device=DEVICE)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i * dt
            
            t1 = torch.tensor([t], device=DEVICE)
            k1 = model.velocity_net(x, t1, global_cond)
            
            t2 = torch.tensor([t + dt/2], device=DEVICE)
            k2 = model.velocity_net(x + dt/2 * k1, t2, global_cond)
            
            x = x + dt * k2
        
        pred_action = model.normalizer.unnormalize(x, 'action')
    
    return pred_action.cpu().numpy()[0]


def infer_fm_heun(model, sample, DEVICE, num_steps=20):
    """
    Heun's method (improved Euler / trapezoidal) with proper normalization.
    Uses average of velocities at start and end of step.
    
    Formula:
    x_tilde = x + dt * f(t, x)
    x_next = x + dt/2 * (f(t, x) + f(t+dt, x_tilde))
    """
    model.eval()
    
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        
        point_features = model.point_encoder(pcd)
        robot_features = model.robot_mlp(norm_agent_pos.reshape(1, -1))
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        x = torch.randn((1, 16, 9), device=DEVICE)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i * dt
            
            t1 = torch.tensor([t], device=DEVICE)
            v1 = model.velocity_net(x, t1, global_cond)
            
            # Predictor step
            x_tilde = x + dt * v1
            
            # Corrector step
            t2 = torch.tensor([t + dt], device=DEVICE)
            v2 = model.velocity_net(x_tilde, t2, global_cond)
            
            # Average velocity
            x = x + dt/2 * (v1 + v2)
        
        pred_action = model.normalizer.unnormalize(x, 'action')
    
    return pred_action.cpu().numpy()[0]


def compare_integrators(model, sample, DEVICE):
    """
    Compare different integration methods with proper normalization.
    """
    gt = sample['action'].numpy()  # Ground truth in real space
    
    results = []
    
    methods = [
        ("Euler-20", infer_fm_euler, {"num_steps": 20}),
        ("Euler-50", infer_fm_euler, {"num_steps": 50}),
        ("Euler-100", infer_fm_euler, {"num_steps": 100}),
        ("Midpoint-20", infer_fm_midpoint, {"num_steps": 20}),
        ("Midpoint-50", infer_fm_midpoint, {"num_steps": 50}),
        ("Heun-20", infer_fm_heun, {"num_steps": 20}),
        ("Heun-50", infer_fm_heun, {"num_steps": 50}),
        ("RK4-10", infer_fm_rk4, {"num_steps": 10}),
        ("RK4-20", infer_fm_rk4, {"num_steps": 20}),
        ("RK4-50", infer_fm_rk4, {"num_steps": 50}),
    ]
    
    for name, func, kwargs in methods:
        errors = []
        times = []
        
        for _ in range(10):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            pred = func(model, sample, DEVICE, **kwargs)
            
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # FDE in cm (final position error)
            fde = np.linalg.norm(pred[-1, :3] - gt[-1, :3]) * 100
            errors.append(fde)
            times.append(elapsed * 1000)
        
        results.append({
            'method': name,
            'FDE_mean': np.mean(errors),
            'FDE_std': np.std(errors),
            'time_ms': np.mean(times)
        })
    
    # Print results
    print("\n" + "="*60)
    print("Integration Method Comparison (with proper normalization)")
    print("="*60)
    print(f"{'Method':<15} {'FDE (cm)':<20} {'Time (ms)':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['method']:<15} {r['FDE_mean']:.2f} ± {r['FDE_std']:.2f}         {r['time_ms']:.1f}")
    
    return results


def visualize_trajectory_smoothness(model, sample, DEVICE):
    """
    Visualize predicted trajectory smoothness with different integrators.
    All in REAL space (meters) for proper interpretation.
    """
    import matplotlib.pyplot as plt
    
    # Get predictions
    pred_euler_20 = infer_fm_euler(model, sample, DEVICE, num_steps=20)
    pred_euler_100 = infer_fm_euler(model, sample, DEVICE, num_steps=100)
    pred_rk4_20 = infer_fm_rk4(model, sample, DEVICE, num_steps=20)
    pred_heun_20 = infer_fm_heun(model, sample, DEVICE, num_steps=20)
    gt = sample['action'].numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Trajectory Smoothness Comparison (Real Space - meters)', fontsize=14)
    
    # Row 1: XYZ positions over prediction horizon
    predictions = [
        ('Ground Truth', gt, 'g-'),
        ('Euler-20', pred_euler_20, 'r--'),
        ('Euler-100', pred_euler_100, 'b:'),
        ('RK4-20', pred_rk4_20, 'm-.'),
        ('Heun-20', pred_heun_20, 'c--'),
    ]
    
    for dim, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[0, dim]
        timesteps = np.arange(16)
        
        for name, pred, style in predictions:
            lw = 2.5 if name == 'Ground Truth' else 1.5
            ax.plot(timesteps, pred[:, dim], style, linewidth=lw, label=name, alpha=0.8)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{label} position (m)')
        ax.set_title(f'{label} Position')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Acceleration (smoothness metric) for each method
    methods_to_analyze = [
        ('Euler-20', pred_euler_20),
        ('RK4-20', pred_rk4_20),
        ('Ground Truth', gt),
    ]
    
    for i, (name, pred) in enumerate(methods_to_analyze):
        ax = axes[1, i]
        
        # Compute velocity (first derivative)
        velocity = np.diff(pred[:, :3], axis=0)
        
        # Compute acceleration (second derivative) 
        acceleration = np.diff(velocity, axis=0)
        accel_magnitude = np.linalg.norm(acceleration, axis=1)
        
        ax.plot(accel_magnitude, 'o-', markersize=4)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Acceleration magnitude (m/step²)')
        ax.set_title(f'{name}\nMean accel: {np.mean(accel_magnitude):.4f}')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at GT mean for reference
        if name != 'Ground Truth':
            gt_vel = np.diff(gt[:, :3], axis=0)
            gt_accel = np.diff(gt_vel, axis=0)
            gt_accel_mag = np.mean(np.linalg.norm(gt_accel, axis=1))
            ax.axhline(y=gt_accel_mag, color='g', linestyle='--', 
                      alpha=0.5, label=f'GT mean: {gt_accel_mag:.4f}')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fm_smoothness_comparison.png', dpi=150)
    plt.close()
    
    print("\n✅ Smoothness comparison saved to 'fm_smoothness_comparison.png'")
    
    # Print summary
    print("\nSmoothness Summary (lower = smoother):")
    print("-"*40)
    for name, pred in [('Euler-20', pred_euler_20), ('Euler-100', pred_euler_100), 
                       ('RK4-20', pred_rk4_20), ('Heun-20', pred_heun_20), ('GT', gt)]:
        vel = np.diff(pred[:, :3], axis=0)
        accel = np.diff(vel, axis=0)
        smoothness = np.mean(np.linalg.norm(accel, axis=1))
        print(f"  {name:<12}: {smoothness:.6f}")


def visualize_3d_trajectory(model, sample, DEVICE, n_samples=5):
    """
    3D visualization comparing multiple predictions.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    gt = sample['action'].numpy()
    obs = sample['agent_pos'].numpy()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g-', linewidth=3, label='Ground Truth')
    
    # Plot observation
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], 'b-', linewidth=2, label='Observation')
    
    # Plot multiple predictions with RK4
    for i in range(n_samples):
        pred = infer_fm_euler(model, sample, DEVICE, num_steps=20)
        alpha = 0.5 if i > 0 else 0.8
        label = 'RK4-20 samples' if i == 0 else None
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r-', 
                linewidth=1.5, alpha=alpha, label=label)
    
    # Mark start and end
    ax.scatter(*gt[0, :3], c='green', s=100, marker='o', label='Start')
    ax.scatter(*gt[-1, :3], c='red', s=100, marker='*', label='Target')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('fm_3d_trajectory.png', dpi=150)
    plt.close()
    
    print("✅ 3D trajectory saved to 'fm_3d_trajectory.png'")


if __name__ == "__main__":
    print("Usage:")
    print("""
    from improved_fm_inference_fixed import (
        compare_integrators, 
        visualize_trajectory_smoothness,
        visualize_3d_trajectory,
        infer_fm_rk4
    )
    from Train_Fork_FM import DP3AgentFlowMatching
    from Data_Loader_Fork import Robot3DDataset
    import torch
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DP3AgentFlowMatching(action_dim=9, robot_state_dim=9).to(DEVICE)
    checkpoint = torch.load('path/to/checkpoint.ckpt', map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Load dataset
    dataset = Robot3DDataset('path/to/data', mode='val')
    sample = dataset[0]
    
    # Compare integration methods
    results = compare_integrators(model, sample, DEVICE)
    
    # Visualize smoothness
    visualize_trajectory_smoothness(model, sample, DEVICE)
    
    # 3D visualization
    visualize_3d_trajectory(model, sample, DEVICE)
    
    # Single inference with RK4
    pred = infer_fm_rk4(model, sample, DEVICE, num_steps=20)
    print(f"Prediction shape: {pred.shape}")  # Should be (16, 9)
    """)