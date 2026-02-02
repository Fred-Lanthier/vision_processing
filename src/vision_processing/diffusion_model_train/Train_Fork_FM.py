"""
Train_Fork_FM.py - Flow Matching Training for 3D Visuomotor Policy

This script implements Flow Matching (FM) as an alternative to DDPM diffusion
for trajectory generation. The key differences from diffusion:

1. TRAINING:
   - Diffusion: x_noisy = sqrt(α_t) * x_0 + sqrt(1-α_t) * noise
   - FM: x_t = t * x_1 + (1-t) * x_0  (linear interpolation)
   
2. TARGET:
   - Diffusion: predict the clean sample x_0 (or noise)
   - FM: predict velocity v = x_1 - x_0
   
3. INFERENCE:
   - Diffusion: iterative denoising (stochastic SDE)
   - FM: ODE integration (deterministic)

The architecture (UNet, DP3Encoder, etc.) remains IDENTICAL to diffusion.
Only the training loop changes.

Reference papers:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- FlowPolicy (Zhang et al., 2024)
- SafeFlow (Dai et al., 2024) - for future safety integration
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import copy
import math
import rospkg
import json
import pickle

# ==============================================================================
# IMPORT DATASET (Same as diffusion version)
# ==============================================================================
from Data_Loader_Fork import Robot3DDataset, seed_everything

# ==============================================================================
# 1. ARCHITECTURE (Identical to diffusion version)
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for time conditioning.
    Converts scalar time t into a high-dimensional embedding.
    
    Math: For each dimension d, we compute:
        emb[d] = sin(t * 10000^(-d/D)) or cos(t * 10000^(-d/D))
    
    This creates a unique "fingerprint" for each timestep that the network
    can use to modulate its behavior. Works for both discrete (0-100) and
    continuous (0-1) time values.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x): return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x): return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x): return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning (Feature-wise Linear Modulation).
    The condition (time + global features) modulates the intermediate features
    via scale and bias: out = scale * out + bias
    """
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2  # scale + bias
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:,0,...], embed[:,1,...]
        out = scale * out + bias 
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D U-Net for trajectory prediction.
    
    In DIFFUSION: predicts the clean sample x_0 (or noise epsilon)
    In FLOW MATCHING: predicts the velocity v = dx/dt
    
    The architecture is IDENTICAL - only the interpretation of the output changes.
    
    Input:  (B, T, D) noisy/interpolated trajectory
    Output: (B, T, D) predicted sample/velocity
    """
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, 
                 down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Time embedding encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        
        # Middle blocks
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # Encoder (downsampling)
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # Decoder (upsampling)
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond=None):
        """
        Args:
            sample: (B, T, D) - interpolated trajectory x_t
            timestep: (B,) or scalar - time value t ∈ [0, 1]
            global_cond: (B, cond_dim) - conditioning features
        
        Returns:
            (B, T, D) - predicted velocity v
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        sample = sample.moveaxis(-1, -2)
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # Time embedding
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        
        # U-Net forward pass
        x = sample
        h = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for res1, res2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        
        # (B, D, T) -> (B, T, D)
        x = x.moveaxis(-1, -2)
        return x


# ==============================================================================
# 2. DP3 ENCODER & NORMALIZER (Identical to diffusion version)
# ==============================================================================

class DP3Encoder(nn.Module):
    """
    Simple MLP encoder for point clouds.
    Uses max-pooling for permutation invariance.
    
    Input:  (B, N, 3) point cloud with N points
    Output: (B, output_dim) compact representation
    """
    def __init__(self, input_dim=3, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU()
        )
        self.projection = nn.Sequential(nn.Linear(256, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]  # Max pooling over points
        return self.projection(x)


class Normalizer(nn.Module):
    """
    Normalizer for robot poses.
    Normalizes position (first 3 dims) to [-1, 1], leaves rotation (6D) unchanged.
    
    Stored as nn.Module buffers so it's saved/loaded with the model checkpoint.
    """
    def __init__(self, stats=None):
        super().__init__()
        self.register_buffer('pos_min', torch.zeros(3))
        self.register_buffer('pos_max', torch.ones(3))
        self.register_buffer('is_initialized', torch.tensor(False, dtype=torch.bool))

        if stats is not None:
            self.load_stats_from_dict(stats)

    def load_stats_from_dict(self, stats):
        print("📥 Injecting normalization stats into model...")
        self.pos_min[:] = torch.tensor(stats['agent_pos']['min'])
        self.pos_max[:] = torch.tensor(stats['agent_pos']['max'])
        self.is_initialized.fill_(True)

    def normalize(self, x, key='agent_pos'): 
        if not self.is_initialized:
            return x
            
        pos = x[..., :3]
        rot = x[..., 3:]
        
        denom = (self.pos_max - self.pos_min).clamp(min=1e-5)
        pos_norm = 2 * (pos - self.pos_min) / denom - 1
        
        return torch.cat([pos_norm, rot], dim=-1)

    def unnormalize(self, x, key='action'):
        if not self.is_initialized:
            return x
            
        pos_norm = x[..., :3]
        rot = x[..., 3:]
        
        denom = (self.pos_max - self.pos_min).clamp(min=1e-5)
        pos = (pos_norm + 1) / 2 * denom + self.pos_min
        
        return torch.cat([pos, rot], dim=-1)


def compute_dataset_stats(dataset):
    """Compute min/max statistics for normalization."""
    print("🔄 Computing normalization statistics...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    min_pos = torch.ones(3) * float('inf')
    max_pos = torch.ones(3) * float('-inf')
    
    for batch in tqdm(loader, desc="Scanning dataset"):
        pos_batch = batch['agent_pos'][..., :3].reshape(-1, 3)
        min_pos = torch.minimum(min_pos, pos_batch.min(dim=0)[0])
        max_pos = torch.maximum(max_pos, pos_batch.max(dim=0)[0])
        
        act_batch = batch['action'][..., :3].reshape(-1, 3)
        min_pos = torch.minimum(min_pos, act_batch.min(dim=0)[0])
        max_pos = torch.maximum(max_pos, act_batch.max(dim=0)[0])

    stats = {
        'agent_pos': {'min': min_pos.tolist(), 'max': max_pos.tolist()},
        'action':    {'min': min_pos.tolist(), 'max': max_pos.tolist()}
    }
    print(f"✅ Stats computed. Min: {min_pos}, Max: {max_pos}")
    return stats


# ==============================================================================
# 3. GLOBAL AGENT (Identical structure, different semantics)
# ==============================================================================

class DP3AgentFlowMatching(nn.Module):
    """
    DP3 Agent using Flow Matching instead of Diffusion.
    
    Architecture is IDENTICAL to DP3AgentRobust.
    The only difference is semantic:
    - In diffusion: noise_pred_net predicts clean sample x_0
    - In flow matching: noise_pred_net predicts velocity v = dx/dt
    
    We keep the name 'noise_pred_net' for compatibility, but it now predicts velocity.
    """
    def __init__(self, action_dim=9, robot_state_dim=9, obs_horizon=2, pred_horizon=16, stats=None):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        self.normalizer = Normalizer(stats)
        self.point_encoder = DP3Encoder(input_dim=3, output_dim=64)
        
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_state_dim * obs_horizon, 128),
            nn.Mish(),
            nn.Linear(128, 64)
        )
        
        # This predicts VELOCITY in Flow Matching (not noise/sample)
        self.velocity_net = ConditionalUnet1D(
            input_dim=action_dim, 
            global_cond_dim=128,  # 64 (point) + 64 (robot)
            down_dims=[256, 512, 1024]
        )
        
    def forward(self, point_cloud, robot_state, x_t, timesteps):
        """
        Forward pass for velocity prediction.
        
        Args:
            point_cloud: (B, N, 3) point cloud observation
            robot_state: (B, obs_horizon, 9) robot state history
            x_t: (B, pred_horizon, 9) interpolated trajectory at time t
            timesteps: (B,) time values t ∈ [0, 1]
        
        Returns:
            (B, pred_horizon, 9) predicted velocity v
        """
        # Encode point cloud
        point_features = self.point_encoder(point_cloud)  # (B, 64)
        
        # Encode robot state
        B = robot_state.shape[0]
        robot_features_flat = self.robot_mlp(robot_state.reshape(B, -1))  # (B, 64)
        
        # Concatenate conditions
        global_cond = torch.cat([point_features, robot_features_flat], dim=-1)  # (B, 128)
        
        # Predict velocity
        return self.velocity_net(x_t, timesteps, global_cond)


# ==============================================================================
# 4. FLOW MATCHING TRAINING LOOP
# ==============================================================================

def main():
    seed_everything(42)
    
    # ==== HYPERPARAMETERS ====
    BATCH_SIZE = 128
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model dimensions
    ACTION_DIM = 9
    ROBOT_STATE_DIM = 9
    OBS_HORIZON = 2
    PRED_HORIZON = 16
    NUM_POINTS = 1024
    
    # Flow Matching specific
    EPSILON = 1e-5  # Small value to avoid boundary issues at t=0 and t=1
    
    config = {
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "model_name": "DP3_FlowMatching_FORK",
        "model_type": "FlowMatching",
        "action_dim": ACTION_DIM,
        "robot_state_dim": ROBOT_STATE_DIM,
        "obs_horizon": OBS_HORIZON,
        "pred_horizon": PRED_HORIZON,
        "num_points": NUM_POINTS,
        "epsilon": EPSILON,
        # SafeFlow parameters (for future inference)
        "num_inference_steps": 20,
        "safeflow_gamma": 1.0,
    }

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'best_val_loss': float('inf')
    }
    
    # ==== PATHS ====
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    models_dir = os.path.join(pkg_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # ==== DATASET ====
    print("⏳ Loading Dataset...")
    
    train_dataset = Robot3DDataset(
        data_path, mode='train', val_ratio=0.2, seed=42,
        num_points=NUM_POINTS, obs_horizon=OBS_HORIZON, 
        pred_horizon=PRED_HORIZON, augment=False
    )
    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=NUM_POINTS, obs_horizon=OBS_HORIZON, 
        pred_horizon=PRED_HORIZON, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    # ==== MODEL ====
    print("🔧 Building Flow Matching Model...")
    model = DP3AgentFlowMatching(
        action_dim=ACTION_DIM, 
        robot_state_dim=ROBOT_STATE_DIM,
        obs_horizon=OBS_HORIZON, 
        pred_horizon=PRED_HORIZON
    ).to(DEVICE)
    
    # EMA (same as diffusion version)
    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
        model_cls=None,
        model_config=None
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        "cosine", 
        optimizer=optimizer, 
        num_warmup_steps=500, 
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )

    print(f"🚀 Flow Matching Training Started on {DEVICE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Val samples: {len(val_dataset)}")
    
    best_val_loss = float('inf')

    # ==== TRAINING LOOP ====
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            # ---- 1. Load and normalize data ----
            pcd = batch['point_cloud'].to(DEVICE, non_blocking=True)  # (B, 1024, 3)
            agent_pos = model.normalizer.normalize(
                batch['agent_pos'].to(DEVICE, non_blocking=True)
            )  # (B, obs_horizon, 9)
            
            # x_1 = target actions (normalized)
            x_1 = model.normalizer.normalize(
                batch['action'].to(DEVICE, non_blocking=True)
            )  # (B, pred_horizon, 9)
            
            B = x_1.shape[0]
            
            # ---- 2. Sample noise x_0 ~ N(0, I) ----
            x_0 = torch.randn_like(x_1)
            
            # ---- 3. Sample time t ~ Uniform(ε, 1-ε) ----
            # Shape (B, 1, 1) for broadcasting over (B, T, D)
            t = torch.rand(B, device=DEVICE) * (1 - 2 * EPSILON) + EPSILON  # t ∈ [ε, 1-ε]
            t_expanded = t.view(B, 1, 1)  # For broadcasting
            
            # ---- 4. Compute interpolated trajectory x_t ----
            # Linear interpolation: x_t = t * x_1 + (1 - t) * x_0
            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
            
            # ---- 5. Compute target velocity ----
            # For linear paths, velocity is constant: v = x_1 - x_0
            v_target = x_1 - x_0
            
            # ---- 6. Predict velocity ----
            v_pred = model(pcd, agent_pos, x_t, t)
            
            # ---- 7. Compute loss ----
            # Simple MSE between predicted and target velocity
            loss = F.mse_loss(v_pred, v_target)
            
            # ---- 8. Backprop ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # ---- 9. Update EMA ----
            ema_model.step(model.parameters())
            
            train_loss_acc += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'decay': f"{ema_model.get_decay(ema_model.optimization_step):.4f}"
            })
        
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # ==== VALIDATION ====
        # Use EMA weights for validation
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        
        model.eval()
        val_loss_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pcd = batch['point_cloud'].to(DEVICE, non_blocking=True)
                agent_pos = model.normalizer.normalize(
                    batch['agent_pos'].to(DEVICE, non_blocking=True)
                )
                x_1 = model.normalizer.normalize(
                    batch['action'].to(DEVICE, non_blocking=True)
                )
                
                B = x_1.shape[0]
                x_0 = torch.randn_like(x_1)
                
                t = torch.rand(B, device=DEVICE) * (1 - 2 * EPSILON) + EPSILON
                t_expanded = t.view(B, 1, 1)
                
                x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
                v_target = x_1 - x_0
                
                v_pred = model(pcd, agent_pos, x_t, t)
                val_loss_acc += F.mse_loss(v_pred, v_target).item()
                
        avg_val = val_loss_acc / len(val_loader) if len(val_loader) > 0 else 0
        history['val_loss'].append(avg_val)
        
        print(f"📊 Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val:.4f}")
        
        # ==== CHECKPOINTING ====
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            history['best_val_loss'] = best_val_loss
            save_name = "flowmatching_best_With.ckpt"
            print("💾 Saved Best EMA Model")
        else:
            save_name = "flowmatching_last_V2.ckpt"
            print("💾 Saved Last EMA Model")
        
        checkpoint_payload = {
            'state_dict': model.state_dict(),
            'history': history,
            'config': config,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'model_class': 'DP3AgentFlowMatching'
        }
        
        torch.save(checkpoint_payload, os.path.join(models_dir, save_name))

        # Restore training weights
        ema_model.restore(model.parameters())

    print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()