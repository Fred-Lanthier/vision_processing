"""
Standard Flow Matching Training for Robot Trajectory Prediction
================================================================

This is STANDARD Flow Matching (not Consistency FM).
Uses 10-15 inference steps with Euler integration.

Key formulas:
- Interpolation: x_t = t * x_1 + (1-t) * x_0
- Target velocity: v = x_1 - x_0 (CONSTANT along the path!)
- Loss: ||v_θ(x_t, t) - (x_1 - x_0)||²

Changes from original:
1. Fixed time scaling for positional embeddings
2. Proper normalizer integration
3. Correct data format handling
4. Improved inference with multiple methods
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
import json
import pickle

# ==============================================================================
# 1. ARCHITECTURE COMPONENTS
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for time t.
    
    IMPORTANT: This expects values in a reasonable range.
    For t ∈ [0, 1], we scale by 1000 to get good frequency coverage.
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
    def forward(self, x): 
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x): 
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x): 
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1D U-Net for velocity prediction."""
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, 
                 down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

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
            sample: (B, T, D) - trajectory
            timestep: (B,) - time values (will be scaled internally)
            global_cond: (B, cond_dim) - conditioning
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        sample = sample.moveaxis(-1, -2)
        
        # Handle timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # =============================================
        # KEY FIX: Scale timesteps for positional embedding
        # The sinusoidal embedding works better with larger values
        # =============================================
        timesteps_scaled = timesteps * 1000.0  # Scale from [0,1] to [0,1000]

        global_feature = self.diffusion_step_encoder(timesteps_scaled)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        
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
# 2. POINT CLOUD ENCODER (DP3-style)
# ==============================================================================

class DP3Encoder(nn.Module):
    """MLP encoder for point clouds with max pooling."""
    def __init__(self, input_dim=3, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(256, output_dim), 
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x: (B, N, 3) or (B, T, N, 3)
        original_shape = x.shape
        
        if len(original_shape) == 4:
            # (B, T, N, 3) -> (B*T, N, 3)
            B, T, N, C = original_shape
            x = x.reshape(B * T, N, C)
        
        x = self.mlp(x)              # (B, N, 256)
        x = torch.max(x, dim=-2)[0]  # (B, 256) - max over points
        x = self.projection(x)       # (B, output_dim)
        
        if len(original_shape) == 4:
            x = x.reshape(B, T, -1)
            
        return x


# ==============================================================================
# 3. SIMPLE NORMALIZER (alternative to LinearNormalizer)
# ==============================================================================

class Normalizer(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        # On enregistre les buffers. Ils seront sauvegardés dans le .ckpt !
        self.register_buffer('pos_min', torch.zeros(3))
        self.register_buffer('pos_max', torch.ones(3))
        # On sauvegarde aussi l'état d'initialisation (booléen)
        self.register_buffer('is_initialized', torch.tensor(False, dtype=torch.bool))

        if stats is not None:
            self.load_stats_from_dict(stats)

    def load_stats_from_dict(self, stats):
        # Cette fonction sert uniquement lors du PREMIER entraînement
        print("📥 Injection des statistiques dans le modèle...")
        self.pos_min[:] = torch.tensor(stats['agent_pos']['min'])
        self.pos_max[:] = torch.tensor(stats['agent_pos']['max'])
        self.is_initialized.fill_(True)

    def normalize(self, x, key='agent_pos'): 
        # Note: key est gardé pour compatibilité, mais ici on gère surtout agent_pos
        if not self.is_initialized:
            return x
            
        # Séparation Pos / Rot (Spécifique à votre format 9D)
        pos = x[..., :3]
        rot = x[..., 3:]
        
        # Formule MinMax [-1, 1]
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


# ==============================================================================
# 4. FLOW MATCHING AGENT
# ==============================================================================

class FlowMatchingAgent(nn.Module):
    """
    Standard Flow Matching Agent.
    
    NOT Consistency FM - uses multiple inference steps (10-15 recommended).
    """
    def __init__(
        self, 
        action_dim=9, 
        obs_horizon=2, 
        pred_horizon=16, 
        encoder_output_dim=64,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        stats = None
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        # Normalizer
        self.normalizer = Normalizer(stats=stats)
        
        # Point cloud encoder
        self.point_encoder = DP3Encoder(input_dim=3, output_dim=encoder_output_dim)
        
        # Robot state encoder
        self.robot_mlp = nn.Sequential(
            nn.Linear(action_dim * obs_horizon, 128),
            nn.Mish(),
            nn.Linear(128, encoder_output_dim)
        )
        
        # Global conditioning dimension
        global_cond_dim = encoder_output_dim * 2  # point + robot
        
        # Velocity prediction network
        self.velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )
        
        print(f"FlowMatchingAgent initialized:")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Obs horizon: {obs_horizon}")
        print(f"  - Pred horizon: {pred_horizon}")
        print(f"  - Global cond dim: {global_cond_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - Total parameters: {total_params:,}")
    
    def encode_obs(self, obs_dict):
        """Encode observations into global conditioning."""
        point_cloud = obs_dict['point_cloud']  # (B, N, 3) or (B, T, N, 3)
        agent_pos = obs_dict['agent_pos']      # (B, obs_horizon, 9)
        
        B = agent_pos.shape[0]
        
        # Encode point cloud
        # If shape is (B, T, N, 3), take last observation
        if len(point_cloud.shape) == 4:
            point_cloud = point_cloud[:, -1]  # (B, N, 3)
        
        point_features = self.point_encoder(point_cloud)  # (B, encoder_dim)
        
        # Encode robot state
        robot_features = self.robot_mlp(agent_pos.reshape(B, -1))  # (B, encoder_dim)
        
        # Concatenate
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        return global_cond
    
    def forward(self, obs_dict, x_t, t):
        """
        Forward pass for velocity prediction.
        
        Args:
            obs_dict: Dictionary with 'point_cloud' and 'agent_pos'
            x_t: (B, pred_horizon, action_dim) - noisy trajectory
            t: (B,) - time values in [0, 1]
        
        Returns:
            velocity: (B, pred_horizon, action_dim)
        """
        global_cond = self.encode_obs(obs_dict)
        velocity = self.velocity_net(x_t, t, global_cond)
        return velocity
    
    def compute_loss_consistency(self, batch, alpha=0.1):
        """
        Standard Flow Matching loss with diagnostics.
        """
        obs = batch['obs']
        actions = batch['action']
        
        device = actions.device
        B = actions.shape[0]
        
        # Normalize (point cloud is NOT normalized — same as diffusion code)
        obs_normalized = {
            'point_cloud': obs['point_cloud'].to(device),
            'agent_pos': self.normalizer.normalize(obs['agent_pos'].to(device))
        }
        x_1 = self.normalizer.normalize(actions.to(device))
        
        # Sample noise
        x_0 = torch.randn_like(x_1)
        
        # Sample time uniformly in [0, 1]
        t = torch.rand(B, device=device)
        delta_t = np.random.uniform(alpha/10, alpha)
        t2 = (t + delta_t).clamp(max=0.9999)
        
        # Linear interpolation
        t_expand = t[:, None, None]
        t2_expand = t2[:, None, None]
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        x_t2 = t2_expand * x_1 + (1 - t2_expand) * x_0

        # Target velocity (CONSTANT - no division!)
        target_velocity = x_1 - x_0
        
        # Predict velocity
        pred_velocity = self.forward(obs_normalized, x_t, t)
        pred_velocity2 = self.forward(obs_normalized, x_t2, t2)
        f1 = x_t + (1 - t_expand) * pred_velocity    # predicted x_1 from (x_t, t)
        f2 = x_t2 + (1 - t2_expand) * pred_velocity2  # predicted x_1 from (x_t2, t2)
        
        # MSE loss
        FM_loss = F.mse_loss(pred_velocity, target_velocity)
        consistency_loss = F.mse_loss(f1, f2.detach())
        loss = FM_loss + alpha * consistency_loss
        
        # Detailed diagnostics
        with torch.no_grad():
            loss_dict = {
                'loss': loss.item(),
                'pred_mean': pred_velocity.mean().item(),
                'pred_std': pred_velocity.std().item(),
                'target_mean': target_velocity.mean().item(),
                'target_std': target_velocity.std().item(),
                # Check if predictions are in reasonable range
                'pred_max': pred_velocity.abs().max().item(),
                'target_max': target_velocity.abs().max().item(),
            }
        
        return loss, loss_dict

    def compute_loss(self, batch, alpha = 0.1):
        """
        Standard Flow Matching loss with diagnostics.
        """
        obs = batch['obs']
        actions = batch['action']
        
        device = actions.device
        B = actions.shape[0]
        
        # Normalize (point cloud is NOT normalized — same as diffusion code)
        obs_normalized = {
            'point_cloud': obs['point_cloud'].to(device),
            'agent_pos': self.normalizer.normalize(obs['agent_pos'].to(device))
        }
        x_1 = self.normalizer.normalize(actions.to(device))
        
        # Sample noise
        x_0 = torch.randn_like(x_1)
        
        # Sample time uniformly in [0, 1]
        t = torch.rand(B, device=device)
        
        # Linear interpolation
        t_expand = t[:, None, None]
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # Target velocity (CONSTANT - no division!)
        target_velocity = x_1 - x_0
        
        # Predict velocity
        pred_velocity = self.forward(obs_normalized, x_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        accel = pred_velocity[:, 2:] - 2 * pred_velocity[:, 1:-1] + pred_velocity[:, :-2]
        smooth_loss = accel.pow(2).mean()
        loss += alpha/10 * smooth_loss

        # Detailed diagnostics
        with torch.no_grad():
            loss_dict = {
                'loss': loss.item(),
                'pred_mean': pred_velocity.mean().item(),
                'pred_std': pred_velocity.std().item(),
                'target_mean': target_velocity.mean().item(),
                'target_std': target_velocity.std().item(),
                # Check if predictions are in reasonable range
                'pred_max': pred_velocity.abs().max().item(),
                'target_max': target_velocity.abs().max().item(),
            }
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_action(self, obs_dict, num_steps=10, method='euler'):
        """
        Generate trajectory using Flow Matching.
        
        Args:
            obs_dict: Dictionary with 'point_cloud' and 'agent_pos'
            num_steps: Number of integration steps (10-15 recommended)
            method: 'euler' or 'midpoint'
        
        Returns:
            action_pred: (B, pred_horizon, action_dim) - predicted trajectory
        """
        device = next(self.parameters()).device
        
        # Move to device and normalize (point cloud NOT normalized — same as diffusion)
        obs = {
            'point_cloud': obs_dict['point_cloud'].to(device),
            'agent_pos': self.normalizer.normalize(
                obs_dict['agent_pos'].to(device)
            )
        }
        
        B = obs['agent_pos'].shape[0]
        
        # Start from noise
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Integration step size
        dt = 1.0 / num_steps
        
        # Integrate from t=0 to t=1
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(B, device=device) * t_val
            
            if method == 'euler':
                # Simple Euler step
                v = self.forward(obs, x, t)
                x = x + v * dt
                
            elif method == 'midpoint':
                # Midpoint method (more accurate)
                v1 = self.forward(obs, x, t)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.ones(B, device=device) * (t_val + dt / 2)
                v2 = self.forward(obs, x_mid, t_mid)
                x = x + v2 * dt
                
            elif method == 'rk4':
                # Runge-Kutta 4 (most accurate but slower)
                v1 = self.forward(obs, x, t)
                
                t2 = torch.ones(B, device=device) * (t_val + dt/2)
                v2 = self.forward(obs, x + v1 * dt/2, t2)
                
                v3 = self.forward(obs, x + v2 * dt/2, t2)
                
                t4 = torch.ones(B, device=device) * (t_val + dt)
                v4 = self.forward(obs, x + v3 * dt, t4)
                
                x = x + (v1 + 2*v2 + 2*v3 + v4) * dt / 6
        
        # Unnormalize
        action_pred = self.normalizer.unnormalize(x)
        
        return {
            'action': action_pred[:, :self.obs_horizon],  # Actions to execute
            'action_pred': action_pred  # Full prediction
        }


def compute_dataset_stats(dataset):
    print("🔄 Calcul des statistiques de normalisation (Itératif)...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=custom_collate_fn)
    
    min_pos = torch.ones(3) * float('inf')
    max_pos = torch.ones(3) * float('-inf')
    
    for batch in tqdm(loader, desc="Scanning dataset"):
        # FM data loader returns nested dict: batch['obs']['agent_pos']
        pos_batch = batch['obs']['agent_pos'][..., :3].reshape(-1, 3)
        min_pos = torch.minimum(min_pos, pos_batch.min(dim=0)[0])
        max_pos = torch.maximum(max_pos, pos_batch.max(dim=0)[0])
        
        act_batch = batch['action'][..., :3].reshape(-1, 3)
        min_pos = torch.minimum(min_pos, act_batch.min(dim=0)[0])
        max_pos = torch.maximum(max_pos, act_batch.max(dim=0)[0])

    stats = {
        'agent_pos': {'min': min_pos.tolist(), 'max': max_pos.tolist()},
        'action':    {'min': min_pos.tolist(), 'max': max_pos.tolist()}
    }
    print(f"✅ Stats calculées. Min: {min_pos}, Max: {max_pos}")
    return stats

# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================

def train_flow_matching(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    ema_model,
    num_epochs,
    device,
    save_dir,
    config,
    STATS
):
    """Training loop for Flow Matching."""
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'best_val_loss': float('inf')
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_acc = 0
        current_alpha = get_alpha(epoch, warmup_epochs=50, max_alpha=0.1)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            # Move to device
            batch = {
                'obs': {
                    'point_cloud': batch['obs']['point_cloud'].to(device),
                    'agent_pos': batch['obs']['agent_pos'].to(device)
                },
                'action': batch['action'].to(device)
            }
            
            # Compute loss
            loss, loss_dict = model.compute_loss_consistency(batch, current_alpha)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            # EMA update
            ema_model.step(model.parameters())
            
            train_loss_acc += loss_dict['loss']
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'ema': f"{ema_model.get_decay(ema_model.optimization_step):.4f}"
            })
        
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        
        model.eval()
        val_loss_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    'obs': {
                        'point_cloud': batch['obs']['point_cloud'].to(device),
                        'agent_pos': batch['obs']['agent_pos'].to(device)
                    },
                    'action': batch['action'].to(device)
                }
                loss, loss_dict = model.compute_loss(batch)
                val_loss_acc += loss.item()

        avg_val_loss = val_loss_acc / len(val_loader)
        history['val_loss'].append(avg_val_loss)        
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history['best_val_loss'] = best_val_loss
            save_name = "best_fm_model_high_dim_CFM_relative.ckpt"
            print("Saved Best EMA Model")
        else:
            save_name = "last_fm_model_high_dim_CFM_relative.ckpt"
            print("Saved Last EMA Model")

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'epoch': epoch,
            'stats': STATS,
            'history': history,
            'best_val_loss': best_val_loss
        }, os.path.join(save_dir, save_name))
        
        # Restauration des poids d'entraînement
        ema_model.restore(model.parameters())

    return history


# ==============================================================================
# 6. EVALUATION METRICS
# ==============================================================================

def evaluate_model(model, dataset, device, num_steps=10, num_samples=100):
    """Evaluate model on validation set."""
    
    model.eval()
    
    all_ade = []  # Average Displacement Error
    all_fde = []  # Final Displacement Error
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]
        
        # Prepare batch
        obs = {
            'point_cloud': sample['obs']['point_cloud'].unsqueeze(0),
            'agent_pos': sample['obs']['agent_pos'].unsqueeze(0)
        }
        gt_action = sample['action'].numpy()
        
        # Predict
        with torch.no_grad():
            result = model.predict_action(obs, num_steps=num_steps, method='midpoint')
        
        pred_action = result['action_pred'].cpu().numpy()[0]
        
        # Compute metrics (position only, first 3 dims)
        pos_pred = pred_action[:, :3]
        pos_gt = gt_action[:, :3]
        
        # ADE: mean error over trajectory
        ade = np.mean(np.linalg.norm(pos_pred - pos_gt, axis=-1))
        
        # FDE: error at final point
        fde = np.linalg.norm(pos_pred[-1] - pos_gt[-1])
        
        all_ade.append(ade)
        all_fde.append(fde)
    
    results = {
        'ade_mean': np.mean(all_ade) * 1000,  # Convert to mm
        'ade_std': np.std(all_ade) * 1000,
        'fde_mean': np.mean(all_fde) * 1000,
        'fde_std': np.std(all_fde) * 1000
    }
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"ADE: {results['ade_mean']:.2f} ± {results['ade_std']:.2f} mm")
    print(f"FDE: {results['fde_mean']:.2f} ± {results['fde_std']:.2f} mm")
    print("="*50)
    
    return results


# ==============================================================================
# 7. COLLATE FUNCTION FOR NESTED DICTS
# ==============================================================================

def custom_collate_fn(batch):
    """Handle nested dictionary batching."""
    result = {
        'obs': {
            'point_cloud': torch.stack([item['obs']['point_cloud'] for item in batch]),
            'agent_pos': torch.stack([item['obs']['agent_pos'] for item in batch])
        },
        'action': torch.stack([item['action'] for item in batch])
    }
    return result

def get_alpha(epoch, warmup_epochs=50, max_alpha=0.1):
    if epoch < warmup_epochs:
        return max_alpha * (epoch / warmup_epochs)
    return max_alpha

# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    from Data_Loader_Fork_FM import Robot3DDataset, seed_everything
    
    seed_everything(42)
    
    # === CONFIG ===
    BATCH_SIZE = 128
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ACTION_DIM = 9
    OBS_HORIZON = 2
    PRED_HORIZON = 16
    NUM_POINTS = 256
    
    # Flow Matching specific
    NUM_INFERENCE_STEPS = 10  # 10-15 recommended for standard FM
    
    config = {
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "model_name": "FlowMatching_Fork_relative",
        "action_dim": ACTION_DIM,
        "obs_horizon": OBS_HORIZON,
        "pred_horizon": PRED_HORIZON,
        "num_points": NUM_POINTS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
    }
    
    # === DATA ===
    import rospkg
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    models_dir = os.path.join(pkg_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("⏳ Loading datasets...")
    train_dataset = Robot3DDataset(
        data_path, mode='train', val_ratio=0.2, seed=42,
        num_points=NUM_POINTS, obs_horizon=OBS_HORIZON, 
        pred_horizon=PRED_HORIZON, augment=True
    )
    STATS = compute_dataset_stats(train_dataset)
    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=NUM_POINTS, obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    # === MODEL ===
    print("🏗️ Building Flow Matching model...")
    model = FlowMatchingAgent(
        action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        encoder_output_dim=64,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        stats=STATS
    ).to(DEVICE)
    
    # EMA
    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6
    )
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )
    
    # === TRAIN ===
    print(f"🚀 Training Flow Matching on {DEVICE}")
    history = train_flow_matching(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_dir=models_dir,
        config=config,
        STATS=STATS
    )
    
    # === FINAL EVALUATION ===
    print("\n🎯 Final Evaluation with EMA weights...")
    ema_model.copy_to(model.parameters())
    
    results = evaluate_model(
        model=model,
        dataset=val_dataset,
        device=DEVICE,
        num_steps=NUM_INFERENCE_STEPS,
        num_samples=100
    )
    
    # Save results
    with open(os.path.join(models_dir, 'fm_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()