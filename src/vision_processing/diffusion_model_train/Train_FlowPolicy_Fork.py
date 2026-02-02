"""
FlowPolicy Training Script with Two-Segment Consistency Flow Matching
======================================================================

This implements the FlowPolicy paper: "Enabling Fast and Robust 3D Flow-based 
Policy via Consistency Flow Matching for Robot Manipulation"

Key Mathematical Concepts:
--------------------------

1. FLOW MATCHING BASICS:
   - We transform noise a_0 ~ N(0,I) to actions a_1 ~ p_data
   - Linear interpolation: a_t = (1-t)*a_0 + t*a_1
   - The network learns velocity: v_θ(a_t, t, condition)
   
2. CONSISTENCY FLOW MATCHING:
   - Enforces velocity consistency along flow trajectories
   - Allows single-step inference (instead of 10-100 steps in diffusion)
   
3. TWO-SEGMENT TRAINING (K=2):
   - Segment 0: t ∈ [0, 0.5]
   - Segment 1: t ∈ [0.5, 1.0]
   - Each segment predicts flow endpoint at (i+1)/K
   
4. LOSS FUNCTION (Equation 8 from paper):
   L = ||f_θ(t, a_t) - f_θ-(t+Δt, a_{t+Δt})||² 
     + α||v_θ(t, a_t) - v_θ-(t+Δt, a_{t+Δt})||²
   
   where: f_θ(t, a_t) = a_t + ((i+1)/K - t) * v_θ(t, a_t)
   
   - f_θ is the predicted flow endpoint
   - v_θ is the velocity prediction
   - θ- is EMA of parameters (target network)
   - α is velocity consistency weight
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

# ==============================================================================
# IMPORT DATASET (Same as original)
# ==============================================================================
# Uncomment the line below when using with ROS
# from Data_Loader_Fork import Robot3DDataset, seed_everything

# For standalone testing, we include seed_everything here
import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. ARCHITECTURE COMPONENTS (Same UNet backbone, but predicts velocity)
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for time conditioning.
    Converts scalar time t into a high-dimensional embedding.
    
    Math: For each dimension d, we compute:
        emb[d] = sin(t * 10000^(-d/D)) or cos(t * 10000^(-d/D))
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
    """
    Residual block with FiLM conditioning (Feature-wise Linear Modulation).
    The condition (time + observation) modulates the features via scale and bias.
    """
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2  # For scale and bias
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
        scale, bias = embed[:,0,...], embed[:,1,...]
        out = scale * out + bias  # FiLM modulation
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D U-Net for action sequence prediction.
    
    In FlowPolicy, this network predicts VELOCITY (not noise like in DDPM).
    
    Input: noisy action sequence (B, T, action_dim)
    Output: velocity prediction (B, T, action_dim)
    Condition: time embedding + observation features
    """
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, 
                 down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Time embedding (also used for flow time t in [0,1])
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
            sample: (B, T, action_dim) - noisy action sequence
            timestep: (B,) - flow time t ∈ [0, 1] (NOT integer diffusion steps!)
            global_cond: (B, cond_dim) - observation features
            
        Returns:
            velocity: (B, T, action_dim) - predicted velocity field
        """
        # Convert from (B, T, C) to (B, C, T) for Conv1d
        sample = sample.moveaxis(-1, -2)
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # Encode time (now continuous t ∈ [0,1])
        global_feature = self.diffusion_step_encoder(timesteps)
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
        
        # Convert back to (B, T, C)
        x = x.moveaxis(-1, -2)
        return x


# ==============================================================================
# 2. OBSERVATION ENCODERS (Same as original)
# ==============================================================================

class DP3Encoder(nn.Module):
    """
    Lightweight PointNet-style encoder for 3D point clouds.
    Uses max-pooling for permutation invariance.
    """
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
        # x: (B, N, 3) -> point cloud with N points
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]  # Global max pooling
        return self.projection(x)


class Normalizer(nn.Module):
    """
    Normalizes positions to [-1, 1] range.
    Statistics are saved as buffers (included in state_dict).
    """
    def __init__(self, stats=None):
        super().__init__()
        self.register_buffer('pos_min', torch.zeros(3))
        self.register_buffer('pos_max', torch.ones(3))
        self.register_buffer('is_initialized', torch.tensor(False, dtype=torch.bool))

        if stats is not None:
            self.load_stats_from_dict(stats)

    def load_stats_from_dict(self, stats):
        print("📥 Loading normalization statistics...")
        self.pos_min[:] = torch.tensor(stats['agent_pos']['min'])
        self.pos_max[:] = torch.tensor(stats['agent_pos']['max'])
        self.is_initialized.fill_(True)

    def normalize(self, x, key='agent_pos'): 
        if not self.is_initialized:
            return x
            
        # Split position (3D) and rotation (6D)
        pos = x[..., :3]
        rot = x[..., 3:]
        
        # MinMax normalization to [-1, 1]
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
# 3. FLOWPOLICY AGENT
# ==============================================================================

class FlowPolicyAgent(nn.Module):
    """
    FlowPolicy Agent using Consistency Flow Matching.
    
    Unlike diffusion policy which predicts noise, FlowPolicy predicts VELOCITY.
    This enables single-step inference for real-time robot control.
    
    Architecture:
        - Point cloud encoder (DP3Encoder)
        - Robot state encoder (MLP)
        - Velocity network (ConditionalUnet1D)
    """
    def __init__(self, action_dim=9, robot_state_dim=9, obs_horizon=2, 
                 pred_horizon=16, stats=None):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        # Normalizer as part of model (saved in state_dict)
        self.normalizer = Normalizer(stats)
        
        # Point cloud encoder
        self.point_encoder = DP3Encoder(input_dim=3, output_dim=64)
        
        # Robot state encoder
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_state_dim * obs_horizon, 128),
            nn.Mish(),
            nn.Linear(128, 64)
        )
        
        # Velocity prediction network (same architecture as noise_pred_net)
        self.velocity_net = ConditionalUnet1D(
            input_dim=action_dim, 
            global_cond_dim=128,  # 64 (point) + 64 (robot)
            down_dims=[256, 512, 1024]
        )
        
    def forward(self, point_cloud, robot_state, noisy_actions, timesteps):
        """
        Forward pass predicts velocity given current state.
        
        Args:
            point_cloud: (B, N, 3) - 3D point cloud observation
            robot_state: (B, obs_horizon, state_dim) - robot state history
            noisy_actions: (B, pred_horizon, action_dim) - noisy action sequence
            timesteps: (B,) - flow time t ∈ [0, 1]
            
        Returns:
            velocity: (B, pred_horizon, action_dim) - predicted velocity
        """
        # Encode point cloud
        point_features = self.point_encoder(point_cloud)  # (B, 64)
        
        # Encode robot state
        B = robot_state.shape[0]
        robot_features = self.robot_mlp(robot_state.reshape(B, -1))  # (B, 64)
        
        # Concatenate conditions
        global_cond = torch.cat([point_features, robot_features], dim=-1)  # (B, 128)
        
        # Predict velocity
        return self.velocity_net(noisy_actions, timesteps, global_cond)
    
    @torch.no_grad()
    def sample(self, point_cloud, robot_state, num_steps=1, device='cuda'):
        """
        Generate action sequence using flow ODE.
        
        FlowPolicy can generate in just 1 step thanks to consistency training!
        
        Args:
            point_cloud: (B, N, 3)
            robot_state: (B, obs_horizon, state_dim) - should be normalized!
            num_steps: Number of integration steps (1 for single-step)
            device: torch device
            
        Returns:
            actions: (B, pred_horizon, action_dim) - generated actions (normalized)
        """
        B = point_cloud.shape[0]
        
        # Start from noise
        a_t = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Encode observations once
        point_features = self.point_encoder(point_cloud)
        robot_features = self.robot_mlp(robot_state.reshape(B, -1))
        global_cond = torch.cat([point_features, robot_features], dim=-1)
        
        if num_steps == 1:
            # Single-step generation (main advantage of FlowPolicy!)
            t = torch.zeros(B, device=device)
            velocity = self.velocity_net(a_t, t, global_cond)
            # Flow endpoint: a_1 = a_0 + (1 - 0) * velocity = a_0 + velocity
            actions = a_t + velocity
        else:
            # Multi-step Euler integration (for comparison/debugging)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((B,), step * dt, device=device)
                velocity = self.velocity_net(a_t, t, global_cond)
                a_t = a_t + dt * velocity
            actions = a_t
            
        return actions


# ==============================================================================
# 4. CONSISTENCY FLOW MATCHING TRAINER
# ==============================================================================

class ConsistencyFlowMatchingTrainer:
    """
    Two-Segment Consistency Flow Matching Trainer.
    
    This implements Equation 8 from the FlowPolicy paper:
    
    L(θ) = E[ ||f_θ^i(t, a_t) - f_θ-^i(t+Δt, a_{t+Δt})||² 
           + α||v_θ^i(t, a_t) - v_θ-^i(t+Δt, a_{t+Δt})||² ]
    
    where:
        - K = 2 (two segments)
        - Segment i covers t ∈ [i/K, (i+1)/K]
        - f_θ^i(t, a_t) = a_t + ((i+1)/K - t) * v_θ(t, a_t)
        - θ- is EMA of θ
    
    Mathematical Intuition:
    -----------------------
    The loss enforces two things:
    
    1. CONSISTENCY: Points at t and t+Δt should predict the same flow endpoint.
       This means the learned flow has straight-line trajectories.
       
    2. VELOCITY CONSISTENCY: The velocity should be the same at t and t+Δt.
       This ensures the velocity field is constant along flow lines.
       
    Together, these allow single-step generation: from any t, we can directly
    predict the endpoint without iterative integration.
    """
    
    def __init__(self, 
                 model: FlowPolicyAgent,
                 device='cuda',
                 num_segments=2,        # K in the paper
                 delta_t=0.05,          # Δt for consistency loss (paper uses larger values)
                 alpha=1.0,             # Weight for velocity consistency
                 ema_decay=0.95,        # EMA decay rate
                 lambda_weights=None):  # Per-segment weights
        """
        Args:
            model: FlowPolicyAgent
            num_segments: K (number of segments, default 2)
            delta_t: Small time step for consistency (Δt)
            alpha: Weight for velocity consistency loss
            ema_decay: EMA decay rate for target network
            lambda_weights: Weights λ_i for each segment (middle segments harder)
        """
        self.model = model
        self.device = device
        self.num_segments = num_segments
        self.delta_t = delta_t
        self.alpha = alpha
        
        # Default lambda weights (can increase middle segment weight)
        if lambda_weights is None:
            self.lambda_weights = [1.0] * num_segments
        else:
            self.lambda_weights = lambda_weights
            
        # Create target network (EMA copy)
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False
            
        self.ema_decay = ema_decay
        
    def update_target_network(self):
        """
        Update target network with EMA of main model parameters.
        θ- = decay * θ- + (1 - decay) * θ
        """
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), 
                                          self.target_model.parameters()):
                target_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1.0 - self.ema_decay)
                
    def compute_interpolation(self, a_src, a_tar, t):
        """
        Compute linear interpolation between source (noise) and target (data).
        
        Math: a_t = (1 - t) * a_src + t * a_tar
        
        This is the standard "CondOT" probability path from flow matching:
        - At t=0: a_0 = a_src (pure noise)
        - At t=1: a_1 = a_tar (clean action)
        
        Args:
            a_src: (B, T, D) - source distribution (noise)
            a_tar: (B, T, D) - target distribution (data)
            t: (B, 1, 1) - time values
            
        Returns:
            a_t: (B, T, D) - interpolated actions
        """
        return (1 - t) * a_src + t * a_tar
    
    def compute_flow_endpoint(self, a_t, velocity, t, segment_idx):
        """
        Compute predicted flow endpoint for segment i.
        
        Math: f_θ^i(t, a_t) = a_t + ((i+1)/K - t) * v_θ(t, a_t)
        
        This predicts where the flow would end at time (i+1)/K.
        For segment 0 (t ∈ [0, 0.5]): predicts endpoint at t=0.5
        For segment 1 (t ∈ [0.5, 1]): predicts endpoint at t=1.0
        
        Args:
            a_t: Current noisy action
            velocity: Predicted velocity
            t: Current time (B, 1, 1)
            segment_idx: Which segment (0 or 1)
            
        Returns:
            endpoint: Predicted flow endpoint
        """
        segment_end = (segment_idx + 1) / self.num_segments
        return a_t + (segment_end - t) * velocity
    
    def compute_loss(self, batch):
        """
        Compute Hybrid Consistency Flow Matching loss.
        
        Based on Theorem 1 from the paper, the consistency loss is EQUIVALENT to:
            "striking a balance between exact velocity estimation and 
             adhering to consistent velocity constraints"
        
        We implement this explicitly as:
        
        L = L_FM + λ_cons × L_consistency
        
        where:
            L_FM = ||v_θ(t, x_t) - v_true||²           [Flow Matching - main signal]
            L_consistency = ||f_θ(t) - f_{θ⁻}(t+Δt)||² [Self-consistency - regularization]
            v_true = x_tar - x_src                      [True OT velocity]
        
        This ensures:
        1. The model learns the correct velocity (FM loss)
        2. The learned flows are straight (consistency loss)
        
        Args:
            batch: Dictionary with point_cloud, agent_pos, action
                
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with component losses for logging
        """
        # Unpack batch
        point_cloud = batch['point_cloud'].to(self.device)
        agent_pos = batch['agent_pos'].to(self.device)
        actions = batch['action'].to(self.device)
        
        B = actions.shape[0]
        
        # Sample noise (source distribution)
        a_src = torch.randn_like(actions)
        
        # Target is the clean action (data distribution)
        a_tar = actions
        
        # ================================================
        # Ground truth velocity for OT path
        # ================================================
        # For linear interpolation x_t = (1-t)*x_src + t*x_tar
        # The velocity dx/dt = x_tar - x_src is CONSTANT
        v_true = a_tar - a_src
        
        total_loss = 0.0
        fm_loss_total = 0.0
        consistency_loss_total = 0.0
        
        # Train on both segments
        for seg_idx in range(self.num_segments):
            # Segment boundaries
            seg_start = seg_idx / self.num_segments
            seg_end = (seg_idx + 1) / self.num_segments
            
            # Sample t uniformly from segment
            t = torch.rand(B, 1, 1, device=self.device) * (seg_end - seg_start - self.delta_t) + seg_start
            t_next = t + self.delta_t
            
            # Compute interpolations on OT path
            a_t = self.compute_interpolation(a_src, a_tar, t)
            a_t_next = self.compute_interpolation(a_src, a_tar, t_next)
            
            # Flatten t for network input
            t_flat = t.squeeze(-1).squeeze(-1)
            t_next_flat = t_next.squeeze(-1).squeeze(-1)
            
            # ================================================
            # Main model prediction at time t
            # ================================================
            velocity_t = self.model(point_cloud, agent_pos, a_t, t_flat)
            
            # ================================================
            # LOSS 1: Flow Matching (main learning signal)
            # ================================================
            # The network should predict the true OT velocity
            fm_loss = F.mse_loss(velocity_t, v_true)
            
            # ================================================
            # LOSS 2: Consistency Regularization
            # ================================================
            # Endpoint predictions should match between t and t+Δt
            endpoint_t = a_t + (seg_end - t) * velocity_t
            
            with torch.no_grad():
                velocity_t_next = self.target_model(point_cloud, agent_pos, a_t_next, t_next_flat)
                endpoint_t_next = a_t_next + (seg_end - t_next) * velocity_t_next
            
            consistency_loss = F.mse_loss(endpoint_t, endpoint_t_next)
            
            # ================================================
            # Combined loss for this segment
            # ================================================
            # FM provides the main signal, consistency encourages straight flows
            segment_loss = self.lambda_weights[seg_idx] * (
                fm_loss + self.alpha * consistency_loss
            )
            
            total_loss += segment_loss
            fm_loss_total += fm_loss.item()
            consistency_loss_total += consistency_loss.item()
        
        # Average over segments
        total_loss = total_loss / self.num_segments
        
        loss_dict = {
            'total': total_loss.item(),
            'fm': fm_loss_total / self.num_segments,
            'cons': consistency_loss_total / self.num_segments,
        }
        
        return total_loss, loss_dict


# ==============================================================================
# 5. UTILITY FUNCTIONS
# ==============================================================================

def compute_dataset_stats(dataset):
    """Compute min/max statistics for normalization."""
    from torch.utils.data import DataLoader
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
# 6. MAIN TRAINING LOOP
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
    
    # FlowPolicy specific hyperparameters
    NUM_SEGMENTS = 2          # K = 2 (two-segment training)
    DELTA_T = 0.05            # Time step for consistency (larger = stronger signal)
    ALPHA = 1.0               # Velocity consistency weight
    EMA_DECAY = 0.95          # EMA decay for target network
    
    config = {
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "model_name": "FlowPolicy_ConsistencyFM_Fork",
        "action_dim": ACTION_DIM,
        "robot_state_dim": ROBOT_STATE_DIM,
        "obs_horizon": OBS_HORIZON,
        "pred_horizon": PRED_HORIZON,
        "num_points": NUM_POINTS,
        "num_segments": NUM_SEGMENTS,
        "delta_t": DELTA_T,
        "alpha": ALPHA,
        "ema_decay": EMA_DECAY,
    }
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'fm_loss': [],
        'cons_loss': [],
        'lr': [],
        'best_val_loss': float('inf')
    }
    
    # ==== DATA LOADING ====
    # Uncomment and modify for your setup:
    import rospkg
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    
    # For testing without ROS, set your path directly:
    # data_path = "/path/to/your/data"  # CHANGE THIS!
    
    print("⏳ Loading Dataset...")
    
    # Import your data loader
    from Data_Loader_Fork import Robot3DDataset
    
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
    
    # ==== MODEL SETUP ====
    print("🔧 Creating FlowPolicy model...")
    model = FlowPolicyAgent(
        action_dim=ACTION_DIM,
        robot_state_dim=ROBOT_STATE_DIM,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON
    ).to(DEVICE)
    
    # Create trainer with Consistency-FM
    trainer = ConsistencyFlowMatchingTrainer(
        model=model,
        device=DEVICE,
        num_segments=NUM_SEGMENTS,
        delta_t=DELTA_T,
        alpha=ALPHA,
        ema_decay=EMA_DECAY
    )
    
    # EMA for saving best model (separate from target network in trainer)
    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )
    
    print(f"🚀 Training FlowPolicy on {DEVICE}")
    print(f"   Segments: {NUM_SEGMENTS}, Δt: {DELTA_T}, α: {ALPHA}")
    
    best_val_loss = float('inf')
    
    # Create output directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # ==== TRAINING LOOP ====
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_acc = 0
        fm_loss_acc = 0
        cons_loss_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            # Normalize inputs
            batch['agent_pos'] = model.normalizer.normalize(
                batch['agent_pos'].to(DEVICE, non_blocking=True)
            )
            batch['action'] = model.normalizer.normalize(
                batch['action'].to(DEVICE, non_blocking=True)
            )
            batch['point_cloud'] = batch['point_cloud'].to(DEVICE, non_blocking=True)
            
            # Compute loss
            loss, loss_dict = trainer.compute_loss(batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            # Update target network (EMA)
            trainer.update_target_network()
            
            # Update EMA for model saving
            ema_model.step(model.parameters())
            
            train_loss_acc += loss_dict['total']
            fm_loss_acc += loss_dict['fm']
            cons_loss_acc += loss_dict['cons']
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'fm': f"{loss_dict['fm']:.4f}",
                'cons': f"{loss_dict['cons']:.4f}",
            })
        
        # Epoch averages
        n_batches = len(train_loader)
        avg_train_loss = train_loss_acc / n_batches
        history['train_loss'].append(avg_train_loss)
        history['fm_loss'].append(fm_loss_acc / n_batches)
        history['cons_loss'].append(cons_loss_acc / n_batches)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # ==== VALIDATION ====
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        
        model.eval()
        val_loss_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch['agent_pos'] = model.normalizer.normalize(
                    batch['agent_pos'].to(DEVICE, non_blocking=True)
                )
                batch['action'] = model.normalizer.normalize(
                    batch['action'].to(DEVICE, non_blocking=True)
                )
                batch['point_cloud'] = batch['point_cloud'].to(DEVICE, non_blocking=True)
                
                val_loss, _ = trainer.compute_loss(batch)
                val_loss_acc += val_loss.item()
        
        avg_val_loss = val_loss_acc / len(val_loader) if len(val_loader) > 0 else 0
        history['val_loss'].append(avg_val_loss)
        
        print(f"📊 Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        # ==== SAVE MODEL ====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_name = "flowpolicy_best_V2.ckpt"
            print("💾 Saved Best Model")
        else:
            save_name = "flowpolicy_last_V2.ckpt"
        
        checkpoint = {
            'state_dict': model.state_dict(),
            'target_state_dict': trainer.target_model.state_dict(),
            'history': history,
            'config': config,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'model_class': 'FlowPolicyAgent'
        }
        
        torch.save(checkpoint, os.path.join(models_dir, save_name))
        
        # Restore training weights
        ema_model.restore(model.parameters())
    
    print("✅ Training Complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()