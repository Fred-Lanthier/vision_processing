"""
Flow Matching Training PICK-AND-PLACE (10D pose+gripper, second-order-ready)
=============================================================
Identique a `Train_Fork_FlowMP_9D.py` (meme architecture U-Net 1D + encodeur
point cloud DP3 + Flow Matching), mais entraine sur les donnees PICK-AND-PLACE
(`datas/PickAndPlace_preprocess`, via `Data_Loader_PickPlace`).

Le nuage d'entree contient pince + cube + etagere ; la pose predite est celle
du TCP (= "fork tip") plus la commande binaire gripper_open. Checkpoints sauves
sous *_pickplace_* dans `models/`.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
        sample = sample.moveaxis(-1, -2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        timesteps_scaled = timesteps * 1000.0

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
        x = x.moveaxis(-1, -2)
        return x


# ==============================================================================
# 2. POINT CLOUD ENCODER (DP3-style)
# ==============================================================================

class DP3Encoder(nn.Module):
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
        original_shape = x.shape
        if len(original_shape) == 4:
            B, T, N, C = original_shape
            x = x.reshape(B * T, N, C)

        x = self.mlp(x)
        x = torch.max(x, dim=-2)[0]
        x = self.projection(x)

        if len(original_shape) == 4:
            x = x.reshape(B, T, -1)

        return x


# ==============================================================================
# 3. ROBUST NORMALIZER (Handles 10D Obs and 10D Actions)
# ==============================================================================

class Normalizer(nn.Module):
    def __init__(self, obs_dim=10, act_dim=10, stats=None):
        super().__init__()
        self.register_buffer('obs_min', torch.zeros(obs_dim))
        self.register_buffer('obs_max', torch.ones(obs_dim))
        self.register_buffer('act_min', torch.zeros(act_dim))
        self.register_buffer('act_max', torch.ones(act_dim))
        self.register_buffer('is_initialized', torch.tensor(False, dtype=torch.bool))

        if stats is not None:
            self.load_stats_from_dict(stats)

    def load_stats_from_dict(self, stats):
        print("📥 Injection des statistiques de normalisation (toutes dimensions)...")
        self.obs_min[:] = torch.tensor(stats['obs']['min'])
        self.obs_max[:] = torch.tensor(stats['obs']['max'])
        self.act_min[:] = torch.tensor(stats['action']['min'])
        self.act_max[:] = torch.tensor(stats['action']['max'])
        self.is_initialized.fill_(True)

    def normalize_obs(self, x):
        if not self.is_initialized: return x
        x_norm = x.clone()
        denom = (self.obs_max[:3] - self.obs_min[:3]).clamp(min=1e-5)
        x_norm[..., :3] = 2 * (x[..., :3] - self.obs_min[:3]) / denom - 1
        if x.shape[-1] > 9:
            grip_denom = (self.obs_max[9:] - self.obs_min[9:]).clamp(min=1e-5)
            x_norm[..., 9:] = 2 * (x[..., 9:] - self.obs_min[9:]) / grip_denom - 1
        return x_norm

    def normalize_act(self, x):
        if not self.is_initialized: return x
        x_norm = x.clone()
        denom = (self.act_max[:3] - self.act_min[:3]).clamp(min=1e-5)
        x_norm[..., :3] = 2 * (x[..., :3] - self.act_min[:3]) / denom - 1
        if x.shape[-1] > 9:
            grip_denom = (self.act_max[9:] - self.act_min[9:]).clamp(min=1e-5)
            x_norm[..., 9:] = 2 * (x[..., 9:] - self.act_min[9:]) / grip_denom - 1
        return x_norm

    def unnormalize_act(self, x):
        if not self.is_initialized: return x
        x_unnorm = x.clone()
        denom = (self.act_max[:3] - self.act_min[:3]).clamp(min=1e-5)
        x_unnorm[..., :3] = (x[..., :3] + 1) / 2 * denom + self.act_min[:3]
        if x.shape[-1] > 9:
            grip_denom = (self.act_max[9:] - self.act_min[9:]).clamp(min=1e-5)
            x_unnorm[..., 9:] = (x[..., 9:] + 1) / 2 * grip_denom + self.act_min[9:]
        return x_unnorm


# ==============================================================================
# 4. FLOW MATCHING AGENT
# ==============================================================================

class FlowMatchingAgent(nn.Module):
    def __init__(
        self,
        obs_dim=10,
        action_dim=10,
        obs_horizon=2,
        pred_horizon=16,
        encoder_output_dim=64,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        stats=None
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.normalizer = Normalizer(obs_dim=obs_dim, act_dim=action_dim, stats=stats)

        self.point_encoder = DP3Encoder(input_dim=3, output_dim=encoder_output_dim)

        self.robot_mlp = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, 128),
            nn.Mish(),
            nn.Linear(128, encoder_output_dim)
        )

        global_cond_dim = encoder_output_dim * 2

        self.velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )

        print(f"FlowMatchingAgent initialized:")
        print(f"  - Obs dim: {obs_dim}")
        print(f"  - Action dim: {action_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - Total parameters: {total_params:,}")

    def encode_obs(self, obs_dict):
        point_cloud = obs_dict['point_cloud']
        agent_pos = obs_dict['agent_pos']
        B = agent_pos.shape[0]

        if len(point_cloud.shape) == 4:
            point_cloud = point_cloud[:, -1]

        point_features = self.point_encoder(point_cloud)
        robot_features = self.robot_mlp(agent_pos.reshape(B, -1))

        global_cond = torch.cat([point_features, robot_features], dim=-1)
        return global_cond

    def forward(self, obs_dict, x_t, t):
        global_cond = self.encode_obs(obs_dict)
        velocity = self.velocity_net(x_t, t, global_cond)
        return velocity

    def compute_loss(self, batch, alpha=0.1):
        obs = batch['obs']
        actions = batch['action']
        device = actions.device
        B = actions.shape[0]

        obs_normalized = {
            'point_cloud': obs['point_cloud'].to(device),
            'agent_pos': self.normalizer.normalize_obs(obs['agent_pos'].to(device))
        }
        x_1 = self.normalizer.normalize_act(actions.to(device))
        x_0 = torch.randn_like(x_1)

        t = torch.rand(B, device=device)
        t_expand = t[:, None, None]
        x_t = t_expand * x_1 + (1 - t_expand) * x_0

        target_velocity = x_1 - x_0
        pred_velocity = self.forward(obs_normalized, x_t, t)

        loss = F.mse_loss(pred_velocity, target_velocity)

        with torch.no_grad():
            loss_dict = {
                'loss': loss.item(),
                'pred_mean': pred_velocity.mean().item(),
                'pred_std': pred_velocity.std().item(),
                'target_mean': target_velocity.mean().item(),
                'target_std': target_velocity.std().item(),
            }

        return loss, loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict, num_steps=10, method='euler'):
        device = next(self.parameters()).device

        obs = {
            'point_cloud': obs_dict['point_cloud'].to(device),
            'agent_pos': self.normalizer.normalize_obs(obs_dict['agent_pos'].to(device))
        }
        B = obs['agent_pos'].shape[0]

        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(B, device=device) * t_val

            if method == 'euler':
                v = self.forward(obs, x, t)
                x = x + v * dt

            elif method == 'midpoint':
                v1 = self.forward(obs, x, t)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.ones(B, device=device) * (t_val + dt / 2)
                v2 = self.forward(obs, x_mid, t_mid)
                x = x + v2 * dt

            elif method == 'rk4':
                v1 = self.forward(obs, x, t)
                t2 = torch.ones(B, device=device) * (t_val + dt/2)
                v2 = self.forward(obs, x + v1 * dt/2, t2)
                v3 = self.forward(obs, x + v2 * dt/2, t2)
                t4 = torch.ones(B, device=device) * (t_val + dt)
                v4 = self.forward(obs, x + v3 * dt, t4)
                x = x + (v1 + 2*v2 + 2*v3 + v4) * dt / 6

        action_pred = self.normalizer.unnormalize_act(x)

        return {
            'action': action_pred,
            'action_pred': action_pred
        }


def compute_dataset_stats(dataset, obs_dim=10, act_dim=10, num_workers=0):
    print("🔄 Calcul des statistiques de normalisation complètes...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers,
                        collate_fn=custom_collate_fn)

    obs_min = torch.ones(obs_dim) * float('inf')
    obs_max = torch.ones(obs_dim) * float('-inf')
    act_min = torch.ones(act_dim) * float('inf')
    act_max = torch.ones(act_dim) * float('-inf')

    for batch in tqdm(loader, desc="Scanning dataset"):
        obs_b = batch['obs']['agent_pos'].reshape(-1, obs_dim)
        obs_min = torch.minimum(obs_min, obs_b.min(dim=0)[0])
        obs_max = torch.maximum(obs_max, obs_b.max(dim=0)[0])

        act_b = batch['action'].reshape(-1, act_dim)
        act_min = torch.minimum(act_min, act_b.min(dim=0)[0])
        act_max = torch.maximum(act_max, act_b.max(dim=0)[0])

    stats = {
        'obs':    {'min': obs_min.tolist(), 'max': obs_max.tolist()},
        'action': {'min': act_min.tolist(), 'max': act_max.tolist()}
    }
    print("✅ Stats calculées et prêtes.")
    return stats


# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================

def train_flow_matching(
    model, train_loader, val_loader, optimizer, lr_scheduler,
    ema_model, num_epochs, device, save_dir, config, STATS
):
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'best_val_loss': float('inf')}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss_acc = 0
        current_alpha = get_alpha(epoch, warmup_epochs=50, max_alpha=0.1)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            batch = {
                'obs': {
                    'point_cloud': batch['obs']['point_cloud'].to(device),
                    'agent_pos': batch['obs']['agent_pos'].to(device)
                },
                'action': batch['action'].to(device)
            }

            loss, loss_dict = model.compute_loss(batch, current_alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            ema_model.step(model.parameters())
            train_loss_acc += loss_dict['loss']

            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'ema': f"{ema_model.get_decay(ema_model.optimization_step):.4f}"
            })

        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history['best_val_loss'] = best_val_loss
            save_name = "best_fm_model_pickplace_10D_1024_rdn.ckpt"
            print("Saved Best EMA Model")
        else:
            save_name = "last_fm_model_pickplace_10D_1024_rdn.ckpt"

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'epoch': epoch,
            'stats': STATS,
            'history': history,
            'best_val_loss': best_val_loss
        }, os.path.join(save_dir, save_name))

        ema_model.restore(model.parameters())

    return history


# ==============================================================================
# 6. EVALUATION METRICS
# ==============================================================================

GRIPPER_BINARY_THRESHOLD = 0.5


def evaluate_model(model, dataset, device, num_steps=10, num_samples=100):
    model.eval()
    all_ade = []
    all_fde = []
    all_gripper_mae = []
    all_gripper_acc = []
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]
        obs = {
            'point_cloud': sample['obs']['point_cloud'].unsqueeze(0),
            'agent_pos': sample['obs']['agent_pos'].unsqueeze(0)
        }
        gt_action = sample['action'].numpy()

        with torch.no_grad():
            result = model.predict_action(obs, num_steps=num_steps, method='midpoint')

        pred_action = result['action_pred'].cpu().numpy()[0]

        pos_pred = pred_action[:, :3]
        pos_gt = gt_action[:, :3]

        ade = np.mean(np.linalg.norm(pos_pred - pos_gt, axis=-1))
        fde = np.linalg.norm(pos_pred[-1] - pos_gt[-1])

        all_ade.append(ade)
        all_fde.append(fde)

        if pred_action.shape[-1] > 9:
            grip_pred = np.clip(pred_action[:, 9], 0.0, 1.0)
            grip_gt = gt_action[:, 9]
            all_gripper_mae.append(np.mean(np.abs(grip_pred - grip_gt)))
            all_gripper_acc.append(np.mean(
                (grip_pred >= GRIPPER_BINARY_THRESHOLD)
                == (grip_gt >= GRIPPER_BINARY_THRESHOLD)
            ))

    results = {
        'ade_mean': np.mean(all_ade) * 1000,
        'ade_std': np.std(all_ade) * 1000,
        'fde_mean': np.mean(all_fde) * 1000,
        'fde_std': np.std(all_fde) * 1000
    }
    if all_gripper_mae:
        results.update({
            'gripper_mae': float(np.mean(all_gripper_mae)),
            'gripper_accuracy': float(np.mean(all_gripper_acc))
        })

    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"ADE: {results['ade_mean']:.2f} ± {results['ade_std']:.2f} mm")
    print(f"FDE: {results['fde_mean']:.2f} ± {results['fde_std']:.2f} mm")
    if all_gripper_mae:
        print(f"Gripper MAE: {results['gripper_mae']:.4f}")
        print(f"Gripper accuracy: {results['gripper_accuracy'] * 100:.2f}%")
    print("="*50)

    return results


def custom_collate_fn(batch):
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
    from Data_Loader_PickPlace import Robot3DDataset, seed_everything
    from diffusers.training_utils import EMAModel
    from diffusers.optimization import get_scheduler

    seed_everything(42)

    # === CONFIG ===
    BATCH_SIZE = 128
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OBS_DIM = 10       # Pos(3) + Rot(6) + gripper_open
    ACTION_DIM = 10    # PosRot(9) + gripper_open
    OBS_HORIZON = 2
    PRED_HORIZON = 16
    NUM_POINTS = 1024
    NUM_INFERENCE_STEPS = 10
    NUM_WORKERS = 0

    config = {
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "model_name": "FlowMatching_PickPlace_10D",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "obs_horizon": OBS_HORIZON,
        "pred_horizon": PRED_HORIZON,
        "num_points": NUM_POINTS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_workers": NUM_WORKERS,
    }

    # === DATA ===
    import rospkg
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'PickAndPlace_preprocess')
    models_dir = os.path.join(pkg_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("⏳ Loading datasets...")
    train_dataset = Robot3DDataset(
        data_path, mode='train', val_ratio=0.2, seed=42,
        num_points=NUM_POINTS, obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON, augment=True
    )
    STATS = compute_dataset_stats(
        train_dataset, obs_dim=OBS_DIM, act_dim=ACTION_DIM, num_workers=NUM_WORKERS
    )
    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=NUM_POINTS, obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn
    )

    # === MODEL ===
    print("🏗️ Building Flow Matching model...")
    model = FlowMatchingAgent(
        obs_dim=OBS_DIM,
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
    print(f"🚀 Training Flow Matching (PickPlace) on {DEVICE}")
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

    with open(os.path.join(models_dir, 'fm_results_pickplace_10D.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
