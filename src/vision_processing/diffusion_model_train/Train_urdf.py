import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import copy
import math
import rospkg
import json
import pickle

# Import Dataset existant (On va le wrapper pour la normalisation)
from Data_Loader_urdf import Robot3DDataset

# ==============================================================================
# 1. ARCHITECTURE ROBOMIMIC (Nettoyée et Intégrée)
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
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        # FiLM modulation
        cond_channels = out_channels * 2
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
        out = scale * out + bias # FiLM Magic
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, 
                 down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Time Embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        
        # Condition globale = Time Emb + Feature Vector (Robot+PCD)
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Down & Up Blocks
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
        # sample: (B, T, C) -> Permute -> (B, C, T)
        sample = sample.moveaxis(-1, -2)
        
        # Timestep processing
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

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
        # (B, C, T) -> (B, T, C)
        x = x.moveaxis(-1, -2)
        return x

# ==============================================================================
# 2. DP3 ENCODER & NORMALIZER
# ==============================================================================

class DP3Encoder(nn.Module):
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
        x = torch.max(x, dim=1)[0]
        return self.projection(x)
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        self.input_pointcloud = input[0].detach()

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

def compute_dataset_stats(dataset):
    print("🔄 Calcul des statistiques de normalisation (Itératif)...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Init avec des valeurs extrêmes inverses
    min_pos = torch.ones(3) * float('inf')
    max_pos = torch.ones(3) * float('-inf')
    
    # On ne normalise PAS l'action, car elle contient la rotation. 
    # On va séparer pos et rot plus tard, ou calculer min/max juste pour pos.
    min_action_pos = torch.ones(3) * float('inf')
    max_action_pos = torch.ones(3) * float('-inf')

    for batch in tqdm(loader, desc="Scanning dataset"):
        # Agent POS (batch, T, 9) -> On regarde seulement les 3 premiers (x,y,z)
        pos_batch = batch['agent_pos'][..., :3].reshape(-1, 3)
        min_pos = torch.minimum(min_pos, pos_batch.min(dim=0)[0])
        max_pos = torch.maximum(max_pos, pos_batch.max(dim=0)[0])
        
        # Action (batch, T, 9) -> idem
        act_batch = batch['action'][..., :3].reshape(-1, 3)
        min_action_pos = torch.minimum(min_action_pos, act_batch.min(dim=0)[0])
        max_action_pos = torch.maximum(max_action_pos, act_batch.max(dim=0)[0])

    stats = {
        'agent_pos': {'min': min_pos.tolist(), 'max': max_pos.tolist()},
        'action':    {'min': min_action_pos.tolist(), 'max': max_action_pos.tolist()}
    }
    print("✅ Stats calculées (Position Only).")
    return stats

# ==============================================================================
# 3. GLOBAL AGENT
# ==============================================================================

class DP3AgentRobust(nn.Module):
    def __init__(self, action_dim=9, robot_state_dim=9, obs_horizon=2, pred_horizon=16, stats=None):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        self.normalizer = Normalizer(stats)

        self.point_encoder = DP3Encoder(input_dim=3, output_dim=64)
        
        # Robot State Encoder
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_state_dim * obs_horizon, 128),
            nn.Mish(),
            nn.Linear(128, 64)
        )
        
        # U-Net Robomimic
        # Global cond = Point (64) + Robot (64) = 128
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim, 
            global_cond_dim=128,
            down_dims=[256, 512, 1024]
        )
        
    def forward(self, point_cloud, robot_state, noisy_actions, timesteps):
        # 1. Encode Vision
        point_features = self.point_encoder(point_cloud) # (B, 64)
        
        # 2. Encode Proprioception
        B = robot_state.shape[0]
        robot_features_flat = self.robot_mlp(robot_state.reshape(B, -1))
        
        global_cond = torch.cat([point_features, robot_features_flat], dim=-1) # (B, 128)
        
        return self.noise_pred_net(noisy_actions, timesteps, global_cond)

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================

def main():
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000 
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ACTION_DIM = 9
    ROBOT_STATE_DIM = 9
    OBS_HORIZON = 2
    PRED_HORIZON = 16
    NUM_POINTS = 1024

    # Diffusion Params
    NOISE_STEPS = 100
    BETA_SCHEDULE = 'squaredcos_cap_v2'
    CLIP_SAMPLE = True
    PREDICTION_TYPE = 'sample'

    config = {
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "model_name": "DP3_Robust_DiffusersEMA_URDF",
        "action_dim": ACTION_DIM,
        "robot_state_dim": ROBOT_STATE_DIM,
        "obs_horizon": OBS_HORIZON,
        "pred_horizon": PRED_HORIZON,
        "num_points": NUM_POINTS,
        "noise_steps": NOISE_STEPS,
        "beta_schedule": BETA_SCHEDULE,
        "clip_sample": CLIP_SAMPLE,
        "prediction_type": PREDICTION_TYPE
    }

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'best_val_loss': float('inf')
    }
    
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    
    # 1. Dataset
    print("⏳ Chargement du Dataset...")
    full_dataset = Robot3DDataset(data_path, mode='all')
    
    # 2. Normalisation
    # stats_path = os.path.join(pkg_path, "normalization_stats_urdf_fork_SAMPLE.json")
    # if os.path.exists(stats_path):
    #     print("📂 Chargement des stats existantes...")
    #     with open(stats_path, 'r') as f:
    #         stats = json.load(f)
    # else:
    #     stats = compute_dataset_stats(full_dataset)
    #     with open(stats_path, 'w') as f:
    #         json.dump(stats, f)
    stats = None
    normalizer = Normalizer(stats)
    
    # Reload datasets en mode train/val
    train_dataset = Robot3DDataset(data_path, mode='train', val_ratio=0.2, seed=42, 
                                    num_points=NUM_POINTS, obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON, augment=True)
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42, 
                                num_points=NUM_POINTS, obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Setup Model
    model = DP3AgentRobust(action_dim=ACTION_DIM, robot_state_dim=ROBOT_STATE_DIM,
                        obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON).to(DEVICE)
    
    # 🔥 INITIALISATION DIFFUSERS EMA 🔥
    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,               # Start low (Warmup)
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
        model_cls=None,
        model_config=None
    )
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NOISE_STEPS,
        beta_schedule=BETA_SCHEDULE,
        clip_sample=CLIP_SAMPLE,
        prediction_type=PREDICTION_TYPE
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    # Warmup à 500 steps comme recommandé
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(train_loader)*NUM_EPOCHS)

    print(f"🚀 Training Started on {DEVICE} with Sample Prediction & Diffusers EMA")
    best_val_loss = float('inf')

    models_dir = os.path.join(pkg_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            # Load Data
            pcd = batch['point_cloud'].to(DEVICE, non_blocking=True)
            # NORMALISATION ON THE FLY
            agent_pos = normalizer.normalize(batch['agent_pos'].to(DEVICE, non_blocking=True), 'agent_pos')
            actions = normalizer.normalize(batch['action'].to(DEVICE, non_blocking=True), 'action')
            
            # Noise generation (Standard Forward Process)
            noise = torch.randn_like(actions)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=DEVICE).long()
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
            
            # Forward
            model_output = model(pcd, agent_pos, noisy_actions, timesteps)
            
            # Loss & Backprop - ✅ SAMPLE PREDICTION LOSS
            # On compare la sortie du modèle aux ACTIONS originales (sample), pas au bruit (epsilon)
            loss = F.mse_loss(model_output, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # 🔥 EMA STEP 🔥
            # Met à jour les poids fantômes
            ema_model.step(model.parameters())
            
            train_loss_acc += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'decay': f"{ema_model.get_decay(ema_model.optimization_step):.4f}"})
        
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # --- VALIDATION (LOGIQUE DE SWAP) ---
        # A. On sauvegarde les poids d'entraînement actuels
        ema_model.store(model.parameters())
        # B. On charge les poids EMA dans le modèle principal pour validation
        ema_model.copy_to(model.parameters())
        
        # Le modèle contient maintenant les poids lissés !
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                pcd = batch['point_cloud'].to(DEVICE, non_blocking=True)
                agent_pos = normalizer.normalize(batch['agent_pos'].to(DEVICE, non_blocking=True), 'agent_pos')
                actions = normalizer.normalize(batch['action'].to(DEVICE, non_blocking=True), 'action')
                
                noise = torch.randn_like(actions)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=DEVICE).long()
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                # Validation avec Sample Prediction
                pred = model(pcd, agent_pos, noisy_actions, timesteps)
                val_loss_acc += F.mse_loss(pred, actions).item() # Compare à actions, pas noise
                
        avg_val = val_loss_acc / len(val_loader) if len(val_loader) > 0 else 0
        history['val_loss'].append(avg_val)
        print(f"Stats: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val:.4f}")
        
        # --- SAVING ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Sauvegarde le modèle courant (qui contient les poids EMA à cause du copy_to)
            save_name = "dp3_policy_best_robust_urdf_FPS_fork_SAMPLE.ckpt"
            print("💾 Saved Best EMA Model")
        else:
            save_name = "dp3_policy_last_robust_urdf_FPS_fork_SAMPLE.ckpt"
            print("💾 Saved Last EMA Model")
        
        checkpoint_payload = {
            'state_dict': model.state_dict(),
            'history': history,
            'config': config,
            'stats': stats, # Les stats de normalisation JSON directes !
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'model_class': 'DP3AgentRobust'
        }

        torch.save(checkpoint_payload, os.path.join(pkg_path, "models", save_name))
        # C. On restaure les poids d'entraînement pour continuer l'entraînement
        ema_model.restore(model.parameters())

        # with open(os.path.join(pkg_path, "pkl_files", "train_history_urdf_FPS_fork_SAMPLE.pkl"), 'wb') as f:
        #     pickle.dump(history, f)
if __name__ == "__main__":
    main()