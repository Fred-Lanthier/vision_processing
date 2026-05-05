"""
Data Loader for Flow Matching — Translation-Centered Version
=============================================================

Key design choice:
  - Translation-only centering at fork tip position
  - Positions centered at (0,0,0), world orientation PRESERVED
  - Model sees orientation changes throughout trajectory

Why this matters for tool-only point clouds (no robot links):
  With full SE(3), after grasp the PCD is literally identical every step.
  With translation-only, the model can see the fork rotating (lifting,
  tilting toward user, etc.) because world orientation is kept.

The policy becomes:
  - Position-invariant (doesn't memorize table coordinates)
  - Gravity-aware (knows which way is up, how tool is oriented)
  - Robot-agnostic (any robot holding the same tool gets same prediction)
"""

import os
import glob
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rotation_matrix_to_ortho6d(matrices):
    x_raw = matrices[..., 0]
    y_raw = matrices[..., 1]
    return np.concatenate([x_raw, y_raw], axis=-1)

class Robot3DDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 mode='train',
                 val_ratio=0.2,       
                 seed=42,             
                 num_points=1024, 
                 pred_horizon=16, 
                 obs_horizon=2, 
                 action_horizon=8,
                 augment=False,
                 data_source='all'):  # 'all', 'real', ou 'sim'
        
        super().__init__()
        self.mode = mode
        self.num_points = num_points
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.seed = seed
        self.augment = augment and (mode == 'train')
        self.data_source = data_source

        all_traj_folders = sorted(glob.glob(os.path.join(root_dir, 'Trajectory*')), 
                                  key=lambda x: int(x.split('_')[-1]))
        
        # 1. Séparation explicite des trajectoires par domaine (Réel vs Simulation)
        real_folders = []
        sim_folders = []
        
        for folder in all_traj_folders:
            folder_name = os.path.basename(folder)
            try:
                traj_id = int(folder_name.split('_')[-1])
                if 1 <= traj_id <= 45:
                    real_folders.append(folder)
                elif 46 <= traj_id <= 64:
                    sim_folders.append(folder)
            except ValueError:
                continue # Ignore les dossiers mal formatés

        # 2. Mélange indépendant pour chaque domaine pour garantir l'aléatoire
        rng = random.Random(self.seed)
        rng.shuffle(real_folders)
        rng.shuffle(sim_folders)
        
        # 3. Calcul des coupes stratifiées (Stratified Split)
        n_val_real = int(len(real_folders) * val_ratio)
        n_train_real = len(real_folders) - n_val_real
        
        n_val_sim = int(len(sim_folders) * val_ratio)
        n_train_sim = len(sim_folders) - n_val_sim

        # 4. Attribution selon le mode (train ou val)
        if mode == 'train':
            selected_folders = real_folders[:n_train_real] + sim_folders[:n_train_sim]
        elif mode == 'val':
            selected_folders = real_folders[n_train_real:] + sim_folders[n_train_sim:]
        else: # mode test ou inference
            selected_folders = real_folders + sim_folders
            
        # 5. Filtrage final selon ce que tu veux envoyer au réseau (data_source)
        if self.data_source == 'real':
            self.traj_folders = [f for f in selected_folders if f in real_folders]
        elif self.data_source == 'sim':
            self.traj_folders = [f for f in selected_folders if f in sim_folders]
        else: # 'all'
            self.traj_folders = selected_folders
            # On mélange une dernière fois pour que les batchs ne soient pas 
            # 100% réels puis 100% sim, mais un beau mélange des deux.
            rng.shuffle(self.traj_folders)

        print(f"📊 Dataset Mode: {mode.upper():<5} | Source: {data_source.upper():<4} | Augment: {self.augment}")
        
        # Petit affichage de débogage pour te confirmer la stratification
        nb_real = sum(1 for f in self.traj_folders if f in real_folders)
        nb_sim = sum(1 for f in self.traj_folders if f in sim_folders)
        print(f"📁 Dossiers retenus : {len(self.traj_folders)} (Réels: {nb_real}, Sim: {nb_sim})")
        print("⏳ Preprocessing and caching to RAM (one-time cost)...")

        self.processed_data = [] 
        MOTION_THRESHOLD = 0.0001

        for folder in tqdm(self.traj_folders, desc="Processing Data"):
            folder_name = os.path.basename(folder)
            traj_id = folder_name.split('_')[-1]
            
            json_path = os.path.join(folder, f'trajectory_{traj_id}.json')
            if not os.path.exists(json_path):
                json_path = os.path.join(folder, f'trajectory{traj_id}.json')
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)
            states = data.get('states', [])
            if len(states) == 0:
                continue

            # 1. Parse full trajectory
            full_traj_poses = []
            full_traj_pcd = []
            raw_positions = []
            time_steps = []
            
            for state in states:
                pos = np.array(state['fork_tip_position'], dtype=np.float32)
                quat = state['fork_tip_orientation']
                rot = R.from_quat(quat)
                mat = rot.as_matrix()
                rot_6d = rotation_matrix_to_ortho6d(mat)
                pose_9d = np.concatenate([pos, rot_6d]).astype(np.float32)
                
                pcd_name = state.get('Merged_Fork_point_cloud')
                pcd_path = os.path.join(folder, f'Merged_Fork_{folder_name}', pcd_name) if pcd_name else None
                
                if pcd_path and os.path.exists(pcd_path):
                    pcd_data = np.load(pcd_path).astype(np.float32)
                else:
                    pcd_data = np.zeros((self.num_points, 3), dtype=np.float32)

                time_step = state["time_step"]

                full_traj_poses.append(pose_9d)
                full_traj_pcd.append(pcd_data)
                raw_positions.append(pos)
                time_steps.append(time_step)

            # 2. Cutoff static tail
            raw_positions = np.array(raw_positions)
            final_pos = raw_positions[-1]
            dist_to_end = np.linalg.norm(raw_positions - final_pos, axis=1)
            cutoff_idx = len(full_traj_poses)
            for i in range(len(full_traj_poses) - 1, 0, -1):
                if dist_to_end[i] > MOTION_THRESHOLD:
                    cutoff_idx = min(i + 5, len(full_traj_poses))
                    break
            
            full_traj_poses = full_traj_poses[:cutoff_idx]
            full_traj_pcd = full_traj_pcd[:cutoff_idx]
            time_steps = time_steps[:cutoff_idx]
            
            if len(full_traj_poses) < (self.obs_horizon + self.pred_horizon):
                continue

            # 3. Sliding window: precompute every sample
            total_steps = len(full_traj_poses)
            
            for current_idx in range(self.obs_horizon - 1, total_steps - self.pred_horizon + 1):
                
                # A. Indices
                obs_indices = list(range(current_idx - self.obs_horizon + 1, current_idx + 1))
                pred_indices = list(range(current_idx, current_idx + self.pred_horizon))

                # B. Data extraction
                raw_pcd = full_traj_pcd[current_idx]
                sampled_pcd = self._sample_point_cloud(raw_pcd)

                obs_poses = np.stack([full_traj_poses[i] for i in obs_indices])
                action_seq = np.stack([full_traj_poses[i] for i in pred_indices])
                
                obs_times = np.array([time_steps[i] for i in obs_indices])
                action_times = np.array([time_steps[i] for i in pred_indices])

                # C. TRANSLATION-ONLY CENTERING
                curr_pos = obs_poses[-1, :3].copy()

                obs_poses_centered = obs_poses.copy()
                obs_poses_centered[:, :3] -= curr_pos

                action_seq_centered = action_seq.copy()
                action_seq_centered[:, :3] -= curr_pos

                # D. CALCUL DES DYNAMIQUES AVEC LE VRAI DELTA T
                # Extraction des intervalles de temps (dt) en évitant la division par 0
                seq_times = np.concatenate([[obs_times[-1]], action_times])
                dt_seq = np.clip(seq_times[1:] - seq_times[:-1], a_min=1e-4, a_max=None)[:, np.newaxis]

                # 1. Vitesse (Dérivée 1ère divisée par le temps)
                action_vel = np.zeros_like(action_seq_centered)
                action_vel[0] = (action_seq_centered[0] - obs_poses_centered[-1]) / dt_seq[0]
                if self.pred_horizon > 1:
                    action_vel[1:] = (action_seq_centered[1:] - action_seq_centered[:-1]) / dt_seq[1:]

                # 2. Accélération (Dérivée 2ème divisée par le temps)
                # On estime la vitesse actuelle (obs_vel) pour calculer la première accélération
                if self.obs_horizon >= 2:
                    dt_obs_last = max(obs_times[-1] - obs_times[-2], 1e-4)
                    curr_vel = (obs_poses_centered[-1] - obs_poses_centered[-2]) / dt_obs_last
                else:
                    curr_vel = np.zeros_like(action_vel[0])
                
                action_acc = np.zeros_like(action_vel)
                action_acc[0] = (action_vel[0] - curr_vel) / dt_seq[0]
                if self.pred_horizon > 1:
                    action_acc[1:] = (action_vel[1:] - action_vel[:-1]) / dt_seq[1:]

                # 3. Concaténation de l'action à 27 dimensions
                action_full = np.concatenate([action_seq_centered, action_vel, action_acc], axis=-1)

                # E. Store ready-to-use sample
                self.processed_data.append({
                    'pcd': sampled_pcd,              
                    'obs': obs_poses_centered,       # (obs_horizon, 9)
                    'act': action_full,              # (pred_horizon, 27)
                })

        print(f"✅ {len(self.processed_data)} samples pre-computed in RAM.\n")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]
        
        pcd = sample['pcd']
        obs = sample['obs']
        act = sample['act']

        if self.augment:
            pcd = pcd.copy()
            obs = obs.copy()
            act = act.copy()
            pcd, obs, act = self._apply_augmentation(pcd, obs, act)
        
        data_dict = {
            'obs': {
                'point_cloud': torch.from_numpy(pcd).float(),
                'agent_pos': torch.from_numpy(obs).float(),
            },
            'action': torch.from_numpy(act).float(),
        }
        
        return data_dict

    def _sample_point_cloud(self, pc):
        if pc.shape[0] == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        if pc.shape[0] <= self.num_points:
            indices = np.arange(pc.shape[0])
            if pc.shape[0] < self.num_points:
                extra_indices = np.random.choice(pc.shape[0], self.num_points - pc.shape[0], replace=True)
                indices = np.concatenate([indices, extra_indices])
            return pc[indices].astype(np.float32)
        
        indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
        return pc[indices].astype(np.float32)

    def _apply_augmentation(self, pcd, obs_poses, action_seq):
        """
        Augmentation adaptée pour les tenseurs d'action à 27D (Pos, Vel, Acc).
        Les vecteurs de vitesse et d'accélération doivent aussi subir la rotation !
        """
        # 1. Random Z-rotation (±15°)
        theta = np.random.uniform(-np.pi/12, np.pi/12)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]], dtype=np.float32)
        
        pcd = pcd @ R_z.T
        
        # Rotation de l'observation (9D)
        obs_poses[:, :3]  = obs_poses[:, :3]  @ R_z.T
        obs_poses[:, 3:6] = obs_poses[:, 3:6] @ R_z.T
        obs_poses[:, 6:9] = obs_poses[:, 6:9] @ R_z.T
        
        # Rotation de l'action (27D)
        for i in range(0, 27, 3):  # Applique la rotation par blocs de 3 dimensions
            action_seq[:, i:i+3] = action_seq[:, i:i+3] @ R_z.T
        
        # 2. Translation jitter (±1cm) - N'affecte QUE les positions (colonnes 0 à 2)
        shift = np.random.uniform(-0.01, 0.01, size=(1, 3)).astype(np.float32)
        pcd += shift
        obs_poses[:, :3] += shift
        action_seq[:, :3] += shift  # Vitesse et accélération sont invariantes à la translation

        # 3. PCD sensor noise (±5mm)
        noise = np.clip(np.random.randn(*pcd.shape).astype(np.float32) * 0.005, -0.01, 0.01)
        pcd += noise

        return pcd, obs_poses, action_seq

    def get_normalizer(self):
        from normalizer import LinearNormalizer
        
        all_point_clouds = []
        all_agent_pos = []
        all_actions = []
        
        print("📊 Computing normalization statistics...")
        for idx in range(min(len(self), 5000)):
            sample = self[idx]
            all_point_clouds.append(sample['obs']['point_cloud'])
            all_agent_pos.append(sample['obs']['agent_pos'])
            all_actions.append(sample['action'])
        
        all_point_clouds = torch.stack(all_point_clouds)
        all_agent_pos = torch.stack(all_agent_pos)
        all_actions = torch.stack(all_actions)
        
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                'point_cloud': all_point_clouds,
                'agent_pos': all_agent_pos,
                'action': all_actions
            },
            last_n_dims=1,
            mode='limits',
            output_max=1.0,
            output_min=-1.0
        )
        
        print("✅ Normalizer fitted!")
        return normalizer

# Custom collate function for nested dictionaries
def custom_collate_fn(batch):
    result = {}
    obs_keys = batch[0]['obs'].keys()
    result['obs'] = {}
    for key in obs_keys:
        result['obs'][key] = torch.stack([item['obs'][key] for item in batch])
    result['action'] = torch.stack([item['action'] for item in batch])
    return result


# --- Test ---
if __name__ == "__main__":
    import rospkg
    rospack = rospkg.RosPack()
    try:
        pkg_path = rospack.get_path('vision_processing')
        data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
        
        print(f"Testing: {data_path}")
        dataset = Robot3DDataset(root_dir=data_path, pred_horizon=16, obs_horizon=2)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample structure:")
            print(f"  obs['point_cloud']: {sample['obs']['point_cloud'].shape}")
            print(f"  obs['agent_pos']:   {sample['obs']['agent_pos'].shape}")
            print(f"  action:             {sample['action'].shape}")
            
            # Verify current pose is at origin
            curr_pos = sample['obs']['agent_pos'][-1, :3]
            print(f"\n  Current position (should be ~0): {curr_pos.numpy()}")
            
            # Verify rotation is NOT identity (world frame preserved)
            curr_rot = sample['obs']['agent_pos'][-1, 3:]
            print(f"  Current rotation 6D (should NOT be [1,0,0,0,1,0]): {curr_rot.numpy()}")
            
    except Exception as e:
        print(f"Error: {e}")