"""
Data Loader for Flow Matching - Corrected Version
==================================================

Changes from original:
1. Returns data in dictionary format compatible with LinearNormalizer
2. Adds get_normalizer() method for proper normalization
3. Separates 'obs' and 'action' clearly
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

# --- UTILITAIRES (Inchangés) ---
def rotation_matrix_to_ortho6d(matrices):
    x_raw = matrices[..., 0]
    y_raw = matrices[..., 1]
    return np.concatenate([x_raw, y_raw], axis=-1)

def rotation_6d_to_matrix(d6):
    d6 = np.array(d6)
    if d6.ndim == 1: d6 = d6[np.newaxis, :]
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    matrix = np.stack((x, y, z), axis=2)
    if matrix.shape[0] == 1: return matrix[0]
    return matrix

def transform_pose_sequence_to_local(pose_seq_9d, T_world_to_local):
    # Version optimisée vectorisée (si possible) ou boucle standard
    # Ici on garde la logique mais on l'appelle une seule fois par item au début
    local_seq = []
    R_inv = T_world_to_local[:3, :3]
    t_inv = T_world_to_local[:3, 3]

    for step in pose_seq_9d:
        pos_world = step[:3]
        pos_local = R_inv @ pos_world + t_inv
        
        rot_6d_world = step[3:]
        R_world = rotation_6d_to_matrix(rot_6d_world)
        R_local = R_inv @ R_world
        rot_6d_local = rotation_matrix_to_ortho6d(R_local)
        
        local_seq.append(np.concatenate([pos_local, rot_6d_local]))
    return np.array(local_seq, dtype=np.float32)


class Robot3DDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 mode='train',
                 val_ratio=0.1,       
                 seed=42,             
                 num_points=1024, 
                 pred_horizon=16, 
                 obs_horizon=2, 
                 action_horizon=8,
                 augment=False):
        
        super().__init__()
        self.mode = mode
        self.num_points = num_points
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.seed = seed
        self.augment = augment and (mode == 'train')

        all_traj_folders = sorted(glob.glob(os.path.join(root_dir, 'Trajectory*')), 
                                  key=lambda x: int(x.split('_')[-1]))
        
        rng = random.Random(self.seed)
        rng.shuffle(all_traj_folders)
        
        n_total = len(all_traj_folders)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        
        if mode == 'train':
            self.traj_folders = all_traj_folders[:n_train]
        elif mode == 'val':
            self.traj_folders = all_traj_folders[n_train:]
        else:
            self.traj_folders = all_traj_folders

        print(f"📊 Dataset Mode: {mode.upper()} | Augment: {self.augment}")
        print("⏳ Prétraitement et mise en cache RAM (One-time cost)...")

        # ON STOCKE TOUT ICI : Plus de 'trajectory_cache' complexe, juste une liste plate d'échantillons prêts
        self.processed_data = [] 
        
        MOTION_THRESHOLD = 0.0001

        for folder in tqdm(self.traj_folders, desc="Processing Data"):
            folder_name = os.path.basename(folder)
            traj_id = folder_name.split('_')[-1]
            
            json_path = os.path.join(folder, f'trajectory_{traj_id}.json')
            if not os.path.exists(json_path): json_path = os.path.join(folder, f'trajectory{traj_id}.json')
            if not os.path.exists(json_path): continue

            with open(json_path, 'r') as f: data = json.load(f)
            states = data.get('states', [])
            if len(states) == 0: continue

            # 1. Chargement brut de toute la trajectoire
            full_traj_poses = []
            full_traj_pcd = []
            raw_positions = []

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
                    # On ne charge pas tout de suite le PCD pour économiser RAM temporaire
                    # On le chargera juste au moment du découpage
                    pcd_data = np.load(pcd_path).astype(np.float32)
                else:
                    pcd_data = np.zeros((self.num_points, 3), dtype=np.float32)

                full_traj_poses.append(pose_9d)
                full_traj_pcd.append(pcd_data)
                raw_positions.append(pos)

            # 2. Cutoff (Statique)
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

            if len(full_traj_poses) < (self.obs_horizon + self.pred_horizon):
                continue

            # 3. DÉCOUPAGE ET TRANSFORMATION (LE COEUR DE L'OPTIMISATION)
            # On boucle sur la fenêtre glissante ICI, pas dans __getitem__
            total_steps = len(full_traj_poses)
            
            for current_idx in range(self.obs_horizon - 1, total_steps - self.pred_horizon + 1):
                
                # A. Récupérer les indices
                obs_indices = list(range(current_idx - self.obs_horizon + 1, current_idx + 1))
                pred_indices = list(range(current_idx, current_idx + self.pred_horizon))

                # B. Récupérer PCD (Déjà Local)
                # On fait le sampling ici pour ne stocker que 1024 points (économie RAM)
                raw_pcd = full_traj_pcd[current_idx]
                sampled_pcd = self._sample_point_cloud(raw_pcd)

                # C. Récupérer Poses (World)
                obs_poses_world = np.stack([full_traj_poses[i] for i in obs_indices])
                action_seq_world = np.stack([full_traj_poses[i] for i in pred_indices])

                # D. CALCULER LA TRANSFORMATION (Une seule fois !)
                current_pose_9d = obs_poses_world[-1]
                curr_pos = current_pose_9d[:3]
                curr_rot_6d = current_pose_9d[3:]
                
                R_curr = rotation_6d_to_matrix(curr_rot_6d)
                T_world_to_curr = np.eye(4)
                T_world_to_curr[:3, :3] = R_curr
                T_world_to_curr[:3, 3] = curr_pos
                T_world_to_local = np.linalg.inv(T_world_to_curr)

                # E. APPLIQUER LA TRANSFORMATION
                obs_poses_local = transform_pose_sequence_to_local(obs_poses_world, T_world_to_local)
                action_seq_local = transform_pose_sequence_to_local(action_seq_world, T_world_to_local)

                # F. STOCKER LE RÉSULTAT FINAL PRÊT À L'EMPLOI
                self.processed_data.append({
                    'pcd': sampled_pcd,          # (1024, 3)
                    'obs': obs_poses_local,      # (2, 9)
                    'act': action_seq_local      # (16, 9)
                })

        print(f"✅ {len(self.processed_data)} échantillons pré-calculés en RAM.\n")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]
        
        pcd = sample['pcd']
        obs = sample['obs']
        act = sample['act']

        # Seule l'augmentation reste ici (car elle doit être aléatoire à chaque époque)
        if self.augment:
             # Copie pour ne pas modifier la donnée en cache
             pcd = pcd.copy()
             pcd, obs, act = self._apply_augmentation(pcd, obs, act)
        
        data_dict = {
            'obs': {
                'point_cloud': torch.from_numpy(pcd).float(),  # (N, 3)
                'agent_pos': torch.from_numpy(obs).float(),      # (obs_horizon, 9)
            },
            'action': torch.from_numpy(act).float()  # (pred_horizon, 9)
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
        
        # Random sampling (you could use FPS for better coverage)
        indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
        return pc[indices].astype(np.float32)

    def _apply_augmentation(self, pcd, obs_poses, action_seq):
        """ =====================================================
        1. Random rotation around Z-axis (±15 degrees)
        =====================================================
        For a tabletop task, gravity (z) is fixed, but the 
        plate/food orientation around z can vary.
        
        The rotation matrix R_z(θ) is:
          [cos θ  -sin θ  0]
          [sin θ   cos θ  0]
          [  0       0    1]
        
        We apply it to:
          - point cloud positions (N, 3)
          - pose positions (dims 0:3)
          - pose 6D rotation (dims 3:9), which are the first two 
            columns of the 3x3 rotation matrix. If original is M,
            augmented is R @ M, so each column gets multiplied by R.
        """
        theta = np.random.uniform(-np.pi/12, np.pi/12)  # ±15°
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]], dtype=np.float32)
        
        # Rotate point cloud: (N, 3) @ R_z.T = (N, 3)
        pcd = pcd @ R_z.T
        
        # Rotate poses (obs and actions)
        for poses in [obs_poses, action_seq]:
            # Position: dims 0:3
            poses[:, :3] = (poses[:, :3] @ R_z.T)
            # 6D rotation: dims 3:6 = column 1, dims 6:9 = column 2
            poses[:, 3:6] = (poses[:, 3:6] @ R_z.T)
            poses[:, 6:9] = (poses[:, 6:9] @ R_z.T)
        
        # =====================================================
        # 2. Global translation shift (±1cm)
        # =====================================================
        shift = np.random.uniform(low=-0.01, high=0.01, size=(1, 3)).astype(np.float32)
        
        pcd = pcd + shift
        obs_poses[:, :3] += shift
        action_seq[:, :3] += shift

        # =====================================================
        # 3. Point cloud jitter (±5mm sensor noise)
        # =====================================================
        noise = np.random.randn(*pcd.shape).astype(np.float32) * 0.005
        noise = np.clip(noise, -0.01, 0.01)
        pcd = pcd + noise

        return pcd, obs_poses, action_seq

    def get_normalizer(self):
        """
        Compute normalizer statistics from the dataset.
        This is called once before training to set up normalization.
        
        Returns a LinearNormalizer that handles:
        - obs['point_cloud']: normalized point clouds
        - obs['agent_pos']: normalized robot states  
        - action: normalized actions
        """
        from normalizer import LinearNormalizer
        
        # Collect all data for statistics
        all_point_clouds = []
        all_agent_pos = []
        all_actions = []
        
        print("📊 Computing normalization statistics...")
        for idx in range(min(len(self), 5000)):  # Sample up to 5000 for efficiency
            sample = self[idx]
            all_point_clouds.append(sample['obs']['point_cloud'])
            all_agent_pos.append(sample['obs']['agent_pos'])
            all_actions.append(sample['action'])
        
        # Stack into tensors
        all_point_clouds = torch.stack(all_point_clouds)  # (N, num_points, 3)
        all_agent_pos = torch.stack(all_agent_pos)        # (N, obs_horizon, 9)
        all_actions = torch.stack(all_actions)            # (N, pred_horizon, 9)
        
        # Create normalizer
        normalizer = LinearNormalizer()
        
        # Fit on data
        # For point clouds: normalize per-dimension (x, y, z)
        normalizer.fit(
            data={
                'point_cloud': all_point_clouds,
                'agent_pos': all_agent_pos,
                'action': all_actions
            },
            last_n_dims=1,  # Normalize last dimension (features)
            mode='limits',  # Scale to [-1, 1]
            output_max=1.0,
            output_min=-1.0
        )
        
        print("✅ Normalizer fitted!")
        return normalizer

    def get_validation_dataset(self):
        """Return validation split of the dataset."""
        return Robot3DDataset(
            root_dir=self.root_dir,
            mode='val',
            val_ratio=0.1,
            seed=self.seed,
            num_points=self.num_points,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            augment=False  # No augmentation for validation
        )


# Custom collate function for nested dictionaries
def custom_collate_fn(batch):
    """
    Collate function that handles nested dictionaries.
    """
    result = {}
    
    # Handle 'obs' dictionary
    obs_keys = batch[0]['obs'].keys()
    result['obs'] = {}
    for key in obs_keys:
        result['obs'][key] = torch.stack([item['obs'][key] for item in batch])
    
    # Handle 'action'
    result['action'] = torch.stack([item['action'] for item in batch])
    
    return result


# --- Test ---
if __name__ == "__main__":
    import rospkg
    rospack = rospkg.RosPack()
    try:
        pkg_path = rospack.get_path('vision_processing')
        data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
        
        print(f"Testing dataset loading from: {data_path}")
        dataset = Robot3DDataset(root_dir=data_path, pred_horizon=16, obs_horizon=2)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample structure:")
            print(f"  obs['point_cloud'] shape: {sample['obs']['point_cloud'].shape}")
            print(f"  obs['agent_pos'] shape: {sample['obs']['agent_pos'].shape}")
            print(f"  action shape: {sample['action'].shape}")
            
    except Exception as e:
        print(f"Error during test: {e}")