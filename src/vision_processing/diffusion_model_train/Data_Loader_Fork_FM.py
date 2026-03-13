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

# --- ROTATION UTILITIES ---
def rotation_matrix_to_ortho6d(matrices):
    x_raw = matrices[..., 0]
    y_raw = matrices[..., 1]
    return np.concatenate([x_raw, y_raw], axis=-1)


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

                full_traj_poses.append(pose_9d)
                full_traj_pcd.append(pcd_data)
                raw_positions.append(pos)

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

            if len(full_traj_poses) < (self.obs_horizon + self.pred_horizon):
                continue

            # 3. Sliding window: precompute every sample
            total_steps = len(full_traj_poses)
            
            for current_idx in range(self.obs_horizon - 1, total_steps - self.pred_horizon + 1):
                
                # A. Indices
                obs_indices = list(range(current_idx - self.obs_horizon + 1, current_idx + 1))
                pred_indices = list(range(current_idx, current_idx + self.pred_horizon))

                # B. Point cloud (already in local frame from preprocessing)
                raw_pcd = full_traj_pcd[current_idx]
                sampled_pcd = self._sample_point_cloud(raw_pcd)

                # C. Poses (world frame)
                obs_poses = np.stack([full_traj_poses[i] for i in obs_indices])
                action_seq = np.stack([full_traj_poses[i] for i in pred_indices])

                # D. TRANSLATION-ONLY CENTERING
                # Subtract current fork-tip position from all positions.
                # Rotations (dims 3:9) stay in WORLD frame — untouched.
                #
                # After this:
                #   obs[-1, :3]     = [0, 0, 0]   (current pos is origin)
                #   obs[-2, :3]     = relative displacement (velocity info)
                #   action[:, :3]   = relative displacements from current pos
                #   obs[:, 3:9]     = world-frame rotations (VARYING signal)
                #   action[:, 3:9]  = world-frame rotations (VARYING signal)
                curr_pos = obs_poses[-1, :3].copy()

                obs_poses_centered = obs_poses.copy()
                obs_poses_centered[:, :3] -= curr_pos

                action_seq_centered = action_seq.copy()
                action_seq_centered[:, :3] -= curr_pos

                # E. Store ready-to-use sample
                self.processed_data.append({
                    'pcd': sampled_pcd,              # (num_points, 3)
                    'obs': obs_poses_centered,       # (obs_horizon, 9)
                    'act': action_seq_centered,      # (pred_horizon, 9)
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
            # if random.random() < 0.5:
            #     # On met à zéro les colonnes 3 à 9 (les rotations 6D)
            #     obs[:, 3:] = 0.0
        
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
        Augmentation for translation-centered data.
        
        Since rotations are in WORLD frame, we can only augment with
        rotations around Z (gravity axis) — physically valid because
        "rotating the whole scene around gravity" is a valid
        transformation for a tabletop task.
        """
        # 1. Random Z-rotation (±15°)
        theta = np.random.uniform(-np.pi/12, np.pi/12)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]], dtype=np.float32)
        
        pcd = pcd @ R_z.T
        
        for poses in [obs_poses, action_seq]:
            poses[:, :3]  = poses[:, :3]  @ R_z.T   # Positions
            poses[:, 3:6] = poses[:, 3:6] @ R_z.T   # Rot col 1
            poses[:, 6:9] = poses[:, 6:9] @ R_z.T   # Rot col 2
        
        # 2. Translation jitter (±1cm)
        shift = np.random.uniform(-0.01, 0.01, size=(1, 3)).astype(np.float32)
        pcd += shift
        obs_poses[:, :3] += shift
        action_seq[:, :3] += shift

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