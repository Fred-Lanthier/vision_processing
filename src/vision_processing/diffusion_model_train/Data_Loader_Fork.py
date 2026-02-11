"""
Data Loader for Flow Matching — Translation-Centered Version
=============================================================

Key change from previous version:
  - OLD: Full SE(3) transform to fork-tip local frame
         → Observation frozen after grasp (tool+food constant in own frame)
  - NEW: Translation-only centering at fork tip position
         → Positions centered at (0,0,0), world orientation PRESERVED
         → Model sees orientation changes throughout trajectory

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
    """Convert 3x3 rotation matrix to 6D continuous representation"""
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
        self.root_dir = root_dir
        self.mode = mode
        self.num_points = num_points
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.seed = seed
        self.augment = augment and (mode == 'train')

        # 1. List trajectory folders
        all_traj_folders = sorted(glob.glob(os.path.join(root_dir, 'Trajectory*')), 
                                  key=lambda x: int(x.split('_')[-1]))
        
        # 2. Train/Val split
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
        
        self.indices = []
        self.trajectory_cache = {} 
        
        MOTION_THRESHOLD = 0.0001

        for folder in self.traj_folders:
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

            # Parse trajectory
            parsed_traj = []
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

                parsed_traj.append({
                    'pose_9d': pose_9d,
                    'pcd_path': pcd_path
                })
                raw_positions.append(pos)

            # Filter static end
            raw_positions = np.array(raw_positions)
            final_pos = raw_positions[-1]
            dist_to_end = np.linalg.norm(raw_positions - final_pos, axis=1)
            
            cutoff_idx = len(parsed_traj)
            for i in range(len(parsed_traj) - 1, 0, -1):
                if dist_to_end[i] > MOTION_THRESHOLD:
                    cutoff_idx = min(i + 5, len(parsed_traj))
                    break
            
            parsed_traj = parsed_traj[:cutoff_idx]
            
            if len(parsed_traj) < (self.obs_horizon + self.pred_horizon):
                continue

            self.trajectory_cache[folder_name] = parsed_traj
            
            total_steps = len(parsed_traj)
            for i in range(self.obs_horizon - 1, total_steps - self.pred_horizon + 1):
                self.indices.append((folder_name, i))

        print(f"✅ {len(self.indices)} sequences indexed.\n")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_name, current_idx = self.indices[idx]
        traj_data = self.trajectory_cache[traj_name]

        obs_indices = list(range(current_idx - self.obs_horizon + 1, current_idx + 1))
        pred_indices = list(range(current_idx, current_idx + self.pred_horizon))

        # Load point cloud (already translation-centered from preprocessing)
        curr_data = traj_data[current_idx]
        pcd_path = curr_data['pcd_path']
        
        if pcd_path and os.path.exists(pcd_path):
            point_cloud = np.load(pcd_path)
        else:
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)

        point_cloud = self._sample_point_cloud(point_cloud)

        # Load poses (world frame)
        obs_poses = np.stack([traj_data[i]['pose_9d'] for i in obs_indices])
        action_seq = np.stack([traj_data[i]['pose_9d'] for i in pred_indices])

        # =====================================================
        # TRANSLATION-ONLY CENTERING
        # =====================================================
        # Subtract current fork tip position from all positions.
        # Rotations (dims 3:9) stay in WORLD frame — untouched.
        #
        # After this:
        #   obs[-1][:3]  = [0, 0, 0]  (current position is origin)
        #   obs[-2][:3]  = relative displacement (velocity info)
        #   action[:, :3] = relative displacements from current pos
        #   obs[:, 3:9]  = world-frame rotations (VARYING signal!)
        #   action[:, 3:9] = world-frame rotations (VARYING signal!)
        
        curr_pos = obs_poses[-1, :3].copy()  # Current fork tip position
        
        obs_poses_centered = obs_poses.copy()
        obs_poses_centered[:, :3] -= curr_pos
        
        action_seq_centered = action_seq.copy()
        action_seq_centered[:, :3] -= curr_pos

        # Data augmentation
        if self.augment:
            point_cloud, obs_poses_centered, action_seq_centered = self._apply_augmentation(
                point_cloud, obs_poses_centered, action_seq_centered
            )

        data_dict = {
            'obs': {
                'point_cloud': torch.from_numpy(point_cloud).float(),           # (N, 3)
                'agent_pos': torch.from_numpy(obs_poses_centered).float(),      # (obs_horizon, 9)
            },
            'action': torch.from_numpy(action_seq_centered).float()             # (pred_horizon, 9)
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
        rotations around Z (gravity axis) — this is physically valid 
        because "rotating the whole scene around gravity" is a valid 
        transformation for a tabletop task.
        
        We apply:
          1. Random Z-rotation (±15°) to positions AND rotations AND PCD
          2. Small translation jitter (±1cm)  
          3. PCD noise (±5mm)
        """
        # 1. Random Z-rotation
        theta = np.random.uniform(-np.pi/12, np.pi/12)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]], dtype=np.float32)
        
        # PCD
        pcd = pcd @ R_z.T
        
        # Poses: positions AND rotations (both are in world frame)
        for poses in [obs_poses, action_seq]:
            poses[:, :3] = poses[:, :3] @ R_z.T      # Positions
            poses[:, 3:6] = poses[:, 3:6] @ R_z.T    # Rot col 1
            poses[:, 6:9] = poses[:, 6:9] @ R_z.T    # Rot col 2
        
        # 2. Translation jitter
        shift = np.random.uniform(-0.01, 0.01, size=(1, 3)).astype(np.float32)
        pcd = pcd + shift
        obs_poses[:, :3] += shift
        action_seq[:, :3] += shift

        # 3. PCD sensor noise
        noise = np.clip(np.random.randn(*pcd.shape).astype(np.float32) * 0.005, -0.01, 0.01)
        pcd = pcd + noise

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

    def get_validation_dataset(self):
        return Robot3DDataset(
            root_dir=self.root_dir,
            mode='val',
            val_ratio=0.1,
            seed=self.seed,
            num_points=self.num_points,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            augment=False
        )


def custom_collate_fn(batch):
    result = {}
    obs_keys = batch[0]['obs'].keys()
    result['obs'] = {}
    for key in obs_keys:
        result['obs'][key] = torch.stack([item['obs'][key] for item in batch])
    result['action'] = torch.stack([item['action'] for item in batch])
    return result


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