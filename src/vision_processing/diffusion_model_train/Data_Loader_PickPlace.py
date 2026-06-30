"""
Data Loader PICK-AND-PLACE — Flow Matching, Translation-Centered (10D)
=====================================================================
Variante de `Data_Loader_Fork_FlowMP_9D.py` pour les donnees PICK-AND-PLACE
(`datas/PickAndPlace_preprocess`, sortie de `Data_preprocess_PickPlace.py`).

Le format de sortie du preprocess est IDENTIQUE a celui de la fourchette :
  * states[].fork_tip_position / fork_tip_orientation + gripper_open
    -> action 10D pose+gripper (le "fork tip" EST le TCP de la pince ici).
  * states[].Merged_Fork_point_cloud -> Merged_Fork_<folder>/Merged_Fork_XXXX.npy
    nuage 1024x3 deja centre en translation sur le TCP (orientation monde gardee),
    contenant pince + cube (cible a saisir) + etagere (cible de depot).

=> Toute la logique (fenetre glissante, centrage translation, augmentation) est
   reprise telle quelle. SEULE difference : le split train/val.

Split : la version fourchette codait en dur des plages d'ID (1-45 reel, 46-64 sim)
et IGNORAIT tout ID hors 1-64. Le pick-and-place est 100% simulation et ses IDs
ne suivent pas ces plages -> on fait un simple split aleatoire stratifie sur
TOUTES les trajectoires (aucune trajectoire n'est silencieusement ignoree).
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

GRIPPER_OPEN_THRESHOLD = 0.03


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


def get_gripper_open_label(state):
    """Binary gripper label: 1.0=open, 0.0=closed."""
    if state.get('gripper_open') is not None:
        return float(state['gripper_open'])

    cmd = state.get('gripper_command')
    if cmd is not None:
        return 1.0 if float(cmd) > 0.0 else 0.0

    qpos = state.get('joint_positions', [])
    if len(qpos) >= 9:
        opening = 0.5 * (float(qpos[7]) + float(qpos[8]))
        return 1.0 if opening >= GRIPPER_OPEN_THRESHOLD else 0.0

    return 1.0


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
                 data_source='all'):  # conserve pour compat (ignore : tout est sim)

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

        # --- Split train/val aleatoire sur TOUTES les trajectoires --------- #
        # (pas de distinction reel/sim : tout le pick-and-place est simule)
        rng = random.Random(self.seed)
        folders = list(all_traj_folders)
        rng.shuffle(folders)

        n_val = int(len(folders) * val_ratio)
        n_train = len(folders) - n_val

        if mode == 'train':
            self.traj_folders = folders[:n_train]
        elif mode == 'val':
            self.traj_folders = folders[n_train:]
        else:  # 'test' / inference : tout
            self.traj_folders = folders

        print(f"📊 Dataset Mode: {mode.upper():<5} | Augment: {self.augment}")
        print(f"📁 Dossiers retenus : {len(self.traj_folders)} / {len(all_traj_folders)}")
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
                gripper_open = np.array([get_gripper_open_label(state)], dtype=np.float32)
                pose_10d = np.concatenate([pos, rot_6d, gripper_open]).astype(np.float32)

                pcd_name = state.get('Merged_Fork_point_cloud')
                pcd_path = os.path.join(folder, f'Merged_Fork_{folder_name}', pcd_name) if pcd_name else None

                if pcd_path and os.path.exists(pcd_path):
                    pcd_data = np.load(pcd_path).astype(np.float32)
                else:
                    pcd_data = np.zeros((self.num_points, 3), dtype=np.float32)

                time_step = state["time_step"]

                full_traj_poses.append(pose_10d)
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

                # D. Action a 10 dimensions: pose 9D + gripper_open
                action_full = action_seq_centered

                # E. Store ready-to-use sample
                self.processed_data.append({
                    'pcd': sampled_pcd,
                    'obs': obs_poses_centered,       # (obs_horizon, 10)
                    'act': action_full,              # (pred_horizon, 10)
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
        """Augmentation pour les tenseurs d'action 10D.

        La rotation Z autour du TCP tourne nuage ET poses de facon coherente : la
        geometrie relative (pince/cube/etagere) est preservee, c'est une simple
        rotation de toute la scene autour de la verticale. Le gripper_open reste
        inchange.
        """
        # 1. Random Z-rotation (±15°)
        theta = np.random.uniform(-np.pi / 12, np.pi / 12)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]], dtype=np.float32)

        pcd = pcd @ R_z.T

        # Rotation de l'observation (pose 9D seulement)
        obs_poses[:, :3]  = obs_poses[:, :3]  @ R_z.T
        obs_poses[:, 3:6] = obs_poses[:, 3:6] @ R_z.T
        obs_poses[:, 6:9] = obs_poses[:, 6:9] @ R_z.T

        # Rotation de l'action (pose 9D seulement) par blocs de 3
        for i in range(0, 9, 3):
            action_seq[:, i:i + 3] = action_seq[:, i:i + 3] @ R_z.T

        # 2. Translation jitter (±1cm) — n'affecte QUE les positions
        shift = np.random.uniform(-0.01, 0.01, size=(1, 3)).astype(np.float32)
        pcd += shift
        obs_poses[:, :3] += shift
        action_seq[:, :3] += shift

        # 3. PCD sensor noise (±5mm)
        noise = np.clip(np.random.randn(*pcd.shape).astype(np.float32) * 0.005, -0.01, 0.01)
        pcd += noise

        return pcd, obs_poses, action_seq


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
        data_path = os.path.join(pkg_path, 'datas', 'PickAndPlace_preprocess')

        print(f"Testing: {data_path}")
        dataset = Robot3DDataset(root_dir=data_path, pred_horizon=16, obs_horizon=2)
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample structure:")
            print(f"  obs['point_cloud']: {sample['obs']['point_cloud'].shape}")
            print(f"  obs['agent_pos']:   {sample['obs']['agent_pos'].shape}")
            print(f"  action:             {sample['action'].shape}")

            curr_pos = sample['obs']['agent_pos'][-1, :3]
            print(f"\n  Current position (should be ~0): {curr_pos.numpy()}")

            curr_rot = sample['obs']['agent_pos'][-1, 3:9]
            print(f"  Current rotation 6D (world-frame, not identity): {curr_rot.numpy()}")
            print(f"  Current gripper_open: {sample['obs']['agent_pos'][-1, 9].item():.1f}")

    except Exception as e:
        print(f"Error: {e}")
