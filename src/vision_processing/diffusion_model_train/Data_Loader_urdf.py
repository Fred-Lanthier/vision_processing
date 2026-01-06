import os
import glob
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import fpsample

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# --- UTILITAIRES MATHÉMATIQUES ROTATION 6D ---
def rotation_matrix_to_ortho6d(matrices):
    """ Convertit Matrice 3x3 -> Représentation 6D continue """
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
                 augment=False): # Flag d'augmentation
        
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.num_points = num_points
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        # On n'augmente QUE pendant l'entrainement
        self.augment = augment and (mode == 'train')

        # 1. Listing des fichiers
        all_traj_folders = sorted(glob.glob(os.path.join(root_dir, 'Trajectory*')), 
                                  key=lambda x: int(x.split('_')[-1]))
        
        # 2. Split Train/Val
        rng = random.Random(seed)
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
        
        # Seuil pour considérer que le robot a fini de bouger (1mm)
        MOTION_THRESHOLD = 0.00001

        for folder in self.traj_folders:
            folder_name = os.path.basename(folder)
            traj_id = folder_name.split('_')[-1]
            
            # Gestion des noms de fichiers
            json_path = os.path.join(folder, f'trajectory_{traj_id}.json')
            if not os.path.exists(json_path):
                 json_path = os.path.join(folder, f'trajectory{traj_id}.json')
            if not os.path.exists(json_path): continue

            with open(json_path, 'r') as f:
                data = json.load(f)
            
            states = data.get('states', [])
            if len(states) == 0: continue

            # --- A. PARSING & CONVERSION 6D ---
            parsed_traj = []
            raw_positions = [] # Pour le filtrage statique

            for state in states:
                pos = np.array(state['end_effector_position'], dtype=np.float32)
                quat = state['end_effector_orientation'] # [x, y, z, w]
                
                # Conversion Quat -> Matrice -> 6D
                rot = R.from_quat(quat)
                mat = rot.as_matrix()
                rot_6d = rotation_matrix_to_ortho6d(mat) # (6,)
                
                # Pose finale 9D = [x, y, z, r1...r6]
                pose_9d = np.concatenate([pos, rot_6d]).astype(np.float32)
                
                pcd_name = state.get('Merged_urdf_point_cloud')
                pcd_path = os.path.join(folder, f'Merged_urdf_{folder_name}', pcd_name) if pcd_name else None

                parsed_traj.append({
                    'pose_9d': pose_9d,
                    'pcd_path': pcd_path
                })
                raw_positions.append(pos)

            # --- B. FILTRAGE STATIQUE (Couper la fin) ---
            raw_positions = np.array(raw_positions)
            final_pos = raw_positions[-1]
            # Distance de chaque point par rapport à la fin
            dist_to_end = np.linalg.norm(raw_positions - final_pos, axis=1)
            
            # On remonte du futur vers le passé. Dès qu'on bouge > 1mm, c'est la fin du mouvement utile.
            cutoff_idx = len(parsed_traj)
            for i in range(len(parsed_traj) - 1, 0, -1):
                if dist_to_end[i] > MOTION_THRESHOLD:
                    # On garde un buffer de 5 frames pour la décélération
                    cutoff_idx = min(i + 5, len(parsed_traj))
                    break
            
            # On coupe !
            parsed_traj = parsed_traj[:cutoff_idx]
            
            # Si trajectoire devenue trop courte, on jette
            if len(parsed_traj) < (self.obs_horizon + self.pred_horizon):
                continue

            self.trajectory_cache[folder_name] = parsed_traj
            
            # Indexation
            total_steps = len(parsed_traj)
            for i in range(self.obs_horizon - 1, total_steps - self.pred_horizon + 1):
                self.indices.append((folder_name, i))

        print(f"✅ {len(self.indices)} séquences indexées (Nettoyées & Converties).\n")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_name, current_idx = self.indices[idx]
        traj_data = self.trajectory_cache[traj_name]

        obs_indices = list(range(current_idx - self.obs_horizon + 1, current_idx + 1))
        pred_indices = list(range(current_idx, current_idx + self.pred_horizon))

        # --- LOAD PCD ---
        curr_data = traj_data[current_idx]
        pcd_path = curr_data['pcd_path']
        
        if pcd_path and os.path.exists(pcd_path):
            point_cloud = np.load(pcd_path)
        else:
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)

        point_cloud = self._sample_point_cloud(point_cloud)

        # --- LOAD POSES ---
        obs_poses = np.stack([traj_data[i]['pose_9d'] for i in obs_indices])
        action_seq = np.stack([traj_data[i]['pose_9d'] for i in pred_indices])

        # --- C. DATA AUGMENTATION (Train Only) ---
        if self.augment:
            point_cloud, obs_poses, action_seq = self._apply_augmentation(point_cloud, obs_poses, action_seq)

        data_dict = {
            'point_cloud': torch.from_numpy(point_cloud).float(),
            'agent_pos': torch.from_numpy(obs_poses).float(),
            'action': torch.from_numpy(action_seq).float()
        }
        
        return data_dict

    def _sample_point_cloud(self, pc):
        """
        Échantillonnage optimisé utilisant Farthest Point Sampling (FPS).
        Garantit une couverture uniforme de l'espace.
        """
        # Sécurité pour les nuages de points vides
        if pc.shape[0] == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        # Cas où on a moins de points que demandé
        # if pc.shape[0] <= self.num_points:
        #     indices = np.arange(pc.shape[0])
        #     if pc.shape[0] < self.num_points:
        #         # Padding par répétition si nécessaire
        #         extra_indices = np.random.choice(pc.shape[0], self.num_points - pc.shape[0], replace=True)
        #         indices = np.concatenate([indices, extra_indices])
        #     return pc[indices].astype(np.float32)

        # --- UTILISATION DE FPSAMPLE ---
        # bucket_fps_indices est l'algorithme le plus rapide pour les grands PCD
        # pc doit être un array numpy de forme (N, 3)
        # indices = fpsample.bucket_fps_kdline_sampling(pc.astype(np.float32), self.num_points, h=5)
        
        return pc[:self.num_points].astype(np.float32)

    def _apply_augmentation(self, pcd, obs_poses, action_seq):
        """
        Applique un Shift aléatoire et un Bruit sur le PCD.
        Robuste aux erreurs de calibration et force l'apprentissage relatif.
        """
        # 1. Shift Global (+/- 2cm)
        # On déplace tout le monde (Robot + Env) ensemble
        shift = np.random.uniform(low=-0.05, high=0.05, size=(1, 3))
        
        pcd = pcd + shift
        obs_poses[:, :3] += shift
        action_seq[:, :3] += shift # Ne pas oublier de shifter l'action future !

        # 2. Jitter PCD (+/- 5mm)
        # Bruit uniquement sur la perception pour robustesse capteur
        noise = np.random.randn(*pcd.shape) * 0.005
        noise = np.clip(noise, -0.01, 0.01)
        pcd = pcd + noise

        return pcd, obs_poses, action_seq

# --- Exemple d'utilisation (Debug) ---
if __name__ == "__main__":
    import rospkg
    rospack = rospkg.RosPack()
    try:
        pkg_path = rospack.get_path('vision_processing')
        data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
        
        # Test
        print(f"Testing dataset loading from: {data_path}")
        dataset = Robot3DDataset(root_dir=data_path, pred_horizon=16, obs_horizon=2)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample keys:", sample.keys())
            print("PC shape:", sample['point_cloud'].shape)   # Attendu: (1024, 3)
            print("Agent Obs shape:", sample['agent_pos'].shape) # Attendu: (2, 9)
            print("Action shape:", sample['action'].shape)       # Attendu: (16, 9)
    except Exception as e:
        print(f"Error during test: {e}")