#!/usr/bin/env python3
"""
Flow Matching Inference Node for FORK — SafeFlow Edition
=========================================================

Extends the Translation-Centered FM node with:
  1. Fork safety points (convex hull + FPS from fork.STL)
  2. SDF-based FMBF correction during ODE integration
  3. Terminal safety projection after ODE

SafeFlow integration (Approach 2 + Terminal Projection):
─────────────────────────────────────────────────────────
  During ODE (for t >= t_star):
    - Denormalize trajectory to state space
    - For each waypoint, transform 5 representative fork points to container frame
    - Query SDF, find worst violation
    - Compute closed-form QP correction (SafeFlow eq.16)
    - Normalize correction back, add to velocity

  After ODE (terminal projection):
    - Check 50 representative fork points at each waypoint
    - Project any remaining violations onto the safe boundary

Frame Chain:
    Container Frame (SDF lives here)
        ↕  T_world_container
    World Frame
        ↕  translation by fork_pos
    Fork-Centered Frame (model predicts here)
        ↕  normalizer (affine: mean/std)
    Normalized Space (ODE integration)
"""
import rospy
import numpy as np
import torch
import torch.nn.functional as F
import collections
import os
import sys
import rospkg
import tf
import tf.transformations as tft
import trimesh
from scipy.spatial import ConvexHull
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path

from utils import compute_T_child_parent_xacro

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

try:
    from Train_Fork_FM import FlowMatchingAgent, Normalizer
except ImportError as e:
    rospy.logerr(f"Import Error: {e}")
    sys.exit(1)


# ==============================================================
# MATH UTILITIES
# ==============================================================

def rotation_matrix_to_ortho6d(matrix):
    """Extract first two columns of a rotation matrix as 6D representation."""
    return np.concatenate([matrix[..., 0], matrix[..., 1]], axis=-1)


def ortho6d_to_rotation_matrix(d6):
    """Convert 6D rotation representation back to a 3x3 rotation matrix.
    Uses Gram-Schmidt to ensure orthogonality."""
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def gram_schmidt_6d(col1, col2):
    """Orthonormalize two 3D vectors using Gram-Schmidt.
    Returns normalized col1 and col2 orthogonal to col1."""
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = col2 - np.dot(col2, col1) * col1
    col2 = col2 / (np.linalg.norm(col2) + 1e-8)
    return col1, col2


# ==============================================================
# FARTHEST POINT SAMPLING
# ==============================================================

def farthest_point_sampling(points, n_samples):
    """Greedy FPS: iteratively pick the point farthest from all selected points.
    This maximizes the minimum inter-point distance → best spatial coverage.

    Args:
        points:    (N, 3) candidate points
        n_samples: how many to select
    Returns:
        indices: (n_samples,) into `points`
    """
    N = points.shape[0]
    n_samples = min(n_samples, N)

    centroid = points.mean(axis=0)
    first_idx = np.argmax(np.linalg.norm(points - centroid, axis=1))

    selected = [first_idx]
    min_dists = np.linalg.norm(points - points[first_idx], axis=1)

    for _ in range(n_samples - 1):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        new_dists = np.linalg.norm(points - points[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return np.array(selected)


# ==============================================================
# SDF LOADER (lightweight — only needs query + load from .npz)
# ==============================================================

class SDFTensorField(torch.nn.Module):
    def __init__(self, npz_path, device):
        super().__init__()
        data = np.load(npz_path)
        
        # Le SDF doit être de forme (1, 1, D, H, W) pour grid_sample.
        sdf_grid = torch.from_numpy(data['sdf_grid'].astype(np.float32)).to(device)
        self.sdf_tensor = sdf_grid.unsqueeze(0).unsqueeze(0) 
        
        self.bounds_min = torch.from_numpy(data['bounds_min']).float().to(device)
        self.bounds_max = torch.from_numpy(data['bounds_max']).float().to(device)
        self.voxel_size = float(data['voxel_size'])
        
        # FIX : Extraction de l'attribut manquant pour l'affichage
        if 'n_voxels' in data:
            self.n_voxels = data['n_voxels']
        else:
            # Fallback automatique si le fichier npz est plus ancien
            self.n_voxels = sdf_grid.shape 

    def forward(self, points_container):
        """
        points_container: (Batch, N_points, 3)
        Retourne les valeurs SDF: (Batch, N_points)
        """
        center = (self.bounds_max + self.bounds_min) / 2.0
        half_extent = (self.bounds_max - self.bounds_min) / 2.0
        
        # Normalisation [-1, 1]
        grid_coords = (points_container - center) / half_extent
        
        # FIX MAJEUR : Inversion des axes pour grid_sample (X, Y, Z) -> (Z, Y, X)
        # grid_sample map le 1er élément à W (nz), le 2ème à H (ny), et le 3ème à D (nx)
        grid_coords = torch.stack([
            grid_coords[..., 2],  # Le Z physique indexera la profondeur (nz)
            grid_coords[..., 1],  # Le Y physique indexera la hauteur (ny)
            grid_coords[..., 0]   # Le X physique indexera la largeur (nx)
        ], dim=-1)
        
        # Reshape pour grid_sample: (Batch, 1, 1, N_points, 3)
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(1)
        
        # grid_sample (trilinéaire, rapide, différentiable)
        sampled = F.grid_sample(
            self.sdf_tensor.expand(points_container.shape[0], -1, -1, -1, -1),
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        return sampled.squeeze(1).squeeze(1).squeeze(1) # (Batch, N_points)


# ==============================================================
# FORK SAFETY POINTS PRECOMPUTATION
# ==============================================================

def precompute_fork_safety_points(stl_path, scale, visual_origin_transform,
                                  n_fast=5, n_full=50):
    """Load the fork STL, apply transforms, extract safety points via FPS.

    The fork tip origin [0,0,0] is always included in the fast set.

    Args:
        stl_path:   path to fork.STL
        scale:      [sx, sy, sz] from xacro (typically [0.001, 0.001, 0.001])
        visual_origin_transform: 4x4 matrix from xacro visual origin
        n_fast:     points for ODE loop (convex hull + FPS)
        n_full:     points for terminal check (all vertices + FPS)

    Returns:
        fork_points_fast: (n_fast+1, 3) — includes fork tip at index 0
        fork_points_full: (n_full, 3)
    """
    mesh = trimesh.load(stl_path, force='mesh')
    mesh.apply_scale(scale)
    mesh.apply_transform(visual_origin_transform)

    vertices = mesh.vertices.copy()  # (N, 3) in fork_tip frame

    # --- Convex hull → FPS for fast set ---
    hull = ConvexHull(vertices)
    hull_vertices = vertices[np.unique(hull.simplices.flatten())]
    fps_hull_idx = farthest_point_sampling(hull_vertices, n_fast)
    fps_points = hull_vertices[fps_hull_idx]

    # Always include the fork tip origin as the first point.
    # If FPS already picked a point very close to [0,0,0], replace it;
    # otherwise prepend it.
    fork_tip = np.array([[0.0, 0.0, 0.0]])
    dists_to_origin = np.linalg.norm(fps_points, axis=1)
    if np.min(dists_to_origin) < 0.005:
        # A point is already within 5mm of origin — replace it
        closest_idx = np.argmin(dists_to_origin)
        fps_points[closest_idx] = fork_tip[0]
        fork_points_fast = fps_points
    else:
        # Prepend origin
        fork_points_fast = np.vstack([fork_tip, fps_points])

    # --- FPS on all vertices for full set ---
    fps_all_idx = farthest_point_sampling(vertices, n_full)
    fork_points_full = vertices[fps_all_idx]

    # Also ensure fork tip is in the full set
    dists_to_origin_full = np.linalg.norm(fork_points_full, axis=1)
    if np.min(dists_to_origin_full) > 0.005:
        fork_points_full = np.vstack([fork_tip, fork_points_full[:n_full - 1]])

    return fork_points_fast.astype(np.float32), fork_points_full.astype(np.float32)


# ==============================================================
# FMBF SAFETY FILTER
# ==============================================================

class FMBFSafetyFilterPyTorch:
    def __init__(self, sdf_tensor_field, T_container_world, fork_points_fast, fork_points_full, action_mean, action_std, phi_0, t_star, max_terminal_shift, device):
        self.sdf = sdf_tensor_field
        self.device = device
        self.t_star = t_star
        self.phi_0 = phi_0
        self.max_terminal_shift = max_terminal_shift
        
        # Paramètres de normalisation
        self.mean = torch.tensor(action_mean, device=device).float()
        self.std = torch.tensor(action_std, device=device).float()
        
        # Points locaux (Fast pour l'ODE, Full pour la projection terminale)
        self.fork_points_local = torch.tensor(fork_points_fast, device=device).float()
        self.fork_points_full_local = torch.tensor(fork_points_full, device=device).float()
        
        # Matrice de passage World -> Container
        self.T_cw = torch.tensor(T_container_world, device=device).float()
        self.R_cw = self.T_cw[:3, :3]
        self.t_cw = self.T_cw[:3, 3]

    def ortho6d_to_rot_batch(self, d6):
        x_raw = d6[..., 0:3]
        y_raw = d6[..., 3:6]
        x = F.normalize(x_raw, dim=-1)
        z = F.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack([x, y, z], dim=-1)

    def compute_correction(self, x_norm, v_norm, t_ode, fork_pos_world):
        """
        Calcule la correction FMBF exacte dans l'espace Euclidien (Physique),
        puis la ramène dans l'espace normalisé pour le solveur ODE.
        """
        # CRITIQUE : L'inférence principale tourne dans un bloc "with torch.no_grad():".
        # Il faut explicitement réactiver les gradients ici pour construire le graphe SDF.
        with torch.enable_grad():
            # 1. Obtenir l'état physique avec suivi des gradients
            x_state = (x_norm * self.std + self.mean).clone().detach().requires_grad_(True)
            
            pos_centered = x_state[0, :, :3]
            rot_6d = x_state[0, :, 3:9]
            
            # 2. Cinématique
            pos_world = pos_centered + torch.tensor(fork_pos_world, device=self.device).float()
            R_world = self.ortho6d_to_rot_batch(rot_6d) # (H, 3, 3)
            
            # (H, 3, 3) @ (3, N) -> (H, 3, N). Transpose -> (H, N, 3)
            rotated_pts = torch.matmul(R_world, self.fork_points_local.T).transpose(1, 2)
            fork_world = pos_world.unsqueeze(1) + rotated_pts
            
            # 3. Passage au repère conteneur
            fork_container = torch.matmul(fork_world, self.R_cw.T) + self.t_cw
            
            # 4. Évaluation du SDF
            h_values = self.sdf(fork_container) # (H, N_points)
            
            # Vitesse projetée dans l'espace physique
            v_state = v_norm * self.std
            corrections_state = torch.zeros_like(x_state)
            
            # Saturation du blow-up pour l'intégrateur Euler
            phi_val = min(1.0 / (1.0 - t_ode + 1e-3), 50.0)
            
            for k in range(h_values.shape[0]): # Pour chaque waypoint de l'horizon
                min_h_val, worst_idx = torch.min(h_values[k], dim=0)
                
                if min_h_val < 0.01: # Seuil d'activation de la barrière
                    # PyTorch calcule le gradient Euclidien EXACT
                    grad_g = torch.autograd.grad(min_h_val, x_state, retain_graph=True)[0]
                    g_k = grad_g[0, k, :] # Vecteur 9D 
                    
                    g_norm_sq = torch.dot(g_k, g_k)
                    if g_norm_sq > 1e-8:
                        a_k = torch.dot(g_k, v_state[0, k, :]) + phi_val * min_h_val
                        
                        if a_k < 0:
                            # Projection orthogonale (Correction QP)
                            corrections_state[0, k, :] = -g_k * (a_k / g_norm_sq)

            # 5. Retour vers l'espace normalisé 
            corrections_norm = corrections_state / self.std
            corrections_norm = torch.clamp(corrections_norm, -3.0, 3.0) 
            
            # On détache le tenseur avant de le retourner à l'environnement "no_grad"
            return corrections_norm.detach()

    def get_debug_violations(self, fork_pos_world, rot_matrix_world):
        """Utilitaire pour RViz : Calcule les violations SDF sur le GPU et renvoie à NumPy."""
        with torch.no_grad():
            pos = torch.tensor(fork_pos_world, device=self.device).float()
            R = torch.tensor(rot_matrix_world, device=self.device).float()
            
            # Utilisation de l'ensemble complet des points pour le debug
            fork_world = pos + torch.matmul(self.fork_points_full_local, R.T)
            fork_container = torch.matmul(fork_world, self.R_cw.T) + self.t_cw
            
            h_values = self.sdf(fork_container.unsqueeze(0)).squeeze(0)
            
        return fork_world.cpu().numpy(), h_values.cpu().numpy()
    
    def terminal_projection(self, traj_np, max_iters=10, lr=0.005):
        """
        Filtre de sécurité terminal: Repousse physiquement les waypoints lissés hors du SDF.
        traj_np: numpy array (Horizon, 9) contenant [pos_x, pos_y, pos_z, rot_6d]
        """
        traj = torch.tensor(traj_np, device=self.device, dtype=torch.float32)
        n_fixes = 0
        
        # On projette chaque waypoint de la trajectoire finale
        for k in range(traj.shape[0]):
            pos_world = traj[k, :3].clone().requires_grad_(True)
            rot_6d = traj[k, 3:9].clone() 
            
            R_world = self.ortho6d_to_rot_batch(rot_6d.unsqueeze(0))[0]
            
            for i in range(max_iters):
                # Évaluation cinématique avec TOUS les points de sécurité (fork_points_full_local)
                fork_world = pos_world + torch.matmul(self.fork_points_full_local, R_world.T)
                fork_container = torch.matmul(fork_world, self.R_cw.T) + self.t_cw
                
                h_vals = self.sdf(fork_container.unsqueeze(0)).squeeze(0)
                min_h, _ = torch.min(h_vals, dim=0)
                
                # Si le point le plus profond est sûr (marge de 2mm), on arrête
                if min_h >= 0.002: 
                    break
                    
                # Violation détectée : on calcule le gradient pour repousser la position
                loss = -min_h 
                grad_pos = torch.autograd.grad(loss, pos_world)[0]
                
                with torch.no_grad():
                    # On déplace la position dans le sens opposé au gradient de pénétration
                    pos_world -= lr * grad_pos
                    
                    # On limite le déplacement total pour éviter les sauts brutaux
                    shift_norm = torch.norm(pos_world - traj[k, :3])
                    if shift_norm > self.max_terminal_shift:
                        direction = (pos_world - traj[k, :3]) / shift_norm
                        pos_world = traj[k, :3] + direction * self.max_terminal_shift
                        break # Décalage maximal atteint
                        
                    pos_world.requires_grad_(True)
                n_fixes += 1
                
            traj[k, :3] = pos_world.detach()
            
        return traj.cpu().numpy(), n_fixes
# ==============================================================
# TEMPORAL ENSEMBLER (unchanged)
# ==============================================================

class TemporalEnsembler:
    def __init__(self, pred_horizon, action_dim, buffer_size=3, weights='exponential'):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.buffer = collections.deque(maxlen=buffer_size)
        self.weights_type = weights
        self.step_counter = 0

    def add_prediction(self, prediction):
        self.buffer.append((self.step_counter, prediction.copy()))
        self.step_counter += 1

    def get_ensembled_trajectory(self, num_steps):
        if len(self.buffer) == 0:
            return None

        trajectory = []
        for i in range(num_steps):
            actions, weights = [], []
            for (timestamp, prediction) in self.buffer:
                age = self.step_counter - timestamp - 1
                pred_index = age + i
                if 0 <= pred_index < self.pred_horizon:
                    actions.append(prediction[pred_index])
                    w = np.exp(-0.5 * age) if self.weights_type == 'exponential' else 1.0
                    weights.append(w)
            if not actions:
                if trajectory:
                    trajectory.append(trajectory[-1])
                continue
            actions = np.array(actions)
            weights = np.array(weights)
            weights /= weights.sum()
            trajectory.append(np.average(actions, axis=0, weights=weights))

        return np.array(trajectory)


# ==============================================================
# ROS NODE
# ==============================================================

class FlowMatchingInferenceNodeFork:
    def __init__(self):
        rospy.init_node("flow_matching_inference_node_fork")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # URDF transforms
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')

        # =====================================================
        # LOAD MODEL
        # =====================================================
        model_name = rospy.get_param("~model_name", "last_fm_model_high_dim_CFM_relative.ckpt")
        self.model_path = os.path.join(pkg_path, 'models', model_name)

        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model not found: {self.model_path}")
            sys.exit(1)

        rospy.loginfo(f"Loading Model: {model_name}")
        payload = torch.load(self.model_path, map_location=self.device)

        cfg = payload.get('config', {})
        self.obs_horizon = cfg.get('obs_horizon', 2)
        self.pred_horizon = cfg.get('pred_horizon', 16)
        self.action_dim = cfg.get('action_dim', 9)
        self.num_points = cfg.get('num_points', 256)

        # Rebuild model with SAME architecture as training
        self.model = FlowMatchingAgent(
            action_dim=self.action_dim,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            encoder_output_dim=cfg.get('encoder_output_dim', 64),
            diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 256),
            down_dims=cfg.get('down_dims', [256, 512, 1024]),
            kernel_size=cfg.get('kernel_size', 5),
            n_groups=cfg.get('n_groups', 8),
        ).to(self.device)

        self.model.load_state_dict(payload['model_state_dict'])

        if not self.model.normalizer.is_initialized and 'stats' in payload:
            self.model.normalizer.load_stats_from_dict(payload['stats'])

        self.model.eval()
        rospy.loginfo(f"Normalizer initialized: {self.model.normalizer.is_initialized.item()}")

        # =====================================================
        # EXTRACT NORMALIZER STATS (needed for FMBF corrections)
        # =====================================================
        # The model uses a MinMax normalizer to scale the FIRST 3 DIMS to [-1, 1].
        # We compute the equivalent affine properties (mean and std) to allow
        # the FMBF velocity corrections to scale back and forth correctly.
        
        self.action_std = np.ones(self.action_dim)
        self.action_mean = np.zeros(self.action_dim)

        try:
            # On extrait les tenseurs de ton Normalizer MinMax
            pos_min = self.model.normalizer.pos_min.cpu().numpy().flatten()
            pos_max = self.model.normalizer.pos_max.cpu().numpy().flatten()
            
            # Équivalence mathématique :
            # Écart-type équivalent = (max - min) / 2
            # Moyenne équivalente = (max + min) / 2
            self.action_std[:3] = (pos_max - pos_min) / 2.0
            self.action_mean[:3] = (pos_max + pos_min) / 2.0
            
            rospy.loginfo(f"[Safety] MinMax stats successfully converted to Mean/Std.")
            rospy.loginfo(f"[Safety] action_std[:3] = {self.action_std[:3]}")
            rospy.loginfo(f"[Safety] action_mean[:3] = {self.action_mean[:3]}")

        except AttributeError as e:
            rospy.logwarn(f"[Safety] Could not extract normalizer stats: {e}")
            rospy.logwarn("[Safety] Using identity (no normalization correction).")

        # Log std[:3] — this is critical for understanding FMBF scaling.
        # If std[:3] ≈ 0.02, then a 0.3 m/s correction → 15 in normalized space.
        rospy.loginfo(f"[Safety] action_std[:3] = {self.action_std[:3]}")
        rospy.loginfo(f"[Safety] action_mean[:3] = {self.action_mean[:3]}")

        # max_correction_norm: max magnitude of FMBF correction in NORMALIZED space.
        # The velocity_net outputs ~1-3 in normalized space. The correction must
        # be strong enough to OVERPOWER the learned velocity when it pushes
        # toward the bowl. At 1.0, the correction can only match the velocity;
        # at 3.0, it can dominate it.
        self.max_correction_norm = rospy.get_param("~safety_max_correction_norm", 3.0)

        # =====================================================
        # ODE SOLVER CONFIG
        # =====================================================
        self.ode_method = rospy.get_param("~ode_method", "euler")
        self.num_ode_steps = rospy.get_param("~num_ode_steps", 20)
        rospy.loginfo(f"ODE: {self.ode_method}, {self.num_ode_steps} steps")

        # ROS setup (tf_listener MUST be created before safety init — TF lookup)
        self.tf_listener = tf.TransformListener()
        self.obs_queue = collections.deque(maxlen=self.obs_horizon)
        self.latest_cloud = None

        # ROS subscribers / publishers
        self.sub_cloud = rospy.Subscriber(
            "/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1
        )
        self.pub_trajectory = rospy.Publisher(
            "/diffusion/target_trajectory", PoseArray, queue_size=1
        )

        self.pub_path = rospy.Publisher(
            "/diffusion/target_path", Path, queue_size=1
        )

        # Debug visualization publishers
        self.pub_safety_points = rospy.Publisher(
            "/safety/fork_points", PointCloud2, queue_size=1
        )
        self.pub_safety_violations = rospy.Publisher(
            "/safety/violations", PointCloud2, queue_size=1
        )
        self.pub_sdf_visual = rospy.Publisher(
            "/safety/sdf_visual", PointCloud2, queue_size=1, latch=True)

        # =====================================================
        # SAFETY: FORK POINTS + SDF + FMBF
        # =====================================================
        self.enable_safety = rospy.get_param("~enable_safety", True)

        if self.enable_safety:
            self._init_safety(package_path, xacro_file)
        else:
            self.safety_filter = None
            rospy.loginfo("[Safety] DISABLED — set ~enable_safety:=true to activate")

        # Ensembler
        self.ensembler = TemporalEnsembler(
            pred_horizon=self.pred_horizon,
            action_dim=self.action_dim,
            buffer_size=3
        )

        # =====================================================
        # TIMERS (Séparation de la physique et de l'inférence)
        # =====================================================
        self.latest_fork_pos = None
        self.latest_fork_rot = None
        
        # 1. Timer d'Observation (RAPIDE) : Doit correspondre EXACTEMENT 
        # à la fréquence de tes données d'entraînement (ex: 10 Hz)
        self.state_rate = 10.0
        rospy.Timer(rospy.Duration(1.0 / self.state_rate), self.update_state_history)
        
        # 2. Timer d'Inférence (LENT) : Tourne aussi vite que le GPU le permet
        self.inference_rate = 10.0 
        rospy.Timer(rospy.Duration(1.0 / self.inference_rate), self.control_loop)
        
        rospy.loginfo("Node Ready (Flow Matching — Asynchronous State Tracking)")

    # =====================================================
    # SAFETY INITIALIZATION
    # =====================================================

    def _init_safety(self, package_path, xacro_file):
        """Load SDF, precompute fork safety points, build FMBF filter."""

        # --- 1. Load fork STL and compute safety points ---
        stl_path = os.path.join(
            package_path, 'src', 'vision_processing',
            'diffusion_model_train', 'fork.STL'
        )
        if not os.path.exists(stl_path):
            rospy.logerr(f"[Safety] Fork STL not found: {stl_path}")
            self.safety_filter = None
            return

        # Build the visual origin transform from xacro values
        #   <origin xyz="-0.033 -0.02 0.0171378" rpy="0 ${27.5*pi/180} 0"/>
        T_visual = self._build_visual_origin()
        scale = [0.001, 0.001, 0.001]  # mm → meters (from xacro)

        n_fast = rospy.get_param("~safety_n_fast", 5)
        n_full = rospy.get_param("~safety_n_full", 50)

        fork_fast, fork_full = precompute_fork_safety_points(
            stl_path, scale, T_visual, n_fast=n_fast, n_full=n_full
        )
        rospy.loginfo(f"[Safety] Fork points: {fork_fast.shape[0]} fast, "
                      f"{fork_full.shape[0]} full")

        # --- 2. Load SDF ---
        sdf_path = rospy.get_param("~sdf_path", "")
        if not sdf_path:
            print("[Safety] No ~sdf_path provided, using default from package")
            sdf_path = os.path.join(package_path, 'models', 'sdf_field_No_outter_tiny.npz')
            print(f"[Safety] Default SDF path: {sdf_path}")
        if not os.path.exists(sdf_path):
            rospy.logerr(f"[Safety] SDF not found: {sdf_path}")
            self.safety_filter = None
            return

        print(f"[Safety] Loading SDF from: {sdf_path}")
        # FIX : Utilisation du device global du noeud ROS pour éviter les crashs de type
        sdf = SDFTensorField(sdf_path, self.device)
        print(f"[Safety] SDF loaded: grid {sdf.n_voxels}, "
              f"voxel={sdf.voxel_size*1000:.1f}mm")

        # --- 3. Container pose in world frame ---
        # T_container_world transforms points FROM world TO container frame.
        # We get this from TF: the container_link frame is published by the URDF.
        # This ensures the SDF frame always matches the visual in Gazebo/RViz.
        self.container_frame = rospy.get_param("~container_frame", "container_link")

        rospy.loginfo(f"[Safety] Waiting for TF: world → {self.container_frame} ...")
        try:
            self.tf_listener.waitForTransform(
                self.container_frame, '/world', rospy.Time(0), rospy.Duration(10.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                self.container_frame, '/world', rospy.Time(0)
            )
            # This gives us T_container_world directly:
            #   p_container = R @ p_world + t
            T_container_world = tft.quaternion_matrix(rot)
            T_container_world[:3, 3] = trans

            rospy.loginfo(f"[Safety] T_container_world from TF:")
            rospy.loginfo(f"  trans: ({trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f})")
            rospy.loginfo(f"  rot:   ({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f})")

        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            rospy.logwarn(f"[Safety] TF lookup failed: {e}")
            rospy.logwarn("[Safety] Falling back to ~container_pose parameter")

            container_pose = rospy.get_param("~container_pose", [0, 0, 0, 0, 0, 0])
            x, y, z, roll, pitch, yaw = container_pose
            T_world_container = np.eye(4)
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy_v, sy_v = np.cos(yaw), np.sin(yaw)
            R = np.array([
                [cy_v*cp, cy_v*sp*sr - sy_v*cr, cy_v*sp*cr + sy_v*sr],
                [sy_v*cp, sy_v*sp*sr + cy_v*cr, sy_v*sp*cr - cy_v*sr],
                [    -sp,                cp*sr,                cp*cr ]
            ])
            T_world_container[:3, :3] = R
            T_world_container[:3, 3] = [x, y, z]
            T_container_world = np.linalg.inv(T_world_container)

            rospy.loginfo(f"[Safety] Fallback container pose: "
                          f"xyz=({x:.3f},{y:.3f},{z:.3f}) "
                          f"rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})")

        # --- 4. Build FMBF filter ---
        phi_0 = rospy.get_param("~safety_phi0", 1.0)
        t_star = rospy.get_param("~safety_t_star", 0.5)
        max_terminal_shift = rospy.get_param("~safety_max_terminal_shift", 0.005)

        self.safety_filter = FMBFSafetyFilterPyTorch(
            sdf_tensor_field=sdf,
            T_container_world=T_container_world,
            fork_points_fast=fork_fast,
            fork_points_full=fork_full,
            action_mean=self.action_mean,       # Requis pour la Jacobienne Euclidienne
            action_std=self.action_std,         # Requis pour la Jacobienne Euclidienne
            phi_0=phi_0,
            t_star=t_star,
            max_terminal_shift=max_terminal_shift,
            device=self.device                  # Requis pour placer les tenseurs sur GPU
        )
        if self.safety_filter is not None:
            rospy.loginfo("[Safety] FMBF Safety Filter initialized successfully.")
        rospy.loginfo(f"[Safety] FMBF active: φ₀={phi_0}, t*={t_star}, "
                      f"max_corr_norm={self.max_correction_norm}, "
                      f"max_shift={max_terminal_shift*1000:.1f}mm")

        self._publish_sdf_visualization()

    def _build_visual_origin(self):
        """Build the 4x4 visual origin transform for the fork STL.
        From xacro: xyz=(-0.033, -0.02, 0.0171378), rpy=(0, 27.5°, 0)"""
        tx, ty, tz = -0.033, -0.02, 0.0171378
        angle_y = 27.5 * np.pi / 180.0
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    # =====================================================
    # CALLBACKS
    # =====================================================

    def cloud_callback(self, msg):
        import sensor_msgs.point_cloud2 as pc2
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array(points, dtype=np.float32)

        if points.shape[0] == 0:
            return

        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            self.latest_cloud = points[indices]
        else:
            extra = np.random.choice(points.shape[0],
                                     self.num_points - points.shape[0], replace=True)
            self.latest_cloud = np.concatenate([points, points[extra]], axis=0)

    def get_current_fork_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                '/world', '/panda_hand_tcp', rospy.Time(0))
            T_world_tcp = tft.quaternion_matrix(rot)
            T_world_tcp[:3, 3] = trans
            T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
            return T_world_fork[:3, 3], T_world_fork[:3, :3]
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return None, None

    # =====================================================
    # MAIN CONTROL LOOP
    # =====================================================

    def update_state_history(self, event):
        """Thread rapide (10Hz) pour maintenir l'historique de vitesse intact."""
        pos, rot = self.get_current_fork_pose()
        if pos is None:
            return
            
        self.latest_fork_pos = pos
        self.latest_fork_rot = rot
        
        rot_6d = rotation_matrix_to_ortho6d(rot)
        curr_pose_9d = np.concatenate([pos, rot_6d.flatten()])
        self.obs_queue.append(curr_pose_9d)
        
    def control_loop(self, event):
        if len(self.obs_queue) < self.obs_horizon or self.latest_cloud is None or self.latest_fork_pos is None:
            return

        # ── Debug: publish fork safety points at CURRENT pose ──
        if self.safety_filter is not None:
            self._publish_safety_debug(self.latest_fork_pos, self.latest_fork_rot)

        # On "gèle" l'état au moment où l'inférence commence
        snapshot_fork_pos = self.latest_fork_pos.copy()
        obs_seq_world = np.stack(self.obs_queue)

        # =====================================================
        # TRANSLATION-ONLY CENTERING
        # =====================================================
        pcd_centered = self.latest_cloud - snapshot_fork_pos
        obs_seq_centered = obs_seq_world.copy()
        obs_seq_centered[:, :3] -= snapshot_fork_pos

        # =====================================================
        # FLOW MATCHING INFERENCE WITH FMBF
        # =====================================================
        t_pcd = torch.from_numpy(pcd_centered).unsqueeze(0).float().to(self.device)
        t_obs = torch.from_numpy(obs_seq_centered).unsqueeze(0).float().to(self.device)
        t_obs_norm = self.model.normalizer.normalize(t_obs)

        with torch.no_grad():
            obs_for_encoding = {'point_cloud': t_pcd, 'agent_pos': t_obs_norm}
            global_cond = self.model.encode_obs(obs_for_encoding)

            x = torch.randn(1, self.pred_horizon, self.action_dim, device=self.device)
            dt = 1.0 / self.num_ode_steps

            for i in range(self.num_ode_steps):
                t_val = i * dt
                t_tensor = torch.tensor([t_val], device=self.device)

                if self.ode_method == 'euler':
                    v = self.model.velocity_net(x, t_tensor, global_cond)
                elif self.ode_method == 'midpoint':
                    t_mid = torch.tensor([t_val + dt / 2], device=self.device)
                    v1 = self.model.velocity_net(x, t_tensor, global_cond)
                    x_mid = x + v1 * (dt / 2)
                    v = self.model.velocity_net(x_mid, t_mid, global_cond)

                if (self.safety_filter is not None and t_val >= self.safety_filter.t_star):
                    v = self._apply_fmbf_correction(x, v, t_val, snapshot_fork_pos)

                x = x + v * dt

            action_centered = self.model.normalizer.unnormalize(x)
            action_centered_np = action_centered[0].cpu().numpy()

        for t in range(action_centered_np.shape[0]):
            col1, col2 = gram_schmidt_6d(action_centered_np[t, 3:6], action_centered_np[t, 6:9])
            action_centered_np[t, 3:6] = col1
            action_centered_np[t, 6:9] = col2

        action_world_np = action_centered_np.copy()
        action_world_np[:, :3] += snapshot_fork_pos

        # =====================================================
        # FIX: LATENCY TRIMMING (CROPPING THE PAST)
        # =====================================================
        # L'inférence a pris du temps. Où est le robot *exactement maintenant* ?
        actual_pos, _ = self.get_current_fork_pose()
        if actual_pos is not None:
            distances = np.linalg.norm(action_world_np[:, :3] - actual_pos, axis=1)
            closest_idx = np.argmin(distances)
            
            # Si le robot est déjà rendu au waypoint 4, on supprime les waypoints 0, 1, 2, 3 !
            # RViz n'affichera que le futur et le follower ne regardera plus en arrière.
            start_idx = max(0, closest_idx)
            action_world_np = action_world_np[start_idx:]

        if action_world_np.shape[0] < 2:
            return # La trajectoire entière est déjà dans le passé, on annule.

        raw_traj = action_world_np.copy()

        if self.safety_filter is not None:
            smooth_traj, n_fixes = self.safety_filter.terminal_projection(raw_traj)
            if n_fixes > 0:
                rospy.loginfo(f"[Safety] Terminal: corrected {n_fixes} waypoints")
        else:
            smooth_traj = raw_traj

        self.publish_trajectory(smooth_traj)

    # =====================================================
    # FMBF CORRECTION (called inside ODE loop)
    # =====================================================

    def _apply_fmbf_correction(self, x_norm, v_norm, t_ode, fork_pos_world):
        """Apply FMBF corrections fully on GPU using the frozen snapshot position"""
        correction_tensor = self.safety_filter.compute_correction(
            x_norm, v_norm, t_ode, fork_pos_world
        )
        return v_norm + correction_tensor
    
    # =====================================================
    # SAFETY DEBUG VISUALIZATION
    # =====================================================

    def _publish_safety_debug(self, fork_pos, fork_rot):
        """Publish fork safety points for RViz visualization."""
        import sensor_msgs.point_cloud2 as pc2
        from std_msgs.msg import Header
        import struct

        if self.safety_filter is None:
            return

        # 1. Obtenir les points et valeurs h via PyTorch
        fork_world, h_values = self.safety_filter.get_debug_violations(fork_pos, fork_rot)

        header = Header()
        header.frame_id = "world"
        header.stamp = rospy.Time.now()

        def pack_rgb(r, g, b):
            a = 255
            packed = (a << 24) | (r << 16) | (g << 8) | b
            return struct.unpack('f', struct.pack('I', packed))[0]

        color_green = pack_rgb(0, 255, 0)
        color_red = pack_rgb(255, 0, 0)

        fields_rgb = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1),
        ]

        # 2. Points verts (Tous les points de la fourchette)
        points_green = [[float(p[0]), float(p[1]), float(p[2]), color_green] for p in fork_world]
        self.pub_safety_points.publish(pc2.create_cloud(header, fields_rgb, points_green))

        # 3. Points rouges (Violations uniquement : h < 0)
        violations = fork_world[h_values < 0]
        if violations.shape[0] > 0:
            points_red = [[float(v[0]), float(v[1]), float(v[2]), color_red] for v in violations]
            self.pub_safety_violations.publish(pc2.create_cloud(header, fields_rgb, points_red))

            worst_h = np.min(h_values)
            n_viol = np.sum(h_values < 0)
            # Limiter l'affichage des logs pour ne pas spammer la console à 10 Hz
            rospy.loginfo_throttle(1.0, f"[Safety VIZ] {n_viol}/{len(h_values)} points en collision, pire h={worst_h*1000:.1f}mm")

    def _publish_sdf_visualization(self):
        """Extracts and publishes ONLY the SDF 'no-go zone' boundary for RViz."""
        import sensor_msgs.point_cloud2 as pc2
        from std_msgs.msg import Header
        import struct

        if self.safety_filter is None or self.safety_filter.sdf is None:
            return

        sdf = self.safety_filter.sdf
        
        # Extraction dynamique depuis la forme du Tenseur PyTorch: (1, 1, nx, ny, nz)
        shape = sdf.sdf_tensor.shape
        nx, ny, nz = shape[2], shape[3], shape[4]
        voxel_size = sdf.voxel_size
        
        # Copie CPU pour le traitement NumPy
        bounds_min = sdf.bounds_min.cpu().numpy()

        # Générer les coordonnées de la grille 3D
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')

        points_x = bounds_min[0] + (ix + 0.5) * voxel_size
        points_y = bounds_min[1] + (iy + 0.5) * voxel_size
        points_z = bounds_min[2] + (iz + 0.5) * voxel_size

        points_container = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3)
        
        # CORRECTION DU CRASH: Extraction des valeurs depuis le tenseur PyTorch
        sdf_values = sdf.sdf_tensor.cpu().numpy().reshape(-1)

        # ── SÉLECTION DE LA "NO-GO ZONE" ──
        mask = (sdf_values <= 0.0) & (sdf_values >= -0.12)
        
        boundary_points = points_container[mask]

        header = Header()
        header.frame_id = self.container_frame 
        header.stamp = rospy.Time.now()

        def pack_rgb(r, g, b):
            a = 255
            packed = (a << 24) | (r << 16) | (g << 8) | b
            return struct.unpack('f', struct.pack('I', packed))[0]

        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1),
        ]

        color_red = pack_rgb(255, 0, 0)
        points_list = []
        
        for i in range(boundary_points.shape[0]):
            points_list.append([
                float(boundary_points[i, 0]),
                float(boundary_points[i, 1]),
                float(boundary_points[i, 2]),
                color_red
            ])

        cloud_msg = pc2.create_cloud(header, fields, points_list)
        self.pub_sdf_visual.publish(cloud_msg)
        rospy.loginfo(f"[Safety VIZ] Publié {len(points_list)} points de la no-go zone SDF sur /safety/sdf_visual")

    # =====================================================
    # PUBLISH
    # =====================================================

    def publish_trajectory(self, trajectory):
        msg_pose_array = PoseArray()
        msg_pose_array.header.frame_id = "world"
        msg_pose_array.header.stamp = rospy.Time.now()

        # Initialisation du message Path
        msg_path = Path()
        msg_path.header = msg_pose_array.header

        for pose_9d in trajectory:
            pos = pose_9d[:3]
            rot_6d = pose_9d[3:]
            rot_mat = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]
            M = np.eye(4)
            M[:3, :3] = rot_mat
            quat = tft.quaternion_from_matrix(M)

            # Création de la Pose basique
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            msg_pose_array.poses.append(p)

            # Création de la PoseStamped pour le Path
            ps = PoseStamped()
            ps.header = msg_path.header
            ps.pose = p
            msg_path.poses.append(ps)

        rospy.loginfo(f"Fork tip: {np.linalg.norm(self.latest_fork_pos):.4f}")
        
        # Publication simultanée
        self.pub_trajectory.publish(msg_pose_array)
        self.pub_path.publish(msg_path)


if __name__ == "__main__":
    import sensor_msgs.point_cloud2 as pc2
    try:
        FlowMatchingInferenceNodeFork()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass