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
import collections
import os
import sys
import rospkg
import tf
import tf.transformations as tft
import trimesh
from scipy.spatial import ConvexHull
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose

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

class SDFField:
    """Loads a precomputed SDF from .npz and provides fast queries.
    The SDF is a 3D voxel grid with trilinear interpolation.

    Convention:
        SDF > 0 → safe (outside forbidden zone)
        SDF < 0 → forbidden (collision zone)
        SDF = 0 → on the boundary
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.sdf_grid = data['sdf_grid']               # (Nx, Ny, Nz)
        self.bounds_min = data['bounds_min']             # (3,) world origin of grid
        self.bounds_max = data['bounds_max']             # (3,)
        self.voxel_size = float(data['voxel_size'])      # meters per voxel
        self.n_voxels = data['n_voxels']                 # (3,) grid dimensions
        self.inner_radius = float(data.get('inner_radius', 0.00))
        self.outer_radius = float(data.get('outer_radius', 0.0))

        rospy.loginfo(f"[SDF] Loaded: grid {self.n_voxels}, "
                      f"voxel={self.voxel_size*1000:.1f}mm, "
                      f"inner={self.inner_radius*1000:.1f}mm, "
                      f"outer={self.outer_radius*1000:.1f}mm")

    def query(self, points):
        """Query SDF at given points (in container frame).

        Args:
            points: (N, 3) array in container frame coordinates
        Returns:
            distances: (N,) SDF values. Negative = inside forbidden zone.
            gradients: (N, 3) normalized SDF gradients pointing toward safe region.
        """
        points = np.atleast_2d(points).astype(np.float64)

        # Convert world coordinates → grid coordinates (continuous)
        grid_coords = (points - self.bounds_min) / self.voxel_size - 0.5

        distances = self._trilinear_interpolate(grid_coords)
        gradients = self._compute_gradient(grid_coords)
        return distances, gradients

    def _trilinear_interpolate(self, grid_coords):
        """Trilinear interpolation on the voxel grid."""
        gc = np.clip(grid_coords, 0, np.array(self.n_voxels) - 1.001)
        i0 = np.floor(gc).astype(int)
        i1 = np.minimum(i0 + 1, np.array(self.n_voxels) - 1)
        t = gc - i0

        # 8 corner values
        d000 = self.sdf_grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        d001 = self.sdf_grid[i0[:, 0], i0[:, 1], i1[:, 2]]
        d010 = self.sdf_grid[i0[:, 0], i1[:, 1], i0[:, 2]]
        d011 = self.sdf_grid[i0[:, 0], i1[:, 1], i1[:, 2]]
        d100 = self.sdf_grid[i1[:, 0], i0[:, 1], i0[:, 2]]
        d101 = self.sdf_grid[i1[:, 0], i0[:, 1], i1[:, 2]]
        d110 = self.sdf_grid[i1[:, 0], i1[:, 1], i0[:, 2]]
        d111 = self.sdf_grid[i1[:, 0], i1[:, 1], i1[:, 2]]

        # Interpolate along x, then y, then z
        c00 = d000 * (1 - t[:, 0]) + d100 * t[:, 0]
        c01 = d001 * (1 - t[:, 0]) + d101 * t[:, 0]
        c10 = d010 * (1 - t[:, 0]) + d110 * t[:, 0]
        c11 = d011 * (1 - t[:, 0]) + d111 * t[:, 0]
        c0 = c00 * (1 - t[:, 1]) + c10 * t[:, 1]
        c1 = c01 * (1 - t[:, 1]) + c11 * t[:, 1]
        return c0 * (1 - t[:, 2]) + c1 * t[:, 2]

    def _compute_gradient(self, grid_coords):
        """Central difference gradient, normalized to unit length."""
        eps = 0.5  # half-voxel offset for central differences
        d_xp = self._trilinear_interpolate(grid_coords + [eps, 0, 0])
        d_xm = self._trilinear_interpolate(grid_coords - [eps, 0, 0])
        d_yp = self._trilinear_interpolate(grid_coords + [0, eps, 0])
        d_ym = self._trilinear_interpolate(grid_coords - [0, eps, 0])
        d_zp = self._trilinear_interpolate(grid_coords + [0, 0, eps])
        d_zm = self._trilinear_interpolate(grid_coords - [0, 0, eps])

        grad = np.stack([(d_xp - d_xm), (d_yp - d_ym), (d_zp - d_zm)], axis=1)
        grad /= (2 * eps * self.voxel_size)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        return grad / np.maximum(norms, 1e-10)


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

class FMBFSafetyFilter:
    """Flow Matching Barrier Function safety filter.

    Implements SafeFlow's closed-form QP correction (eq.16) using SDF:
      h(s) = SDF(s)                          (barrier function)
      b = ∇SDF(s)                            (gradient)
      a = dot(b, v) + φ(t, h) * h           (constraint value)
      u = -b * a / ||b||²  if a < 0         (minimal correction)

    The blow-up function φ ensures convergence to safe set:
      φ(t, h) = φ₀           if h >= 0  (already safe)
      φ(t, h) = 1/(1-t)      if h < 0   (unsafe → force out before t=1)

    Args:
        sdf:                SDFField instance (in container frame)
        T_container_world:  4x4 transform FROM world TO container frame
                            i.e. p_container = T_container_world @ [p_world; 1]
        fork_points_fast:   (M, 3) representative points in fork_tip frame
        fork_points_full:   (N, 3) dense points in fork_tip frame
        phi_0:              CBF gain for safe states (default 1.0)
        t_star:             ODE time to start FMBF (default 0.5)
    """

    def __init__(self, sdf, T_container_world,
                 fork_points_fast, fork_points_full,
                 phi_0=1.0, t_star=0.5,
                 max_terminal_shift=0.005):
        self.sdf = sdf
        self.fork_points_fast = fork_points_fast    # (M, 3) in fork_tip frame
        self.fork_points_full = fork_points_full    # (N, 3) in fork_tip frame
        self.phi_0 = phi_0
        self.t_star = t_star
        self.max_terminal_shift = max_terminal_shift

        # Precompute frame transforms
        self.T_container_world = T_container_world
        self.R_container_world = T_container_world[:3, :3]
        self.R_world_container = self.R_container_world.T
        self.t_container = T_container_world[:3, 3]

    def phi(self, t, h):
        """Bounded gain function.
        
        Why NOT 1/(1-t):
        ─────────────────
        The 1/(1-t) blow-up from the SafeFlow paper gives theoretical convergence
        guarantees, but in practice it creates corrections that are 10-100x the 
        learned velocity. After dividing by std to go back to normalized space, 
        these corrections overwhelm the velocity_net output and destabilize the ODE.
        
        The terminal projection catches whatever the soft phi misses.
        
        Ramp: φ₀ at t=0  →  3·φ₀ at t=1
        """
        return 1/(1-t) if h < 0 else self.phi_0

    def _points_to_container_frame(self, points_world):
        """Transform points from world frame to container (SDF) frame.

        Math: p_container = R_container_world @ p_world + t_container
              which is the same as T_world_container @ [p_world; 1]

        Args:
            points_world: (N, 3)
        Returns:
            points_container: (N, 3)
        """
        return (self.R_container_world @ points_world.T).T + self.t_container

    def _grad_to_world_frame(self, grad_container):
        """Rotate SDF gradient from container frame to world frame.

        Since gradient is a direction (not a position), only rotation applies.
        grad_world = R_world_container @ grad_container

        Args:
            grad_container: (3,) or (N, 3)
        Returns:
            grad_world: same shape
        """
        if grad_container.ndim == 1:
            return self.R_world_container @ grad_container
        return (self.R_world_container @ grad_container.T).T

    def compute_fork_world_points(self, pos_world, R_world):
        """Compute world positions of representative fork points.

        Each representative point offset_j is in the fork_tip local frame.
        Its world position is:  p_j = pos_world + R_world @ offset_j

        Args:
            pos_world:    (3,) fork tip position in world frame
            R_world:      (3, 3) fork tip rotation in world frame
            use_full:     if True, use 50-point set; else use 5-point set
        Returns:
            points_world: (M, 3)
        """
        # Vectorized: (M, 3) = (3,) + (M, 3) @ (3, 3).T
        # But we want R @ offset for each offset, so: (M, 3) = offsets @ R.T
        return pos_world + self.fork_points_fast @ R_world.T

    def compute_fork_world_points_full(self, pos_world, R_world):
        """Same as above but with the 50-point set for terminal check."""
        return pos_world + self.fork_points_full @ R_world.T

    def compute_correction(self, waypoint_pos_world, waypoint_rot_world,
                           velocity_pos_state, t_ode):
        """Compute FMBF correction for one waypoint (closed-form QP).

        Steps:
            1. Transform 5 fork representative points to container frame
            2. Query SDF → get h and ∇h for each point
            3. Find worst violation (min h)
            4. Compute a = dot(∇h, v_pos) + φ(t, h) * h
            5. If a < 0: u = -∇h * a / ||∇h||²  (minimal correction)

        Args:
            waypoint_pos_world:   (3,) position in world frame
            waypoint_rot_world:   (3, 3) rotation in world frame
            velocity_pos_state:   (3,) position velocity in centered frame
            t_ode:                current ODE time in [0, 1)

        Returns:
            correction:  (3,) velocity correction in centered frame (world orientation)
                         Zero if no correction needed.
        """
        # Step 1: Fork points in world frame
        fork_world = self.compute_fork_world_points(waypoint_pos_world,
                                                     waypoint_rot_world)

        # Step 2: Transform to container frame and query SDF
        fork_container = self._points_to_container_frame(fork_world)
        h_values, grad_container = self.sdf.query(fork_container)

        # Step 3: Worst violation
        worst_idx = np.argmin(h_values)
        h_worst = h_values[worst_idx]
        grad_worst_container = grad_container[worst_idx]  # (3,) in container frame

        # Step 4: Transform gradient to world frame
        # Since we use translation-only centering, world and centered frame
        # share the same orientation → gradient in world = gradient in centered
        grad_world = self._grad_to_world_frame(grad_worst_container)

        # Step 5: Compute a_k = dot(b_k, v_k) + φ(t, h_k) * h_k
        a_k = np.dot(grad_world, velocity_pos_state) + self.phi(t_ode, h_worst) * h_worst

        # Step 6: Closed-form QP solution (SafeFlow eq.16)
        if a_k >= 0:
            return np.zeros(3)  # constraint satisfied, no correction
        else:
            grad_norm_sq = np.dot(grad_world, grad_world)
            if grad_norm_sq < 1e-12:
                return np.zeros(3)  # degenerate gradient, skip
            return -grad_world * a_k / grad_norm_sq

    def terminal_projection(self, trajectory_world, max_iters=3):
        """Terminal safety: iterative projection — NO shift clamp.

        This runs AFTER the ODE and AFTER the ensembler. No risk of
        destabilizing the ODE integration, so we apply the FULL correction
        needed to reach h=0 in each pass.

        Multiple passes handle rigid-body coupling (fixing one fork point
        can push another into violation).
        """
        trajectory_safe = trajectory_world.copy()
        total_corrections = 0

        for iteration in range(max_iters):
            n_corrections = 0
            for k in range(trajectory_safe.shape[0]):
                pos_world = trajectory_safe[k, :3]
                rot_6d = trajectory_safe[k, 3:9]
                R_world = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]

                fork_world = self.compute_fork_world_points_full(pos_world, R_world)
                fork_container = self._points_to_container_frame(fork_world)
                h_values, grad_container = self.sdf.query(fork_container)

                worst_idx = np.argmin(h_values)
                h_worst = h_values[worst_idx]

                if h_worst < 0:
                    grad_world = self._grad_to_world_frame(grad_container[worst_idx])
                    grad_norm = np.linalg.norm(grad_world)
                    if grad_norm > 1e-8:
                        # Full correction: push to h=0 plus 1mm margin
                        shift_mag = (-h_worst + 0.001) / grad_norm
                        trajectory_safe[k, :3] += (grad_world / grad_norm) * shift_mag
                        n_corrections += 1

            total_corrections += n_corrections
            if n_corrections == 0:
                break

        return trajectory_safe, total_corrections


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
        model_name = rospy.get_param("~model_name", "last_fm_model_high_dim_CFM_relative_dropout.ckpt")
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

        self.control_rate = rospy.get_param("~control_rate", 10.0)
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)
        rospy.loginfo("Node Ready (Flow Matching — SafeFlow Edition)")

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
            sdf_path = os.path.join(package_path, 'models', 'sdf_field_No_outter.npz')
            print(f"[Safety] Default SDF path: {sdf_path}")
        if not os.path.exists(sdf_path):
            rospy.logerr(f"[Safety] SDF not found: {sdf_path}")
            self.safety_filter = None
            return

        print(f"[Safety] Loading SDF from: {sdf_path}")
        sdf = SDFField(sdf_path)
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

        self.safety_filter = FMBFSafetyFilter(
            sdf=sdf,
            T_container_world=T_container_world,
            fork_points_fast=fork_fast,
            fork_points_full=fork_full,
            phi_0=phi_0,
            t_star=t_star,
            max_terminal_shift=max_terminal_shift,
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

    def control_loop(self, event):
        if self.latest_cloud is None:
            return

        self.fork_pos, self.fork_rot = self.get_current_fork_pose()
        if self.fork_pos is None:
            return

        # ── Debug: publish fork safety points at CURRENT pose ──
        if self.safety_filter is not None:
            self._publish_safety_debug(self.fork_pos, self.fork_rot)

        rot_6d = rotation_matrix_to_ortho6d(self.fork_rot)
        curr_pose_9d = np.concatenate([self.fork_pos, rot_6d.flatten()])

        self.obs_queue.append(curr_pose_9d)
        if len(self.obs_queue) < self.obs_horizon:
            return

        # =====================================================
        # TRANSLATION-ONLY CENTERING
        # =====================================================
        pcd_centered = self.latest_cloud - self.fork_pos
        obs_seq_world = np.stack(self.obs_queue)
        obs_seq_centered = obs_seq_world.copy()
        obs_seq_centered[:, :3] -= self.fork_pos

        # =====================================================
        # FLOW MATCHING INFERENCE WITH FMBF
        # =====================================================
        t_pcd = torch.from_numpy(pcd_centered).unsqueeze(0).float().to(self.device)
        t_obs = torch.from_numpy(obs_seq_centered).unsqueeze(0).float().to(self.device)
        t_obs_norm = self.model.normalizer.normalize(t_obs)

        with torch.no_grad():
            # Encode observation ONCE
            obs_for_encoding = {'point_cloud': t_pcd, 'agent_pos': t_obs_norm}
            global_cond = self.model.encode_obs(obs_for_encoding)

            # ODE integration with FMBF correction
            x = torch.randn(1, self.pred_horizon, self.action_dim,
                             device=self.device)
            dt = 1.0 / self.num_ode_steps

            for i in range(self.num_ode_steps):
                t_val = i * dt
                t_tensor = torch.tensor([t_val], device=self.device)

                # --- Compute learned velocity ---
                if self.ode_method == 'euler':
                    v = self.model.velocity_net(x, t_tensor, global_cond)
                elif self.ode_method == 'midpoint':
                    t_mid = torch.tensor([t_val + dt / 2], device=self.device)
                    v1 = self.model.velocity_net(x, t_tensor, global_cond)
                    x_mid = x + v1 * (dt / 2)
                    v = self.model.velocity_net(x_mid, t_mid, global_cond)

                # --- FMBF safety correction (only for t >= t_star) ---
                if (self.safety_filter is not None
                        and t_val >= self.safety_filter.t_star):
                    v = self._apply_fmbf_correction(x, v, t_val)

                # --- Step ---
                x = x + v * dt

            # Unnormalize back to physical units
            action_centered = self.model.normalizer.unnormalize(x)
            action_centered_np = action_centered[0].cpu().numpy()

        # =====================================================
        # GRAM-SCHMIDT ORTHONORMALIZATION ON ROTATIONS
        # =====================================================
        for t in range(action_centered_np.shape[0]):
            col1, col2 = gram_schmidt_6d(
                action_centered_np[t, 3:6],
                action_centered_np[t, 6:9]
            )
            action_centered_np[t, 3:6] = col1
            action_centered_np[t, 6:9] = col2

        # =====================================================
        # BACK TO WORLD FRAME
        # =====================================================
        action_world_np = action_centered_np.copy()
        action_world_np[:, :3] += self.fork_pos

        # ── DIAGNOSTICS ──
        span = action_centered_np[-1, :3] - action_centered_np[0, :3]
        rospy.loginfo(f"[FM] Centered span: [{span[0]:.4f}, {span[1]:.4f}, "
                      f"{span[2]:.4f}] = {np.linalg.norm(span)*100:.1f}cm")
        rospy.loginfo(f"[FM] World pos[0]: {action_world_np[0,:3]}")
        rospy.loginfo(f"[FM] World pos[-1]: {action_world_np[-1,:3]}")

        self.ensembler.add_prediction(action_world_np)
        smooth_traj = self.ensembler.get_ensembled_trajectory(num_steps=8)

        if smooth_traj is not None:
            # Terminal projection AFTER ensembling — the ensembler can blend
            # safe trajectories into unsafe ones, so we must re-check here.
            if self.safety_filter is not None:
                smooth_traj, n_fixes = self.safety_filter.terminal_projection(
                    smooth_traj)
                if n_fixes > 0:
                    rospy.loginfo(f"[Safety] Terminal: corrected {n_fixes} waypoints")

            self.publish_trajectory(smooth_traj)

    # =====================================================
    # FMBF CORRECTION (called inside ODE loop)
    # =====================================================

    def _apply_fmbf_correction(self, x_norm, v_norm, t_ode):
        """Apply FMBF corrections
        """
        x_np = x_norm[0].cpu().numpy()       # (H, 9) normalized
        v_np = v_norm[0].cpu().numpy()        # (H, 9) normalized

        # Denormalize to state space
        x_state = x_np * self.action_std + self.action_mean
        v_state = v_np * self.action_std

        corrections = np.zeros((self.pred_horizon, 3))

        for k in range(self.pred_horizon):
            pos_centered = x_state[k, :3]
            pos_world = pos_centered + self.fork_pos

            rot_6d = x_state[k, 3:9]
            R_world = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]

            v_pos_state = v_state[k, :3]

            u_state = self.safety_filter.compute_correction(
                pos_world, R_world, v_pos_state, t_ode
            )
            corrections[k] = u_state

        # ── Convert to normalized space ──
        corrections_norm = corrections / (self.action_std[:3] + 1e-10)

        # Build full 9D correction (only position dims)
        correction_full = np.zeros_like(v_np)
        correction_full[:, :3] = corrections_norm

        correction_tensor = torch.from_numpy(correction_full).unsqueeze(0).float().to(self.device)
        return v_norm + correction_tensor

    # =====================================================
    # SAFETY DEBUG VISUALIZATION
    # =====================================================

    def _publish_safety_debug(self, fork_pos, fork_rot):
        """Publish fork safety points for RViz visualization.

        Publishes two PointCloud2 topics:
          /safety/fork_points   — ALL 50 safety points (solid green)
          /safety/violations    — only points with h < 0 (solid red)
        """
        import sensor_msgs.point_cloud2 as pc2
        from std_msgs.msg import Header
        import struct

        sf = self.safety_filter

        # Transform FULL set (50 points) to world frame using CURRENT fork pose
        fork_world = sf.compute_fork_world_points_full(fork_pos, fork_rot)

        # Query SDF to find violations
        fork_container = sf._points_to_container_frame(fork_world)
        h_values, _ = sf.sdf.query(fork_container)

        header = Header()
        header.frame_id = "world"
        header.stamp = rospy.Time.now()

        # Helper to pack RGB colors into a float32 for PointCloud2
        def pack_rgb(r, g, b):
            a = 255
            packed = (a << 24) | (r << 16) | (g << 8) | b
            return struct.unpack('f', struct.pack('I', packed))[0]

        color_green = pack_rgb(0, 255, 0)
        color_red = pack_rgb(255, 0, 0)

        # Fields definition for X, Y, Z and RGB
        fields_rgb = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1),
        ]

        # 1. Publish ALL safety points (Green)
        points_green = []
        for i in range(fork_world.shape[0]):
            points_green.append([
                float(fork_world[i, 0]),
                float(fork_world[i, 1]),
                float(fork_world[i, 2]),
                color_green
            ])
        cloud_msg_green = pc2.create_cloud(header, fields_rgb, points_green)
        self.pub_safety_points.publish(cloud_msg_green)

        # 2. Publish only VIOLATION points (Red)
        violations = fork_world[h_values < 0]
        if violations.shape[0] > 0:
            points_red = []
            for v in violations:
                points_red.append([
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                    color_red
                ])
            viol_msg = pc2.create_cloud(header, fields_rgb, points_red)
            self.pub_safety_violations.publish(viol_msg)

            # Log the worst violation for diagnosis
            worst_h = np.min(h_values)
            n_viol = np.sum(h_values < 0)
            rospy.loginfo(f"[Safety VIZ] {n_viol}/{len(h_values)} points in violation, "
                          f"worst h={worst_h:.4f} ({worst_h*1000:.1f}mm)")

    def _publish_sdf_visualization(self):
        """Extracts and publishes ONLY the SDF 'no-go zone' boundary for RViz."""
        import sensor_msgs.point_cloud2 as pc2
        from std_msgs.msg import Header
        import struct

        if self.safety_filter is None or self.safety_filter.sdf is None:
            return

        sdf = self.safety_filter.sdf
        nx, ny, nz = sdf.n_voxels
        voxel_size = sdf.voxel_size
        bounds_min = sdf.bounds_min

        # Générer les coordonnées de la grille 3D
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')

        points_x = bounds_min[0] + (ix + 0.5) * voxel_size
        points_y = bounds_min[1] + (iy + 0.5) * voxel_size
        points_z = bounds_min[2] + (iz + 0.5) * voxel_size

        points_container = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3)
        sdf_values = sdf.sdf_grid.reshape(-1)

        # ── SÉLECTION DE LA "NO-GO ZONE" ──
        # SDF <= 0.0 : On est à l'intérieur du conteneur (collision).
        # SDF >= -0.010 : On limite à une coquille de 1 cm d'épaisseur.
        # Pourquoi une coquille ? Si on publiait le volume plein, les points 
        # transparents s'empileraient visuellement et le résultat serait opaque.
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

        # On colore tout en rouge uni
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
        msg = PoseArray()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()

        for pose_9d in trajectory:
            pos = pose_9d[:3]
            rot_6d = pose_9d[3:]
            rot_mat = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]
            M = np.eye(4)
            M[:3, :3] = rot_mat
            quat = tft.quaternion_from_matrix(M)

            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            msg.poses.append(p)

        rospy.loginfo(f"Fork tip: {np.linalg.norm(self.fork_pos):.4f}")
        self.pub_trajectory.publish(msg)


if __name__ == "__main__":
    import sensor_msgs.point_cloud2 as pc2
    try:
        FlowMatchingInferenceNodeFork()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass