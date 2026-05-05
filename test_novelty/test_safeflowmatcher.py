import os
import time
import torch
import numpy as np
import pybullet as pb
import rospkg

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Modèles et Utilitaires
from src.vision_processing.diffusion_model_train.Train_Fork_FM import FlowMatchingAgent
from vision_processing import fast_ik_module
from third_party.SafeFlowMatcher.diffuser.models.rdf_cbf import RDF_CBF
from third_party.RDF.panda_layer.panda_layer import PandaLayer
from third_party.RDF.rdf_core import RDFCore
from third_party.RDF.rdf_barrier import RDF_Barrier

# ==============================================================================
# 1. UTILITAIRES DE CONVERSION 9D <-> SE(3)
# ==============================================================================

def decode_9d_to_se3_gpu(traj_9d):
    """
    Orthogonalisation Gram-Schmidt 100% vectorisée sur GPU.
    Zéro boucle for, zéro transfert CPU pendant le calcul.
    """
    original_shape = traj_9d.shape
    if len(original_shape) == 3:
        B, H, D = original_shape
        traj = traj_9d.reshape(B * H, D)
    else:
        traj = traj_9d
        
    pos = traj[:, :3]
    c1 = traj[:, 3:6]
    c2 = traj[:, 6:9]
    
    # Gram-Schmidt vectorisé
    b1 = torch.nn.functional.normalize(c1, dim=-1)
    dot_product = torch.sum(b1 * c2, dim=-1, keepdim=True)
    b2 = c2 - dot_product * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Construction du batch de matrices 4x4
    N = traj.shape[0]
    T = torch.eye(4, device=traj_9d.device).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, 0] = b1
    T[:, :3, 1] = b2
    T[:, :3, 2] = b3
    T[:, :3, 3] = pos
    
    return T # Retourne un tenseur [N, 4, 4] sur le GPU

def decode_9d_to_se3(traj_9d):
    """
    Convertit un tenseur (B, H, 9) en une liste de matrices 4x4 (SE3).
    Format 9D : [x, y, z, r11, r12, r21, r22, r31, r32]
    Attention : La convention ici est orientée COLONNES.
    """
    if isinstance(traj_9d, torch.Tensor):
        traj_np = traj_9d.detach().cpu().numpy()
    else:
        traj_np = np.array(traj_9d)
        
    original_shape = traj_np.shape
    if len(original_shape) == 3:
        B, H, D = original_shape
        traj_np = traj_np.reshape(B * H, D)
    else:
        H, D = original_shape
        B = 1
        
    N = traj_np.shape[0]
    poses_se3 = []
    
    for i in range(N):
        pos = traj_np[i, :3]
        rot_6d = traj_np[i, 3:]
        
        # Gram-Schmidt (colonnes)
        c1 = rot_6d[:3]
        c2 = rot_6d[3:]
        
        b1 = c1 / np.linalg.norm(c1)
        b2 = c2 - np.dot(b1, c2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        
        R = np.column_stack((b1, b2, b3))
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        poses_se3.append(T)
        
    return poses_se3, B, H

def encode_se3_to_9d(poses_se3, B, H):
    """
    Convertit une liste de matrices 4x4 (SE3) en un tenseur (B, H, 9).
    """
    N = len(poses_se3)
    traj_9d = np.zeros((N, 9), dtype=np.float32)
    
    for i in range(N):
        T = poses_se3[i]
        pos = T[:3, 3]
        R = T[:3, :3]
        
        c1 = R[:, 0]
        c2 = R[:, 1]
        
        traj_9d[i, :3] = pos
        traj_9d[i, 3:6] = c1
        traj_9d[i, 6:9] = c2
        
    traj_9d = traj_9d.reshape(B, H, 9)
    return torch.tensor(traj_9d, device='cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# 2. CLASSE SAFE FLOW MATCHER
# ==============================================================================

class SafeFlowMatcherPipeline:
    def __init__(self, model_path, urdf_path, ee_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔄 Chargement du FlowMatchingAgent depuis {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        self.fm_agent = FlowMatchingAgent(
            action_dim=self.config.get('action_dim', 9),
            obs_horizon=self.config.get('obs_horizon', 2),
            pred_horizon=self.config.get('pred_horizon', 16),
            stats=checkpoint.get('stats', None)
        ).to(self.device)
        self.fm_agent.load_state_dict(checkpoint['model_state_dict'])
        self.fm_agent.eval()
        print("✅ FlowMatchingAgent chargé.")

        print("🔄 Chargement FastIK...")
        self.ik_module = fast_ik_module.FastIK(urdf_path, ee_name)
        print("✅ FastIK chargé.")

        print("🔄 Chargement RDF (PandaLayer + BP_8.pt)...")
        self.robot_layer = PandaLayer(self.device)
        rospack = rospkg.RosPack()
        model_path_rdf = rospack.get_path('vision_processing') + '/third_party/RDF/models/BP_8.pt'
        model_rdf = torch.load(model_path_rdf, map_location=self.device, weights_only=False)
        self.rdf_core = RDFCore(8, -1.0, 1.0, self.robot_layer, self.device, model_rdf)
        self.rdf_barrier = RDF_Barrier(self.rdf_core, d_safe=0.05)
        print("✅ RDF chargé.")
        
        # Warm-start IK persistant entre les pas de correction
        self._q_warm = None

    def fk_9d_single(self, q_single):
        """
        FK pour un seul vecteur articulaire q [7] -> pose 9D [9].
        Compatible avec torch.func.jacfwd / vmap.
        Note: on opere en float32 explicitement.
        """
        q_b = q_single.unsqueeze(0)  # [1, 7], dtype preserve par l'appelant
        base = torch.eye(4, device=self.device, dtype=q_single.dtype).unsqueeze(0)
        T = self.robot_layer.get_transformations_each_link(base, q_b)[-1][0]  # [4,4]
        pos = T[:3, 3]    # [3]
        r1  = T[:3, 0]    # [3]
        r2  = T[:3, 1]    # [3]
        return torch.cat([pos, r1, r2])  # [9]

    def generate_baseline(self, obs_dict, num_steps=10):
        """
        Génère une trajectoire nue (sans CBF) avec num_steps pas d'Euler réguliers.
        """
        B = obs_dict['agent_pos'].shape[0]
        H = self.fm_agent.pred_horizon
        D = self.fm_agent.action_dim
        
        obs_norm = {
            'point_cloud': obs_dict['point_cloud'].to(self.device),
            'agent_pos': self.fm_agent.normalizer.normalize(obs_dict['agent_pos'].to(self.device))
        }
        
        torch.manual_seed(42)
        x_0 = torch.randn(B, H, D, device=self.device)
        
        A_nom = x_0.clone()
        dt = 1.0 / num_steps
        history = [] # Snap intermédiaires
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.ones(B, device=self.device) * t_val
            with torch.no_grad():
                v = self.fm_agent.forward(obs_norm, A_nom, t)
            A_nom = A_nom + v * dt
            history.append(self.fm_agent.normalizer.unnormalize(A_nom).detach().cpu())
            
        return self.fm_agent.normalizer.unnormalize(A_nom), history

    def generate_safe(self, obs_dict, curr_pos, q_start,
                      Tp=1, Tc=4, P_obs=None,
                      alpha=2.0, eps=0.5, rho=0.9, delta=0.01,
                      lam_jac=1e-4, t_w=0.9):
        """
        PC-Integrator SafeFlowMatcher avec CBF dans l'espace des poses 9D.

        Mapping correct : grad_h_x = (JJ^T + lam*I)^{-1} J grad_h_q
        QP en forme fermée dans R9 (pas de ProxSuite).
        Relaxation schedule w_t pour une convergence progressive.
        """
        B  = obs_dict['agent_pos'].shape[0]
        H  = self.fm_agent.pred_horizon   # 16 waypoints
        D  = self.fm_agent.action_dim      # 9
        N  = B * H                         # points à traiter
        dev = self.device

        obs_norm = {
            'point_cloud': obs_dict['point_cloud'].to(dev),
            'agent_pos': self.fm_agent.normalizer.normalize(obs_dict['agent_pos'].to(dev))
        }

        torch.manual_seed(42)
        x_0 = torch.randn(B, H, D, device=dev)

        # ── PREDICTOR : un seul grand pas Euler depuis le bruit ────────────
        t_zero = torch.zeros(B, device=dev)
        with torch.no_grad():
            v_pred = self.fm_agent.forward(obs_norm, x_0, t_zero)
        A_tau = x_0 + 1.0 * v_pred          # [B, H, 9]  (espace normalisé)

        # ── CORRECTOR ─────────────────────────────────────────────────────
        dt_tau = 1.0 / Tc
        prof   = {'fm': 0.0, 'ik': 0.0, 'rdf': 0.0, 'jac': 0.0, 'math': 0.0}
        history_safe = []

        # Warm-start IK (persistant entre pas de correction)
        # FastIK attend des q en 9D (le bras a 9 joints dans le URDF fork)
        q_warm = (np.zeros((N, 9)) if self._q_warm is None
                  else self._q_warm[:N])

        for i in range(Tc):
            tau = (i + 0.5) * dt_tau          # midpoint rule (plus stable)
            t   = torch.full((B,), tau, device=dev)

            # 1. Champ FM normalisé (vanishing time-scale)
            t0 = time.perf_counter()
            with torch.no_grad():
                v_theta = self.fm_agent.forward(obs_norm, A_tau, t)
            prof['fm'] += (time.perf_counter() - t0) * 1000
            v_tilde_norm = alpha * (1.0 - tau) * v_theta  # [B, H, 9] normalisé

            if P_obs is None:
                A_tau = A_tau + v_tilde_norm * dt_tau
                history_safe.append(
                    self.fm_agent.normalizer.unnormalize(A_tau).detach().cpu())
                continue

            # 2. Dénormaliser pour avoir les poses physiques (en mètres)
            A_unnorm = self.fm_agent.normalizer.unnormalize(A_tau)  # [B, H, 9]
            A_world  = A_unnorm.clone()
            A_world[..., :3] += curr_pos.to(dev)
            x_flat = A_world.reshape(N, 9)           # [N, 9] poses cibles

            # v_tilde en espace physique (pour le QP fermé)
            A_next_unnorm = self.fm_agent.normalizer.unnormalize(
                A_tau + v_tilde_norm * dt_tau
            )
            A_next_world = A_next_unnorm.clone()
            A_next_world[..., :3] += curr_pos.to(dev)
            v_world_flat = (A_next_world - A_world).reshape(N, 9)  # [N, 9]

            # 3. IK batchée (cibles physiques → angles articulaires)
            t0 = time.perf_counter()
            se3_np = decode_9d_to_se3_gpu(x_flat).detach().cpu().numpy()
            se3_list = [se3_np[j] for j in range(N)]
            q_np = self.ik_module.solve_batch(se3_list, q_warm[0])  # [N, 9]
            q_warm = q_np                             # warm-start pour prochain pas
            prof['ik'] += (time.perf_counter() - t0) * 1000

            q = torch.from_numpy(q_np[:, :7]).float().to(dev).requires_grad_(True)  # [N, 7]

            # 4. RDF : h et grad_h dans l'espace ARTICULAIRE
            t0 = time.perf_counter()
            h_q, grad_h_q, _ = self.rdf_barrier(q, P_obs)  # h:[N], grad:[N,7]
            prof['rdf'] += (time.perf_counter() - t0) * 1000

            # 5. Jacobienne FK 9x7 vmap-ee (UNE seule passe GPU, forward-mode AD)
            t0 = time.perf_counter()
            q_float = q.detach().float()  # [N, 7] float32 garanti
            with torch.no_grad():
                J = torch.func.vmap(
                    torch.func.jacfwd(self.fk_9d_single)  # forward-mode: preserve float32
                )(q_float)                    # [N, 9, 7] float32
            prof['jac'] += (time.perf_counter() - t0) * 1000

            # 6. Mapping articulaire → pose : grad_h_x = (JJ^T + lam I)^{-1} J grad_h_q
            t0 = time.perf_counter()
            with torch.no_grad():
                JJt = torch.bmm(J, J.transpose(1, 2))   # [N, 9, 9]
                JJt = JJt + lam_jac * torch.eye(9, device=dev).unsqueeze(0)
                rhs = torch.bmm(J, grad_h_q.unsqueeze(-1))  # [N, 9, 1]
                grad_h_x = torch.linalg.solve(JJt, rhs).squeeze(-1)  # [N, 9]

                # 7. QP fermé dans R9 (solution analytique à une contrainte)
                # Relaxation schedule : lisse au début, strict à la fin
                if tau <= t_w:
                    w_t = 200.0 * (1.0 - np.exp(3.0 * (tau / t_w - 1.0)))
                else:
                    w_t = 0.0

                diff = h_q - delta
                beta = eps * torch.sign(diff) * diff.abs().pow(rho)   # [N]

                dot    = (grad_h_x * v_world_flat).sum(-1)             # [N]
                constr = dot + beta                                     # >= 0 si safe

                denom  = (grad_h_x * grad_h_x).sum(-1) + w_t ** 2     # [N]
                lam_qp = torch.clamp(-constr / denom.clamp_min(1e-8), min=0.0)
                u_world = v_world_flat + lam_qp.unsqueeze(-1) * grad_h_x  # [N, 9]

            n_viol = (h_q < 0).sum().item()
            if i % 2 == 0:
                print(f"  [Step {i:02d}] min(h): {h_q.min().item():.4f} | "
                      f"Violations: {n_viol}/{N} | w_t: {w_t:.1f}")

            # 8. Convertir u_world (déplac. physique) → v normalisé puis intégrer
            # u_world est en mètres/pas ; on en déduit A_world_next puis on renormalise
            A_world_next = A_world + u_world.reshape(B, H, 9)
            A_world_next[..., :3] -= curr_pos.to(dev)    # revenir en relatif
            A_tau_next = self.fm_agent.normalizer.normalize(A_world_next)
            # v_safe_norm est implicite dans le delta
            A_tau = A_tau_next
            prof['math'] += (time.perf_counter() - t0) * 1000

            history_safe.append(
                self.fm_agent.normalizer.unnormalize(A_tau).detach().cpu())

        # Sauvegarder le warm-start pour le prochain appel
        self._q_warm = q_warm if P_obs is not None else self._q_warm

        return self.fm_agent.normalizer.unnormalize(A_tau), prof, history_safe


# ==============================================================================
# 3. TEST DÉGÉNÉRÉ (SANS OBSTACLE)
# ==============================================================================

def run_degenerate_test():
    print("=" * 60)
    print("🚀 NOUVEAU TEST DÉGÉNÉRÉ (PC INTEGRATOR)")
    print("=" * 60)
    
    model_path = "./models/last_fm_model_high_dim_CFM_relative.ckpt"
    rospack = rospkg.RosPack()
    vp_path = rospack.get_path('vision_processing')
    # CHANGEMENT IMPORTANT : On utilise le même URDF que la formation / RDF (Panda Fork) !
    urdf_path = vp_path + "/urdf/panda_fork.urdf"
    ee_name = "fork_tip"
    
    if not os.path.exists(model_path):
        print("❌ Modèle introuvable.")
        return
        
    pipeline = SafeFlowMatcherPipeline(model_path, urdf_path, ee_name)
    
    # Mock observation
    B = 1
    curr_pos = torch.tensor([0.5, 0.0, 0.3], dtype=torch.float32)
    obs_dict = {
        'point_cloud': torch.zeros(B, 256, 3), 
        'agent_pos': torch.zeros(B, 2, 9) 
    }
    
    q_start = np.zeros(9)
    
    print("\n🔄 Génération Baseline (10 pas Euler purs)...")
    A_baseline, _ = pipeline.generate_baseline(obs_dict, num_steps=2)
    
    print("🔄 Génération SafeFlowMatcher (1 pred + 50 corr vanishing)...")
    A_safe, _, _ = pipeline.generate_safe(obs_dict, curr_pos, q_start, Tp=1, Tc=50)
    
    # Validation Finale
    end_baseline = A_baseline[0, -1, :3]
    end_safe = A_safe[0, -1, :3]
    dist = torch.norm(end_baseline - end_safe).item()
    
    print("\n🎯 RÉSULTATS DU TEST DÉGÉNÉRÉ")
    print("-" * 40)
    print(f"Position finale Baseline : {end_baseline.tolist()}")
    print(f"Position finale Safe     : {end_safe.tolist()}")
    print(f"Distance à la cible      : {dist*1000:.2f} mm")
    
    if dist < 0.01: # Moins de 1 cm
        print("✅ SUCCÈS ! Le PC Integrator atterrit au même endroit que la Baseline.")
    else:
        print("❌ ÉCHEC ! L'intégration diverge trop.")
    print("=" * 60)

def run_obstacle_test():
    print("=" * 60)
    print("🚀 NOUVEAU TEST OBSTACLE (PHASE 4)")
    print("=" * 60)
    
    model_path = "./models/best_fm_model_high_dim_CFM_relative_dropout.ckpt"
    rospack = rospkg.RosPack()
    vp_path = rospack.get_path('vision_processing')
    urdf_path = vp_path + "/urdf/panda_fork.urdf"
    ee_name = "fork_tip"
    
    if not os.path.exists(model_path):
        print("❌ Modèle introuvable.")
        return
        
    pipeline = SafeFlowMatcherPipeline(model_path, urdf_path, ee_name)
    
    B = 1
    curr_pos = torch.tensor([0.5, 0.0, 0.3], dtype=torch.float32)
    obs_dict = {
        'point_cloud': torch.zeros(B, 256, 3), 
        'agent_pos': torch.zeros(B, 2, 9) 
    }
    q_start = np.zeros(9)
    
    print("\n🔄 Génération Baseline (Nominal)...")
    A_baseline, history_baseline = pipeline.generate_baseline(obs_dict, num_steps=2)
    
    # Création d'un obstacle synthétique (Sphère) directement sur le chemin nominal
    # On prend le milieu de la trajectoire baseline
    mid_pos = A_baseline[0, 8, :3].detach().cpu().numpy() + curr_pos.numpy() + np.array([0.0, -0.10, 0.10])
    
    # Sphère de 100 points
    phi = np.random.uniform(0, np.pi, 100)
    theta = np.random.uniform(0, 2 * np.pi, 100)
    r = 0.05
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    P_obs = np.column_stack((x, y, z)) + mid_pos
    P_obs_tensor = torch.tensor(P_obs, dtype=torch.float32, device='cuda')
    
    print(f"\n🧱 Obstacle placé au milieu du trajet à : {mid_pos}")
    
    print("🔄 Génération SafeFlowMatcher (CBF ACTIVÉE)...")
    t0 = time.perf_counter()
    A_safe, prof, history_safe = pipeline.generate_safe(obs_dict, curr_pos, q_start, Tp=1, Tc=10, P_obs=P_obs_tensor)
    dt_total = (time.perf_counter() - t0) * 1000
    
    print("\n📊 PROFILING DE LA BOUCLE CORRECTOR (10 itérations)")
    print("-" * 40)
    print(f"🧠 FM Forward : {prof['fm']:.2f} ms")
    print(f"⚙️ IK Batch   : {prof['ik']:.2f} ms")
    print(f"🛡️ RDF Eval   : {prof['rdf']:.2f} ms")
    print(f"📐 Jac_9D     : {prof['jac']:.2f} ms")
    print(f"🧮 Math/Proj  : {prof['math']:.2f} ms")
    
    print("\n🎯 RÉSULTATS DU TEST OBSTACLE")
    print("-" * 40)
    print(f"⏱️ Temps total de génération : {dt_total:.2f} ms")
    
    dist_baseline_to_obs = torch.norm(A_baseline[0, :, :3] + curr_pos.to('cuda') - torch.tensor(mid_pos, device='cuda'), dim=1).min().item()
    dist_safe_to_obs = torch.norm(A_safe[0, :, :3] + curr_pos.to('cuda') - torch.tensor(mid_pos, device='cuda'), dim=1).min().item()
    
    print(f"📏 Distance min Baseline <-> Obstacle : {dist_baseline_to_obs*1000:.2f} mm")
    print(f"📏 Distance min Safe     <-> Obstacle : {dist_safe_to_obs*1000:.2f} mm")
    
    if dist_safe_to_obs > dist_baseline_to_obs + 0.01:
        print("✅ SUCCÈS ! La trajectoire a été déviée pour éviter l'obstacle.")
    else:
        print("❌ ÉCHEC ! L'évitement n'est pas suffisant.")
    print("=" * 60)

    print("\n\U0001f5a5\ufe0f Lancement de la visualisation 3D (PyVista) avec slider d'\u00e9tapes...")
    try:
        import pyvista as pv

        curr_pos_np = curr_pos.numpy()
        obs_cloud = pv.PolyData(P_obs)

        n_baseline = len(history_baseline)
        n_safe = len(history_safe)
        n_total = n_baseline + n_safe

        baseline_steps = [h[0, :, :3].numpy() + curr_pos_np for h in history_baseline]
        safe_steps     = [h[0, :, :3].numpy() + curr_pos_np for h in history_safe]

        # ----------------------------------------------------------------
        # Pre-compute robot meshes at the final pose of each trajectory
        # ----------------------------------------------------------------
        def get_robot_meshes_for_traj(A_final_unnorm):
            """Returns trimesh robot meshes at the last waypoint of a trajectory."""
            A_world = A_final_unnorm.clone()
            A_world[..., :3] += curr_pos.to(pipeline.device)
            last_se3 = decode_9d_to_se3_gpu(A_world[0, -1:, :].reshape(1, 9))
            last_se3_np = last_se3.detach().cpu().numpy()[0]
            q_last = pipeline.ik_module.solve_batch([last_se3_np], q_start)[0]
            q_7dof = torch.tensor(q_last[:7], dtype=torch.float32,
                                  device=pipeline.device).unsqueeze(0)
            pose_base = torch.eye(4, device=pipeline.device).unsqueeze(0)
            meshes = pipeline.robot_layer.get_forward_robot_mesh(pose_base, q_7dof)[0]
            return meshes, q_last

        def trimesh_to_pv(tm):
            """Converts a trimesh Mesh to a PyVista PolyData."""
            faces = np.hstack([np.full((len(tm.faces), 1), 3), tm.faces]).ravel()
            return pv.PolyData(np.array(tm.vertices, dtype=np.float32), faces)

        print("  \U0001f916 Computing IK for baseline final pose...")
        meshes_baseline, q_bl = get_robot_meshes_for_traj(A_baseline)
        print("  \U0001f916 Computing IK for safe final pose...")
        meshes_safe,     q_sf = get_robot_meshes_for_traj(A_safe)

        pv_meshes_baseline = [trimesh_to_pv(m) for m in meshes_baseline]
        pv_meshes_safe     = [trimesh_to_pv(m) for m in meshes_safe]

        # ----------------------------------------------------------------
        # Build the PyVista scene
        # ----------------------------------------------------------------
        plotter = pv.Plotter()
        plotter.set_background('#0d0d1a')

        # Reference ground plane
        grid = pv.Plane(center=(0.4, 0.0, 0.0), direction=(0, 0, 1),
                        i_size=1.2, j_size=1.2)
        plotter.add_mesh(grid, color='#1a1a2e', opacity=0.4)

        # Obstacle
        plotter.add_points(obs_cloud, color='#ff4444', point_size=14,
                           render_points_as_spheres=True, name='Obstacle')

        # Final trajectories (dim reference)
        plotter.add_mesh(pv.lines_from_points(baseline_steps[-1]),
                         color='#4488ff', line_width=2, opacity=0.25, name='BaseFinal')
        plotter.add_mesh(pv.lines_from_points(safe_steps[-1]),
                         color='#44ff88', line_width=2, opacity=0.25, name='SafeFinal')

        # Robot baseline (grey) at final pose
        for j, m in enumerate(pv_meshes_baseline):
            plotter.add_mesh(m, color='#aaaaaa', opacity=0.50,
                             smooth_shading=True, name=f'RobotBase_{j}')

        # Robot safe (bright green) at final pose
        for j, m in enumerate(pv_meshes_safe):
            plotter.add_mesh(m, color='#00ff88', opacity=0.80,
                             smooth_shading=True, name=f'RobotSafe_{j}')

        plotter.add_text(
            f'Baseline  (gris)   q=[{", ".join(f"{v:.2f}" for v in q_bl[:7])}]\n'
            f'Safe      (vert)   q=[{", ".join(f"{v:.2f}" for v in q_sf[:7])}]',
            font_size=10, color='white', position='upper_left', name='InfoLabel'
        )

        # Dynamic actors for the trajectory slider
        dummy_pts = np.zeros((2, 3), dtype=np.float32)
        plotter.add_mesh(pv.lines_from_points(dummy_pts),
                         color='white', line_width=5, name='DynamicLine')
        plotter.add_points(pv.PolyData(dummy_pts), color='white',
                           point_size=10, render_points_as_spheres=True,
                           name='DynamicPts')
        plotter.add_text('Step 0', font_size=13, color='white',
                         position='upper_right', name='StepLabel')

        def update_step(value):
            step = int(value)
            if step < n_baseline:
                pts = baseline_steps[step]
                color = '#4488ff'
                phase = f'Baseline \u2014 Step {step+1}/{n_baseline}'
            else:
                s = step - n_baseline
                pts = safe_steps[s]
                color = '#44ff88'
                phase = f'Corrector CBF \u2014 Step {s+1}/{n_safe}'
            plotter.remove_actor('DynamicLine')
            plotter.remove_actor('DynamicPts')
            plotter.add_mesh(pv.lines_from_points(pts), color=color,
                             line_width=5, name='DynamicLine')
            plotter.add_points(pv.PolyData(pts), color=color, point_size=12,
                               render_points_as_spheres=True, name='DynamicPts')
            plotter.remove_actor('StepLabel')
            plotter.add_text(phase, font_size=13, color='white',
                             position='upper_right', name='StepLabel')

        plotter.add_slider_widget(
            update_step,
            rng=[0, n_total - 1],
            value=n_total - 1,   # Start at the final step to show robot poses
            title="Pas d'integration",
            pointa=(0.1, 0.05),
            pointb=(0.9, 0.05),
            style='modern',
        )
        update_step(n_total - 1)
        plotter.add_axes()
        plotter.show(title='SafeFlowMatcher \u2014 Pose finale du robot')

    except ImportError:
        print("\u26a0\ufe0f PyVista n'est pas installe.")

if __name__ == "__main__":
    # run_degenerate_test()
    run_obstacle_test()
