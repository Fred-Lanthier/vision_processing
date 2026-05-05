import torch
import numpy as np
import time

# Imports de vos dépendances C++ et Bernstein
# Ces imports supposent que vous lancez depuis le workspace src/vision_processing/
try:
    from vision_processing import fast_ik_module
    from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
    from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
    from third_party.SDF_Bernstein_Basis.bernstein_barrier import BernsteinBarrier
    from third_party.RDF.urdf_layer import URDFLayer
except ImportError as e:
    print("Veuillez vous assurer que le module C++ est bien build et l'environnement sourcé :", e)


class SADFlowerPlanner:
    def __init__(self, urdf_path, voxel_dir, weights_dir, device='cuda'):
        self.device = device
        
        self.ik_solver = fast_ik_module.FastIK(urdf_path, "panda_hand")
        
        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=None,
            voxel_dir=voxel_dir
        )
        
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(weights_dir, robot_name='panda')
        link_names = ['panda_link0', 'panda_link1', 'panda_link2',
                    'panda_link3', 'panda_link4', 'panda_link5',
                    'panda_link6', 'panda_link7']
        weight_handler.add_models(link_names, robot_name='panda')
        
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer,
                                            self.device, link_names)
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=0.08)
        
        self.I7 = torch.eye(7, device=self.device).float()  # ← juste ça

    def decode_9d_to_se3(self, A_9d):
        """
        Fonction utilitaire: Convertit un vecteur 9D [p_x, p_y, p_z, r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
        vers une matrice 4x4 (SE3).
        A adapter selon l'origine de votre représentation de rotation (ex: orthonormalisation Gram-Schmidt).
        """
        mat = np.eye(4)
        mat[:3, 3] = A_9d[:3]
        
        # Reconstitution d'une base orthogonale à partir des deux vecteurs directeurs
        r1 = A_9d[3:6]
        r1 = r1 / (np.linalg.norm(r1) + 1e-8)
        r2 = A_9d[6:9]
        r2 = r2 - np.dot(r1, r2) * r1  # Gram-Schmidt
        r2 = r2 / (np.linalg.norm(r2) + 1e-8)
        r3 = np.cross(r1, r2)
        
        mat[:3, 0] = r1
        mat[:3, 1] = r2
        mat[:3, 2] = r3
        return mat

    def generate_safe_trajectory(self, fm_agent, normalizer, obs_norm, curr_pos, q_init, P_obs_static, 
                                 N_STEPS=10, T0=0.6, C=0.5, lam=1e-5):
        """
        Boucle d'inférence SAD-Flower Modifiée : ~10 pas ODE
        """
        B = 1 # ou inférer depuis obs_norm
        H = 16 # Horizon des waypoints
        dt = 1.0 / N_STEPS
        
        # Initialisation du bruit latent : τ_0 ~ p_0
        A = torch.randn(B, H, 9, device=self.device)
        
        # --- PHASE 0 : WARM-START IK INITIAL ---
        # Warm-start : config courante répétée
        Q_warm_np = np.tile(q_init.cpu().numpy()[:7], (H, 1))  # [H, 7]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # --- BOUCLE ODE PRINCIPALE ---
        for i in range(N_STEPS):
            t = i * dt
            t_tensor = torch.full((B,), t, device=self.device)

            with torch.no_grad():
                v = fm_agent.forward(obs_norm, A, t_tensor)   # [B, H, 9]

            if t < T0:
                A = A + v * dt

            else:
                phi_t = C / (1.0 - t + 1e-4) ** 2

                A_unnorm    = normalizer.unnormalize(A)
                A_world_np  = A_unnorm.squeeze(0).cpu().numpy()
                A_world_np[..., :3] += curr_pos.cpu().numpy()

                A_next_unnorm = normalizer.unnormalize(A + v * dt)
                A_next_world  = A_next_unnorm.squeeze(0).cpu().numpy()
                A_next_world[..., :3] += curr_pos.cpu().numpy()
                v_world = torch.from_numpy(A_next_world - A_world_np).float().to(self.device)

                se3_list = [self.decode_9d_to_se3(A_world_np[k]) for k in range(H)]
                Q, Js = self.ik_solver.solve_batch_with_jacobians(se3_list, Q_warm_np[0])
                Q_warm_np = Q
                q_batch = torch.from_numpy(Q[:, :7]).float().to(self.device).requires_grad_(True)
                J_batch = torch.from_numpy(np.stack(Js)).float().to(self.device)[:, :, :7]  # 1 seul transfert GPU

                h_q, grad_h_q, _ = self.barrier(q_batch, P_obs_static)
                h_q      = h_q.detach()
                grad_h_q = grad_h_q.detach()

                JtJ      = torch.bmm(J_batch.transpose(1,2), J_batch) + lam * self.I7.unsqueeze(0)
                grad_h_x = torch.bmm(J_batch,
                            torch.linalg.solve(JtJ, grad_h_q.unsqueeze(-1))).squeeze(-1)

                dot    = (grad_h_x * v_world).sum(-1)
                constr = dot + phi_t * h_q
                denom  = (grad_h_x**2).sum(-1).clamp(min=1e-8)
                lam_qp = torch.clamp(-constr / denom, min=0)
                u_world = lam_qp.unsqueeze(-1) * grad_h_x

                A_corr_world = A_world_np + u_world.detach().cpu().numpy()
                A_corr_world[..., :3] -= curr_pos.cpu().numpy()
                u_norm = normalizer.normalize(
                    torch.from_numpy(A_corr_world).float().to(self.device).unsqueeze(0)
                ) - A

                A = A + (v + u_norm) * dt
        
        torch.cuda.synchronize()
        print(f"✅ Trajectoire générée en {time.perf_counter() - start_time:.4f}s [{N_STEPS} étapes]")
        
        # Retourne A_world final
        return A
