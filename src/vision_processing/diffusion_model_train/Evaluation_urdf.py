import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import rospkg
import json
from scipy.spatial.transform import Rotation as R

# Import du modèle et des outils
from Train_urdf import DP3AgentRobust, Normalizer
from Data_Loader_urdf import Robot3DDataset

# --- UTILITAIRES ROTATION ---
def ortho6d_to_rotation_matrix(d6):
    """ Reconstruit la matrice de rotation (3x3) à partir de la 6D """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    
    # Normalisation (Gram-Schmidt)
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    
    # (..., 3, 3)
    result = np.stack([x, y, z], axis=-1)
    return result

def compute_errors(gt_action, pred_action):
    """ Calcule les erreurs de position (cm) et rotation (degrés) """
    # 1. Erreur Position (Euclidienne)
    pos_gt = gt_action[:, :3]
    pos_pred = pred_action[:, :3]
    pos_err = np.linalg.norm(pos_gt - pos_pred, axis=1) # (T,)
    mean_pos_err_cm = np.mean(pos_err) * 100 # Convertir en cm
    
    # 2. Erreur Rotation (Angulaire)
    rot_gt = ortho6d_to_rotation_matrix(gt_action[:, 3:])
    rot_pred = ortho6d_to_rotation_matrix(pred_action[:, 3:])
    
    rot_err_deg = []
    for i in range(len(rot_gt)):
        r_gt = R.from_matrix(rot_gt[i])
        r_pred = R.from_matrix(rot_pred[i])
        # Calcul de l'angle entre les deux rotations
        diff = r_gt * r_pred.inv()
        deg = diff.magnitude() * (180 / np.pi)
        rot_err_deg.append(deg)
        
    mean_rot_err_deg = np.mean(rot_err_deg)
    
    return mean_pos_err_cm, mean_rot_err_deg

def evaluate_robust():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Evaluation Avancée sur {DEVICE}")

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, "dp3_policy_last_robust_urdf.ckpt")
    stats_path = os.path.join(pkg_path, "normalization_stats.json")

    # 1. Chargement Stats
    if not os.path.exists(stats_path):
        print("❌ Erreur: Stats manquantes.")
        return
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    normalizer = Normalizer(stats)

    # 2. Dataset
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42)
    print(f"📦 Validation Set: {len(val_dataset)} séquences")

    # 3. Modèle
    model = DP3AgentRobust(action_dim=9, robot_state_dim=9, obs_horizon=2, pred_horizon=16).to(DEVICE)
    
    if not os.path.exists(ckpt_path):
        print("❌ Erreur: Checkpoint manquant.")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # 4. Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # 5. Évaluation sur 10 exemples aléatoires
    np.random.seed(42)
    indices = np.random.choice(len(val_dataset), 10, replace=False)
    # indices = np.array([150,175,200])
    for i, idx in enumerate(indices):
        print(f"\n--- Séquence {idx} ---")
        sample = val_dataset[idx]
        
        pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
        raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
        raw_gt_action = sample['action'].numpy()
        
        # --- Inférence ---
        with torch.no_grad():
            norm_agent_pos = normalizer.normalize(raw_agent_pos, 'agent_pos')
            
            p_feat = model.point_encoder(pcd)
            r_feat = model.robot_mlp(norm_agent_pos.reshape(1, -1))
            global_cond = torch.cat([p_feat, r_feat], dim=-1)
            
            noisy_action = torch.randn((1, 16, 9), device=DEVICE)
            noise_scheduler.set_timesteps(100)
            
            for t in noise_scheduler.timesteps:
                timesteps = torch.tensor([t], device=DEVICE).long()
                noise_pred = model.noise_pred_net(noisy_action, timesteps, global_cond)
                noisy_action = noise_scheduler.step(noise_pred, t, noisy_action).prev_sample

            pred_action = normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0]

        # --- Métriques ---
        pos_err, rot_err = compute_errors(raw_gt_action, pred_action)
        print(f"📊 Erreur Position: {pos_err:.2f} cm | Erreur Rotation: {rot_err:.2f}°")

        # --- Visualisation ---
        visualize_advanced(
            pcd[0].cpu().numpy(), 
            raw_agent_pos[0].cpu().numpy(), 
            raw_gt_action, 
            pred_action, 
            idx,
            pos_err,
            rot_err
        )

def visualize_advanced(pcd, obs_hist, gt_action, pred_action, idx, pos_err, rot_err):
    """
    Génère 4 vues (3D, Top, Front, Side) et affiche l'orientation.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Seq {idx} | Err Pos: {pos_err:.1f}cm | Err Rot: {rot_err:.1f}°', fontsize=16)

    # Subplot 1: 3D Isometric
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    setup_plot_3d(ax3d, pcd, obs_hist, gt_action, pred_action, "Vue 3D Isométrique")

    # Subplot 2: Top View (X-Y)
    ax_top = fig.add_subplot(2, 2, 2)
    setup_plot_2d(ax_top, pcd, obs_hist, gt_action, pred_action, 0, 1, "Vue de Haut (X-Y)")

    # Subplot 3: Front View (X-Z) 
    ax_front = fig.add_subplot(2, 2, 3)
    setup_plot_2d(ax_front, pcd, obs_hist, gt_action, pred_action, 0, 2, "Vue de Face (X-Z) [Hauteur]")

    # Subplot 4: Side View (Y-Z) 
    ax_side = fig.add_subplot(2, 2, 4)
    setup_plot_2d(ax_side, pcd, obs_hist, gt_action, pred_action, 1, 2, "Vue de Côté (Y-Z) [Hauteur]")

    save_name = f"eval_advanced_seq{idx}.png"
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print(f"✅ Image sauvegardée : {save_name}")

def setup_plot_3d(ax, pcd, obs, gt, pred, title):
    ax.scatter(pcd[::5, 0], pcd[::5, 1], pcd[::5, 2], s=1, c='gray', alpha=0.2)
    ax.plot(obs[:,0], obs[:,1], obs[:,2], c='blue', label='Passé')
    ax.plot(gt[:,0], gt[:,1], gt[:,2], c='green', linestyle='--', label='Vérité')
    ax.plot(pred[:,0], pred[:,1], pred[:,2], c='red', linewidth=2, label='Pred')
    
    mats = ortho6d_to_rotation_matrix(pred[::4, 3:]) 
    pos = pred[::4, :3]
    dirs = mats[:, :, 0] * 0.05 
    ax.quiver(pos[:,0], pos[:,1], pos[:,2], dirs[:,0], dirs[:,1], dirs[:,2], color='orange', alpha=0.8)

    ax.set_title(title)
    ax.legend()

def setup_plot_2d(ax, pcd, obs, gt, pred, dim1, dim2, title):
    ax.scatter(pcd[::5, dim1], pcd[::5, dim2], s=1, c='gray', alpha=0.1)
    ax.plot(obs[:, dim1], obs[:, dim2], c='blue', marker='.', markersize=5)
    ax.plot(gt[:, dim1], gt[:, dim2], c='green', linestyle='--')
    ax.plot(pred[:, dim1], pred[:, dim2], c='red', linewidth=2, alpha=0.8)
    
    ax.scatter(pred[0, dim1], pred[0, dim2], c='lime', marker='*', s=100, zorder=5)
    ax.scatter(pred[-1, dim1], pred[-1, dim2], c='red', marker='x', s=80, zorder=5)
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')

if __name__ == "__main__":
    # Décommente la ligne suivante pour visualiser des exemples complets
    evaluate_robust()