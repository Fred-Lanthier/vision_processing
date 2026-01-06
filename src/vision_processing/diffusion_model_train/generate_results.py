import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import json
import pandas as pd
from tqdm import tqdm
import rospkg
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import time
import seaborn as sns

# Import de tes classes locales
from Train_urdf import DP3AgentRobust, Normalizer
from Data_Loader_urdf import Robot3DDataset

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_train_history():
    print("📊 Génération du Graphique d'Historique d'Entraînement...")
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    pkl_file = os.path.join(package_path, 'train_history_urdf_FPS.pkl')
    
    with open(pkl_file, 'rb') as f:
        history = pickle.load(f)
    
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f"Expérience: {history['config']['model_name']}")
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '01_train_history.svg'), format='svg')
    plt.show()
    return history

def infer_single(model, normalizer, sample, scheduler, DEVICE):
    model.eval()
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE) # Toujours nécessaire pour le modèle
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        norm_agent_pos = normalizer.normalize(raw_agent_pos, 'agent_pos')
        global_cond = torch.cat([model.point_encoder(pcd), 
                                model.robot_mlp(norm_agent_pos.reshape(1, -1))], dim=-1)

        noisy_action = torch.randn((1, 16, 9), device=DEVICE)

        for t in scheduler.timesteps:
            timesteps = torch.tensor([t], device=DEVICE).long()
            noise_pred = model.noise_pred_net(noisy_action, timesteps, global_cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample
        pred_action = normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0]
        gt_action = sample['action'].numpy()

        # Compute Errors
        final_dist_error = np.linalg.norm(pred_action[-1, :2] - gt_action[-1, :2])
        gt_action = sample['action'].cpu().numpy()

        final_dist = np.linalg.norm(pred_action[-1, :2] - gt_action[-1, :2])
        
        # Angle Error (Using 6D rotation converted to matrix)
        gt_rot_6d = gt_action[-1, 3:]
        gt_rot_mat = ortho6d_to_rotation_matrix(gt_rot_6d)
        pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:])
        
        # R_diff = R_gt * R_pred^T
        r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
        # Clip to avoid numerical errors with arccos
        cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
        final_angle = np.arccos(cos_theta) * (180 / np.pi) # In Degrees
        
        return pred_action, final_dist, final_angle

def plot_multimodality_spaghetti(model, normalizer, sample, scheduler, DEVICE, idx, n_samples=100):
    print(f"🍝 Génération du Spaghetti Plot (Sans PCD) pour Seq {idx}...")
    plt.figure(figsize=(10, 10))
    gt = sample['action'].numpy()
    obs = sample['agent_pos'].numpy()

    # 1. Dessiner les prédictions (Spaghettis)
    inference_times = []
    final_dists = []
    final_angles = []
    for i in range(n_samples):
        start_time = time.time()
        pred, final_dist, final_angle = infer_single(model, normalizer, sample, scheduler, DEVICE)
        inference_times.append(time.time() - start_time)
        final_dists.append(final_dist)
        final_angles.append(final_angle)
        plt.plot(pred[:, 0], pred[:, 1], color='red', alpha=0.08, linewidth=1.5, label='Prédictions' if i == 0 else "")
    
    print(f"⏱️ Inference time: {np.mean(inference_times):.4f}s ± {np.std(inference_times):.4f}s")

    # 2. Dessiner le passé et la vérité terrain par-dessus
    plt.plot(obs[:,0], obs[:,1], 'b-o', label='Passé (Observation)', linewidth=2.5)
    plt.plot(gt[:, 0], gt[:, 1], 'g--', label='Vérité Terrain (GT)', linewidth=3)

    plt.title(f"Analyse de la Multi-modalité (Trajectoires seules)\nSéquence {idx} - DDIM 10 steps", fontsize=14)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.1)
    plt.savefig(f'02_spaghetti_no_pcd_seq{idx}.png', dpi=300)
    plt.close()

    # 3. Calculer les erreurs finales
    final_dist_error = np.mean(final_dists) * 100   
    final_angle_error = np.mean(final_angles)
    final_dist_std = np.std(final_dists) * 100
    final_angle_std = np.std(final_angles)
    print(f"Dist Error: {final_dist_error:.4f} ± {final_dist_std:.4f} cm")
    print(f"Angle Error: {final_angle_error:.4f} ± {final_angle_std:.4f} deg")

def make_diffusion_gif(model, normalizer, sample, scheduler, DEVICE, idx):
    print(f"🎬 Génération du GIF (Sans PCD, Pause 3s finale) pour Seq {idx}...")
    model.eval()
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    
    frames = []
    with torch.no_grad():
        norm_agent_pos = normalizer.normalize(raw_agent_pos, 'agent_pos')
        global_cond = torch.cat([model.point_encoder(pcd), 
                                model.robot_mlp(norm_agent_pos.reshape(1, -1))], dim=-1)

        noisy_action = torch.randn((1, 16, 9), device=DEVICE)
        frames.append(normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0])

        for t in scheduler.timesteps:
            noise_pred = model.noise_pred_net(noisy_action, torch.tensor([t], device=DEVICE).long(), global_cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample
            frames.append(normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0])

    # Ajout de la pause de 3 secondes (30 frames à 100ms)
    last_frame = frames[-1]
    for _ in range(30):
        frames.append(last_frame)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(f):
        ax.clear()
        path = frames[f]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=3)
        
        step_text = f"DDIM Step: {f}" if f < (len(frames)-30) else "Prédiction Finale"
        ax.set_title(f"Processus de Génération\n{step_text}")
        
        # Centrage automatique de la vue sur la trajectoire
        ax.set_xlim(raw_agent_pos[0,0,0].cpu()-0.3, raw_agent_pos[0,0,0].cpu()+0.3)
        ax.set_ylim(raw_agent_pos[0,0,1].cpu()-0.3, raw_agent_pos[0,0,1].cpu()+0.3)
        ax.set_zlim(0, 0.8)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save(f'03_diffusion_clean_pause_seq{idx}.gif', writer='pillow')
    plt.close()

def run_quantitative_analysis(model, normalizer, val_dataset, scheduler, DEVICE, n_samples=50):
    model.eval()
    results = []
    inference_times = []

    print(f"📊 Analyse quantitative en cours ({n_samples} échantillons/séquence)...")

    # On limite à 20-30 séquences pour ne pas que ce soit trop long en réunion
    num_eval_sequences = min(30, len(val_dataset))
    indices = np.linspace(0, len(val_dataset)-1, num_eval_sequences, dtype=int)

    for idx in tqdm(indices):
        sample = val_dataset[idx]
        gt_action = sample['action'].numpy() # (T, 9)
        
        # --- Multi-Inférence et Chronométrage ---
        seq_preds = []
        for _ in range(n_samples):
            start_time = time.perf_counter()
            pred = infer_single(model, normalizer, sample, scheduler, DEVICE)
            inference_times.append(time.perf_counter() - start_time)
            seq_preds.append(pred)
        
        seq_preds = np.array(seq_preds) # (50, 16, 9)

        # --- Calcul des Métriques ---
        # 1. Erreur de Position Finale (FDE)
        gt_final_pos = gt_action[-1, :3]
        pred_final_poses = seq_preds[:, -1, :3]
        fde_dist = np.linalg.norm(pred_final_poses - gt_final_pos, axis=1) # (50,)
        
        # 2. Erreur de Position Moyenne (ADE) sur toute la trajectoire
        gt_pos_all = gt_action[:, :3]
        pred_pos_all = seq_preds[:, :, :3]
        ade_dist = np.mean(np.linalg.norm(pred_pos_all - gt_pos_all, axis=2), axis=1) # (50,)

        # 3. Erreur de Rotation Finale (Distance angulaire)
        # On utilise la trace de la matrice pour l'angle : theta = arccos((Tr(R_diff)-1)/2)
        gt_rot_6d = gt_action[-1, 3:]
        gt_rot_mat = ortho6d_to_rotation_matrix(gt_rot_6d)
        
        rot_errors = []
        for s in range(n_samples):
            pred_rot_mat = ortho6d_to_rotation_matrix(seq_preds[s, -1, 3:])
            # R_diff = R_gt * R_pred^T
            r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
            # Clip pour éviter les erreurs de précision flottante avec arccos
            cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle_err = np.arccos(cos_theta) * (180 / np.pi)
            rot_errors.append(angle_err)

        # 4. Variance Inter-échantillons (Dispersion)
        # Moyenne des distances euclidiennes entre les prédictions et leur propre moyenne
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        variance_paths = np.mean(np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))

        results.append({
            'seq_idx': idx,
            'mean_fde_cm': np.mean(fde_dist) * 100,
            'best_fde_cm': np.min(fde_dist) * 100,
            'mean_ade_cm': np.mean(ade_dist) * 100,
            'mean_rot_err_deg': np.mean(rot_errors),
            'path_variance_cm': variance_paths * 100
        })

    # --- Agrégation Finale ---
    df = pd.DataFrame(results)
    
    summary = {
        "Position ADE (Moyenne)": f"{df['mean_ade_cm'].mean():.2f} cm",
        "Position FDE (Moyenne)": f"{df['mean_fde_cm'].mean():.2f} cm",
        "Position FDE (Best-of-50)": f"{df['best_fde_cm'].mean():.2f} cm",
        "Rotation Error": f"{df['mean_rot_err_deg'].mean():.2f} °",
        "Dispersion (Incertitude)": f"{df['path_variance_cm'].mean():.2f} cm",
        "Temps Inférence (Moyen)": f"{np.mean(inference_times)*1000:.1f} ms",
        "Temps Inférence (Std)": f"{np.std(inference_times)*1000:.1f} ms"
    }

    return summary, df

def plot_error_distributions(df):
    """
    Génère des histogrammes pour visualiser la répartition des erreurs ADE et FDE.
    """
    print("📊 Génération des histogrammes de distribution d'erreurs...")
    
    # Configuration du style
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Histogramme de l'ADE (Erreur Moyenne)
    sns.histplot(df['mean_ade_cm'], bins=15, kde=True, ax=ax1, color="royalblue")
    ax1.axvline(df['mean_ade_cm'].mean(), color='red', linestyle='--', 
                label=f"Moyenne: {df['mean_ade_cm'].mean():.2f} cm")
    ax1.set_title("Distribution de l'ADE (Précision globale)", fontsize=14)
    ax1.set_xlabel("Erreur (cm)")
    ax1.set_ylabel("Nombre de séquences")
    ax1.legend()

    # 2. Histogramme du FDE (Erreur Finale)
    sns.histplot(df['mean_fde_cm'], bins=15, kde=True, ax=ax2, color="orange")
    ax2.axvline(df['mean_fde_cm'].mean(), color='red', linestyle='--', 
                label=f"Moyenne: {df['mean_fde_cm'].mean():.2f} cm")
    ax2.set_title("Distribution du FDE (Précision à la cible)", fontsize=14)
    ax2.set_xlabel("Erreur (cm)")
    ax2.set_ylabel("Nombre de séquences")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('04_distribution_erreurs.png', dpi=300)
    plt.show()

def ortho6d_to_rotation_matrix(d6):
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, "dp3_policy_last_robust_urdf_FPS.ckpt")
    stats_path = os.path.join(pkg_path, "normalization_stats_urdf.json")

    with open(stats_path, 'r') as f: stats = json.load(f)
    normalizer = Normalizer(stats)
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42)
    
    model = DP3AgentRobust(action_dim=9, robot_state_dim=9).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')
    scheduler.set_timesteps(10)
    
    sample_idx = 25 
    sample = val_dataset[sample_idx]
    
    generate_train_history()
    plot_multimodality_spaghetti(model, normalizer, sample, scheduler, DEVICE, sample_idx)
    make_diffusion_gif(model, normalizer, sample, scheduler, DEVICE, sample_idx)
    # summary, df = run_quantitative_analysis(model, normalizer, val_dataset, scheduler, DEVICE)
    # plot_error_distributions(df)

    # print("\n✅ Visuels épurés (Trajectoires seules) générés avec succès.")
    # print(summary)

    # worst_ade_row = df.loc[df['mean_ade_cm'].idxmax()]
    # worst_idx = int(worst_ade_row['seq_idx'])
    # worst_error = worst_ade_row['mean_ade_cm']

    # print(f"🚨 Outlier détecté !")
    # print(f"Indice de la séquence : {worst_idx}")
    # print(f"Erreur ADE : {worst_error:.2f} cm")

    # # Trouver aussi la séquence avec la pire erreur de rotation
    # worst_rot_row = df.loc[df['mean_rot_err_deg'].idxmax()]
    # worst_rot_idx = int(worst_rot_row['seq_idx'])
    # print(f"🔄 Pire rotation à l'indice : {worst_rot_idx} ({worst_rot_row['mean_rot_err_deg']:.2f}°)")

    # # Trouver aussi la séquence avec la pire erreur de FDE
    # worst_fde_row = df.loc[df['mean_fde_cm'].idxmax()]
    # worst_fde_idx = int(worst_fde_row['seq_idx'])
    # print(f"🔄 Pire FDE à l'indice : {worst_fde_idx} ({worst_fde_row['mean_fde_cm']:.2f} cm)")

    # Générer les visuels pour les séquences avec les pires erreurs
    # worst_ade_sample = val_dataset[worst_idx]
    # worst_rot_sample = val_dataset[worst_rot_idx]
    # worst_fde_sample = val_dataset[worst_fde_idx]
    # plot_multimodality_spaghetti(model, normalizer, worst_ade_sample, scheduler, DEVICE, worst_idx)
    # make_diffusion_gif(model, normalizer, worst_ade_sample, scheduler, DEVICE, worst_idx)
if __name__ == "__main__":
    main()