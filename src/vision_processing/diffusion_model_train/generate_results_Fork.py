import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Nécessaire pour les plots 3D
import pickle
import json
import pandas as pd
from tqdm import tqdm
import rospkg
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import time
import seaborn as sns

# # Import de tes classes locales
# from Train_urdf import DP3AgentRobust, Normalizer
# from Data_Loader_urdf import Robot3DDataset

from Train_Fork import DP3AgentRobust, Normalizer
from Data_Loader_Fork import Robot3DDataset

# ==============================================================================
# 1. UTILITAIRES MATHÉMATIQUES & CONFIGURATION
# ==============================================================================

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ortho6d_to_rotation_matrix(d6):
    """
    Convertit la représentation 6D (B, T, 6) en Matrice de Rotation (B, T, 3, 3)
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)

def compute_angular_error(matrix1, matrix2):
    """
    Calcule l'erreur angulaire (en degrés) entre deux matrices de rotation.
    """
    # R_diff = R1 * R2^T
    r_diff = np.matmul(matrix1, matrix2.transpose(0, 1, 3, 2))
    trace = np.trace(r_diff, axis1=-2, axis2=-1)
    val = (trace - 1.0) / 2.0
    cos_theta = np.clip(val, -1.0, 1.0) # Clip numérique de sécurité
    return np.arccos(cos_theta) * (180.0 / np.pi)

# ==============================================================================
# 2. LOGIQUE D'INFÉRENCE (Issue du Code 1)
# ==============================================================================

def infer_single(model, normalizer, sample, scheduler, DEVICE):
    """
    Exécute une inférence unique et retourne la prédiction brute ainsi que les erreurs finales.
    """
    model.eval()
    pcd = sample['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['agent_pos'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        norm_agent_pos = normalizer.normalize(raw_agent_pos, 'agent_pos')
        global_cond = torch.cat([model.point_encoder(pcd),
                                 model.robot_mlp(norm_agent_pos.reshape(1, -1))], dim=-1)

        noisy_action = torch.randn((1, 16, 9), device=DEVICE)

        for t in scheduler.timesteps:
            timesteps = torch.tensor([t], device=DEVICE).long()
            noise_pred = model.noise_pred_net(noisy_action, torch.tensor([t], device=DEVICE).long(), global_cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample
        
        pred_action = normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0]
        gt_action = sample['action'].numpy()

        # --- Calcul des Erreurs pour cette inférence ---
        # 1. Distance Finale
        final_dist = np.linalg.norm(pred_action[-1, :3] - gt_action[-1, :3])
        
        # 2. Erreur Angulaire Finale
        gt_rot_6d = gt_action[-1, 3:]
        # Expansion des dims pour matcher la fonction (B, T, 6) -> ici on traite juste le dernier pas
        gt_rot_mat = ortho6d_to_rotation_matrix(gt_rot_6d[None, None, :])[0, 0]
        pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:][None, None, :])[0, 0]
        
        r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
        cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
        final_angle = np.arccos(cos_theta) * (180 / np.pi)
        
    return pred_action, final_dist, final_angle

# ==============================================================================
# 3. VISUALISATIONS (Fusion Code 1 & Code 2)
# ==============================================================================

def generate_train_history():
    print("📊 Génération du Graphique d'Historique d'Entraînement...")
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    # Chemin spécifique du Code 2
    pkl_file = os.path.join(package_path, 'pkl_files', 'train_history_fork_only_SAMPLE_NO_AUG.pkl')
    
    if not os.path.exists(pkl_file):
        print(f"⚠️ Fichier historique introuvable : {pkl_file}")
        return None

    with open(pkl_file, 'rb') as f:
        history = pickle.load(f)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.yscale('log')
    plt.title(f"Convergence - {history['config'].get('model_name', 'DP3')}")
    plt.xlabel("Époques")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('01_train_history.svg', format='svg')
    plt.close()
    return history

def plot_3d_multimodality_with_orientation(model, normalizer, sample, scheduler, DEVICE, idx, n_samples=20):
    """
    Affiche 3D complet (Code 2) : Nuage de points + Spaghettis + Orientation.
    Utilise infer_single pour la cohérence.
    """
    print(f"🍝 Génération du Plot 3D Complet (Pos + Orient) pour Seq {idx}...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # A. Environnement
    pcd = sample['point_cloud'].numpy()
    if len(pcd) > 512:
        indices = np.random.choice(len(pcd), 512, replace=False)
        pcd_vis = pcd[indices]
    else:
        pcd_vis = pcd
    ax.scatter(pcd_vis[:,0], pcd_vis[:,1], pcd_vis[:,2], c=pcd_vis[:,2], cmap='Greys', s=5, alpha=0.2, label='Env (PCD)')

    # B. Vérité Terrain
    gt = sample['action'].numpy()
    obs = sample['agent_pos'].numpy()
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g--', linewidth=3, label='Vérité Terrain (GT)')
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], 'b-o', linewidth=2, label='Historique')

    # C. Inférences Multiples
    final_preds = []
    for _ in range(n_samples):
        # Utilisation de la fonction d'inférence modulaire
        pred_action, _, _ = infer_single(model, normalizer, sample, scheduler, DEVICE)
        final_preds.append(pred_action)
        ax.plot(pred_action[:, 0], pred_action[:, 1], pred_action[:, 2], color='red', alpha=0.15, linewidth=1)

    # D. Orientation Finale (Visualisation)
    last_pred = final_preds[0][-1] 
    gt_last = gt[-1]

    # Repère GT (Vert)
    gt_rot = ortho6d_to_rotation_matrix(gt_last[3:][None, None, :])[0, 0]
    origin = gt_last[:3]
    scale = 0.05
    ax.quiver(origin[0], origin[1], origin[2], gt_rot[0,0], gt_rot[1,0], gt_rot[2,0], color='green', lw=2, length=scale)
    
    # Repère Pred (Rouge)
    pred_rot = ortho6d_to_rotation_matrix(last_pred[3:][None, None, :])[0, 0]
    origin_p = last_pred[:3]
    ax.quiver(origin_p[0], origin_p[1], origin_p[2], pred_rot[0,0], pred_rot[1,0], pred_rot[2,0], color='red', lw=2, length=scale, label='Orient. Pred')

    ax.set_title(f"Seq {idx} : Analyse 3D (Rouge=Pred, Vert=GT)")
    # Auto-scale centré sur le robot
    center = np.mean(gt, axis=0)[:3]
    radius = 0.2
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    plt.legend()
    plt.savefig(f'02_3d_spaghetti_seq{idx}.png', dpi=200)
    plt.close()

def plot_multimodality_spaghetti_2d(model, normalizer, sample, scheduler, DEVICE, idx, n_samples=50):
    """
    Analyse 2D (Code 1) : Utile pour voir la dispersion XY sans le bruit du PCD.
    """
    print(f"🍝 Génération du Spaghetti Plot 2D pour Seq {idx}...")
    plt.figure(figsize=(10, 10))
    gt = sample['action'].numpy()
    obs = sample['agent_pos'].numpy()

    final_dists = []
    final_angles = []

    for i in range(n_samples):
        pred, final_dist, final_angle = infer_single(model, normalizer, sample, scheduler, DEVICE)
        final_dists.append(final_dist)
        final_angles.append(final_angle)
        plt.plot(pred[:, 0], pred[:, 1], color='red', alpha=0.1, linewidth=1.5, label='Prédictions' if i == 0 else "")
    
    plt.plot(obs[:,0], obs[:,1], 'b-o', label='Passé', linewidth=2.5)
    plt.plot(gt[:, 0], gt[:, 1], 'g--', label='Vérité Terrain', linewidth=3)
    plt.title(f"Dispersion 2D - Seq {idx} (N={n_samples})")
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig(f'02_spaghetti_2d_clean_seq{idx}.png', dpi=300)
    plt.close()

def plot_rotation_error_over_time(gt_action, pred_action, idx):
    """
    Affiche l'écart angulaire timestep par timestep (Code 2).
    """
    gt_tensor = torch.from_numpy(gt_action).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_action).unsqueeze(0)
    
    gt_rot = ortho6d_to_rotation_matrix(gt_tensor[..., 3:])[0]
    pred_rot = ortho6d_to_rotation_matrix(pred_tensor[..., 3:])[0]
    
    errors = []
    for t in range(len(gt_action)):
        r_diff = np.dot(gt_rot[t], pred_rot[t].T)
        val = (np.trace(r_diff) - 1.0) / 2.0
        cos_theta = np.clip(val, -1.0, 1.0)
        err_deg = np.arccos(cos_theta) * (180.0 / np.pi)
        errors.append(err_deg)
        
    plt.figure(figsize=(8, 4))
    plt.plot(errors, 'r-o', label='Erreur Angulaire')
    plt.axhline(y=15, color='orange', linestyle='--', label='Seuil (15°)')
    plt.xlabel("Timestep")
    plt.ylabel("Erreur (°)")
    plt.title(f"Dérive Angulaire - Seq {idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'05_rotation_error_seq{idx}.png')
    plt.close()

def make_diffusion_gif(model, normalizer, sample, scheduler, DEVICE, idx):
    """
    Version fusionnée : Visuels riches du Code 2 (GT, PCD) + Pause "Clean" du Code 1.
    """
    print(f"🎬 Génération du GIF pour Seq {idx}...")
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

    # Pause à la fin (Code 1 feature)
    for _ in range(15): frames.append(frames[-1])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    gt = sample['action'].numpy()
    center = np.mean(gt[:, :3], axis=0)
    
    def update(f):
        ax.clear()
        path = frames[f]
        # Robot Path
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=3, label='Diffusion')
        # GT Ghost (Code 2 feature)
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color='green', alpha=0.3, linewidth=1, label='Target')
        # PCD (Code 2 feature)
        pc_np = sample['point_cloud'].numpy()[::5]
        ax.scatter(pc_np[:,0], pc_np[:,1], pc_np[:,2], s=1, c='gray', alpha=0.1)

        step_text = f"Step: {f}/{len(frames)-15}" if f < (len(frames)-15) else "Final Prediction"
        ax.set_title(f"Génération Diffusive\n{step_text}")
        ax.set_xlim(center[0]-0.2, center[0]+0.2)
        ax.set_ylim(center[1]-0.2, center[1]+0.2)
        ax.set_zlim(center[2]-0.2, center[2]+0.2)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save(f'03_diffusion_process_seq{idx}.gif', writer='pillow')
    plt.close()

# ==============================================================================
# 4. ANALYSE QUANTITATIVE (Fusion Stats Code 1 + Seuils Code 2)
# ==============================================================================

def run_quantitative_analysis(model, normalizer, val_dataset, scheduler, DEVICE, n_samples=20):
    model.eval()
    results = []
    
    # Seuils de succès (Code 2)
    THRESHOLD_POS = 0.02 # 2 cm
    THRESHOLD_ROT = 15.0 # 15 degrés
    
    print(f"📊 Analyse quantitative ({n_samples} samples/seq)...")
    # On limite pour ne pas surcharger en démo, sinon utiliser range(len(val_dataset))
    num_eval_sequences = min(30, len(val_dataset)) 
    indices = np.linspace(0, len(val_dataset)-1, num_eval_sequences, dtype=int)

    for idx in tqdm(indices):
        sample = val_dataset[idx]
        
        # --- Multi-Inférence (Code 1 logic) ---
        seq_preds = []
        seq_final_angles = []
        seq_final_dists = []
        
        for _ in range(n_samples):
            pred, final_dist, final_angle = infer_single(model, normalizer, sample, scheduler, DEVICE)
            seq_preds.append(pred)
            seq_final_dists.append(final_dist)
            seq_final_angles.append(final_angle)
        
        seq_preds = np.array(seq_preds) # (N, 16, 9)
        
        # --- Calcul Stats (Code 1 & 2 Fusionnés) ---
        # Moyennes des erreurs
        mean_fde = np.mean(seq_final_dists)
        mean_rot = np.mean(seq_final_angles)
        
        # Success Rate (Code 2) : Combien d'échantillons ont réussi ?
        success_count = sum([(d < THRESHOLD_POS and a < THRESHOLD_ROT) for d, a in zip(seq_final_dists, seq_final_angles)])
        success_rate = (success_count / n_samples) * 100
        
        # ADE (Average Displacement Error)
        gt_action = sample['action'].numpy()
        gt_pos_all = gt_action[:, :3]
        pred_pos_all = seq_preds[:, :, :3]
        ade_dist = np.mean(np.linalg.norm(pred_pos_all - gt_pos_all, axis=2), axis=1) # Pour chaque sample
        mean_ade = np.mean(ade_dist)

        # Variance/Dispersion (Code 1)
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        path_variance = np.mean(np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))

        results.append({
            'seq_idx': idx,
            'ADE': mean_ade * 100,      # cm
            'FDE': mean_fde * 100,      # cm
            'RotErr': mean_rot,         # deg
            'SuccessRate': success_rate,# %
            'Dispersion': path_variance * 100 # cm
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("RESUME DES PERFORMANCES (Validation Subset)")
    print("="*50)
    print(f"Success Rate Moyen : {df['SuccessRate'].mean():.2f} %")
    print(f"Position Error (ADE): {df['ADE'].mean():.2f} cm")
    print(f"Rotation Error (Fin): {df['RotErr'].mean():.2f} deg")
    print(f"Dispersion Moyenne  : {df['Dispersion'].mean():.2f} cm")
    print("="*50)

    return df

def plot_error_distributions(df):
    """
    Code 2 (Scatter plot) est très pertinent pour corréler Position et Rotation.
    """
    print("📊 Génération histogrammes erreurs...")
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(df['SuccessRate'], bins=10, kde=False, ax=ax1, color="green")
    ax1.set_title("Distribution du Taux de Succès par Séquence")
    ax1.set_xlabel("Succès (%)")

    sns.scatterplot(data=df, x='ADE', y='RotErr', ax=ax2, hue='SuccessRate', palette='viridis', alpha=0.8)
    ax2.set_title("Corrélation Erreur Position vs Rotation")
    ax2.set_xlabel("Erreur Position (cm)")
    ax2.set_ylabel("Erreur Rotation (deg)")

    plt.tight_layout()
    plt.savefig('04_metrics_distribution.png', dpi=300)
    plt.close()

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Utilisation du device : {DEVICE}")
    
    # 1. Chemins (Code 2 specifics)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    
    # Chemins spécifiques demandés
    ckpt_path = os.path.join(pkg_path, "models", "dp3_policy_best_diffusers_fork_only_SAMPLE_NO_AUG.ckpt") 
    stats_path = os.path.join(pkg_path, "normalization_stats_fork_only.json")

    # 2. Chargement Données & Modèle
    print("📂 Chargement modèle et stats...")
    with open(stats_path, 'r') as f: stats = json.load(f)
    normalizer = Normalizer(stats)
    
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42)
    print(f"✅ Validation Set : {len(val_dataset)} séquences")

    model = DP3AgentRobust(action_dim=9, robot_state_dim=9).to(DEVICE)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ ERREUR : Checkpoint introuvable : {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False) 
    print("✅ Poids chargés.")

    scheduler = DDIMScheduler(num_train_timesteps=100, 
                                beta_schedule='squaredcos_cap_v2', 
                                prediction_type='sample', 
                                clip_sample=True)
    scheduler.set_timesteps(10)

    # 3. Visualisation sur une séquence spécifique
    sample_idx = 25 # Comme dans Code 1
    if len(val_dataset) > sample_idx:
        sample = val_dataset[sample_idx]
        
        # A. Graphique Historique (Code 2)
        generate_train_history()
        
        # B. 3D Spaghetti + Orientation (Code 2)
        plot_3d_multimodality_with_orientation(model, normalizer, sample, scheduler, DEVICE, sample_idx)
        
        # C. 2D Spaghetti Clean (Code 1 - Bonus)
        plot_multimodality_spaghetti_2d(model, normalizer, sample, scheduler, DEVICE, sample_idx)

        # D. GIF (Fusion)
        make_diffusion_gif(model, normalizer, sample, scheduler, DEVICE, sample_idx)

    # 4. Analyse Globale
    df = run_quantitative_analysis(model, normalizer, val_dataset, scheduler, DEVICE, n_samples=20)
    plot_error_distributions(df)
    
    # 5. Outlier Detection (Code 2 Logic)
    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    print(f"\n🚨 PIRE SEQUENCE (Idx {int(worst_seq['seq_idx'])}) :")
    print(f"   Succès: {worst_seq['SuccessRate']:.1f}% | ADE: {worst_seq['ADE']:.2f}cm | Rot: {worst_seq['RotErr']:.1f}°")
    
    # Génération visuels pour la pire séquence
    idx_worst = int(worst_seq['seq_idx'])
    worst_sample = val_dataset[idx_worst]
    
    print(f"📸 Génération visuels pour l'outlier Seq {idx_worst}...")
    plot_3d_multimodality_with_orientation(model, normalizer, worst_sample, scheduler, DEVICE, idx_worst)
    
    # Générer une vraie prédiction pour le plot d'erreur temporelle
    pred_worst, _, _ = infer_single(model, normalizer, worst_sample, scheduler, DEVICE)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)
    
    print("\n✅ Analyse terminée. Vérifie les images PNG générées.")

if __name__ == "__main__":
    main()