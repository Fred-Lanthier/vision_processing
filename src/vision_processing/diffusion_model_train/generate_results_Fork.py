import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import pandas as pd
from tqdm import tqdm
import rospkg
import time
import seaborn as sns
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Import de vos modules locaux
# Assurez-vous que Train_Fork.py et Data_Loader_Fork.py sont dans le même dossier ou le PYTHONPATH
from Train_Fork import DP3AgentRobust, Normalizer
from Data_Loader_Fork import Robot3DDataset, seed_everything

# ==============================================================================
# 1. UTILITAIRES MATHÉMATIQUES
# ==============================================================================

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

def plot_training_history(history):
    """ Affiche les courbes de perte si disponibles dans le checkpoint """
    if history is None: return
    print("📊 Génération du Graphique d'Historique...")
    plt.figure(figsize=(10, 5))
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.yscale('log')
    plt.title("Convergence de l'entraînement")
    plt.xlabel("Époques")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('01_train_history.svg', format='svg')
    plt.close()

# ==============================================================================
# 2. LOGIQUE D'INFÉRENCE & TIMING
# ==============================================================================

def infer_single(model, sample, scheduler, DEVICE):
    """
    Exécute une inférence complète et retourne les prédictions + le temps d'exécution.
    """
    model.eval()
    
    # Préparation des tenseurs (Batch size = 1)
    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)
    
    # --- DÉBUT CHRONO ---
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # 1. Normalisation (interne au modèle)
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        
        # 2. Encodage Vision + Proprio
        global_cond = torch.cat([model.point_encoder(pcd),
                                 model.robot_mlp(norm_agent_pos.reshape(1, -1))], dim=-1)

        # 3. Boucle de Diffusion Inverse
        noisy_action = torch.randn((1, 16, 9), device=DEVICE)

        for t in scheduler.timesteps:
            # Prédiction du bruit
            noise_pred = model.noise_pred_net(noisy_action, torch.tensor([t], device=DEVICE).long(), global_cond)
            # Step du scheduler
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample
        
        # 4. Un-normalization
        pred_action = model.normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0]
    
    # --- FIN CHRONO ---
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) # en secondes

    # --- Calcul des Métriques ---
    gt_action = sample['action'].numpy()

    # 1. Distance Euclidienne Finale (FDE)
    final_dist = np.linalg.norm(pred_action[-1, :3] - gt_action[-1, :3])
    
    # 2. Erreur Angulaire Finale
    gt_rot_6d = gt_action[-1, 3:]
    gt_rot_mat = ortho6d_to_rotation_matrix(gt_rot_6d[None, None, :])[0, 0]
    pred_rot_mat = ortho6d_to_rotation_matrix(pred_action[-1, 3:][None, None, :])[0, 0]
    
    r_diff = np.dot(gt_rot_mat, pred_rot_mat.T)
    cos_theta = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
    final_angle = np.arccos(cos_theta) * (180 / np.pi)
        
    return pred_action, final_dist, final_angle, inference_time

# ==============================================================================
# 3. VISUALISATIONS COMPLÈTES (Repère Local)
# ==============================================================================

def plot_3d_local_frame(model, sample, scheduler, DEVICE, idx, n_samples=10):
    """
    Affiche le nuage de points et les trajectoires dans le repère LOCAL.
    """
    print(f"🍝 Génération du Plot 3D (Repère Local) pour Seq {idx}...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # A. Environnement
    pcd = sample['obs']['point_cloud'].numpy()
    if len(pcd) > 512:
        indices = np.random.choice(len(pcd), 512, replace=False)
        pcd_vis = pcd[indices]
    else:
        pcd_vis = pcd
    ax.scatter(pcd_vis[:,0], pcd_vis[:,1], pcd_vis[:,2], c=pcd_vis[:,2], cmap='Greys', s=5, alpha=0.2, label='Env (Local)')

    # B. Vérité Terrain
    gt = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g--', linewidth=3, label='Vérité Terrain (GT)')
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], 'b-o', linewidth=2, label='Historique')

    # C. Inférences Multiples
    final_preds = []
    for _ in range(n_samples):
        pred_action, _, _, _ = infer_single(model, sample, scheduler, DEVICE)
        final_preds.append(pred_action)
        ax.plot(pred_action[:, 0], pred_action[:, 1], pred_action[:, 2], color='red', alpha=0.15, linewidth=1)

    # D. Orientation Finale (Visualisation)
    last_pred = final_preds[0][-1] 
    gt_last = gt[-1]

    # Repères (Quivers)
    gt_rot = ortho6d_to_rotation_matrix(gt_last[3:][None, None, :])[0, 0]
    origin = gt_last[:3]
    scale = 0.05
    ax.quiver(origin[0], origin[1], origin[2], gt_rot[0,0], gt_rot[1,0], gt_rot[2,0], color='green', lw=2, length=scale)
    
    pred_rot = ortho6d_to_rotation_matrix(last_pred[3:][None, None, :])[0, 0]
    origin_p = last_pred[:3]
    ax.quiver(origin_p[0], origin_p[1], origin_p[2], pred_rot[0,0], pred_rot[1,0], pred_rot[2,0], color='red', lw=2, length=scale, label='Orient. Pred')

    ax.set_title(f"Seq {idx} : Vue Centrée sur la Fourchette (Local Frame)\n(0,0,0) = Position Actuelle")
    
    # Zoom auto
    center = np.mean(gt[:, :3], axis=0)
    radius = 0.15 
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    plt.legend()
    plt.savefig(f'02_3d_local_seq{idx}.png', dpi=150)
    plt.close()

def plot_multimodality_spaghetti_2d(model, sample, scheduler, DEVICE, idx, n_samples=50):
    """ Affiche la dispersion XY """
    print(f"🍝 Génération du Spaghetti Plot 2D pour Seq {idx}...")
    plt.figure(figsize=(10, 10))
    gt = sample['action'].numpy()
    obs = sample['obs']['agent_pos'].numpy()

    for i in range(n_samples):
        pred, _, _, _ = infer_single(model, sample, scheduler, DEVICE)
        plt.plot(pred[:, 0], pred[:, 1], color='red', alpha=0.1, linewidth=1.5, label='Prédictions' if i == 0 else "")
    
    plt.plot(obs[:,0], obs[:,1], 'b-o', label='Passé', linewidth=2.5)
    plt.plot(gt[:, 0], gt[:, 1], 'g--', label='Vérité Terrain', linewidth=3)
    plt.title(f"Dispersion 2D (Local Frame) - Seq {idx}")
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig(f'02_spaghetti_2d_seq{idx}.png', dpi=150)
    plt.close()

def plot_rotation_error_over_time(gt_action, pred_action, idx):
    """ Affiche l'écart angulaire timestep par timestep """
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

def make_diffusion_gif(model, sample, scheduler, DEVICE, idx):
    """ Génère un GIF du processus de diffusion """
    print(f"🎬 Génération du GIF pour Seq {idx}...")
    model.eval()
    pcd = sample['obs']['point_cloud'].unsqueeze(0).to(DEVICE)
    raw_agent_pos = sample['obs']['agent_pos'].unsqueeze(0).to(DEVICE)
    frames = []
    
    with torch.no_grad():
        norm_agent_pos = model.normalizer.normalize(raw_agent_pos, 'agent_pos')
        global_cond = torch.cat([model.point_encoder(pcd),
                                 model.robot_mlp(norm_agent_pos.reshape(1, -1))], dim=-1)

        noisy_action = torch.randn((1, 16, 9), device=DEVICE)
        frames.append(model.normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0])

        for t in scheduler.timesteps:
            noise_pred = model.noise_pred_net(noisy_action, torch.tensor([t], device=DEVICE).long(), global_cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample
            frames.append(model.normalizer.unnormalize(noisy_action, 'action').cpu().numpy()[0])

    for _ in range(10): frames.append(frames[-1]) # Pause à la fin

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    gt = sample['action'].numpy()
    center = np.mean(gt[:, :3], axis=0) # Sera proche de 0
    
    def update(f):
        ax.clear()
        path = frames[f]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=3, label='Diffusion')
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color='green', alpha=0.3, linewidth=1, label='Target')
        
        # PCD Subsampled
        pc_np = sample['obs']['point_cloud'].numpy()[::5]
        ax.scatter(pc_np[:,0], pc_np[:,1], pc_np[:,2], s=1, c='gray', alpha=0.1)

        step_text = f"Step: {f}/{len(frames)-10}" if f < (len(frames)-10) else "Final Prediction"
        ax.set_title(f"Génération Diffusive\n{step_text}")
        
        # Limites fixes autour de 0 (Local Frame)
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_zlim(-0.15, 0.15)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save(f'03_diffusion_process_seq{idx}.gif', writer='pillow')
    plt.close()

# ==============================================================================
# 4. ANALYSE QUANTITATIVE & TEMPORELLE
# ==============================================================================

def run_quantitative_analysis(model, val_dataset, scheduler, DEVICE, n_samples=20):
    model.eval()
    results = []
    times = []
    
    THRESHOLD_POS = 0.02 # 2cm
    THRESHOLD_ROT = 15.0 # 15 deg
    
    # Nombre de séquences à évaluer (ex: 50 pour aller vite, ou len(val_dataset))
    num_eval = min(50, len(val_dataset)) 
    indices = np.linspace(0, len(val_dataset)-1, num_eval, dtype=int)
    
    print(f"📊 Analyse de {num_eval} séquences (x{n_samples} inférences chacune)...")

    for idx in tqdm(indices):
        sample = val_dataset[idx]
        seq_dists = []
        seq_angles = []
        seq_preds = [] # Pour dispersion
        
        for _ in range(n_samples):
            pred, final_dist, final_angle, dt = infer_single(model, sample, scheduler, DEVICE)
            seq_dists.append(final_dist)
            seq_angles.append(final_angle)
            seq_preds.append(pred)
            times.append(dt)
        
        seq_preds = np.array(seq_preds)
        
        # Moyennes
        mean_fde = np.mean(seq_dists)
        mean_rot = np.mean(seq_angles)
        
        # Dispersion
        mean_pred_path = np.mean(seq_preds[:, :, :3], axis=0)
        path_variance = np.mean(np.linalg.norm(seq_preds[:, :, :3] - mean_pred_path, axis=2))

        # Succès
        success_count = sum([(d < THRESHOLD_POS and a < THRESHOLD_ROT) for d, a in zip(seq_dists, seq_angles)])
        success_rate = (success_count / n_samples) * 100
        
        # ADE
        gt_pos = sample['action'].numpy()[:, :3]
        pred_pos_all = seq_preds[:, :, :3]
        ade_dist = np.mean(np.linalg.norm(pred_pos_all - gt_pos, axis=2))

        results.append({
            'seq_idx': idx,
            'ADE': ade_dist * 100,      # cm
            'FDE': mean_fde * 100,      # cm
            'RotErr': mean_rot,         # deg
            'SuccessRate': success_rate,# %
            'Dispersion': path_variance * 100 # cm
        })

    df = pd.DataFrame(results)
    avg_time = np.mean(times)
    freq = 1.0 / avg_time if avg_time > 0 else 0

    print("\n" + "="*60)
    print("🚀 RÉSULTATS DE VALIDATION & PERFORMANCE")
    print("="*60)
    print(f"Temps moyen d'inférence : {avg_time*1000:.2f} ms")
    print(f"Fréquence d'inférence   : {freq:.2f} Hz")
    print("-" * 60)
    print(f"Success Rate Moyen      : {df['SuccessRate'].mean():.2f} %")
    print(f"Erreur Position (ADE)   : {df['ADE'].mean():.2f} cm")
    print(f"Erreur Position (FDE)   : {df['FDE'].mean():.2f} cm")
    print(f"Erreur Rotation (Fin)   : {df['RotErr'].mean():.2f} deg")
    print(f"Dispersion Moyenne      : {df['Dispersion'].mean():.2f} cm")
    print("="*60)

    # Histogramme Temps
    plt.figure(figsize=(6, 4))
    plt.hist(np.array(times)*1000, bins=30, color='purple', alpha=0.7)
    plt.xlabel('Temps (ms)')
    plt.title(f'Latence Inférence (Moy: {avg_time*1000:.1f}ms)')
    plt.savefig('06_inference_latency.png')
    plt.close()

    return df

def plot_error_distributions(df):
    """ Plots Histogrammes et Scatter """
    print("📊 Génération histogrammes erreurs...")
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(df['SuccessRate'], bins=10, kde=False, ax=ax1, color="green")
    ax1.set_title("Distribution du Taux de Succès")
    ax1.set_xlabel("Succès (%)")

    sns.scatterplot(data=df, x='ADE', y='RotErr', ax=ax2, hue='SuccessRate', palette='viridis', alpha=0.8)
    ax2.set_title("Position vs Rotation Error")
    ax2.set_xlabel("ADE (cm)")
    ax2.set_ylabel("Rotation Error (deg)")

    plt.tight_layout()
    plt.savefig('04_metrics_distribution.png', dpi=300)
    plt.close()

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    seed_everything(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device : {DEVICE}")
    
    # 1. Chemins
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(pkg_path, "models", "Last_Fork_256_points_Relative.ckpt") # ou Last_Fork.ckpt

    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint introuvable : {ckpt_path}, essai Last_Fork...")
        ckpt_path = os.path.join(pkg_path, "models", "Last_Fork.ckpt")
    
    # 2. Dataset (Pre-compute RAM)
    print(f"📂 Chargement Dataset : {data_path}")
    val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42)
    print(f"✅ Validation Set : {len(val_dataset)} séquences")
    
    # 3. Modèle
    print(f"🧠 Chargement Modèle : {ckpt_path}")
    model = DP3AgentRobust(action_dim=9, robot_state_dim=9).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    if 'history' in checkpoint: plot_training_history(checkpoint['history'])
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    # 4. Scheduler (DDIM Rapide pour l'inférence)
    scheduler = DDIMScheduler(num_train_timesteps=100, 
                              beta_schedule='squaredcos_cap_v2', 
                              prediction_type='sample', 
                              clip_sample=True)
    scheduler.set_timesteps(10) # 10 steps = Rapide

    # 5. Visualisation Test (Outliers & Random)
    if len(val_dataset) > 0:
        idx = 25
        plot_3d_local_frame(model, val_dataset[idx], scheduler, DEVICE, idx)
        make_diffusion_gif(model, val_dataset[idx], scheduler, DEVICE, idx)
        plot_multimodality_spaghetti_2d(model, val_dataset[idx], scheduler, DEVICE, idx)

    # 6. Benchmark Complet
    df = run_quantitative_analysis(model, val_dataset, scheduler, DEVICE, n_samples=10)
    plot_error_distributions(df)

    # 7. Pire Séquence (Outlier)
    worst_seq = df.loc[df['SuccessRate'].idxmin()]
    idx_worst = int(worst_seq['seq_idx'])
    print(f"\n🚨 PIRE SEQUENCE (Idx {idx_worst}) : Succès {worst_seq['SuccessRate']:.1f}%")
    
    worst_sample = val_dataset[idx_worst]
    plot_3d_local_frame(model, worst_sample, scheduler, DEVICE, idx_worst)
    
    # Plot erreur temporelle sur un sample de la pire sequence
    pred_worst, _, _, _ = infer_single(model, worst_sample, scheduler, DEVICE)
    plot_rotation_error_over_time(worst_sample['action'].numpy(), pred_worst, idx_worst)
    
    print("\n✅ Analyse terminée. Images sauvegardées.")

if __name__ == "__main__":
    main()