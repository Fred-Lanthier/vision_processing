import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg

# Import de ta classe Dataset 
# Assure-toi que le fichier s'appelle bien dataset.py, sinon change l'import
from Data_Loader_urdf import Robot3DDataset

def visualize_sequence(dataset, idx=None, save_dir=None):
    """
    Charge une séquence et affiche :
    1. Le nuage de points COMPLET (Merged_XXXX.npy) sans sous-échantillonnage.
    2. L'historique de l'effecteur (Bleu).
    3. La prédiction future (Rouge).
    """
    if idx is None:
        idx = np.random.randint(0, len(dataset))
    
    # Récupération des infos internes du dataset pour trouver le fichier exact
    # dataset.indices est une liste de tuples (traj_name, start_index)
    traj_name, current_idx = dataset.indices[idx]
    
    print(f"👀 Visualisation Séquence {idx}")
    print(f"   📂 Trajectoire : {traj_name}")
    print(f"   ⏱️  Start Index : {current_idx}")

    # Récupération des données Tenseurs (pour la trajectoire robot)
    data = dataset[idx]
    obs_pos = data['agent_pos'][:, :3].numpy()   # (obs_horizon, 3)
    pred_pos = data['action'][:, :3].numpy()     # (pred_horizon, 3)

    # Récupération du Nuage de Points ORIGINAL (Merged_XXXX.npy)
    # On contourne le __getitem__ qui fait le sampling à 1024 points
    traj_data_cache = dataset.trajectory_cache[traj_name]
    step_data = traj_data_cache[current_idx]
    pcd_path = step_data['pcd_path']
    
    full_pcd = None
    if pcd_path and os.path.exists(pcd_path):
        print(f"   ☁️  Chargement PCD : {os.path.basename(pcd_path)}")
        full_pcd = np.load(pcd_path)
    else:
        print(f"   ⚠️ PCD introuvable : {pcd_path}")
        # Fallback sur le pcd samplé du dataset si le fichier n'est pas là
        full_pcd = data['point_cloud'].numpy()

    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Affichage du Nuage de Points (Merged_XXXX.npy)
    if full_pcd is not None and len(full_pcd) > 0:
        # Optimisation pour Matplotlib (qui rame si > 10k points)
        # On affiche max 5000 points pour la fluidité, mais pris sur l'ensemble
        step = max(1, len(full_pcd) // 5000)
        display_pcd = full_pcd[::step]
        
        # Astuce : Colorer par hauteur (Z) pour mieux voir le relief
        sc = ax.scatter(display_pcd[:, 0], display_pcd[:, 1], display_pcd[:, 2], 
                   c=display_pcd[:, 2], cmap='viridis', s=2, alpha=0.6, 
                   label=f'Merged PCD ({os.path.basename(pcd_path)})')
        
        # Barre de couleur pour comprendre l'échelle Z
        plt.colorbar(sc, ax=ax, label='Hauteur Z (m)', shrink=0.5, pad=0.1)
    
    # 2. Affichage Observation (Passé - Bleu)
    ax.plot(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2], 
            c='blue', linewidth=3, label='Passé (Obs)')
    ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2], 
               c='blue', s=50, marker='o')

    # 3. Affichage Action (Futur - Rouge)
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
            c='red', linewidth=3, label='Futur (Pred)')
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
               c='red', s=40, marker='^')
    
    # Point de départ t=0
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], 
               c='lime', s=150, marker='*', edgecolors='black', label='Start (t=0)')

    # --- ESTHÉTIQUE ---
    ax.set_xlabel('X (Monde)')
    ax.set_ylabel('Y (Monde)')
    ax.set_zlabel('Z (Monde)')
    ax.set_title(f'Visualisation Trajectoire: {traj_name} | Step {current_idx}\nFichier: {os.path.basename(pcd_path)}')
    ax.legend(loc='upper right')
    
    # Égalisation des axes pour ne pas déformer la 3D
    # On prend tous les points pour calculer les limites
    all_points = full_pcd if full_pcd is not None else obs_pos
    if len(all_points) > 0:
        mid_x = (np.max(all_points[:,0]) + np.min(all_points[:,0])) * 0.5
        mid_y = (np.max(all_points[:,1]) + np.min(all_points[:,1])) * 0.5
        mid_z = (np.max(all_points[:,2]) + np.min(all_points[:,2])) * 0.5
        
        max_range = np.max([
            np.max(all_points[:,0]) - np.min(all_points[:,0]),
            np.max(all_points[:,1]) - np.min(all_points[:,1]),
            np.max(all_points[:,2]) - np.min(all_points[:,2])
        ]) / 2.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=30, azim=45)

    # SAUVEGARDE SSH
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Nom de fichier explicite
    filename = f"viz_{traj_name}_step{current_idx:04d}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"✅ Image sauvegardée : {save_path}")

if __name__ == "__main__":
    # Setup
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')

    print("⏳ Chargement du Dataset...")
    dataset = Robot3DDataset(root_dir=data_path, pred_horizon=16, obs_horizon=2)
    
    # Générer 3 exemples aléatoires pour vérifier
    idxs = [1, 35, 69]
    for i in range(3):
        print(f"\n--- Exemple {i+1}/3 ---")
        visualize_sequence(dataset, idx=idxs[i])

    
    # Création des Datasets
train_dataset = Robot3DDataset(data_path, mode='train', val_ratio=0.1)
val_dataset = Robot3DDataset(data_path, mode='val', val_ratio=0.1)

# Création des DataLoaders PyTorch
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, # Pas besoin de shuffle en validation
    num_workers=4
)