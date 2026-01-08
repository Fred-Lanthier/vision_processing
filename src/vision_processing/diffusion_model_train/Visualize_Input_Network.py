import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
import json
from Data_Loader_Fork import Robot3DDataset

def visualize_input_movie():
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision_processing')
    data_path = os.path.join(pkg_path, 'datas', 'Trajectories_preprocess')
    
    # On charge le dataset sans augmentation pour voir la réalité
    dataset = Robot3DDataset(data_path, mode='all', augment=False)
    
    # Choisissons une séquence problématique (ex: une qui monte)
    # On prend une séquence au hasard vers le milieu d'une trajectoire
    idx = np.random.randint(0, len(dataset))
    
    print(f"🎬 Génération du film pour la séquence {idx}...")
    
    # On va récupérer les frames successives pour voir l'évolution
    # Attention: le dataset est indexé par fenêtres glissantes.
    # Pour voir une trajectoire continue, on doit prendre les indices contigus.
    
    # Trouvons le début de la trajectoire de cet index
    traj_name, start_step = dataset.indices[idx]
    print(f"Trajectoire: {traj_name}, Start Step: {start_step}")
    
    # On va afficher 10 frames consécutives
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for t in range(10):
        # On recrée l'index pour t+i
        # Note: ceci est approximatif, on suppose que les indices sont contigus dans dataset.indices
        # Pour faire propre, on recharge depuis le cache
        step = start_step + t
        traj_data = dataset.trajectory_cache[traj_name]
        
        if step >= len(traj_data): break
        
        data_step = traj_data[step]
        
        # PCD
        if data_step['pcd_path'] and os.path.exists(data_step['pcd_path']):
            pcd = np.load(data_step['pcd_path'])
        else:
            continue
            
        # Pose Robot
        pose = data_step['pose_9d'][:3]
        
        ax.clear()
        # Affichage PCD (Sous-échantillonné)
        ax.scatter(pcd[::10, 0], pcd[::10, 1], pcd[::10, 2], s=1, c='gray', alpha=0.5)
        
        # Affichage Robot (Grosse boule rouge)
        ax.scatter(pose[0], pose[1], pose[2], s=200, c='red', marker='o', label='Robot EE')
        
        ax.set_xlim(pose[0]-0.3, pose[0]+0.3)
        ax.set_ylim(pose[1]-0.3, pose[1]+0.3)
        ax.set_zlim(0, 0.6)
        ax.set_title(f"Step {step} (Regarde si la nourriture bouge !)")
        ax.legend()
        
        # Pause pour animation
        plt.pause(0.5)
        
        # Sauvegarde frame
        plt.savefig(f"debug_frame_{t}.png")

if __name__ == "__main__":
    visualize_input_movie()