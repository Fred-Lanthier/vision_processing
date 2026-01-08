import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospkg
import os
from scipy.spatial.transform import Rotation as R

def draw_frame(ax, T, label, length=0.1):
    """Dessine les axes X (rouge), Y (vert), Z (bleu) d'une matrice T"""
    origin = T[:3, 3]
    x_axis = T[:3, :3] @ np.array([length, 0, 0])
    y_axis = T[:3, :3] @ np.array([0, length, 0])
    z_axis = T[:3, :3] @ np.array([0, 0, length])

    ax.quiver(*origin, *x_axis, color='r', linewidth=2)
    ax.quiver(*origin, *y_axis, color='g', linewidth=2)
    ax.quiver(*origin, *z_axis, color='b', linewidth=2)
    ax.text(*(origin + z_axis), label, color='black', fontsize=9)

def pose_to_matrix(pos, quat):
    """Convertit pos [x,y,z] et quat [x,y,z,w] en matrice 4x4"""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def quick_3d_view(robot_path, ee_pos, ee_quat):
    """
    Visualisation 3D validant le passage du repère TCP (JSON) 
    au repère Fork_Tip (Objectif modèle)
    """
    if not os.path.exists(robot_path):
        print(f"Erreur : fichier introuvable {robot_path}")
        return
    pcd = np.load(robot_path)

    # 1. Matrice de l'effecteur (TCP) issue du JSON
    T_world_ee = pose_to_matrix(ee_pos, ee_quat)

    # 2. Matrice Relative : TCP -> Fork_Tip
    # On compense la longueur de la pince (0.1034) car ee_pos est déjà au TCP
    T_ee_fork = np.eye(4)
    # Rotation calculée précédemment (compensant l'inclinaison de la fourchette)
    rot_rel = R.from_euler('xyz', [0, -207.5, 0.0], degrees=True)
    T_ee_fork[:3, :3] = rot_rel.as_matrix()
    
    # Translation relative (du centre des doigts vers la pointe des dents)
    # Z_joint_total (0.195) - Z_pince (0.1034) = 0.0916m
    T_ee_fork[:3, 3] = [-0.0055, -0.00, 0.233-0.1034] 

    # 3. Calcul de la pose CIBLE pour le modèle (Fork Tip dans World)
    T_world_fork = T_world_ee @ T_ee_fork

    # --- Visualisation ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Affichage du nuage de points (en gris)
    ax.scatter(pcd[::5, 0], pcd[::5, 1], pcd[::5, 2], s=1, alpha=0.2, c='gray')

    # Affichage des repères
    draw_frame(ax, np.eye(4), "World", length=0.1)
    draw_frame(ax, T_world_ee, "TCP (JSON)", length=0.1)
    draw_frame(ax, T_world_fork, "Fork_Tip (Target)", length=0.15)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')  
    ax.set_zlabel('Z (m)')
    
    # Centrage sur la zone d'intérêt
    ax.set_xlim([ee_pos[0] - 0.5, ee_pos[0] + 0.5])
    ax.set_ylim([ee_pos[1] - 0.5, ee_pos[1] + 0.5])
    ax.set_zlim([ee_pos[2] - 0.5, ee_pos[2] + 0.5])
    
    ax.view_init(elev=0, azim=90)
    plt.savefig("dreamitate_validation.png")
    plt.show()
    print(f"✅ Fork Tip World Pose: \n{T_world_fork}")
    print("📸 Visualisation sauvegardée.")

def main():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')

    # Données du premier pas de temps
    ee_pos = [0.5065196, 5.4009e-05, 0.335793]
    ee_quat = [0.999999, -9.4914e-05, -0.001316, -0.000102]

    robot_path = os.path.join(package_path, 'datas', "Trajectories_preprocess", 
                              "Trajectory_18", "Merged_urdf_fork_Trajectory_18", "Merged_urdf_fork_0070.npy")
    
    quick_3d_view(robot_path, ee_pos, ee_quat)

if __name__ == "__main__":
    main()