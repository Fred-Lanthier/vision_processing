import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import numpy as np
import os
import rospkg

# 1. Configuration des chemins ROS
rospack = rospkg.RosPack()
package_path = rospack.get_path('vision_processing')
stl_path = os.path.join(package_path, 'src', "vision_processing", "SDF", "03_surface_aligned.obj")

print(f"Chargement du fichier : {stl_path}")
mesh = trimesh.load(stl_path)

# 2. Transformations du Mesh (Modifie l'objet réel)
# Rotation de 180° sur Y (pour l'orienter vers le haut si nécessaire)
angles = np.array([0, 0, 0])
rotation_matrix = trimesh.transformations.euler_matrix(*np.radians(angles))
mesh.apply_transform(rotation_matrix)

# Calcul du centrage basé sur les nouvelles bornes
bounds = mesh.bounds
min_vals, max_vals = bounds[0], bounds[1]

# Calcul des offsets pour centrer en X,Y et poser sur Z=0

offset_x = -(min_vals[0] + max_vals[0]) / 2.0
offset_y = -(min_vals[1] + max_vals[1]) / 2.0
offset_z = -min_vals[2]
print(offset_x, offset_y, offset_z)
# mesh.apply_translation([offset_x, offset_y, offset_z])
mesh.apply_translation([0, 0, 0.0])

# 3. Échantillonnage et Sauvegarde
# On génère le nuage de points à partir du mesh transformé
point_cloud = mesh.sample(10000)
output_path = os.path.join(package_path, 'src', "vision_processing", "SDF", "resultat_mesh_oriente.npy")
np.save(output_path, point_cloud)

print("\n--- Statistiques finales ---")
print(f"Dimensions (L x l x h): {mesh.extents}")
print(f"Fichier sauvegardé : {output_path}")

# 4. Visualisation Interactive
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Affichage du nuage de points
steps = 1
ax.scatter(point_cloud[::steps, 0], point_cloud[::steps, 1], point_cloud[::steps, 2], s=1, c='gray', alpha=0.4)

# Dessin des axes à l'origine
length = 0.05
ax.quiver(0, 0, 0, length, 0, 0, color='r', label='X (Rouge)')
ax.quiver(0, 0, 0, 0, length, 0, color='g', label='Y (Vert)')
ax.quiver(0, 0, 0, 0, 0, length, color='b', label='Z (Bleu)')

# Ajustement automatique des limites de la vue
all_pts = np.vstack([point_cloud, [0,0,0]])
max_range = np.ptp(all_pts, axis=0).max() / 2.0
mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) * 0.5
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
ax.azim = 90
ax.elev = 0
ax.set_title("Visualisation 3D : Objet centré et orienté")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()

# Cette fois, la fenêtre va s'ouvrir correctement
plt.show()