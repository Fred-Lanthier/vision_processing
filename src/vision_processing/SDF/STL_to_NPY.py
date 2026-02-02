import matplotlib
matplotlib.use('Agg') # Indispensable pour travailler via SSH sans serveur X
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import numpy as np
import os
import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path('vision_processing')

stl_path = os.path.join(package_path, 'src', "vision_processing", "SDF", "mon_bol_parfait.obj")

mesh = trimesh.load(stl_path)

# Appliquer le scale (0.001) pour être en mètres comme dans ROS
# mesh.apply_scale(0.001)

# 2. Analyse numérique (très utile en SSH)
print(f"Bornes du mesh (min): {mesh.bounds[0]}")
print(f"Bornes du mesh (max): {mesh.bounds[1]}")
print(f"Dimensions (L x l x h): {mesh.extents}")

# 3. Préparer la visualisation
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Extraire un échantillon de points pour la visualisation (plus rapide qu'un rendu complet)
points = mesh.sample(10000)

angles = np.array([0, 180, 0])
rotation_matrix = trimesh.transformations.euler_matrix(*np.radians(angles))[:3, :3]
points = points @ rotation_matrix.T

print("Avant translation : ")
print((np.min(points[:,0]) + np.max(points[:,0])) / 2.0)
print((np.min(points[:,1]) + np.max(points[:,1])) / 2.0)
print(np.min(points[:,2]))

points[:,0] = points[:,0] - (np.min(points[:,0]) + np.max(points[:,0])) / 2
points[:,1] = points[:,1] - (np.min(points[:,1]) + np.max(points[:,1])) / 2
points[:,2] = points[:,2] - np.min(points[:,2])

print("Après translation : ")
print((np.min(points[:,0]) + np.max(points[:,0])) / 2.0)
print((np.min(points[:,1]) + np.max(points[:,1])) / 2.0)
print(np.min(points[:,2]))
# Afficher le nuage de points de la fourchette
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='gray', alpha=0.5)

# 4. Dessiner les axes à l'origine (0,0,0)
length = 0.075 # 5cm pour bien voir les axes
ax.quiver(0, 0, 0, length, 0, 0, color='r', label='X')
ax.quiver(0, 0, 0, 0, length, 0, color='g', label='Y')
ax.quiver(0, 0, 0, 0, 0, length, color='b', label='Z')

# Ajuster les limites pour bien voir l'origine et l'objet
all_points = np.vstack([points, [0,0,0]])
max_range = np.ptp(all_points, axis=0).max() / 2.0
mid = (all_points.max(axis=0) + all_points.min(axis=0)) * 0.5
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
ax.view_init(elev=30, azim=45)

ax.set_title("Visualisation de l'origine du STL (Mètres)")
ax.legend()

# Sauvegarder
plt.savefig('debug_origin.png')
print("Image 'debug_origin.png' générée.")