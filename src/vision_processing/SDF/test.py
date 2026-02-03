import numpy as np
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

def depth_to_mesh_clean(input_file, output_file):
    print(f"--- Traitement avancé pour : {input_file} ---")
    
    # 1. Chargement
    depth_image = np.load(input_file)
    h, w = depth_image.shape
    print(f"Dimensions : {w}x{h}")

    # ---------------------------------------------------------
    # ÉTAPE CRUCIALE : REPROJECTION 3D (Pixel -> Mètres)
    # ---------------------------------------------------------
    # Sans intrinsèques réelles, on estime une caméra standard (FOV ~60-70°)
    # Si vous avez les intrinsèques (fx, fy, cx, cy), remplacez les valeurs ci-dessous.
    fx = w  # Estimation grossière de la focale (souvent proche de la largeur)
    fy = w
    cx = w / 2
    cy = h / 2

    # Grille de coordonnées
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Masque pour ignorer les points invalides (Z=0 ou trop loin)
    mask = (depth_image > 0.1) & (depth_image < 3.0) # On ignore < 10cm et > 3m
    
    # Projection Pinhole : Z * (u - cx) / fx
    z_m = depth_image[mask]
    x_m = (u[mask] - cx) * z_m / fx
    y_m = (v[mask] - cy) * z_m / fy
    
    points_3d = np.column_stack((x_m, y_m, z_m)).astype(np.float32)
    print(f"Points valides après filtrage : {len(points_3d)}")

    # Conversion MeshLib
    pc = mn.pointCloudFromPoints(points_3d)

    # ---------------------------------------------------------
    # TRIANGULATION & NETTOYAGE
    # ---------------------------------------------------------
    # Paramètres de triangulation
    t_params = mm.TriangulatePointCloudSettings()
    # maxEdgeLen est VITAL : empêche de relier un point à 1m d'un autre
    # On met 5cm (0.05) comme distance max entre deux points connectés
    t_params.maxEdgeLen = 0.05 
    
    print("Triangulation avec contrainte de distance...")
    mesh = mm.triangulatePointCloud(pc, t_params)

    # ---------------------------------------------------------
    # POST-TRAITEMENT
    # ---------------------------------------------------------
    print("Lissage (Relaxation)...")
    # Le 'relax' est souvent meilleur que le 'denoise' pour les depth maps organiques
    relax_params = mm.MeshRelaxParams()
    relax_params.iterations = 5
    relax_params.force = 0.1
    mm.relax(mesh, relax_params)

    # Sauvegarde
    mm.saveMesh(mesh, output_file)
    print(f"Sauvegardé : {output_file}")

if __name__ == "__main__":
    traiter = depth_to_mesh_clean
    traiter("Images_Test/resultat_depth_mask.npy", "Images_Test/resultat_final.obj")