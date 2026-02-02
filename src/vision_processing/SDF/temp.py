from meshlib import mrmeshpy as mm
import numpy as np
import os

def load_npy_to_pointcloud(npy_path):
    """
    Charge un fichier .npy (N, 3) et le convertit en mrmeshpy.PointCloud
    """
    print(f"Chargement du fichier NumPy : {npy_path}")
    data = np.load(npy_path)
    
    # Vérifications de sécurité
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Le fichier .npy doit être de forme (N, 3). Forme actuelle : {data.shape}")
    
    # Si c'est en float64, on passe en float32 (standard pour la 3D)
    data = data.astype(np.float32)
    
    # Création du vecteur de points pour MeshLib
    # Note : MeshLib utilise des types C++, il faut donc remplir un std_vector
    vec_points = mm.std_vector_Vector3f()
    
    # Optimisation : on pré-alloue la mémoire si possible, sinon on boucle
    # (Pour de très gros nuages >1M points, cette boucle peut prendre quelques secondes)
    for p in data:
        vec_points.append(mm.Vector3f(p[0], p[1], p[2]))
        
    # Création de l'objet PointCloud
    pc = mm.PointCloud()
    pc.addPoint(vec_points)
    
    return pc

def process_pipeline(input_file, output_file):
    # 1. Lire le nuage de points (Adapté pour NPY)
    if input_file.endswith('.npy'):
        pc = load_npy_to_pointcloud(input_file)
    else:
        # Fallback pour les fichiers classiques (.ply, .pcd)
        pc = mm.loadPointCloud(input_file)

    if not pc or pc.validPointsCount() == 0:
        print("Erreur : Nuage vide ou non chargé.")
        return

    print(f"1. Nuage chargé : {pc.validPointsCount()} points")

    # --- LE RESTE DU PIPELINE RESTE IDENTIQUE ---
    
    # 2. Nettoyage initial (Outliers)
    # Pour un nuage brut venant d'un .npy, il est souvent utile d'invalider les points nuls (0,0,0)
    # qui traînent parfois dans les buffers de caméra.
    
    # 3. Calcul et Orientation des Normales
    print("3. Calcul des normales...")
    # Paramètres par défaut souvent suffisants
    mm.calcOrientedNormals(pc, mm.CalcOrientedNormalsParams())

    # 4. Mesh Generation (Poisson)
    print("4. Reconstruction de surface (Poisson)...")
    mesh_params = mm.PointCloudToMeshParams()
    mesh_params.surfaceProfile = mm.PointCloudToMeshParams.SurfaceProfile.Smooth
    mesh = mm.pointCloudToMesh(pc, mesh_params)
    
    # 5. Remplir les trous
    print("5. Remplissage des trous...")
    mm.fillHoles(mesh, metric=mm.getUniversalMetric(mesh))

    # 6. Nettoyage Topologique (Watertight)
    print("6. Nettoyage (Remove Disconnected)...")
    mm.removeDisconnectedComponents(mesh)

    # 7. Remeshing (Propre & Généralisable)
    print("7. Remeshing isotrope...")
    remesh_params = mm.RemeshSettings()
    remesh_params.maxEdgeLen = mm.findAvgEdgeLength(mesh) 
    remesh_params.iterations = 3
    mm.isotropicRemesh(mesh, remesh_params)

    # 8. Lissage
    print("8. Lissage final...")
    mm.laplacianSmooth(mesh, 2)

    # Sauvegarde
    mm.saveMesh(mesh, output_file)
    print(f"--- Succès : {output_file} généré ---")

if __name__ == "__main__":
    INPUT_NPY = "Images_Test/resultat_depth_mask.npy"  # Votre fichier source
    OUTPUT_STL = "resultat_propre.stl"
    
    # Création d'un fichier dummy pour tester si vous n'en avez pas
    if not os.path.exists(INPUT_NPY):
        print("Création d'un fichier .npy de test...")
        # Génère une sphère bruitée pour l'exemple
        dummy_points = np.random.randn(1000, 3).astype(np.float32)
        # Normaliser pour faire une sphère creuse
        norms = np.linalg.norm(dummy_points, axis=1, keepdims=True)
        dummy_points = dummy_points / norms
        np.save(INPUT_NPY, dummy_points)

    process_pipeline(INPUT_NPY, OUTPUT_STL)