import numpy as np
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn  # Module nécessaire pour l'interopérabilité NumPy
from pathlib import Path

def traiter_nuage_npy(input_file, output_file):
    print(f"--- Début du traitement pour : {input_file} ---")

    # ---------------------------------------------------------
    # 1. Lire mon nuage de points (.npy)
    # ---------------------------------------------------------
    # On charge les données brutes avec numpy
    raw_points = np.load(input_file)
    
    # Vérification de sécurité : s'assurer que c'est du float32 (format standard pour MeshLib)
    # et que la forme est bien (N, 3) pour X, Y, Z
    if raw_points.dtype != np.float32:
        raw_points = raw_points.astype(np.float32)
    
    # Conversion du tableau NumPy vers l'objet PointCloud de MeshLib
    pc = mn.pointCloudFromPoints(raw_points)
    
    print(f"1. Nuage chargé depuis .npy ({pc.validPoints.count()} points)")

    # ---------------------------------------------------------
    # 2. Remove les outliers (Supprimer les points aberrants)
    # ---------------------------------------------------------
    bbox = pc.computeBoundingBox()
    diag = bbox.diagonal()
    
    outlier_params = mm.OutlierParams()
    # Rayon de recherche : 1% de la diagonale de l'objet
    outlier_params.radius = diag * 0.01 
    
    # Détection et suppression
    outliers_mask = mm.findOutliers(pc, outlier_params)
    pc.validPoints.subtract(outliers_mask)
    print(f"2. Outliers supprimés (Reste {pc.validPoints.count()} points)")

    # ---------------------------------------------------------
    # 3. Créé un premier jet de mesh (Triangulation)
    # ---------------------------------------------------------
    # On transforme les points en triangles
    mesh = mm.triangulatePointCloud(pc)
    print("3. Triangulation effectuée")

    # ---------------------------------------------------------
    # 4. Remplir les trous
    # ---------------------------------------------------------
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    print(f"4. Remplissage de {len(hole_edges)} trous...")

    for e in hole_edges:
        params = mm.FillHoleParams()
        params.metric = mm.getUniversalMetric(mesh)
        mm.fillHole(mesh, e, params)

    # ---------------------------------------------------------
    # 5. Lisser (Enlever le bruit)
    # ---------------------------------------------------------
    mm.meshDenoiseViaNormals(mesh)
    print("5. Lissage terminé")

    # ---------------------------------------------------------
    # Sauvegarde
    # ---------------------------------------------------------
    mm.saveMesh(mesh, output_file)
    print(f"--- Terminé. Résultat sauvegardé dans : {output_file} ---")

# Exécution
if __name__ == "__main__":
    # Assurez-vous que votre fichier .npy contient un tableau de forme (N, 3)
    fichier_entree = "Images_Test/resultat_depth_mask.npy"  
    fichier_sortie = "Resultat_Final.stl"
    
    if Path(fichier_entree).exists():
        traiter_nuage_npy(fichier_entree, fichier_sortie)
    else:
        print(f"Erreur : Le fichier {fichier_entree} est introuvable.")