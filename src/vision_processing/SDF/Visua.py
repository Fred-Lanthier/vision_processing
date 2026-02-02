import pyvista as pv
import os

def visualize_comparison(original_file, repaired_file):
    print(f"👀 Chargement des fichiers...")
    
    if not os.path.exists(repaired_file):
        print(f"❌ Fichier introuvable : {repaired_file}")
        return

    # 1. Chargement
    # On charge l'original s'il existe pour comparer
    has_original = os.path.exists(original_file)
    if has_original:
        mesh_orig = pv.read(original_file)
    
    mesh_new = pv.read(repaired_file)

    # 2. Configuration du Plotter
    # shape=(1, 2) crée deux fenêtres côte à côte
    p = pv.Plotter(shape=(1, 2) if has_original else (1, 1))

    # --- Fenêtre de Gauche (Original) ---
    if has_original:
        p.subplot(0, 0)
        p.add_text("Avant : Avec Trous (BPA)", font_size=10)
        # On affiche les arêtes (show_edges=True) pour bien voir les trous
        p.add_mesh(mesh_orig, color="cyan", pbr=True, metallic=0.6, roughness=0.2,show_edges=False)
        p.add_axes()

    # --- Fenêtre de Droite (Réparé) ---
    p.subplot(0, 1 if has_original else 0)
    p.add_text("Après : Bouché + Lissé (MeshLab)", font_size=10)
    
    # Utilisation du rendu PBR (Physically Based Rendering)
    # Cela donne un aspect métallique qui aide à juger si le lissage Taubin est bon
    p.add_mesh(mesh_new, color="cyan", pbr=True, metallic=0.6, roughness=0.2, show_edges=False)
    
    # On ajoute une lumière pour voir les reflets sur la surface lisse
    light = pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), color='white')
    p.add_light(light)

    # 3. Synchronisation des caméras
    # Quand vous tournez l'un, l'autre tourne aussi !
    if has_original:
        p.link_views()

    print("🎥 Ouverture de la fenêtre de visualisation...")
    print("   -> Clic gauche + glisser : Tourner")
    print("   -> Molette : Zoomer")
    print("   -> Shift + Clic : Déplacer")
    p.show()

if __name__ == "__main__":
    # Vos noms de fichiers
    file_before = "Images_Test/Mesh.obj"          # Sortie du script BPA (avec trous)
    file_after = "Images_Test/Mesh.obj"  # Sortie du script MeshLab (réparé)
    
    visualize_comparison(file_before, file_after)