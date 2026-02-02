import pymeshlab
import os

def final_polish_pipeline(input_mesh_path, output_mesh_path, target_faces=10000):
    print(f"💎 [Polishing] Finition surface dérivable : {input_mesh_path}")
    
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(input_mesh_path)
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return

    print(f"   Mesh initial : {ms.current_mesh().face_number()} faces")

    # --- ÉTAPE 1 : BOUCHAGE SÉLECTIF ---
    print("1. Bouchage des micro-trous (sans toucher au bord du bol)...")
    try:
        ms.apply_filter('meshing_close_holes', maxholesize=300)
    except Exception as e:
        print(f"⚠️ Pas de trous détectés ou erreur : {e}")

    # --- ÉTAPE 2 : NETTOYAGE PRÉALABLE ---
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_repair_non_manifold_vertices')

    # --- ÉTAPE 3 : LISSAGE GLOBAL (HC LAPLACIAN) ---
    print("2. Lissage HC Laplacian (Tout le mesh + Bords)...")
    try:
        ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving', 
                        angledeg=270,
                        iterations=20,
                        selection=False)
        print("   -> Lissage appliqué sur tout le volume.")
    except Exception as e:
        print(f"❌ Erreur Lissage : {e}")

    # --- ÉTAPE 4 : DÉCIMATION ---
    # Reduce mesh complexity for faster SDF computation
    # Quadric edge collapse preserves shape well
    current_faces = ms.current_mesh().face_number()
    if current_faces > target_faces:
        print(f"3. Décimation : {current_faces} → {target_faces} faces...")
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=target_faces,
                        preserveboundary=True,    # Keep the rim intact
                        preservenormal=True,      # Keep surface orientation
                        preservetopology=True,    # No holes created
                        qualitythr=0.5)
        print(f"   -> Résultat : {ms.current_mesh().face_number()} faces")

    # Sauvegarde
    ms.save_current_mesh(output_mesh_path)
    print(f"✅ Sauvegardé sous : {output_mesh_path}")

if __name__ == "__main__":
    input_file = "mon_bol_propre.obj"
    output_file = "mon_bol_parfait.obj"
    
    if os.path.exists(input_file):
        final_polish_pipeline(input_file, output_file, target_faces=10000)
    else:
        print(f"❌ Fichier introuvable : {input_file}")