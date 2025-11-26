import requests
import argparse
import base64
import cv2
import numpy as np
import os
import time

def detect_food(image_path, server_url):
    print(f"📤 Envoi vers {server_url}...")
    start_req = time.time()
    with open(image_path, 'rb') as f:
        try:
            response = requests.post(server_url, files={'image': f})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur connexion : {e}")
            return None
    print(f"📥 Réponse reçue ({time.time() - start_req:.2f}s)")
    return response.json()

def save_base64_image(base64_string, output_path):
    """Helper pour décoder et sauvegarder une image base64"""
    if not base64_string:
        print(f"⚠️ Pas de données d'image pour {output_path}")
        return
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(output_path, img)
        print(f"   Generation de : {output_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde {output_path} : {e}")

def main():
    parser = argparse.ArgumentParser(description="Client SAM 3")
    parser.add_argument('--image', type=str, required=True, help='Chemin image')
    parser.add_argument('--ip', type=str, default='localhost', help='IP serveur')
    args = parser.parse_args()
    
    url = f"http://{args.ip}:5000/detect"
    if not os.path.exists(args.image):
        print("❌ Image introuvable.")
        return

    result = detect_food(args.image, url)
    
    if result and result.get('success'):
        print("\n✅ RÉSULTATS :")
        print(f"   🍎 Sélection : {result['selected_food']}")
        print(f"   🎯 Centroïde : {result['centroid']}")
        
        # --- Sauvegarde des deux images ---
        print("\n💾 Sauvegarde des images résultats...")
        save_base64_image(result.get('overlay_base64'), "client_resultat_overlay.png")
        save_base64_image(result.get('rainbow_base64'), "client_result_rainbow_map.png")
        print("Done.")
            
    else:
        print("❌ Échec détection.")
        if result: print(f"   Serveur : {result.get('error')}")

if __name__ == "__main__":
    main()