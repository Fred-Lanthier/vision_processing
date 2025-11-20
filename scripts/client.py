import requests
import argparse
import base64
import cv2
import numpy as np
import os
import time

def detect_food(image_path, server_url='http://localhost:5000/detect'):
    print(f"📤 Envoi de l'image {image_path} vers {server_url}...")
    start_req = time.time()
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(server_url, files=files)
            response.raise_for_status() # Lève une erreur si status != 200
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur de connexion au serveur : {e}")
            return None
            
    print(f"📥 Réponse reçue en {time.time() - start_req:.2f}s")
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Client for SAM 3 Food Detection")
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--ip', type=str, default='localhost', help='IP of the server')
    args = parser.parse_args()
    
    url = f"http://{args.ip}:5000/detect"
    
    # Vérification image
    if not os.path.exists(args.image):
        print("❌ L'image n'existe pas.")
        return

    result = detect_food(args.image, url)
    
    if result and result.get('success'):
        print("\n✅ RÉSULTATS SAM 3 :")
        print(f"   🍎 Aliment sélectionné : {result['selected_food']}")
        print(f"   🔢 Nombre de morceaux : {result['total_pieces']}")
        print(f"   🎯 Centroïde (px)     : {result['centroid']}")
        print(f"   📐 Aire (px²)         : {result['selected_piece_area']}")
        print(f"   ⏱️  Temps calcul GPU   : {result['computation_time']:.3f}s")
        
        # --- Sauvegarde de l'image résultat renvoyée ---
        if 'processed_image_base64' in result:
            img_data = base64.b64decode(result['processed_image_base64'])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            output_name = "client_result_sam3.jpg"
            cv2.imwrite(output_name, img)
            print(f"\n🖼️  Image annotée sauvegardée sous : {output_name}")
            
    else:
        print("❌ Échec de la détection ou réponse vide.")
        if result:
            print(f"   Message serveur: {result.get('error')}")

if __name__ == "__main__":
    main()