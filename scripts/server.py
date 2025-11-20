#!/home/flanthier/Documents/src/vision_processing/venv_sam3/bin/python3
from flask import Flask, request, jsonify
import os
import cv2
import base64
import numpy as np
import sys

# Ajouter le dossier courant au path pour trouver SAM3_Gemini.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer notre classe SAM 3
from SAM3_Gemini import FoodSegmenterNative

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
# Remplacez par votre vraie clé ici !
GOOGLE_API_KEY = "AIzaSyBiI8ij7bP_P0CUHkxv8W_cjMOCNa24-7I"

# ==========================================
# CHARGEMENT UNIQUE DU MODÈLE
# ==========================================
print("🚀 [SERVER] Loading SAM 3 + Gemini models...")
if "AIza" not in GOOGLE_API_KEY:
    print("❌ ERREUR CRITIQUE : Clé API manquante dans flask_server_sam3.py")
    sys.exit(1)

segmenter = FoodSegmenterNative(google_api_key=GOOGLE_API_KEY)
print("✅ [SERVER] Models ready and listening on port 5000!")

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint pour détecter la nourriture avec SAM 3"""
    temp_path = "temp_server_upload.jpg"
    
    try:
        # 1. Récupération de l'image
        if 'image' in request.files:
            image_file = request.files['image']
            image_file.save(temp_path)
        else:
            return jsonify({"error": "No image provided in 'image' field"}), 400
        
        # 2. Lancer la détection via notre classe SAM 3
        # Cela génère aussi les images overlay/rainbow sur le disque du serveur
        result = segmenter.process_image(temp_path)
        
        if result["success"]:
            print(f"🔍 Detected: {result['selected_food']}")
            
            # 3. Encodage de l'image annotée en Base64 pour le renvoi au client
            # (Permet au client de voir le résultat visuel sans accéder au disque du serveur)
            processed_img = result['processed_image']
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 4. Construction de la réponse JSON
            # Note: On doit convertir les types NumPy (int64, float32) en types Python natifs
            response_data = {
                "success": True,
                "selected_food": result['selected_food'],
                "total_pieces": int(result['total_pieces']),
                "centroid": [float(c) for c in result['centroid']],
                "selected_piece_area": int(result['selected_piece_area']),
                "computation_time": float(result['computation_time']),
                "processed_image_base64": img_base64 # L'image résultat
            }
            
            return jsonify(response_data)
        else:
            return jsonify({
                "success": False, 
                "error": "No food detected by Gemini/SAM3",
                "computation_time": float(result['computation_time'])
            }), 404
            
    except Exception as e:
        print(f"❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Nettoyage du fichier temporaire d'upload
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # host='0.0.0.0' permet d'être accessible depuis une autre machine (ex: robot vers PC GPU)
    app.run(host='0.0.0.0', port=5000, debug=False)