#!/home/flanthier/Documents/src/vision_processing/venv_sam3/bin/python3
from flask import Flask, request, jsonify
import os
import cv2
import base64
import numpy as np
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError: pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SAM3_Gemini import FoodSegmenterNative

app = Flask(__name__)

# Config Clé API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or "AIza" not in GOOGLE_API_KEY:
    print("❌ ERREUR CRITIQUE : Clé API Google invalide ou manquante !")
    sys.exit(1)

# Chargement Modèle
print("🚀 [SERVER] Loading SAM 3 + Gemini models...")
segmenter = FoodSegmenterNative(google_api_key=GOOGLE_API_KEY)
print("✅ [SERVER] Ready on port 5000!")

def encode_image_to_base64(cv_image):
    """Helper pour convertir une image OpenCV BGR en string Base64"""
    if cv_image is None:
        return None
    # Encoder en JPG pour le transport (plus léger que PNG)
    success, buffer = cv2.imencode('.jpg', cv_image)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/detect', methods=['POST'])
def detect():
    temp_path = "temp_server_upload.jpg"
    try:
        if 'image' in request.files:
            image_file = request.files['image']
            image_file.save(temp_path)
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Lancement du traitement
        result = segmenter.process_image(temp_path)
        
        if result["success"]:
            # --- Encodage des deux images ---
            overlay_b64 = encode_image_to_base64(result['selected_overlay_image'])
            rainbow_b64 = encode_image_to_base64(result['selected_rainbow_image'])

            response_data = {
                "success": True,
                "selected_food": result['selected_food'],
                "total_pieces": int(result['total_pieces']),
                "centroid": [float(c) for c in result['centroid']],
                "selected_piece_area": int(result['selected_piece_area']),
                "computation_time": float(result['computation_time']),
                # NOUVEAUX CHAMPS BASE64
                "overlay_base64": overlay_b64,
                "rainbow_base64": rainbow_b64
            }
            return jsonify(response_data)
        else:
            return jsonify({
                "success": False, 
                "error": "No food detected",
                "computation_time": float(result['computation_time'])
            }), 404
            
    except Exception as e:
        print(f"❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)