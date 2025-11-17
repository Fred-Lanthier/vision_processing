#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
from TEST_single import SingleFoodDetector

app = Flask(__name__)

# Charger le modèle UNE SEULE FOIS au démarrage
print("🚀 Loading models at startup...")
detector = SingleFoodDetector(model_type="vit_l", model_name="sam_vit_l.pth")
print("✅ Models ready!")

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint pour détecter la nourriture"""
    try:
        # Récupérer l'image
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = f"/tmp/{image_file.filename}"
            image_file.save(image_path)
        elif 'image_path' in request.json:
            image_path = request.json['image_path']
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Détecter
        result = detector.detect_individual_food_pieces(image_path)
        
        if result:
            print(result['selected_piece'])
            
            # Convert NumPy types to Python native types
            selected_piece = {
                'mask': result['selected_piece']['mask'].tolist(),  # Convert array to list
                'area': int(result['selected_piece']['area'])  # Convert int64 to int
            }
            
            return jsonify({
                "success": True,
                "selected_food": result['selected_food'],
                "total_pieces": result['total_pieces'],
                "centroid": result['centroid'],
                "computation_time": result['computation_time'],
                "selected_piece": selected_piece
            })
        else:
            return jsonify({"success": False, "error": "No detections found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)