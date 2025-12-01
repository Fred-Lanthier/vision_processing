import gradio as gr
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
import cv2

# Importez votre classe Scanner ici
# Assurez-vous que SAM3_FatSecret_PlateCalib.py est dans le même dossier
from calories_counter_without_measurement import NutrientScannerV5

load_dotenv()

# --- INITIALISATION GLOBALE ---
print("⏳ Initialisation du moteur IA (Cela peut prendre quelques secondes)...")
try:
    scanner = NutrientScannerV5()
    print("✅ Moteur prêt !")
except Exception as e:
    print(f"❌ Erreur d'initialisation : {e}")
    scanner = None

def process_image_gradio(image_input):
    """
    Fonction wrapper qui connecte Gradio à votre classe Scanner
    """
    if scanner is None:
        return None, "❌ Le moteur IA n'a pas pu démarrer. Vérifiez la console."
        
    if image_input is None:
        return None, "Veuillez prendre une photo."

    temp_filename = "temp_gradio_input.jpg"
    pil_image = Image.fromarray(image_input)
    pil_image.save(temp_filename)

    try:
        # Appel de l'analyse
        # Retourne maintenant un DICTIONNAIRE complet
        global_results = scanner.analyze_plate(temp_filename)
        
        # --- DEBUG RAPIDE ---
        print(f"🔍 Type retourné : {type(global_results)}")

        if not global_results or not global_results.get('items'):
            return None, "⚠️ Aucun aliment détecté ou erreur de calibration."

        # 3. Formatter le résultat pour l'affichage
        annotated_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # --- NOUVEAU FORMATTAGE ---
        # On utilise les totaux déjà calculés par le scanner
        summary_text = "# 🥗 Résultat Nutritionnel\n\n"
        
        # Liste des aliments (contenue dans la clé 'items')
        items_list = global_results['items']
        
        for item in items_list:
            name = item.get('name', 'Inconnu')
            cals = item.get('calories', 0)
            weight = item.get('weight_g', 0)
            vol = item.get('volume_cm3', 0)
            # Gestion sécurisée des macros si elles manquent
            macros = item.get('macros', {'protein': 0, 'carbs': 0, 'fat': 0})
            
            summary_text += f"### 🍔 {name.title()}\n"
            summary_text += f"- **Calories:** {cals} kcal\n"
            summary_text += f"- **Poids:** {weight} g ({vol:.1f} cm³)\n"
            summary_text += f"- **Macros:** P: {macros['protein']}g | G: {macros['carbs']}g | L: {macros['fat']}g\n\n"
        
        summary_text += "---\n"
        summary_text += f"# 🔥 TOTAL: {global_results.get('total_calories', 0)} kcal\n"
        summary_text += f"#### 🥩 Protéines: {global_results.get('total_protein_g', 0)}g\n"
        summary_text += f"#### 🍞 Glucides: {global_results.get('total_carbs_g', 0)}g\n"
        summary_text += f"#### 🥑 Lipides: {global_results.get('total_fat_g', 0)}g"
        
        # Convertir l'image annotée en RGB pour Gradio
        final_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        return final_img_rgb, summary_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Erreur technique : {str(e)}"

# --- INTERFACE GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown("# 🍎 AI Calorie Scanner 3D")
    gr.Markdown("Prenez une photo de votre assiette (avec une pièce de monnaie pour l'échelle !).")
    
    with gr.Row():
        with gr.Column():
            # Input: Webcam ou Upload
            input_img = gr.Image(sources=["webcam", "upload"], label="Votre Assiette", type="numpy")
            btn = gr.Button("🔍 Analyser", variant="primary")
        
        with gr.Column():
            # Output: Image Résultat + Texte
            output_img = gr.Image(label="Analyse", interactive=False)
            output_info = gr.Markdown(label="Détails")
            
    btn.click(fn=process_image_gradio, inputs=input_img, outputs=[output_img, output_info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)