import pickle
import matplotlib.pyplot as plt
import torch
import rospkg
import numpy as np
import os

# --- FUNCTIONS ---
def load_pickle_history():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    pkl_file = os.path.join(package_path, 'train_history.pkl')
    
    with open(pkl_file, 'rb') as f:
        history = pickle.load(f)
    
    return history

def main():
    data = load_pickle_history()
    
    print("--- Configuration de l'entraînement ---")
    for key, value in data['config'].items():
        print(f"{key}: {value}")
    
    # 2. Résumé des performances
    # print(f"\nMeilleure Val Loss: {data['best_val_loss']:.6f} à l'époque {data['best_epoch']}")

    # 3. Graphique rapide
    plt.plot(data['train_loss'], label='Train')
    plt.plot(data['val_loss'], label='Val')
    plt.title(f"Expérience: {data['config']['model_name']}")
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_history.svg'), format='svg')
    plt.show()

if __name__ == "__main__":
    main()
