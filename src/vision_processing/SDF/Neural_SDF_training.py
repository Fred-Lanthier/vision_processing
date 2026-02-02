import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
import numpy as np
import time
import os
import gc

# --- CONFIGURATION CHIRURGICALE ---
MESH_PATH = "mon_bol_parfait.obj"
MODEL_PATH = "udf_bol_sharp.pth"
NUM_SAMPLES = 300000       # Beaucoup de points
BATCH_SIZE = 8192          # Batch plus petit pour mieux converger
EPOCHS = 250               # Plus d'époques pour affiner les détails
LR = 5e-4                  # Learning rate plus doux pour ne pas osciller
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. RÉSEAU AMÉLIORÉ (SIREN-like ou Sharp MLP) ---
class SDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        # On augmente la taille pour capter les détails fins
        self.net = nn.Sequential(
            nn.Linear(3, 256), nn.Softplus(beta=100), # Softplus est dérivable mais plus "droit" que Tanh
            nn.Linear(256, 256), nn.Softplus(beta=100),
            nn.Linear(256, 256), nn.Softplus(beta=100),
            nn.Linear(256, 1)   # Sortie linéaire
        )

    def forward(self, x):
        return self.net(x)

# --- 2. GÉNÉRATION DES DONNÉES (MODE PRÉCISION) ---
def generate_sharp_data(mesh_path, n_samples):
    print(f"⏳ Chargement et analyse fine du maillage...")
    mesh = trimesh.load(mesh_path)
    
    # Normalisation stricte
    centroid = mesh.centroid
    scale = np.max(mesh.extents) / 1.5 # On laisse de la marge
    print(f"📏 Scale: {scale:.4f}")
    
    mesh_norm = mesh.copy()
    mesh_norm.apply_translation(-centroid)
    mesh_norm.apply_scale(1.0 / scale)

    print("📍 Stratégie d'échantillonnage : FOCUS SURFACE...")
    
    # A. SUR LA SURFACE (40%) - Distance = 0 stricte
    points_surface, _ = trimesh.sample.sample_surface(mesh_norm, int(n_samples * 0.4))
    
    # B. VERY NEAR (40%) - C'est ici que ça se joue !
    # On met un bruit minuscule (3mm à l'échelle réelle) pour apprendre le gradient immédiat
    # Si scale ~ 0.15 (15cm), 0.02 en bruit norm = 3mm réel
    points_very_near = points_surface + np.random.normal(0, 0.015, points_surface.shape)
    
    # C. FAR (20%) - Juste pour savoir que loin = loin
    points_far = np.random.uniform(-1.0, 1.0, (int(n_samples * 0.2), 3))
    
    query_points = np.vstack([points_surface, points_very_near, points_far]).astype(np.float32)
    
    # Calcul des distances (UDF)
    print("⚙️ Calcul Ground Truth...")
    chunk_size = 10000
    udf_values = []
    for i in range(0, query_points.shape[0], chunk_size):
        batch = query_points[i:i+chunk_size]
        _, dists, _ = mesh_norm.nearest.on_surface(batch)
        udf_values.append(dists)
    
    udf_gt = np.concatenate(udf_values)
    
    # PONDÉRATION (WEIGHTING)
    # On veut que le réseau ait peur de se tromper près du mur, mais s'en fiche un peu s'il se trompe loin.
    # On crée un vecteur de poids W.
    # Poids = 1.0 partout, sauf près de la surface où c'est 10.0
    weights = np.ones_like(udf_gt)
    weights[udf_gt < 0.05] = 10.0 
    
    del mesh, mesh_norm
    gc.collect()

    return (
        torch.tensor(query_points, device=DEVICE),
        torch.tensor(udf_gt, device=DEVICE, dtype=torch.float32).unsqueeze(1),
        torch.tensor(weights, device=DEVICE, dtype=torch.float32).unsqueeze(1),
        centroid, scale
    )

# --- 3. ENTRAÎNEMENT AVEC PONDÉRATION ---
def train_sharp():
    X, Y, W, center, scale = generate_sharp_data(MESH_PATH, NUM_SAMPLES)
    
    model = SDFNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("\n🔥 Démarrage Entraînement Haute Fidélité...")
    
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X.shape[0])
        total_loss = 0
        
        for i in range(0, X.shape[0], BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_x, batch_y, batch_w = X[indices], Y[indices], W[indices]
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            # Loss pondérée : (Pred - Realité)^2 * Importance
            # On force la distance à être positive avec abs() dans la loss pour aider le réseau
            loss = torch.mean(batch_w * (torch.abs(pred) - batch_y)**2)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Weighted Loss: {total_loss / (X.shape[0]/BATCH_SIZE):.2e}")

    # Sauvegarde
    state = {'state_dict': model.state_dict(), 'center': center, 'scale': scale}
    torch.save(state, MODEL_PATH)
    print(f"💾 Modèle Sharp sauvegardé : {MODEL_PATH}")

if __name__ == "__main__":
    train_sharp()