import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
print("Chargement des données...")
data = np.load("maze_trajectories.npz")
dataset_np = data['trajectories']  # [N, 80, 2]
maze = data['maze']
horizon = dataset_np.shape[1]

# Conversion en tenseur PyTorch
dataset = torch.tensor(dataset_np, dtype=torch.float32)
x_1_data = dataset.view(-1, horizon * 2)

# ============================================================
# 2. ARCHITECTURE (Rectified Flow / x1-prediction)
# ============================================================
class RectifiedTrajectoryNet(nn.Module):
    def __init__(self, channels=2, horizon=80):
        super().__init__()
        self.horizon = horizon
        self.channels = channels
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.ELU(),
            nn.Linear(64, channels * horizon)
        )
        
        # Convolutions 1D plus profondes pour la complexité du labyrinthe
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, padding=2), nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ELU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.ELU(),
            nn.Conv1d(64, channels, kernel_size=5, padding=2)
        )
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        b = x_t.shape[0]
        x_c = x_t.view(b, self.channels, self.horizon)
        t_emb = self.time_mlp(t).view(b, self.channels, self.horizon)
        return self.net(x_c + t_emb).view(b, -1)
    
    def step_euler(self, x_t: Tensor, t_start: float, t_end: float) -> Tensor:
        dt = t_end - t_start
        t_s = torch.full((x_t.shape[0], 1), t_start, device=x_t.device)
        x_1_pred = self(t_s, x_t)
        v_t = (x_1_pred - x_t) / max(1.0 - t_start, 1e-4)
        return x_t + v_t * dt

# ============================================================
# 3. ENTRAÎNEMENT
# ============================================================
model = RectifiedTrajectoryNet(horizon=horizon)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 3000
batch_size = 128

print(f"Entraînement du modèle sur {len(x_1_data)} trajectoires...")
for step in range(epochs):
    idx = torch.randint(0, len(x_1_data), (batch_size,))
    batch_x_1 = x_1_data[idx]
    
    # Bruit centré sur le labyrinthe (environ [6.5, 6.5]) pour aider l'apprentissage
    x_0 = torch.randn_like(batch_x_1) + torch.tensor([6.5, 6.5]).repeat(horizon).unsqueeze(0)
    
    t = torch.rand(batch_size, 1)
    x_t = (1 - t) * x_0 + t * batch_x_1
    
    optimizer.zero_grad()
    pred_x_1 = model(t=t, x_t=x_t)
    loss = loss_fn(pred_x_1, batch_x_1)
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 500 == 0:
        print(f"Step {step+1}/{epochs} | Loss: {loss.item():.4f}")

# ============================================================
# 4. FAST-SAFEFLOW : WARPING & INPAINTING
# ============================================================
# Les ellipses définies dans ton script
ellipses_params = [
    ([4.0, 5.5], 1.8, 1.0),
    ([9.0, 4.0], 1.5, 0.8),
    ([7.0, 8.5], 0.8, 1.2)
]

def apply_analytical_cbf(tau, ellipses, margin=0.05):
    """
    Répulse les points hors des ellipses en utilisant le gradient exact 
    de la fonction h(s) = (s-c)^T Q (s-c) - 1.
    """
    tau_safe = tau.clone()
    for center, a, b in ellipses:
        c = torch.tensor(center, dtype=torch.float32)
        Q = torch.tensor([[1.0/(a**2), 0], [0, 1.0/(b**2)]], dtype=torch.float32)
        
        # On itère pour pousser les points jusqu'à ce qu'ils soient hors de l'ellipse
        for _ in range(15):
            diff = tau_safe - c
            # Calcul de h(s)
            h_val = torch.sum((diff @ Q) * diff, dim=1) - 1.0
            
            mask = h_val < margin
            if not mask.any():
                break
            
            # Le gradient repousse le point de la manière la plus courte possible
            grad = 2.0 * (diff[mask] @ Q)
            grad = grad / torch.norm(grad, dim=1, keepdim=True)
            
            # Poussée géométrique
            tau_safe[mask] = tau_safe[mask] + grad * 0.1
            
    return tau_safe

def apply_maze_cbf(tau, maze, cell_size=1.0, margin=0.1):
    """
    Répulse les points hors des murs du labyrinthe en utilisant le 
    Signed Distance Field (SDF) analytique des boîtes (AABB).
    """
    tau_safe = tau.clone()
    rows, cols = maze.shape
    
    # 1. Sécurité globale : Ne pas sortir de la carte
    tau_safe[:, 0] = torch.clamp(tau_safe[:, 0], margin, cols * cell_size - margin)
    tau_safe[:, 1] = torch.clamp(tau_safe[:, 1], margin, rows * cell_size - margin)

    # 2. Extraction des centres des murs
    walls = []
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                walls.append([(c + 0.5) * cell_size, (r + 0.5) * cell_size])
    
    if not walls: 
        return tau_safe
        
    walls = torch.tensor(walls, dtype=torch.float32)
    h = cell_size / 2.0  # Demi-taille du mur
    
    # Itération pour repousser les points
    for _ in range(15):
        # Calcul de la distance entre tous les points et tous les murs
        diff = tau_safe.unsqueeze(1) - walls.unsqueeze(0)  # Shape: [N_points, N_walls, 2]
        abs_diff = torch.abs(diff)
        
        # Vecteur de distance au bord de la boîte
        d = abs_diff - h
        
        # Distance SDF analytique pour une boîte
        dist_out = torch.norm(torch.clamp(d, min=0.0), dim=-1)
        dist_in = torch.min(torch.max(d, dim=-1)[0], torch.zeros_like(d[:,:,0]))
        sdf = dist_out + dist_in  # Shape: [N_points, N_walls]
        
        # On vérifie si un point est trop près d'un mur (< margin)
        mask = sdf < margin
        if not mask.any():
            break  # Tout est sécurisé !
            
        # Pour chaque point en collision, on le pousse hors de son mur le plus proche
        for idx in torch.where(mask.any(dim=1))[0]:
            w_idx = torch.argmin(sdf[idx])  # Le mur le plus incriminé
            
            # Calcul du gradient exact de répulsion
            if dist_out[idx, w_idx] > 0:
                # Le point est à l'extérieur du mur, mais dans la zone de marge
                grad = torch.clamp(d[idx, w_idx], min=0.0) * torch.sign(diff[idx, w_idx])
                grad = grad / (torch.norm(grad) + 1e-8)
            else:
                # Le point a pénétré à l'intérieur du mur (Alerte rouge)
                grad = torch.zeros(2)
                # On le pousse vers la face la plus proche
                if d[idx, w_idx, 0] > d[idx, w_idx, 1]:
                    grad[0] = torch.sign(diff[idx, w_idx, 0])
                else:
                    grad[1] = torch.sign(diff[idx, w_idx, 1])
                    
            # Poussée géométrique proportionnelle à la pénétration
            penetration = margin - sdf[idx, w_idx]
            tau_safe[idx] = tau_safe[idx] + grad * penetration * 1.05 # +5% pour être sûr de sortir
            
    return tau_safe


model.eval()
with torch.no_grad():
    # Phase A : Génération Nominale
    x_curr = torch.randn(1, horizon * 2) + torch.tensor([6.5, 6.5]).repeat(horizon)
    num_steps = 20
    for i in range(num_steps):
        t_start = i / num_steps
        t_end = (i + 1) / num_steps
        x_curr = model.step_euler(x_curr, t_start, t_end)
    
    tau_nominal = x_curr.view(horizon, 2)
    
    # Phase B : Warping Géométrique DOUBLE (Ellipses + Labyrinthe)
    # 1. On esquive les ellipses
    tau_warp = apply_analytical_cbf(tau_nominal, ellipses_params, margin=0.2)
    # 2. Si l'esquive nous a poussés dans un mur, on se repousse du mur !
    tau_warp = apply_maze_cbf(tau_warp, maze, cell_size=1.0, margin=0.15)
    
    # Phase C : Inpainting Temporel (CBF-lite)
    t_edit = 0.65 
    x_0_edit = torch.randn(1, horizon * 2) + torch.tensor([6.5, 6.5]).repeat(horizon)
    x_edit_t = (1 - t_edit) * x_0_edit + t_edit * tau_warp.view(1, horizon * 2)
    
    num_edit_steps = 15
    dt_edit = (1.0 - t_edit) / num_edit_steps
    
    for i in range(num_edit_steps):
        t_start = t_edit + i * dt_edit
        t_end = t_edit + (i + 1) * dt_edit
        
        x_edit_t = model.step_euler(x_edit_t, t_start, t_end)
        
        # Projection de sécurité à chaque pas (Effet élastique contraint)
        tau_temp = x_edit_t.view(horizon, 2)
        
        # On applique la sécurité sur les deux environnements
        tau_temp = apply_analytical_cbf(tau_temp, ellipses_params, margin=0.05)
        tau_temp = apply_maze_cbf(tau_temp, maze, cell_size=1.0, margin=0.05)
        
        x_edit_t = tau_temp.view(1, horizon * 2)

    tau_smoothed = x_edit_t.view(horizon, 2)

# ============================================================
# 5. VISUALISATION FINALE
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
cell_size = 1.0
rows, cols = maze.shape

# Dessin des murs du labyrinthe
for r in range(rows):
    for c in range(cols):
        if maze[r, c] == 1:
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='steelblue', alpha=0.9))

# Dessin des obstacles elliptiques
for center, a, b in ellipses_params:
    ax.add_patch(Ellipse(center, 2*a, 2*b, color='navy', alpha=0.6))

# Tracé
ax.plot(tau_nominal[:, 0].numpy(), tau_nominal[:, 1].numpy(), 'r--', linewidth=2, label='Nominale (Collision)')
ax.plot(tau_warp[:, 0].numpy(), tau_warp[:, 1].numpy(), 'orange', linewidth=2, marker='.', markersize=6, label='Warping (Saccadée)')
ax.plot(tau_smoothed[:, 0].numpy(), tau_smoothed[:, 1].numpy(), 'g-', linewidth=4, label='Fast-SafeFlow (Lissée)')

# Départ et Arrivée
ax.plot(tau_nominal[0, 0].numpy(), tau_nominal[0, 1].numpy(), 'go', markersize=8, label='Start')
ax.plot(tau_nominal[-1, 0].numpy(), tau_nominal[-1, 1].numpy(), 'r*', markersize=12, label='Goal')

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')
ax.set_title("Validation Fast-SafeFlow dans un Labyrinthe Multimodal")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("fast_safeflow_results.png")