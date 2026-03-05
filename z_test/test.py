import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal

# 1. Configuration des données
# Création d'une grille de points (x, y)
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Paramètres des distributions (Sigma arbitraire de 0.6 pour bien voir la séparation)
sigma = 0.6
cov_matrix = [[sigma**2, 0], [0, sigma**2]]

# Génération des deux normales
rv1 = multivariate_normal([1, 0], cov_matrix)
rv2 = multivariate_normal([-1, 0], cov_matrix)

# Somme des deux distributions (Distribution conjointe Z)
Z = rv1.pdf(pos) + rv2.pdf(pos)

# 2. Calcul des intégrales (Marginales)
# p_X(x) = intégrale de Z par rapport à y
# axis=0 correspond à l'axe y dans notre meshgrid (lignes)
p_X = np.trapz(Z, y, axis=0)

# p_Y(y) = intégrale de Z par rapport à x
# axis=1 correspond à l'axe x dans notre meshgrid (colonnes)
p_Y = np.trapz(Z, x, axis=1)

# 3. Création du graphique
fig = plt.figure(figsize=(10, 10))

# Définition de la grille: 4x4
# On laisse un peu d'espace pour les labels
gs = gridspec.GridSpec(4, 4, wspace=0.1, hspace=0.1)

# --- Graphique principal (Distribution conjointe 2D/3D) ---
# Position: en bas à droite (occupe 3x3 cases)
ax_main = fig.add_subplot(gs[1:4, 1:4])
contour = ax_main.contourf(X, Y, Z, levels=20, cmap='viridis')
ax_main.set_xlabel('x')
ax_main.set_ylabel('y')
ax_main.grid(True, linestyle='--', alpha=0.3)
ax_main.text(1, 0, 'Centre (1,0)', color='white', ha='center', va='center', fontweight='bold')
ax_main.text(-1, 0, 'Centre (-1,0)', color='white', ha='center', va='center', fontweight='bold')

# --- Graphique du haut (Marginale X) ---
# Position: en haut à droite (occupe 1x3 cases)
ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.plot(x, p_X, color='crimson', linewidth=2)
ax_top.fill_between(x, p_X, color='crimson', alpha=0.3)
ax_top.set_ylabel('$p_X(x)$')
ax_top.set_title('Distribution Complète et Marginales', fontsize=14)
# On cache les labels x du graphique du haut pour éviter la redondance
plt.setp(ax_top.get_xticklabels(), visible=False)
ax_top.grid(True, linestyle='--', alpha=0.3)

# --- Graphique de gauche (Marginale Y) ---
# Position: en bas à gauche (occupe 3x1 cases)
ax_left = fig.add_subplot(gs[1:4, 0], sharey=ax_main)
# Attention: on plot (p_Y, y) pour que la courbe soit verticale
ax_left.plot(p_Y, y, color='dodgerblue', linewidth=2)
ax_left.fill_betweenx(y, 0, p_Y, color='dodgerblue', alpha=0.3)
ax_left.set_xlabel('$p_Y(y)$')
# On inverse l'axe x pour que "le haut" de la courbe pointe vers le graphique principal
ax_left.invert_xaxis()
# On cache les labels y du graphique de gauche (sauf si on veut les voir)
# Ici je les laisse car ils sont à l'extérieur
ax_left.grid(True, linestyle='--', alpha=0.3)

plt.show()