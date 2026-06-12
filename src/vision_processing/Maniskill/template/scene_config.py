"""
scene_config.py
===============
Configuration data-driven des objets de la scene (equivalent generique de
`FOOD_BUILD_CONFIGS`) et des parametres de randomisation par episode.

Chaque objet se decrit par un dict. Type "mesh" (fichier .glb/.obj) ou
primitive "box"/"sphere". Ajouter un objet = ajouter une entree, aucune logique
a reecrire.
"""

from dataclasses import dataclass, field


@dataclass
class ObjectConfig:
    """Un asset spawnable dans la scene."""
    name: str
    type: str = "mesh"                      # "mesh" | "box" | "sphere"
    mesh_path: str = ""                     # si type == "mesh"
    scale: tuple = (1.0, 1.0, 1.0)          # mesh: scale; box: demi-tailles; sphere: rayon en [0]
    half_z: float = 0.0                      # demi-hauteur pour poser sur la surface
    local_p: tuple = (0.0, 0.0, 0.0)         # offset local de pose (corrige le mesh)
    local_euler_xyz_deg: tuple = (0.0, 0.0, 0.0)
    color: tuple = (0.5, 0.5, 0.5, 1.0)
    yaw_offset: bool = False                 # flag exploitable par la politique


@dataclass
class RandomizationConfig:
    """Parametres de domain randomization appliques dans `_initialize_episode`."""
    x_range: tuple = (-0.23, -0.11)          # position de spawn en x
    y_range: tuple = (-0.14, 0.02)           # position de spawn en y
    random_yaw: bool = True                  # yaw aleatoire de l'objet
    use_pedestal: bool = True                # activer le piedestal aleatoire
    pedestal_prob: float = 0.5               # probabilite d'avoir un piedestal
    pedestal_height_range: tuple = (0.04, 0.10)
    pedestal_half_xy: float = 0.05
    pedestal_half_z: float = 0.04


# --------------------------------------------------------------------------- #
# Exemple : reprend tes assets de nourriture (a remplacer par n'importe quoi).
# --------------------------------------------------------------------------- #
EXAMPLE_OBJECT_CONFIGS = {
    "cube_red": ObjectConfig(
        name="cube_red", type="box", scale=(0.02, 0.02, 0.02),
        half_z=0.02, color=(0.8, 0.1, 0.1, 1.0),
    ),
    "cube_blue": ObjectConfig(
        name="cube_blue", type="box", scale=(0.025, 0.025, 0.025),
        half_z=0.025, color=(0.1, 0.2, 0.8, 1.0),
    ),
    "ball_green": ObjectConfig(
        name="ball_green", type="sphere", scale=(0.02, 0.0, 0.0),
        half_z=0.02, color=(0.1, 0.7, 0.2, 1.0),
    ),
}
