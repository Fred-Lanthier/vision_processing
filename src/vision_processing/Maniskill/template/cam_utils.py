"""
cam_utils.py
============
Helpers pour orienter une camera dans la convention SAPIEN/ManiSkill.

Convention camera SAPIEN (verifiee via la matrice sapien2opencv du pipeline
original) :
  * axe optique (forward) = +x local
  * up                    = +z local
  * left                  = +y local   (donc right = -y)

`dir_to_quat_wxyz` construit le quaternion (w,x,y,z) qui oriente la camera de
sorte que son +x pointe selon `forward`, son +z aligne au mieux sur `up`.
`look_at_quat_wxyz` fait pareil a partir d'un point a regarder.

Les vecteurs sont exprimes dans le repere du LIEN DE MONTAGE (= repere base si
la camera est montee sur la base fixe du robot).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def dir_to_quat_wxyz(forward, up=(0.0, 0.0, 1.0)):
    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, dtype=float)

    left = np.cross(u, f)               # +y = left
    if np.linalg.norm(left) < 1e-6:     # forward colineaire a up -> autre ref
        u = np.array([1.0, 0.0, 0.0])
        left = np.cross(u, f)
    left /= np.linalg.norm(left)
    up_corr = np.cross(f, left)         # +z = up

    Rm = np.stack([f, left, up_corr], axis=1)  # colonnes = axes locaux x,y,z
    q = R.from_matrix(Rm).as_quat()            # xyzw
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def look_at_quat_wxyz(cam_pos, target_pos, up=(0.0, 0.0, 1.0)):
    forward = np.asarray(target_pos, dtype=float) - np.asarray(cam_pos, dtype=float)
    return dir_to_quat_wxyz(forward, up)


def rpy_to_quat_wxyz(roll, pitch, yaw):
    """Convertit un rpy URDF (extrinseque xyz) en quaternion (w, x, y, z).

    Sert a reproduire exactement une pose de camera definie dans un URDF
    (ex: la camera wrist de panda_fork.urdf).
    """
    q = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()  # lowercase = extrinseque
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
