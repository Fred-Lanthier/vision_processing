"""
policy_base.py
==============
Squelette de POLITIQUE EXPERTE scriptee, vectorisee (batch).

`ScriptedPolicy` est l'interface abstraite : `reset()` puis `act()` a chaque
step. Une politique concrete implemente sa machine a etats (FSM) en tenseurs
`(B, ...)` exactement comme les 3 phases de ton `FoodSpikingEnv`.

On fournit aussi :
  * des helpers de controle proportionnel (position / angle) ;
  * `ReachHoldPolicy`, un exemple minimal et fonctionnel (approche l'objet et
    maintient la position) pour que `collect.py` tourne de bout en bout.

L'action attendue par `pd_ee_delta_pose` (Panda) est de dimension 7 :
  [dx, dy, dz, drx, dry, drz, gripper].
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ACTION_DIM = 7  # pd_ee_delta_pose Panda : 3 pos + 3 rot + 1 gripper


def wrap_deg(err):
    """Ramene une erreur angulaire (deg) dans [-180, 180]."""
    err = torch.where(err > 180, err - 360, err)
    err = torch.where(err < -180, err + 360, err)
    return err


def quat_wxyz_to_euler_deg(q_wxyz, device):
    """Quaternion (B,4 wxyz, tenseur) -> euler XYZ deg (3 tenseurs roll/pitch/yaw)."""
    q = q_wxyz.cpu().numpy()
    q_xyzw = np.concatenate((q[:, 1:], q[:, 0:1]), axis=1)
    e = R.from_quat(q_xyzw).as_euler("xyz", degrees=True)
    roll = torch.tensor(e[:, 0], dtype=torch.float32, device=device)
    pitch = torch.tensor(e[:, 1], dtype=torch.float32, device=device)
    yaw = torch.tensor(e[:, 2], dtype=torch.float32, device=device)
    return roll, pitch, yaw


class ScriptedPolicy(ABC):
    """Interface d'une politique experte vectorisee."""

    @abstractmethod
    def reset(self, env):
        """Initialise les buffers d'etat (B,) a partir de l'env reset."""

    @abstractmethod
    def act(self, env, obs):
        """Retourne (action: (B, ACTION_DIM), done_mask: (B,) bool)."""

    # -- helpers communs --------------------------------------------------- #
    @staticmethod
    def goto(action, idx, current_p, target_p, gain=5.0, max_speed=0.05):
        """Remplit action[idx, :3] pour aller de current_p vers target_p."""
        direction = target_p - current_p
        dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
        speed = torch.clamp(dist * gain, max=max_speed)
        action[idx, :3] = (direction / dist) * speed
        return dist.squeeze(-1)


class ReachHoldPolicy(ScriptedPolicy):
    """Exemple minimal : amene l'EE au-dessus de l'objet et maintient.

    Demontre le pattern FSM vectorise sans la complexite du scooping. A copier
    puis enrichir (phases supplementaires : grasp, transport, place...).
    """

    def __init__(self, hover_height=0.05, reach_tol=0.01, hold_target=15):
        self.hover_height = hover_height
        self.reach_tol = reach_tol
        self.hold_target = hold_target

    def reset(self, env):
        uw = env.unwrapped
        B, dev = uw.num_envs, uw.device
        self.B, self.dev = B, dev
        self.hold = torch.zeros(B, dtype=torch.int32, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

    def act(self, env, obs):
        uw = env.unwrapped
        links = {l.name: l for l in uw.agent.robot.get_links()}
        ee = links[uw.agent.ee_link_name]
        ee_pos = ee.pose.p

        obj_p, _ = uw.get_active_object_poses()
        target = obj_p.clone()
        target[:, 2] += self.hover_height

        action = torch.zeros((self.B, ACTION_DIM), dtype=torch.float32, device=self.dev)
        action[:, 6] = 1.0  # gripper ouvert

        active = ~self.done
        if active.any():
            idx = active.nonzero(as_tuple=True)[0]
            dist = self.goto(action, idx, ee_pos[idx], target[idx])

            arrived = idx[dist < self.reach_tol]
            self.hold[arrived] += 1
            action[arrived, :3] = 0.0
            self.done[arrived[self.hold[arrived] > self.hold_target]] = True

        return action, self.done.clone()
