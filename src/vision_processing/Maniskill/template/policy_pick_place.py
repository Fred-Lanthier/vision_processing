"""
policy_pick_place.py
====================
Politique experte scriptee PICK-AND-PLACE (vectorisee, batch).

Machine a etats a 8 phases :
  1 HOVER     : se placer au-dessus du cube, gripper ouvert
  2 DESCEND   : descendre jusqu'au centre du cube
  3 GRASP     : fermer le gripper et maintenir quelques pas
  4 LIFT      : remonter avec le cube
  5 TRANSPORT : translater au-dessus de l'etagere (a droite)
  6 PLACE     : descendre poser le cube sur l'etagere
  7 RELEASE   : ouvrir le gripper et maintenir
  8 RETREAT   : remonter, puis termine

Choix de design : on ne pilote QUE la position (action[:, :3]) + le gripper
(action[:, 6]) ; les deltas de rotation restent nuls. Le keyframe de homing
oriente deja le gripper vers le bas, et `pd_ee_delta_pose` conserve cette
orientation -> grasp top-down stable, sans controle d'orientation explicite.

IMPORTANT - echelle des actions : pour `pd_ee_delta_pose`, l'action de position
est NORMALISEE dans [-1, 1] (mappee sur +-0.1 m via pos_lower/upper du
controleur), PAS en metres. `max_step` est donc une fraction de cette plage :
max_step=0.5 -> jusqu'a 0.05 m de deplacement par pas de controle. Une valeur
trop faible (ex: 0.04) rend la politique ~20x trop lente et l'episode est
tronque avant la fin.

Gripper (mimic Panda, action normalisee) : +1 = ouvert, -1 = ferme.
"""

import torch

from policy_base import ScriptedPolicy

HOVER, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, RELEASE, RETREAT = range(1, 9)


class PickPlacePolicy(ScriptedPolicy):
    def __init__(self,
                 approach_h=0.12, grasp_dz=0.0, lift_h=0.12,
                 transport_h=0.18, place_dz=0.005,
                 reach_tol=0.012, xy_tol=0.02,
                 pos_gain=5.0, max_step=0.13,
                 grasp_hold=10, release_hold=10):
        self.approach_h = approach_h
        self.grasp_dz = grasp_dz
        self.lift_h = lift_h
        self.transport_h = transport_h
        self.place_dz = place_dz
        self.reach_tol = reach_tol
        self.xy_tol = xy_tol
        self.pos_gain = pos_gain
        self.max_step = max_step
        self.grasp_hold = grasp_hold
        self.release_hold = release_hold

    def reset(self, env):
        uw = env.unwrapped
        B, dev = uw.num_envs, uw.device
        self.B, self.dev = B, dev
        self.phase = torch.ones(B, dtype=torch.long, device=dev)
        self.timer = torch.zeros(B, dtype=torch.int32, device=dev)
        self.locked = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

    def act(self, env, obs):
        uw = env.unwrapped
        B, dev = self.B, self.dev
        links = {l.name: l for l in uw.agent.robot.get_links()}
        tcp = links[uw.agent.ee_link_name].pose.p
        cube_p, _ = uw.get_cube_pose()
        place = uw.place_target
        ph = self.phase

        # ---- Cible de position + commande gripper par phase --------------- #
        target = tcp.clone()
        grip = torch.ones(B, device=dev)  # ouvert par defaut

        m = ph == HOVER
        target[m, 0], target[m, 1] = cube_p[m, 0], cube_p[m, 1]
        target[m, 2] = cube_p[m, 2] + self.approach_h

        m = ph == DESCEND
        target[m, 0], target[m, 1] = cube_p[m, 0], cube_p[m, 1]
        target[m, 2] = cube_p[m, 2] + self.grasp_dz

        m = ph == GRASP
        target[m] = self.locked[m]
        grip[m] = -1.0

        m = ph == LIFT
        target[m, 0], target[m, 1] = tcp[m, 0], tcp[m, 1]
        target[m, 2] = self.lift_h
        grip[m] = -1.0

        m = ph == TRANSPORT
        target[m, 0], target[m, 1] = place[m, 0], place[m, 1]
        target[m, 2] = self.transport_h
        grip[m] = -1.0

        m = ph == PLACE
        target[m] = place[m]
        target[m, 2] = place[m, 2] + self.place_dz
        grip[m] = -1.0

        m = ph == RELEASE
        target[m] = self.locked[m]
        grip[m] = 1.0

        m = ph == RETREAT
        target[m, 0], target[m, 1] = tcp[m, 0], tcp[m, 1]
        target[m, 2] = self.transport_h
        grip[m] = 1.0

        # ---- Action (position-only + gripper) ----------------------------- #
        action = torch.zeros((B, 7), dtype=torch.float32, device=dev)
        delta = (target - tcp) * self.pos_gain
        action[:, :3] = torch.clamp(delta, -self.max_step, self.max_step)
        action[:, 6] = grip

        # ---- Timers des phases de maintien -------------------------------- #
        hold = (ph == GRASP) | (ph == RELEASE)
        self.timer[hold] += 1
        self.timer[~hold] = 0

        # ---- Transitions -------------------------------------------------- #
        dist = torch.norm(target - tcp, dim=1)
        xy = torch.norm((target - tcp)[:, :2], dim=1)
        tol = self.reach_tol
        new = ph.clone()

        new[(ph == HOVER) & (dist < tol)] = DESCEND

        m = (ph == DESCEND) & (dist < tol)
        new[m] = GRASP
        self.locked[m] = tcp[m]

        new[(ph == GRASP) & (self.timer > self.grasp_hold)] = LIFT
        new[(ph == LIFT) & (dist < tol)] = TRANSPORT
        new[(ph == TRANSPORT) & (xy < self.xy_tol)] = PLACE

        m = (ph == PLACE) & (dist < tol)
        new[m] = RELEASE
        self.locked[m] = tcp[m]

        new[(ph == RELEASE) & (self.timer > self.release_hold)] = RETREAT
        self.done[(ph == RETREAT) & (dist < tol)] = True

        self.phase = new
        return action, self.done.clone()
