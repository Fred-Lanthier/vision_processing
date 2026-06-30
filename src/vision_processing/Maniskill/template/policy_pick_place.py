"""
policy_pick_place.py
====================
Politique experte scriptee PICK-AND-PLACE (vectorisee, batch).

Machine a etats a 8 phases :
  1 HOVER     : suivre une droite depuis la pose initiale vers le cube, jusqu'a
               1 cm AU-DESSUS de son sommet, gripper ouvert
  2 DESCEND   : suivre une courte trajectoire lisse depuis ce point jusqu'au
               centre du cube, pour eviter un changement brutal de direction
  3 GRASP     : fermer le gripper et maintenir quelques pas
  4 LIFT      : depart en ligne droite vers l'etagere a 60 deg par rapport a la
               table (= 30 deg de la verticale)
  5 TRANSPORT : rejoindre le dessus de l'etagere par un ARC lisse (parabole /
               demi-cercle). La hauteur est calculee pour respecter les angles
               de depart/arrivee demandes, pas tiree au hasard
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

import math
import torch

from policy_base import ScriptedPolicy

HOVER, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, RELEASE, RETREAT = range(1, 9)


class PickPlacePolicy(ScriptedPolicy):
    def __init__(self,
                 approach_h=0.01, grasp_dz=0.0, lift_h=0.12, cube_half=0.02,
                 transport_h=0.18, place_dz=0.005,
                 lift_rise=0.10,
                 departure_angle_min=60.0, departure_angle_max=60.0,
                 transport_start_angle_deg=60.0, transport_end_angle_deg=-60.0,
                 place_hover=0.06, arc_mode="parabola",
                 arc_ds=0.05, arc_on_path_tol=0.035,
                 approach_ds=0.08, approach_on_path_tol=0.015,
                 lift_ds=0.10, lift_on_path_tol=0.015,
                 descend_ds=0.12, descend_on_path_tol=0.01,
                 reach_tol=0.012, xy_tol=0.02,
                 pos_gain=5.0, max_step=0.13,
                 grasp_hold=10, release_hold=10):
        # approach_h is now the hover clearance ABOVE THE CUBE TOP (default 1 cm):
        # the HOVER follows an explicit straight segment to a point 1 cm over the
        # cube, then DESCEND drops the short remaining distance to grasp.
        self.approach_h = approach_h
        self.cube_half = cube_half
        self.grasp_dz = grasp_dz
        self.approach_ds = approach_ds
        self.approach_on_path_tol = approach_on_path_tol
        # Smooth HOVER -> DESCEND blend. The TCP first reaches the point 1 cm above
        # the cube, then a short carrot path moves from that point to the grasp
        # point with zero slope at the beginning and end (smoothstep).
        self.descend_ds = descend_ds
        self.descend_on_path_tol = descend_on_path_tol
        self.lift_h = lift_h
        self.lift_ds = lift_ds
        self.lift_on_path_tol = lift_on_path_tol
        self.lift_rise = lift_rise
        self.departure_angle_min = departure_angle_min
        self.departure_angle_max = departure_angle_max
        self.transport_h = transport_h
        self.place_dz = place_dz
        # Arc de transport. Pour une parabole z(s) = z0 + dz*s + A*4*s*(1-s),
        # on choisit dz et A pour respecter les pentes voulues au depart et a
        # l'arrivee. Cela evite les arcs aleatoires trop raides qui deviennent
        # presque verticaux.
        self.transport_start_angle = math.radians(transport_start_angle_deg)
        self.transport_end_angle = math.radians(transport_end_angle_deg)
        # `place_hover` = hauteur de fin d'arc au-dessus du depot (avant la
        # descente PLACE). `arc_mode` = forme du sommet :
        #   "parabola" -> 4 s (1-s)            (montee douce, depart en pente)
        #   "circle"   -> 2 sqrt(s (1-s))      (calotte elliptique, tangentes verticales)
        self.place_hover = place_hover
        self.arc_mode = arc_mode
        # Suivi de chemin (carrot) pour l'arc : on avance le parametre de chemin
        # `arc_ds` par pas SEULEMENT quand le TCP est a moins de `arc_on_path_tol`
        # du point vise. Ainsi le TCP TRACE l'arc au lieu de couper les portions
        # raides (sur un arc haut, la pente dz/dxy depasse ce que le clamp par axe
        # autorise ; sans ce gating, le sommet est sous-atteint). Les arcs hauts
        # prennent juste plus de pas.
        self.arc_ds = arc_ds
        self.arc_on_path_tol = arc_on_path_tol
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
        self.approach_start = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.approach_goal = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.approach_s_cmd = torch.zeros(B, dtype=torch.float32, device=dev)
        self.approach_initialized = torch.zeros(B, dtype=torch.bool, device=dev)
        self.descend_start = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.descend_goal = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.descend_s_cmd = torch.zeros(B, dtype=torch.float32, device=dev)
        self.lift_start = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.lift_goal = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        self.lift_s_cmd = torch.zeros(B, dtype=torch.float32, device=dev)
        # Point de depart de l'arc de transport (TCP a la fin du LIFT).
        self.transport_start = torch.zeros((B, 3), dtype=torch.float32, device=dev)
        # Departure angle (rad), randomized per env in [min, max] (>=45 deg).
        a0 = math.radians(self.departure_angle_min)
        a1 = math.radians(self.departure_angle_max)
        self.departure_angle = a0 + torch.rand(B, device=dev) * (a1 - a0)
        # Parametre de chemin du carrot le long de l'arc de transport [0, 1].
        self.s_cmd = torch.zeros(B, dtype=torch.float32, device=dev)
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
        if m.any():
            init = m & (~self.approach_initialized)
            self.approach_start[init] = tcp[init]
            self.approach_s_cmd[init] = 0.0
            self.approach_initialized[init] = True

            goal = self.approach_goal.clone()
            goal[m, 0], goal[m, 1] = cube_p[m, 0], cube_p[m, 1]
            goal[m, 2] = cube_p[m, 2] + self.cube_half + self.approach_h
            self.approach_goal[m] = goal[m]

            s = self.approach_s_cmd[m]
            target[m] = self.approach_start[m] * (1.0 - s).unsqueeze(1) \
                + goal[m] * s.unsqueeze(1)

        m = ph == DESCEND
        if m.any():
            goal = self.descend_goal.clone()
            goal[m, 0], goal[m, 1] = cube_p[m, 0], cube_p[m, 1]
            goal[m, 2] = cube_p[m, 2] + self.grasp_dz
            self.descend_goal[m] = goal[m]

            s = self.descend_s_cmd[m]
            s_smooth = s * s * (3.0 - 2.0 * s)
            target[m] = self.descend_start[m] * (1.0 - s_smooth).unsqueeze(1) \
                + goal[m] * s_smooth.unsqueeze(1)

        m = ph == GRASP
        target[m] = self.locked[m]
        grip[m] = -1.0

        m = ph == LIFT
        if m.any():
            s = self.lift_s_cmd[m]
            target[m] = self.lift_start[m] * (1.0 - s).unsqueeze(1) \
                + self.lift_goal[m] * s.unsqueeze(1)
            grip[m] = -1.0

        m = ph == TRANSPORT
        if m.any():
            # Suivi de chemin : la cible est le point de l'arc au parametre
            # commande s_cmd (le "carrot"), pas l'etagere directement. xy et z
            # avancent ENSEMBLE le long de l'arc, donc la forme est preservee meme
            # si l'arc est raide. s_cmd n'avance que quand le TCP a rejoint le
            # carrot (voir transitions) -> le TCP trace l'arc sans le couper.
            start_xy = self.transport_start[m, :2]
            start_z = self.transport_start[m, 2]
            place_xy = place[m, :2]
            path_len = torch.norm(place_xy - start_xy, dim=1).clamp(min=1e-6)
            slope0 = math.tan(self.transport_start_angle)
            slope1 = math.tan(self.transport_end_angle)
            end_z_from_angles = start_z + 0.5 * path_len * (slope0 + slope1)
            min_end_z = place[m, 2] + self.place_hover
            end_z = torch.maximum(end_z_from_angles, min_end_z)
            dz = end_z - start_z
            s = self.s_cmd[m]
            if self.arc_mode == "circle":
                bump = 2.0 * torch.sqrt(torch.clamp(s * (1.0 - s), min=0.0))
            else:  # parabola
                bump = 4.0 * s * (1.0 - s)
            # With the unclamped end_z, this satisfies both requested endpoint
            # slopes. If safety clamps end_z above the box, preserve the initial
            # slope and let the final slope adapt.
            arc_amp = ((slope0 * path_len - dz) / 4.0).clamp(min=0.02)
            carrot_xy = start_xy + (place_xy - start_xy) * s.unsqueeze(1)
            target[m, 0] = carrot_xy[:, 0]
            target[m, 1] = carrot_xy[:, 1]
            target[m, 2] = start_z + dz * s + arc_amp * bump
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

        in_approach = ph == HOVER
        on_approach_path = in_approach & (dist < self.approach_on_path_tol)
        self.approach_s_cmd[on_approach_path] = torch.clamp(
            self.approach_s_cmd[on_approach_path] + self.approach_ds,
            max=1.0,
        )

        approach_goal_dist = torch.norm(self.approach_goal - tcp, dim=1)
        m = (ph == HOVER) & (self.approach_s_cmd >= 0.999) & (approach_goal_dist < tol)
        new[m] = DESCEND
        self.descend_start[m] = tcp[m]
        self.descend_goal[m, 0], self.descend_goal[m, 1] = cube_p[m, 0], cube_p[m, 1]
        self.descend_goal[m, 2] = cube_p[m, 2] + self.grasp_dz
        self.descend_s_cmd[m] = 0.0

        in_descend = ph == DESCEND
        on_descend_path = in_descend & (dist < self.descend_on_path_tol)
        self.descend_s_cmd[on_descend_path] = torch.clamp(
            self.descend_s_cmd[on_descend_path] + self.descend_ds,
            max=1.0,
        )

        descend_goal_dist = torch.norm(self.descend_goal - tcp, dim=1)
        m = (ph == DESCEND) & (self.descend_s_cmd >= 0.999) & (descend_goal_dist < tol)
        new[m] = GRASP
        self.locked[m] = tcp[m]

        m = (ph == GRASP) & (self.timer > self.grasp_hold)
        if m.any():
            new[m] = LIFT
            self.lift_start[m] = tcp[m]
            to_shelf = place[m, :2] - tcp[m, :2]
            dir_xy = to_shelf / torch.norm(to_shelf, dim=1, keepdim=True).clamp(min=1e-6)
            run = (self.lift_rise / torch.tan(self.departure_angle[m])).unsqueeze(1)
            self.lift_goal[m, :2] = tcp[m, :2] + dir_xy * run
            self.lift_goal[m, 2] = tcp[m, 2] + self.lift_rise
            self.lift_s_cmd[m] = 0.0

        in_lift = ph == LIFT
        on_lift_path = in_lift & (dist < self.lift_on_path_tol)
        self.lift_s_cmd[on_lift_path] = torch.clamp(
            self.lift_s_cmd[on_lift_path] + self.lift_ds,
            max=1.0,
        )

        lift_goal_dist = torch.norm(self.lift_goal - tcp, dim=1)
        m = (ph == LIFT) & (self.lift_s_cmd >= 0.999) & (lift_goal_dist < tol)
        new[m] = TRANSPORT
        self.transport_start[m] = tcp[m]   # ancre l'arc au sommet du lift
        self.s_cmd[m] = 0.0                # carrot au depart de l'arc

        # Avance le carrot d'arc_ds, uniquement pour les envs en TRANSPORT dont le
        # TCP a rejoint le carrot (target). Les portions raides attendent que le
        # TCP monte avant d'avancer -> l'arc est trace, pas coupe.
        in_transport = ph == TRANSPORT
        on_path = in_transport & (dist < self.arc_on_path_tol)
        self.s_cmd[on_path] = torch.clamp(self.s_cmd[on_path] + self.arc_ds, max=1.0)

        # Fin de l'arc -> descente PLACE.
        new[(ph == TRANSPORT) & (self.s_cmd >= 0.999)] = PLACE

        m = (ph == PLACE) & (dist < tol)
        new[m] = RELEASE
        self.locked[m] = tcp[m]

        new[(ph == RELEASE) & (self.timer > self.release_hold)] = RETREAT
        self.done[(ph == RETREAT) & (dist < tol)] = True

        self.phase = new
        return action, self.done.clone()
