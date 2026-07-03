"""
Data_preprocess_PickPlace.py
============================
Preprocess des demonstrations PICK-AND-PLACE (ManiSkill) vers le MEME format que
le pipeline fourchette (`Data_preprocess.py` -> `Merged_Fork`), afin de reutiliser
tel quel le data loader d'entrainement `Data_Loader_Fork_FlowMP_9D.py`.

Differences voulues avec le pipeline fourchette :
  * PAS de fourchette : on utilise le nuage de la PINCE panda_hand comme "outil".
  * PAS de SAM : ni segmentation, ni tracking SAM2 du robot. Les objets utiles a
    la politique (cube a saisir + etagere/boite de depot a droite) sont
    reconstruits a partir de leur pose GT enregistree a chaque step (record_actors)
    -- equivalent d'un masque parfait, sans perception.
  * On NE sauvegarde PAS le robot complet : seulement outil (pince) + cube +
    etagere. (Le `Merged_Fork` original etait outil + nourriture ; ici on ajoute
    l'etagere car le depot fait partie de la tache.)
  * Labels ecrits sous les MEMES cles que la fourchette (`fork_tip_position`,
    `fork_tip_orientation`, `Merged_Fork_point_cloud`) + `gripper_open` pour
    entrainer une action jointe 10D pose+gripper.

Centrage : chaque nuage `Merged_Fork_XXXX.npy` est centre en TRANSLATION sur le
TCP du step (orientation monde conservee), exactement comme la fourchette centre
sur la pointe (gravity-aware, position-invariant). Le "fork_tip" EST le TCP ici,
donc T_ee_fork = identite.

Entree :  datas/<RECORD_DIR>/Trajectory_*/trajectory_*.json   (avec states[].actors.cube)
Sortie :  datas/PickAndPlace_preprocess/Trajectory_*/
            trajectory_*.json                       (+ fork_tip_*, Merged_Fork_point_cloud)
            Merged_Fork_Trajectory_*/Merged_Fork_XXXX.npy   (1024x3, centre TCP)
          datas/PickAndPlace_preprocess/pos_and_rot.json

Lancer (depuis ce dossier, venv actif) :
    python3 Data_preprocess_PickPlace.py
"""

import glob
import json
import os

import numpy as np
import rospkg
from scipy.spatial.transform import Rotation as R

try:
    import fpsample
    _HAS_FPS = True
except Exception:
    _HAS_FPS = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RECORD_DIR = "PickPlace_record_TEST"        # dossier source (sortie du collect)
PREPROCESS_DIR = "PickAndPlace_preprocess"  # dossier cible (demande par l'utilisateur)
NUM_POINTS = 1024                           # taille finale du nuage (= model input)
SUBSAMPLE_POS_AND_ROT = 20                  # pas pour pos_and_rot.json (comme l'original)
GRIPPER_OPEN_THRESHOLD = 0.03               # ouverture moyenne des doigts (m)

# Objets de la scene reconstruits dans le nuage (depuis leur pose GT enregistree).
# Ce sont les "masques" parfaits des objets utiles a la politique :
#   * cube  -> cible a SAISIR
#   * shelf -> cible de DEPOT (etagere/boite a droite). Indispensable : sans elle
#              le modele ne percoit pas ou poser le cube.
# Format : name -> (demi-tailles boite [x,y,z] (m), nb de points echantillonnes).
# Les demi-tailles DOIVENT correspondre a env_pick_place.py (CUBE_HALF, SHELF_HALF).
OBJECT_SPECS = {
    "cube":  ([0.02, 0.02, 0.02], 600),
    "shelf": ([0.08, 0.08, 0.005], 900),   # boite 1 cm de haut (cf. SHELF_HALF)
}

# Filtrage de visibilite des objets cibles : ne garde que les faces visibles
# depuis la camera statique du deploiement (back-face culling, exact pour une
# boite convexe). Le pipeline reel (SAM + profondeur) ne voit que la coquille
# faisant face a la camera ; echantillonner la surface COMPLETE creait un
# decalage train/deploiement (centroide du nuage ~1-2 cm plus loin de la
# camera). L'outil (pince) reste complet : au deploiement il est aussi
# reconstruit analytiquement (condition_pcd_pickplace.py), pas vu par camera.
# Camera statique du xacro : base-rel (0.9, -0.19, 0.62) ; base ManiSkill a
# world (-0.615, 0, 0) -> monde ManiSkill (0.285, -0.19, 0.62).
OBJECT_VISIBLE_ONLY = True
STATIC_CAM_POS = np.array([0.285, -0.19, 0.62])

# Repartition indicative des points bruts de l'outil avant FPS (oversampling).
N_TOOL_BODY = 1200
N_TOOL_FINGER = 450     # par doigt

# Geometrie de la pince Panda, exprimee dans le repere du TCP (panda_hand_tcp).
# Convention panda_hand : +z = axe d'approche (vers les doigts), les doigts
# coulissent selon +-y. Le TCP est a +0.1034 m du repere panda_hand le long de +z,
# donc z_tcp = z_hand - 0.1034. Les doigts s'attachent a z_hand = 0.0584 et font
# 0.054 m de long -> pointe a z_hand ~ 0.112 (~ TCP). Le corps de la pince est
# derriere le TCP (z_tcp negatif).
TCP_FROM_HAND_Z = 0.1034
FINGER_BASE_Z_HAND = 0.0584
FINGER_LEN = 0.054
FINGER_HALF_X = 0.012      # demi-largeur du doigt (selon x)
FINGER_THICK_Y = 0.012     # epaisseur du doigt (selon y)
HAND_HALF_X = 0.020        # demi-profondeur du corps (selon x)
HAND_HALF_Y = 0.045        # demi-largeur du corps (selon y)
HAND_BACK_Z_HAND = -0.010  # arriere du corps (repere hand)


# --------------------------------------------------------------------------- #
# Helpers geometrie
# --------------------------------------------------------------------------- #
def _sample_box_surface(half, center, n, rng):
    """Echantillonne n points sur la SURFACE d'une boite (half-sizes, centre)."""
    half = np.asarray(half, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    # Aire par paire de faces ~ produit des deux autres dimensions -> repartition.
    areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
    areas = areas / areas.sum()
    pts = np.empty((n, 3))
    for i in range(n):
        ax = rng.choice(3, p=areas)        # axe normal a la face
        p = (rng.random(3) * 2 - 1) * half  # point dans le volume
        p[ax] = half[ax] * (1 if rng.random() < 0.5 else -1)  # colle sur la face
        pts[i] = p + center
    return pts


def _sample_box_surface_with_normals(half, n, rng):
    """Comme _sample_box_surface (centre origine) mais retourne aussi la normale
    sortante de la face de chaque point (repere local de la boite)."""
    half = np.asarray(half, dtype=np.float64)
    areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
    areas = areas / areas.sum()
    pts = np.empty((n, 3))
    nrm = np.zeros((n, 3))
    for i in range(n):
        ax = rng.choice(3, p=areas)
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        p = (rng.random(3) * 2 - 1) * half
        p[ax] = half[ax] * sgn
        pts[i] = p
        nrm[i, ax] = sgn
    return pts, nrm


def sample_box_visible_world(half, R_obj, p_obj, n, rng, cam_pos, max_tries=10):
    """n points sur les faces d'une boite (pose monde R_obj/p_obj) VISIBLES depuis
    cam_pos (back-face culling : normale monde . (cam - point) > 0). Boite convexe
    -> equivalent exact a l'auto-occlusion. N'inclut pas l'occlusion par d'autres
    objets ni par le robot (approximation assumee)."""
    kept = []
    total = 0
    for _ in range(max_tries):
        loc, nrm = _sample_box_surface_with_normals(half, 3 * n, rng)
        pw = loc @ R_obj.T + p_obj
        nw = nrm @ R_obj.T
        vis = np.einsum('ij,ij->i', nw, cam_pos[None, :] - pw) > 1e-12
        if vis.any():
            kept.append(pw[vis])
            total += int(vis.sum())
            if total >= n:
                break
    if total == 0:  # degenere (ne devrait pas arriver pour une boite)
        loc, _ = _sample_box_surface_with_normals(half, n, rng)
        return loc @ R_obj.T + p_obj
    out = np.vstack(kept)
    return out[:n] if len(out) >= n else out[np.arange(n) % len(out)]


def build_tool_cloud_tcp(q_left, q_right, rng):
    """Nuage de la pince panda_hand dans le repere du TCP (z=approche, +-y=doigts).

    q_left/q_right : ouverture de chaque doigt (m), depuis joint_positions[7:9].
    Retour : (N,3) corps + 2 doigts, centre implicitement sur le TCP (origine).
    """
    def to_tcp_z(z_hand):
        return z_hand - TCP_FROM_HAND_Z

    # Corps de la pince (boite) en repere hand -> tcp.
    body_zc_hand = 0.5 * (HAND_BACK_Z_HAND + FINGER_BASE_Z_HAND)
    body_half = [HAND_HALF_X, HAND_HALF_Y, 0.5 * (FINGER_BASE_Z_HAND - HAND_BACK_Z_HAND)]
    body_center = [0.0, 0.0, to_tcp_z(body_zc_hand)]
    body = _sample_box_surface(body_half, body_center, N_TOOL_BODY, rng)

    # Doigts : boites fines le long de z, a +-(q + epaisseur/2) en y.
    fz_center_hand = FINGER_BASE_Z_HAND + 0.5 * FINGER_LEN
    fz_half = 0.5 * FINGER_LEN
    finger_half = [FINGER_HALF_X, 0.5 * FINGER_THICK_Y, fz_half]

    yl = q_left + 0.5 * FINGER_THICK_Y
    left = _sample_box_surface(finger_half, [0.0, yl, to_tcp_z(fz_center_hand)],
                               N_TOOL_FINGER, rng)
    yr = -(q_right + 0.5 * FINGER_THICK_Y)
    right = _sample_box_surface(finger_half, [0.0, yr, to_tcp_z(fz_center_hand)],
                                N_TOOL_FINGER, rng)

    return np.vstack([body, left, right])


def downsample_fps(points, num_target=NUM_POINTS, h=5):
    """Farthest Point Sampling vers une taille fixe (sinon retour tel quel)."""
    if points.shape[0] <= num_target:
        return points
    if _HAS_FPS:
        idx = fpsample.bucket_fps_kdline_sampling(points.astype(np.float32), num_target, h=h)
        return points[idx]
    # Fallback : tirage aleatoire sans remise.
    idx = np.random.choice(points.shape[0], num_target, replace=False)
    return points[idx]


def quat_wxyz_to_R_and_xyzw(q_wxyz):
    """Quaternion sapien (w,x,y,z) -> (matrice de rotation, quaternion xyzw)."""
    w, x, y, z = q_wxyz
    q_xyzw = [x, y, z, w]
    return R.from_quat(q_xyzw).as_matrix(), q_xyzw


def quat_to_6d(quat_xyzw):
    """Quaternion [x,y,z,w] -> rotation 6D (deux premieres colonnes)."""
    mat = R.from_quat(quat_xyzw).as_matrix()
    return np.concatenate([mat[:, 0], mat[:, 1]])


def get_gripper_open_label(state, q_left, q_right):
    """Retourne 1.0 si la pince doit etre ouverte, 0.0 si elle doit etre fermee."""
    cmd = state.get('gripper_command')
    if cmd is not None:
        return 1.0 if float(cmd) > 0.0 else 0.0

    opening = 0.5 * (float(q_left) + float(q_right))
    return 1.0 if opening >= GRIPPER_OPEN_THRESHOLD else 0.0


# --------------------------------------------------------------------------- #
# Coeur du preprocess
# --------------------------------------------------------------------------- #
def process_trajectory(src_folder, dst_base, rng):
    folder_name = os.path.basename(src_folder)        # "Trajectory_7"
    traj_id = folder_name.split('_')[-1]
    src_json = os.path.join(src_folder, f'trajectory_{traj_id}.json')
    if not os.path.exists(src_json):
        print(f"   skip {folder_name}: pas de json")
        return None

    with open(src_json, 'r') as f:
        data = json.load(f)
    states = data.get('states', [])
    if not states:
        print(f"  skip {folder_name}: pas de states")
        return None

    dst_folder = os.path.join(dst_base, folder_name)
    pcd_dir = os.path.join(dst_folder, f'Merged_Fork_{folder_name}')
    os.makedirs(pcd_dir, exist_ok=True)

    traj_label = {}  # pour pos_and_rot.json

    for state in states:
        step_num = int(state['step_number'])
        step_str = f"{step_num:04d}"

        # --- Pose TCP (= "fork tip") en repere base (= monde fixe ici) -------- #
        tcp = state['links']['panda_hand_tcp']
        p_tcp = np.asarray(tcp['position'], dtype=np.float64)
        R_tcp, quat_xyzw = quat_wxyz_to_R_and_xyzw(tcp['orientation'])

        # --- Nuage outil (pince) : repere TCP -> monde ----------------------- #
        qpos = state.get('joint_positions', [])
        q_left = float(qpos[7]) if len(qpos) >= 9 else 0.04
        q_right = float(qpos[8]) if len(qpos) >= 9 else 0.04
        gripper_open = get_gripper_open_label(state, q_left, q_right)
        tool_tcp = build_tool_cloud_tcp(q_left, q_right, rng)
        tool_world = tool_tcp @ R_tcp.T + p_tcp

        # --- Nuages des objets cibles (cube + etagere) : pose GT -> monde ---- #
        clouds = [tool_world]
        for name, obj in state.get('actors', {}).items():
            spec = OBJECT_SPECS.get(name)
            if spec is None:
                continue
            half, n_pts = spec
            R_obj, _ = quat_wxyz_to_R_and_xyzw(obj['orientation'])
            p_obj = np.asarray(obj['position'], dtype=np.float64)
            if OBJECT_VISIBLE_ONLY:
                clouds.append(sample_box_visible_world(
                    half, R_obj, p_obj, n_pts, rng, STATIC_CAM_POS))
            else:
                obj_local = _sample_box_surface(half, [0, 0, 0], n_pts, rng)
                clouds.append(obj_local @ R_obj.T + p_obj)

        merged_world = np.vstack(clouds)
        merged_world = downsample_fps(merged_world, NUM_POINTS)

        # --- Centrage translation-only sur le TCP (orientation monde gardee) -- #
        merged_centered = (merged_world - p_tcp).astype(np.float32)
        np.save(os.path.join(pcd_dir, f'Merged_Fork_{step_str}.npy'), merged_centered)

        # --- Labels (memes cles que la fourchette) --------------------------- #
        state['fork_tip_position'] = p_tcp.tolist()
        state['fork_tip_orientation'] = list(quat_xyzw)
        state['Merged_Fork_point_cloud'] = f'Merged_Fork_{step_str}.npy'
        state['gripper_open'] = gripper_open

        if step_num % SUBSAMPLE_POS_AND_ROT == 0:
            traj_label[f'step_{step_num}'] = {
                'pos': p_tcp.tolist(),
                'rot_6d': quat_to_6d(quat_xyzw).tolist(),
                'gripper_open': gripper_open,
            }

    # T_ee_fork = identite (le TCP EST la pointe d'outil).
    data['T_ee_fork'] = np.eye(4).tolist()

    with open(os.path.join(dst_folder, f'trajectory_{traj_id}.json'), 'w') as f:
        json.dump(data, f, indent=4)

    print(f"  ok {folder_name}: {len(states)} steps -> {pcd_dir}")
    return f'trajectory_{traj_id}', traj_label


def main():
    pkg = rospkg.RosPack().get_path('vision_processing')
    datas = os.path.join(pkg, 'datas')
    src_base = os.path.join(datas, RECORD_DIR)
    dst_base = os.path.join(datas, PREPROCESS_DIR)
    os.makedirs(dst_base, exist_ok=True)

    src_folders = sorted(
        glob.glob(os.path.join(src_base, 'Trajectory*')),
        key=lambda x: int(x.split('_')[-1]),
    )
    if not src_folders:
        print(f"❌ Aucune trajectoire dans {src_base}")
        return
    if not _HAS_FPS:
        print("⚠️ fpsample indisponible -> downsample aleatoire (FPS recommande).")

    print(f"🔄 Preprocess PickPlace : {len(src_folders)} trajectoires")
    print(f"   {src_base}\n   -> {dst_base}")

    rng = np.random.default_rng(0)
    all_labels = {}
    for src_folder in src_folders:
        res = process_trajectory(src_folder, dst_base, rng)
        if res is not None:
            key, label = res
            all_labels[key] = label

    with open(os.path.join(dst_base, 'pos_and_rot.json'), 'w') as f:
        json.dump(all_labels, f, indent=4)

    print(f"\n✅ Termine. {len(all_labels)} trajectoires -> {dst_base}")


if __name__ == '__main__':
    main()
