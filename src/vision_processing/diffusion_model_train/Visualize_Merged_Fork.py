"""
Visualize_Merged_Fork.py
========================
Visualise UN nuage de points `Merged_Fork_XXXX.npy` d'une trajectoire
preprocessee (sortie de `Data_preprocess_PickPlace.py`).

Le nuage est centre en TRANSLATION sur le TCP (origine) et garde l'orientation
monde. On y trouve : la pince panda_hand (compacte, autour de l'origine), le cube
cible et l'etagere/boite de depot (a -y). Le repere RGB dessine a l'origine est
le TCP (= "fork tip"). Les points sont colores par hauteur z (gravity-aware).

Exemples :
    python3 Visualize_Merged_Fork.py                       # 1er traj, step du milieu
    python3 Visualize_Merged_Fork.py --traj 3 --step 40
    python3 Visualize_Merged_Fork.py --traj 3 --step 40 --save out.png --no-show
    python3 Visualize_Merged_Fork.py --root /chemin/PickAndPlace_preprocess
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401 (Axes3D enregistre la projection 3d)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def find_default_root():
    """datas/PickAndPlace_preprocess via rospkg, sinon chemin relatif."""
    try:
        import rospkg
        pkg = rospkg.RosPack().get_path('vision_processing')
        return os.path.join(pkg, 'datas', 'PickAndPlace_preprocess')
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, '..', '..', '..', 'datas',
                                            'PickAndPlace_preprocess'))


def resolve_pcd_path(root, traj, step):
    """Retourne (chemin_npy, traj_id, step_str) pour le step demande."""
    if traj is None:
        folders = sorted(glob.glob(os.path.join(root, 'Trajectory_*')),
                         key=lambda x: int(x.split('_')[-1]))
        if not folders:
            raise FileNotFoundError(f"Aucune trajectoire dans {root}")
        traj = int(os.path.basename(folders[0]).split('_')[-1])

    folder = os.path.join(root, f'Trajectory_{traj}')
    pcd_dir = os.path.join(folder, f'Merged_Fork_Trajectory_{traj}')
    files = sorted(glob.glob(os.path.join(pcd_dir, 'Merged_Fork_*.npy')))
    if not files:
        raise FileNotFoundError(f"Aucun Merged_Fork_*.npy dans {pcd_dir}")

    if step is None:
        path = files[len(files) // 2]          # step du milieu par defaut
    else:
        path = os.path.join(pcd_dir, f'Merged_Fork_{int(step):04d}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Step {step} absent. Steps dispo : "
                f"{[int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in files]}")

    step_str = os.path.basename(path).split('_')[-1].split('.')[0]
    return path, traj, step_str


def draw_tcp_frame(ax, length=0.05):
    """Repere TCP a l'origine (le nuage est centre dessus)."""
    for vec, col, name in ((np.array([length, 0, 0]), 'r', 'x'),
                           (np.array([0, length, 0]), 'g', 'y'),
                           (np.array([0, 0, length]), 'b', 'z')):
        ax.quiver(0, 0, 0, *vec, color=col, linewidth=2)
    ax.scatter([0], [0], [0], c='k', s=40, marker='*', label='TCP (fork tip)')


def set_equal_aspect(ax, pts):
    """Boite isometrique pour ne pas deformer la geometrie."""
    mins, maxs = pts.min(0), pts.max(0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 + 1e-6
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser(description="Visualise un Merged_Fork point cloud.")
    ap.add_argument('--root', default=None, help="dossier PickAndPlace_preprocess")
    ap.add_argument('--traj', type=int, default=None, help="id de trajectoire")
    ap.add_argument('--step', type=int, default=None, help="numero de step (defaut: milieu)")
    ap.add_argument('--save', default=None, help="chemin PNG de sortie")
    ap.add_argument('--no-show', action='store_true', help="ne pas ouvrir la fenetre")
    args = ap.parse_args()

    root = args.root or find_default_root()
    path, traj, step_str = resolve_pcd_path(root, args.traj, args.step)

    pts = np.load(path).astype(np.float32)
    print(f"📂 {path}")
    print(f"   {pts.shape[0]} points | bornes "
          f"min {np.round(pts.min(0), 3).tolist()} max {np.round(pts.max(0), 3).tolist()}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=pts[:, 2], cmap='viridis', s=6, depthshade=True)
    fig.colorbar(sc, ax=ax, shrink=0.6, label='z (hauteur, m)')

    draw_tcp_frame(ax)
    set_equal_aspect(ax, pts)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(f"Merged_Fork — Trajectory {traj}, step {step_str}  "
                 f"({pts.shape[0]} pts, centre sur le TCP)")
    ax.legend(loc='upper right')
    ax.view_init(elev=20, azim=-0)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"💾 Figure -> {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
