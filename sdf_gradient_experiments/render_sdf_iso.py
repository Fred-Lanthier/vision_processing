#!/usr/bin/env python3
"""Isometric matplotlib render of the Bernstein robot SDF (thesis-styled).

ONE figure, three robot poses side by side (3 subplots), each showing in an
isometric 3D view:
  - the robot surface (grey, ground-truth mesh),
  - the d_safe iso-distance surface of the learned SDF (blue "hero" shell,
    extracted by marching cubes), and
  - the SDF gradient field on that surface (arrows = +grad h, i.e. the outward
    CBF push direction).

Rendered entirely with matplotlib + thesis_style (same fonts/palette/sizing as
every other thesis figure). The dense 3D meshes are rasterized so the SVG stays
compact while the axis text stays vector. Reads the production models read-only;
writes only into ./thesis_results/ as .svg (thesis) + .png (preview).

Run with the venv python from the package root or this folder:
    python3 render_sdf_iso.py
    python3 render_sdf_iso.py --level 0.015          # iso-distance shell [m]
    python3 render_sdf_iso.py --name fig_sdf_iso
"""
import argparse
import os
import sys

import numpy as np
import torch

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thesis_results")
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", ".."))
sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import thesis_style as ts  # noqa: E402
ts.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from skimage import measure  # noqa: E402

D_SAFE = 0.015           # m — the CBF clearance margin (the highlighted shell)
# True isometric camera: equal foreshortening on all three axes, orthographic
# projection (no perspective), exactly the canonical iso view.
ELEV, AZIM = 35.264, -45
N_ARROWS = 60            # gradient arrows per subplot
ARROW_LEN = 0.035        # m

# Three poses (7 joint angles). DEFAULT_Q is the launch spawn pose; the other
# two extend the shoulder/elbow (joints 2 and 4) so the arm silhouette — and the
# SDF shell that follows it — is clearly different in each panel.
POSES = [
    ("(a)", V.DEFAULT_Q.copy()),
    ("(b)", V.DEFAULT_Q + np.array([0.5, 0.55, 0.0, 1.05, 0.0, -0.55, 0.0], np.float32)),
    # forward-leaning reach with an open elbow (a plausible reach-to-plate config)
    ("(c)", V.DEFAULT_Q + np.array([0.3, 0.90, 0.0, 1.20, 0.0, -0.60, 0.0], np.float32)),
]


def compute_pose(core, rl, links, q7, args, dev):
    """Robot mesh, d_safe isosurface (verts/faces) and gradient samples for one pose."""
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(q7, 0.001, dev)
    robot_mesh = V.selected_robot_mesh(rl, pose, q9, links)
    bounds = V.automatic_bounds(robot_mesh, args.padding)
    grid_points, dims, origin = V.make_grid(bounds, args.grid_spacing)
    sdf = V.evaluate_sdf(core, grid_points, pose, q9, args.chunk_size, dev)
    vol = sdf.reshape(dims, order="F")     # vol[i,j,k] -> (xs[i], ys[j], zs[k])

    sp = args.grid_spacing
    verts, faces, _, _ = measure.marching_cubes(vol, level=args.level,
                                                spacing=(sp, sp, sp),
                                                step_size=args.step_size)
    verts = verts + np.asarray(origin, dtype=np.float32)   # index units -> world

    # Gradient samples: a spatially spread subset of the isosurface vertices.
    n = min(N_ARROWS, len(verts))
    pick = np.random.default_rng(1).choice(len(verts), n, replace=False)
    gpts = verts[pick].astype(np.float32)
    _, grad = V.evaluate_sdf_and_grad(core, gpts, pose, q9, args.chunk_size, dev)
    gnorm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad = grad / np.clip(gnorm, 1e-9, None)
    return dict(robot=robot_mesh, verts=verts, faces=faces, gpts=gpts, grad=grad)


def draw_pose(ax, data):
    # Robot surface (grey, ground truth) — full mesh, fairly opaque so it stays
    # visible through the translucent shell drawn over it.
    rc = Poly3DCollection(data["robot"].triangles, facecolor="#9aa0a8",
                          edgecolor="none", alpha=0.9)
    rc.set_rasterized(True)
    ax.add_collection3d(rc)

    # d_safe isosurface (blue hero shell) — the FULL closed marching-cubes
    # surface, kept translucent so the robot inside shows through.
    sc = Poly3DCollection(data["verts"][data["faces"]], facecolor=ts.BLUE,
                          edgecolor="none", alpha=0.13)
    sc.set_rasterized(True)
    ax.add_collection3d(sc)

    # SDF gradient field on the shell (+grad h = outward CBF push direction).
    p, g = data["gpts"], data["grad"]
    q = ax.quiver(p[:, 0], p[:, 1], p[:, 2], g[:, 0], g[:, 1], g[:, 2],
                  length=ARROW_LEN, normalize=True, color=ts.ORANGE,
                  linewidth=0.7, arrow_length_ratio=0.45)
    q.set_rasterized(True)

    # Equal-aspect cube centred on the content.
    pts = np.vstack([data["verts"], data["robot"].vertices])
    ctr = (pts.min(0) + pts.max(0)) / 2.0
    half = (pts.max(0) - pts.min(0)).max() / 2.0 * 1.02
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    # zoom>1 enlarges the 3D content within its axes box, removing most of the
    # whitespace 3D axes leave around the plot.
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.4)
    except TypeError:                  # older matplotlib without the zoom kwarg
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("ortho")          # orthographic -> true isometric look
    # Clean qualitative figure: no axis panes/ticks (their labels are what
    # bled into the neighbouring panels and added the empty top band). The
    # d_safe shell already carries the only scale that matters (1.5 cm).
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=float, default=D_SAFE,
                    help="iso-distance surface [m] (default d_safe = 0.015).")
    ap.add_argument("--grid-spacing", type=float, default=0.008,
                    help="SDF sampling grid spacing [m] (finer = smoother shell).")
    ap.add_argument("--padding", type=float, default=0.06)
    ap.add_argument("--chunk-size", type=int, default=32768)
    ap.add_argument("--step-size", type=int, default=2,
                    help="marching-cubes stride in voxels (>=1). Larger = fewer "
                         "triangles and faster render, still a connected smooth "
                         "surface. 1 = finest.")
    ap.add_argument("--links", default="all",
                    help="which links to render: 'all' (default), 'protected' "
                         "(the CBF set), or a comma-separated list of SDF link "
                         "names (e.g. panda_link5,panda_hand,fork_tip).")
    ap.add_argument("--name", default="fig_sdf_iso")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    link_arg = "all" if args.links in ("all", "all-links") else args.links
    links = V.resolve_link_names(link_arg)
    print(f"[sdf-iso] links: {', '.join(links)}")
    rl, _, core = V.build_sdf_stack(dev, links)

    fig = plt.figure(figsize=(ts.TEXTWIDTH, 2.6))
    for col, (label, q7) in enumerate(POSES):
        print(f"[sdf-iso] pose {label} ...")
        data = compute_pose(core, rl, links, np.asarray(q7, np.float32), args, dev)
        ax = fig.add_subplot(1, len(POSES), col + 1, projection="3d")
        draw_pose(ax, data)
        # panel label inside the panel (no title -> no gap to the legend)
        ax.text2D(0.5, 0.96, fr"$\mathbf{{{label}}}$", transform=ax.transAxes,
                  ha="center", va="top", fontsize=9)

    handles = [
        Patch(facecolor="#b8b8b8", alpha=0.5, label="Surface du robot"),
        Patch(facecolor=ts.BLUE, alpha=0.45,
              label=r"Isosurface $d_{\mathrm{safe}}$ (%.1f cm)" % (args.level * 100)),
        Line2D([], [], color=ts.ORANGE, lw=1.4,
               label=r"Gradient $\nabla h$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.0, wspace=0.0)
    ts.save(fig, OUT, args.name)
    print(f"Saved {os.path.join(OUT, args.name)}.svg / .png")


if __name__ == "__main__":
    main()
