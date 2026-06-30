#!/usr/bin/env python3
"""Build the avec/sans-CBF trajectory comparison figures from two rosbags.

Inputs: one bag recorded with the CBF active and one recorded in
monitor-only + contact-stop mode (baseline), same scene and seed:
    rosbag record /cbf_safety/diagnostics /cbf_safety/diagnostics_layout \
        /cbf_safety/contact_event /joint_states /perception/persistent_obstacles \
        /perception/persistent_target /perception/target_cube \
        /planner/nominal_trajectory

The views are centered on the initial target (the food) with a fixed, configurable
box (--view-up/--view-down/--view-left/--view-right, in cm) so the deviation
of the safe path away from the baseline reads at a glance. The target is the
centroid of the first valid target cloud if recorded, else --target x y z,
else the initial CBF fork-tip/TCP point as an explicit fallback.

Outputs:
  comparison_avec_sans.png  fork-tip paths (XY + XZ) through the obstacle
                            cloud, baseline ending at the contact marker,
                            plus estimated clearance vs time for both runs
  deviation_h.png           CBF tip path colored by clearance, baseline path
                            as the unfiltered reference, and deviation vs
                            normalized path progress

Usage:
    python3 analyze_trajectory.py run_cbf.bag run_baseline.bag [-o outdir]
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2

PANDA_JOINTS = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7']
DEFAULT_TARGET_TOPICS = (
    "/perception/persistent_target",
    "/perception/target_cube",
)


def read_target_centroid(path, topics):
    """Median centroid (base frame) of the first target cloud in the bag.

    The median is robust to the few smeared/outlier points the persistent target
    can carry, so the view recenters on the actual food, not on a stray voxel.
    Using the first valid cloud keeps the plots centered on the target's initial
    coordinates instead of the final grasped/carried position.

    Returns (centroid, topic, t_sec) or (None, None, None).
    """
    topics = list(topics)
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            pts = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)))
            if len(pts):
                return np.median(pts.reshape(-1, 3), axis=0), topic, t.to_sec()
    return None, None, None


def load_run(path):
    """Extract joint trajectory, barrier series and obstacle snapshot."""
    run = {"t_q": [], "q": [], "t_h": [], "h": [],
           "contact_time": None, "obs": None}
    layout = None
    obs_msgs = []
    with rosbag.Bag(path) as bag:
        start = bag.get_start_time()
        for topic, msg, t in bag.read_messages():
            ts = t.to_sec() - start
            if topic == "/cbf_safety/diagnostics_layout":
                layout = json.loads(msg.data)
            elif topic == "/joint_states":
                pos = dict(zip(msg.name, msg.position))
                if all(j in pos for j in PANDA_JOINTS):
                    run["t_q"].append(ts)
                    run["q"].append([pos[j] for j in PANDA_JOINTS])
            elif topic == "/cbf_safety/diagnostics":
                run["t_h"].append(ts)
                run["h"].append(list(msg.data))
            elif topic == "/cbf_safety/contact_event":
                if run["contact_time"] is None:
                    run["contact_time"] = ts
            elif topic == "/perception/persistent_obstacles":
                obs_msgs.append((ts, msg))

    if layout is None:
        sys.exit(f"No diagnostics layout in {path}")
    n_vec = layout["vec_dim"] * len(layout["vec_fields"])
    scal = {n: i for i, n in enumerate(layout["scalar_fields"])}
    diag = np.asarray(run["h"], dtype=np.float64)
    t_h = np.asarray(run["t_h"])
    h = diag[:, n_vec + scal["h"]]
    sel = diag[:, n_vec + scal["selected_count"]]
    valid = (sel > 0) & (np.nan_to_num(h, nan=1e9) < 5.0)
    if "contact_stop" in scal and run["contact_time"] is None:
        stopped = np.nan_to_num(diag[:, n_vec + scal["contact_stop"]]) > 0.5
        if stopped.any():
            run["contact_time"] = float(t_h[np.argmax(stopped)])

    # obstacle snapshot: latest message before contact (or 75% of the run)
    t_snap = run["contact_time"] if run["contact_time"] is not None \
        else 0.75 * t_h[-1]
    best = min(obs_msgs, key=lambda m: abs(m[0] - t_snap)) if obs_msgs else None
    if best is not None:
        pts = np.array(list(pc2.read_points(
            best[1], field_names=("x", "y", "z"), skip_nans=True)))
        run["obs"] = pts

    run["t_q"] = np.asarray(run["t_q"])
    run["q"] = np.asarray(run["q"])
    run["t_h"], run["h"], run["valid"] = t_h, h, valid

    # Trim to the active-task window (first/last cycle with a real obstacle
    # selection): removes the startup idle period and prevents the clearance
    # interpolation from extrapolating over the initial approach.
    tv = t_h[valid]
    if len(tv):
        t0, t1 = max(0.0, tv[0] - 2.0), tv[-1] + 1.0
        kq = (run["t_q"] >= t0) & (run["t_q"] <= t1)
        kh = (t_h >= t0) & (t_h <= t1)
        run["t_q"], run["q"] = run["t_q"][kq] - t0, run["q"][kq]
        run["t_h"], run["h"] = t_h[kh] - t0, h[kh]
        run["valid"] = valid[kh]
        if run["contact_time"] is not None:
            run["contact_time"] -= t0
    return run


def make_fk():
    """Replicate the CBF node's FK setup (URDFLayer on the xacro model)."""
    import torch
    import rospkg
    import xacro
    pkg_path = rospkg.RosPack().get_path('vision_processing')
    sys.path.insert(0, pkg_path)
    from third_party.RDF.urdf_layer import URDFLayer
    doc = xacro.process_file(pkg_path + '/urdf/panda_camera.xacro')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(doc.toxml())
        urdf_path = f.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = URDFLayer(urdf_path=urdf_path, device=device, package_dir=pkg_path,
                      voxel_dir=pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128')

    def tip_positions(q_np):
        out = np.zeros((len(q_np), 3))
        q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
        q9 = torch.cat([q, torch.zeros((q.shape[0], 2), device=device)], dim=-1)
        for i in range(len(q_np)):  # the native FK is single-sample
            poses = layer._native_forward_kinematics(q9[i:i + 1])
            out[i] = poses['fork_tip'][0, :3, 3].detach().cpu().numpy()
        return out

    return tip_positions


def clearance(run, d_safe):
    return run["h"] + d_safe


def truncate_at_contact(run, tip, t_tip):
    if run["contact_time"] is None:
        return tip, t_tip
    keep = t_tip <= run["contact_time"]
    return tip[keep], t_tip[keep]


def arclength_resample(path, n=200):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    s /= s[-1]
    si = np.linspace(0, 1, n)
    return np.stack([np.interp(si, s, path[:, k]) for k in range(3)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cbf_bag")
    ap.add_argument("baseline_bag")
    ap.add_argument("-o", "--outdir", default="trajectory_comparison")
    ap.add_argument("--d-safe", type=float, default=0.015)
    ap.add_argument("--stride", type=int, default=4,
                    help="joint-state subsampling for FK")
    ap.add_argument("--target", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="manual target (food) position [m] to center the views "
                         "on; overrides the recorded target cloud")
    ap.add_argument("--target-topic", action="append", default=None,
                    help="PointCloud2 topic to use for the initial target center. "
                         "Can be repeated. Default: /perception/persistent_target "
                         "and /perception/target_cube")
    ap.add_argument("--view-up", type=float, default=30.0,
                    help="view half-extent above the target [cm]")
    ap.add_argument("--view-down", type=float, default=30.0,
                    help="view half-extent below the target [cm]")
    ap.add_argument("--view-left", type=float, default=30.0,
                    help="view half-extent left of the target [cm]")
    ap.add_argument("--view-right", type=float, default=30.0,
                    help="view half-extent right of the target [cm]")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    os.makedirs(args.outdir, exist_ok=True)
    runs = {"cbf": load_run(args.cbf_bag), "base": load_run(args.baseline_bag)}
    fk = make_fk()
    tips = {}
    for k, r in runs.items():
        q = r["q"][::args.stride]
        t = r["t_q"][::args.stride]
        tip = fk(q)
        tips[k] = truncate_at_contact(r, tip, t)
        print(f"{k}: {len(q)} FK samples, contact at "
              f"{r['contact_time'] if r['contact_time'] else '—'} s")

    obs = runs["base"]["obs"] if runs["base"]["obs"] is not None else runs["cbf"]["obs"]
    if obs is not None and len(obs) > 4000:
        obs = obs[np.random.default_rng(0).choice(len(obs), 4000, replace=False)]

    # Target (food) the views recenter on: manual override, else the first valid
    # recorded target cloud, else the initial CBF fork-tip/TCP point.
    target_source = "manual --target"
    if args.target is not None:
        target = np.asarray(args.target, dtype=float)
    else:
        target_topics = args.target_topic or DEFAULT_TARGET_TOPICS
        target = None
        for bag_path in (args.cbf_bag, args.baseline_bag):
            target, topic, target_time = read_target_centroid(bag_path, target_topics)
            if target is not None:
                target_source = (
                    f"first cloud on {topic} in {os.path.basename(bag_path)} "
                    f"at t={target_time:.3f}s")
                break
        if target is None:
            target = tips["cbf"][0][0]
            target_source = (
                "fallback: initial CBF fork-tip/TCP point; no target cloud was "
                "recorded, so pass --target X Y Z or record /perception/target_cube")
    print(f"target (view center): ({target[0]:.3f}, {target[1]:.3f}, "
          f"{target[2]:.3f}) m")
    print(f"target source: {target_source}")
    # Per-direction half-extents [m]: vertical = up/down, horizontal = left/right.
    m_up, m_down = args.view_up / 100.0, args.view_down / 100.0
    m_left, m_right = args.view_left / 100.0, args.view_right / 100.0

    def center_view(ax, ix, iy):
        ax.set_xlim(target[ix] - m_left, target[ix] + m_right)
        ax.set_ylim(target[iy] - m_down, target[iy] + m_up)
        # 'box' keeps the requested limits AND an equal data scale (it resizes
        # the axes box), so the deviation is not distorted by the asymmetric box.
        ax.set_aspect("equal", adjustable="box")

    C_BASE, C_CBF, C_OBS = ts.ORANGE, ts.BLUE, ts.LIGHTGREY

    # ------- Figure A: avec / sans comparison --------------------------------
    fig = plt.figure(figsize=(ts.TEXTWIDTH, 4.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.8, 1.0])
    views = [("x", "y", 0, 1, "vue de dessus"),
             ("x", "z", 0, 2, "vue de côté"),
             ("y", "z", 1, 2, "vue de face")]
    for col, (nx, ny, ix, iy, title) in enumerate(views):
        ax = fig.add_subplot(gs[0, col])
        if obs is not None:
            ax.scatter(obs[:, ix], obs[:, iy], s=1.0, c=C_OBS, alpha=0.35,
                       label="obstacles (carte persistante)" if col == 0 else None)
        tb, _ = tips["base"]
        tc, _ = tips["cbf"]
        ax.plot(tb[:, ix], tb[:, iy], color=C_BASE, lw=1.5, label="sans CBF")
        ax.plot(tc[:, ix], tc[:, iy], color=C_CBF, lw=1.5, label="avec CBF")
        ax.plot(tc[0, ix], tc[0, iy], "o", color="k", ms=5, label="départ")
        ax.plot(target[ix], target[iy], "*", color=ts.GREEN, ms=13, mec="k",
                mew=0.6, label="cible", zorder=6)
        if runs["base"]["contact_time"] is not None:
            ax.plot(tb[-1, ix], tb[-1, iy], "x", color=C_BASE, ms=11, mew=2.5,
                    label="contact estimé")
        ax.set_xlabel(f"{nx} [m]")
        ax.set_ylabel(f"{ny} [m]")
        ax.set_title(title)
        center_view(ax, ix, iy)
        if col == 0:
            ax_views = ax

    ax = fig.add_subplot(gs[1, :])
    for k, color in (("base", C_BASE), ("cbf", C_CBF)):
        r = runs[k]
        m = r["valid"]
        ax.plot(r["t_h"][m], 1000 * clearance(r, args.d_safe)[m],
                color=color, lw=1.0)
    ax.axhline(1000 * args.d_safe, color="k", lw=0.8, ls=":",
               label=r"$d_{\mathrm{safe}}$")
    ax.axhline(0.0, color="k", lw=0.8, ls="--", label="contact")
    if runs["base"]["contact_time"] is not None:
        ax.axvline(runs["base"]["contact_time"], color=C_BASE, lw=0.8, ls="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("dégagement estimé [mm]")
    # single figure-level legend: paths/markers from the views + reference lines
    h1, l1 = ax_views.get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower left", ncol=4,
               bbox_to_anchor=(0.02, 1.0), bbox_transform=fig.transFigure,
               columnspacing=1.2, handlelength=1.5, handletextpad=0.5)
    fig.tight_layout()
    ts.save(fig, args.outdir, "comparison_avec_sans")

    # ------- Figure B: deviation, path colored by clearance ------------------
    fig = plt.figure(figsize=(ts.TEXTWIDTH, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    tc, t_tc = tips["cbf"]
    r = runs["cbf"]
    cl = np.interp(t_tc, r["t_h"][r["valid"]],
                   1000 * clearance(r, args.d_safe)[r["valid"]])
    pts = tc[:, [0, 2]].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="RdYlGn",
                        norm=plt.Normalize(0.0, 60.0), linewidths=2.2)
    lc.set_array(np.clip(cl[:-1], 0, 60))
    ax.add_collection(lc)
    tb, _ = tips["base"]
    ax.plot(tb[:, 0], tb[:, 2], color=ts.ORANGE, lw=1.1, ls="--",
            label="exécution sans filtre (référence)")
    if obs is not None:
        ax.scatter(obs[:, 0], obs[:, 2], s=1.0, c=C_OBS, alpha=0.3)
    ax.plot(target[0], target[2], "*", color=ts.GREEN, ms=13, mec="k",
            mew=0.6, label="cible", zorder=6)
    center_view(ax, 0, 2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    cb = fig.colorbar(lc, ax=ax, pad=0.01)
    cb.set_label("dégagement estimé [mm]")
    ts.legend_top(ax)

    ax = fig.add_subplot(gs[1, 0])
    rb = arclength_resample(tb)
    rc = arclength_resample(tc)
    s = np.linspace(0, 1, len(rb))
    ax.plot(s, 1000 * np.linalg.norm(rc - rb, axis=1), color=ts.PURPLE, lw=1.4)
    ax.set_xlabel("progression normalisée le long du chemin [-]")
    ax.set_ylabel("déviation [mm]")
    fig.tight_layout()
    ts.save(fig, args.outdir, "deviation_h")

    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
