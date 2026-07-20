#!/usr/bin/env python3
"""Decode and plot /cbf_safety/diagnostics from a rosbag.

Produces the thesis result figures from one experiment bag:
  1. h_soft(t) with zero line and CBF-active / escape shading
  2. Joint-velocity decomposition |dq_ff|, |dq_fb|, |dq_base|, |dq_cbf_delta|,
     |dq_escape|, |dq_safe|
  3. Pre- vs post-filter safe velocity (low-pass effect); per-joint figures
     include the MEASURED velocity from /joint_states when recorded
  4. Post-repair constraint residual (negative = CBF inequality violated
     after filtering/clamping)
  5. Pipeline timing boxplot (+ the 1 kHz reflex layer's per-tick compute
     time from /pipeline/timing/cbf_reflex_compute when recorded)
  5b. Per-stage timing over time: one panel per /pipeline/timing/* topic,
     raw samples + rolling median, to see WHEN each stage is slow
  6. Mode-flag timeline (cap_active, recovery, escape, repair, EMA)
  6b. Reflex Lipschitz validation: the online empirical estimate against the
     configured ~grad_lipschitz L (certificate assumption), + the brake alpha

Usage:
    python3 plot_cbf_diagnostics.py run.bag [-o outdir] [--csv]

Record with:
    rosbag record /cbf_safety/diagnostics /cbf_safety/diagnostics_layout \
        /joint_states /cbf_reflex/status -e '/pipeline/timing/.*'

/cbf_reflex/status is REQUIRED for the Lipschitz-validation figure (thesis:
it is what justifies the value of L in the parameter table).
"""

import argparse
import json
import os
import sys

import numpy as np

import rosbag

DIAG_TOPIC = "/cbf_safety/diagnostics"
LAYOUT_TOPIC = "/cbf_safety/diagnostics_layout"
JOINTS_TOPIC = "/joint_states"
OBSTACLES_TOPIC = "/perception/persistent_obstacles"
REFLEX_STATUS_TOPIC = "/cbf_reflex/status"
TIMING_NS = "/pipeline/timing/"
PANDA_JOINTS = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7']
PANDA_FINGER_JOINTS = ['panda_finger_joint1', 'panda_finger_joint2']
PANDA_KINEMATIC_JOINTS = PANDA_JOINTS + PANDA_FINGER_JOINTS


def load_bag(path):
    layout = None
    rows = []
    stamps = []
    with rosbag.Bag(path) as bag:
        for _, msg, _ in bag.read_messages(topics=[LAYOUT_TOPIC]):
            layout = json.loads(msg.data)
            break
        for _, msg, t in bag.read_messages(topics=[DIAG_TOPIC]):
            rows.append(list(msg.data))
            stamps.append(t.to_sec())
    if layout is None:
        sys.exit(f"No {LAYOUT_TOPIC} message in {path}; "
                 "was the layout topic recorded?")
    if not rows:
        sys.exit(f"No {DIAG_TOPIC} messages in {path}.")
    data = np.asarray(rows, dtype=np.float64)
    t0 = stamps[0]
    t = np.asarray(stamps) - t0

    vec_fields = layout["vec_fields"]
    vec_dim = int(layout["vec_dim"])
    scalar_fields = layout["scalar_fields"]
    n_vec = len(vec_fields) * vec_dim
    expected = n_vec + len(scalar_fields)
    if data.shape[1] != expected:
        sys.exit(f"Message length {data.shape[1]} does not match layout "
                 f"({expected}); node and bag out of sync?")

    series = {}
    for i, name in enumerate(vec_fields):
        series[name] = data[:, i * vec_dim:(i + 1) * vec_dim]
    for i, name in enumerate(scalar_fields):
        series[name] = data[:, n_vec + i]
    return t, series, t0


def _interpolate_joint_samples(target_t, source_t, source_values):
    """Interpolate asynchronous joint samples, keeping the last duplicate.

    ROS may publish the arm and gripper in separate JointState messages.  This
    aligns the measured fingers to each arm timestamp without ever inventing a
    closed-gripper zero. Endpoint values are held, matching ``numpy.interp``.
    """
    target_t = np.asarray(target_t, dtype=np.float64)
    source_t = np.asarray(source_t, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    if source_values.ndim == 1:
        source_values = source_values[:, None]
    if len(source_t) != len(source_values) or len(source_t) == 0:
        raise ValueError("source joint samples must be non-empty and aligned")
    order = np.argsort(source_t, kind="stable")
    source_t = source_t[order]
    source_values = source_values[order]
    # Keep the last message at a duplicate timestamp.
    keep = np.r_[source_t[1:] != source_t[:-1], True]
    source_t = source_t[keep]
    source_values = source_values[keep]
    return np.stack([
        np.interp(target_t, source_t, source_values[:, j])
        for j in range(source_values.shape[1])
    ], axis=1)


def load_motion(path, t0):
    """Read joint trajectory and obstacle clouds, on the diagnostics timebase.

    Returns (t_q, q, obs_t, obs_clouds). When both finger joints were recorded,
    q has columns [arm joints 1--7, finger joints 1--2] and therefore shape
    [T,9]. Arm-only legacy bags return [T,7]; the pick-place clearance overlay
    rejects those bags rather than silently assuming closed fingers.
    """
    import sensor_msgs.point_cloud2 as pc2
    t_q, q_arm, t_finger, q_finger, obs_t, obs = [], [], [], [], [], []
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages(
                topics=[JOINTS_TOPIC, OBSTACLES_TOPIC]):
            ts = t.to_sec() - t0
            if topic == JOINTS_TOPIC:
                pos = dict(zip(msg.name, msg.position))
                if all(j in pos for j in PANDA_FINGER_JOINTS):
                    t_finger.append(ts)
                    q_finger.append([pos[j] for j in PANDA_FINGER_JOINTS])
                if all(j in pos for j in PANDA_JOINTS):
                    t_q.append(ts)
                    q_arm.append([pos[j] for j in PANDA_JOINTS])
            else:
                pts = np.array(list(pc2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True)),
                    dtype=np.float32)
                obs_t.append(ts)
                obs.append(pts.reshape(-1, 3))
    t_q = np.asarray(t_q, dtype=np.float64)
    q_arm = np.asarray(q_arm, dtype=np.float64)
    if len(t_q) and len(t_finger):
        fingers_at_arm = _interpolate_joint_samples(
            t_q, t_finger, q_finger)
        q = np.concatenate((q_arm, fingers_at_arm), axis=1)
    else:
        q = q_arm
    return t_q, q, np.asarray(obs_t), obs


def load_extras(path, t0, velocity_source="fd"):
    """One extra bag pass for the executed-motion / reflex-layer series.

    Returns (t_v, dq_meas, reflex_compute_ms, reflex_period_ms):
      t_v, dq_meas   — measured joint velocities [rad/s] from /joint_states.
                       (None, None) if the topic is missing.
      reflex_*_ms    — per-tick compute time / achieved loop period of the
                       1 kHz reflex brake (cbf_reflex_node.py), empty arrays
                       when the reflex layer was off or not recorded.

    velocity_source: "fd" (default) differentiates the recorded POSITIONS
    (centered difference); "field" trusts the JointState velocity field. The
    default is fd because the Gazebo/ros_control velocity field is not
    reliable here: on run1000, stationary joints (position drift 0.0003 rad
    over 11 s) reported a sustained ~0.6 rad/s — the integral of the reported
    velocity is wildly inconsistent with the recorded positions. Positions
    are trustworthy, so their derivative is the honest executed speed.
    """
    t_v, vel, pos = [], [], []
    reflex_compute, reflex_period = [], []
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages(
                topics=[JOINTS_TOPIC,
                        "/pipeline/timing/cbf_reflex_compute",
                        "/pipeline/timing/cbf_reflex_period"]):
            if topic == JOINTS_TOPIC:
                p = dict(zip(msg.name, msg.position))
                if not all(j in p for j in PANDA_JOINTS):
                    continue
                v = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
                t_v.append(t.to_sec() - t0)
                pos.append([p[j] for j in PANDA_JOINTS])
                vel.append([v.get(j, np.nan) for j in PANDA_JOINTS])
            elif topic.endswith("cbf_reflex_compute"):
                reflex_compute.append(float(msg.data))
            else:
                reflex_period.append(float(msg.data))
    reflex_compute = np.asarray(reflex_compute)
    reflex_period = np.asarray(reflex_period)
    if len(t_v) < 3:
        return None, None, reflex_compute, reflex_period
    t_v = np.asarray(t_v)
    vel = np.asarray(vel)
    if velocity_source == "fd" or np.all(np.isnan(vel)):
        pos = np.asarray(pos)
        vel = np.empty_like(pos)
        # Centered difference; one-sided at the ends. Guard repeated stamps.
        dt2 = t_v[2:] - t_v[:-2]
        dt2[dt2 <= 0] = np.median(dt2[dt2 > 0]) if (dt2 > 0).any() else 2e-2
        vel[1:-1] = (pos[2:] - pos[:-2]) / dt2[:, None]
        vel[0] = vel[1]
        vel[-1] = vel[-2]
    return t_v, vel, reflex_compute, reflex_period


def load_reflex_status(path, t0):
    """Read /cbf_reflex/status = [alpha, h_pred_min, n_brake, age, L_emp].

    L_emp is the reflex node's ONLINE Lipschitz monitor: per cycle, the max
    over near-barrier rows of ||grad_h(q) - grad_h(q')|| / ||q - q'||. The
    certified extrapolation assumes the configured ~grad_lipschitz L bounds
    it, so this series is what validates (or refutes) that assumption.

    Returns (t, alpha, h_pred, n_brake, age, L_emp); empty arrays when the
    topic was not recorded (reflex off, or bag recorded without it).
    """
    t, rows = [], []
    with rosbag.Bag(path) as bag:
        for _, msg, ts in bag.read_messages(topics=[REFLEX_STATUS_TOPIC]):
            d = list(msg.data)
            if len(d) >= 5:
                t.append(ts.to_sec() - t0)
                rows.append(d[:5])
    if not rows:
        return (np.array([]),) * 6
    a = np.asarray(rows, dtype=np.float64)
    return (np.asarray(t), a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4])


def load_pipeline_timing(path, t0):
    """stage -> (t, ms) for every /pipeline/timing/<stage> topic in the bag.

    The messages are std_msgs/Float64 (no header), so the bag receipt time is
    the only timebase available; it is the same one the diagnostics use, so the
    stages line up with the rest of the figures. Empty dict when the timing
    topics were not recorded.
    """
    series = {}
    with rosbag.Bag(path) as bag:
        topics = [tp for tp in bag.get_type_and_topic_info().topics
                  if tp.startswith(TIMING_NS)]
        if not topics:
            return {}
        for topic, msg, ts in bag.read_messages(topics=topics):
            t, v = series.setdefault(topic[len(TIMING_NS):], ([], []))
            t.append(ts.to_sec() - t0)
            v.append(float(msg.data))
    return {k: (np.asarray(a), np.asarray(b, dtype=np.float64))
            for k, (a, b) in series.items()}


def rolling_median(v, window):
    """Centered rolling median of v; returns None when the window does not fit."""
    w = int(window) | 1  # odd, so the window is centered on the sample
    if w < 3 or len(v) < w:
        return None
    padded = np.pad(v, w // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, w)
    return np.median(win, axis=1)


def bin_average(t_grid, t_src, v_src):
    """Anti-aliased downsample of (t_src, v_src[N,D]) onto t_grid.

    Averages all source samples falling in each grid cell (cell edges at the
    midpoints between grid points). Point-sampling (np.interp) a ~1 kHz
    measured stream onto a decimated diagnostics timebase aliases controller
    chatter into fake square waves; the per-cell mean is the honest picture.
    Cells with no source sample fall back to linear interpolation.
    """
    edges = np.concatenate((
        [t_grid[0] - 0.5 * (t_grid[1] - t_grid[0])],
        0.5 * (t_grid[:-1] + t_grid[1:]),
        [t_grid[-1] + 0.5 * (t_grid[-1] - t_grid[-2])]))
    idx = np.digitize(t_src, edges) - 1
    ok = (idx >= 0) & (idx < len(t_grid))
    sums = np.zeros((len(t_grid), v_src.shape[1]))
    counts = np.zeros(len(t_grid))
    np.add.at(sums, idx[ok], v_src[ok])
    np.add.at(counts, idx[ok], 1.0)
    out = np.empty_like(sums)
    filled = counts > 0
    out[filled] = sums[filled] / counts[filled, None]
    if not filled.all():
        for j in range(v_src.shape[1]):
            out[~filled, j] = np.interp(
                t_grid[~filled], t_src, v_src[:, j])
    return out


def _rate(ts):
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return float("nan")
    return (len(ts) - 1) / (ts[-1] - ts[0])


def print_bag_summary(path):
    """Print statistics for the non-diagnostics topics in the bag: the joint
    trajectory, the persistent obstacle cloud and the per-stage pipeline timing
    (/pipeline/timing/*). Best-effort: missing topics are simply skipped.
    """
    js_t, js_pos, js_vel = [], [], []
    obs_t, obs_n = [], []
    timing = {}
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages():
            if topic == JOINTS_TOPIC:
                p = dict(zip(msg.name, msg.position))
                v = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
                if all(j in p for j in PANDA_JOINTS):
                    js_t.append(t.to_sec())
                    js_pos.append([p[j] for j in PANDA_JOINTS])
                    js_vel.append([v.get(j, np.nan) for j in PANDA_JOINTS])
            elif topic == OBSTACLES_TOPIC:
                obs_t.append(t.to_sec())
                obs_n.append(int(msg.width) * int(msg.height))
            elif topic.startswith("/pipeline/timing/"):
                timing.setdefault(topic, []).append(float(msg.data))

    print("\n" + "=" * 72)
    print(f"OTHER RECORDED TOPICS — {os.path.basename(path)}")
    print("=" * 72)

    if js_t:
        js_pos = np.asarray(js_pos)
        js_vel = np.asarray(js_vel)
        span = js_t[-1] - js_t[0]
        print(f"\n/joint_states  (sensor_msgs/JointState)")
        print(f"  {len(js_t)} samples | {_rate(js_t):.1f} Hz | span {span:.1f} s")
        print(f"  {'joint':<7}{'pos min':>9}{'pos max':>9}{'range':>8}{'|vel|max':>10}")
        for j, name in enumerate(PANDA_JOINTS):
            p = js_pos[:, j]
            v = np.abs(js_vel[:, j])
            print(f"  {name[-1]:<7}{p.min():>9.3f}{p.max():>9.3f}"
                  f"{p.max() - p.min():>8.3f}{np.nanmax(v):>10.3f}")
    else:
        print(f"\n/joint_states: not in bag")

    if obs_t:
        obs_n = np.asarray(obs_n)
        print(f"\n/perception/persistent_obstacles  (sensor_msgs/PointCloud2)")
        print(f"  {len(obs_t)} clouds | {_rate(obs_t):.1f} Hz")
        print(f"  points/cloud: min {obs_n.min()}  median "
              f"{int(np.median(obs_n))}  max {obs_n.max()}  mean {obs_n.mean():.0f}")
    else:
        print(f"\n/perception/persistent_obstacles: not in bag")

    if timing:
        print(f"\n/pipeline/timing/*  (std_msgs/Float64, milliseconds)")
        print(f"  {'stage':<26}{'n':>6}{'mean':>8}{'med':>8}{'p95':>8}{'max':>8}")
        for topic in sorted(timing):
            v = np.asarray(timing[topic], dtype=float)
            name = topic.replace("/pipeline/timing/", "")
            zero = " (inactive)" if np.allclose(v, 0.0) else ""
            print(f"  {name:<26}{len(v):>6}{v.mean():>8.2f}{v.std():>8.2f}{np.median(v):>8.2f}"
                  f"{np.percentile(v, 95):>8.2f}{v.max():>8.2f}{zero}")
    else:
        print(f"\n/pipeline/timing/*: not in bag")
    print()


def real_clearance_series(path, t0, t, dummy, samples_per_link,
                          robot_variant="pickplace"):
    """Offline true min distance robot->obstacles, aligned to diagnostics time t.

    Returns ``(distance, details)`` aligned to t, or ``None`` if inputs are
    unavailable. ``details['link']`` identifies the closest protected mesh.
    """
    try:
        import real_distance as rdist
        t_q, q, obs_t, obs = load_motion(path, t0)
        if len(t_q) < 2 or len(obs_t) == 0:
            print("[real_distance] joint_states / persistent_obstacles missing "
                  "from bag; skipping real-clearance overlay.")
            return None
        if q.ndim != 2 or q.shape[1] != len(PANDA_KINEMATIC_JOINTS):
            print("[real_distance] both measured Panda finger joints are "
                  "required for the physical pick-place overlay; this bag "
                  f"contains q with shape {q.shape}. Skipping instead of "
                  "assuming closed fingers.")
            return None
        # Joint angles interpolated onto the diagnostics timestamps.
        q_at_t = np.stack([
            np.interp(t, t_q, q[:, j]) for j in range(q.shape[1])
        ], axis=1)
        # Nearest obstacle message per diagnostics step.
        obs_index = np.abs(obs_t[None, :] - t[:, None]).argmin(axis=1)
        rc = rdist.RealClearance(
            samples_per_link=samples_per_link,
            robot_variant=robot_variant)
        print(f"[real_distance] protected meshes: {rc.found_links}")
        d, details = rc.clearance(
            q_at_t, obs, obs_index, return_details=True)
        d[dummy] = np.nan
        details["link"][dummy] = None
        valid = np.isfinite(d)
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            min_index = valid_indices[np.argmin(d[valid])]
            print("[real_distance] global minimum: "
                  f"{d[min_index]:.4f} m on {details['link'][min_index]} "
                  f"at t={t[min_index]:.2f} s")
        return d, details
    except Exception as e:
        print(f"[real_distance] skipped real-clearance overlay: {e!r}")
        return None


def norm(v):
    return np.linalg.norm(v, axis=1)


def shade_active(ax, t, mask, color, label):
    """Shade contiguous True regions of mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    starts = ([0] if mask[0] else []) + [i + 1 for i in edges if not mask[i]]
    ends = [i + 1 for i in edges if mask[i]] + ([len(mask) - 1] if mask[-1] else [])
    first = True
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=color, alpha=0.15,
                   label=label if first else None)
        first = False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", help="rosbag file with /cbf_safety/diagnostics")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output directory (default: <bagname>_diag)")
    ap.add_argument("--csv", action="store_true",
                    help="also dump a flat CSV of all decoded series")
    ap.add_argument("--tau-safe", type=float, default=0.2,
                    help="output low-pass constant used to replay the filter "
                         "and reconstruct the repair correction dq_rep")
    ap.add_argument("--tau-nom", type=float, default=0.05,
                    help="input low-pass constant (cbf_filter_tau); used only "
                         "to label the input-filter panel, which is exact "
                         "because both its input and output are logged")
    ap.add_argument("--max-joint-velocity", type=float, default=0.7,
                    help="velocity clamp applied before replaying the output "
                         "filter, to match the node which filters the clamped "
                         "solver output (cbf node L1601)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="display smoothing window in seconds for the velocity "
                         "decomposition (raw shown faintly behind; 0 = off)")
    ap.add_argument("--d-safe", type=float, default=0.015,
                    help="safety margin used at runtime; the offline real "
                         "clearance is shifted by it to compare with h on the "
                         "same barrier-value axis (h_real = d_real - d_safe)")
    ap.add_argument("--grasp-d-safe", type=float, default=0.002,
                    help="grasped-target group safety margin (grasp_d_safe); used "
                         "to annotate the h_groups figure. h_food already has it "
                         "folded in (h_food=0 at clearance=grasp_d_safe).")
    ap.add_argument("--real-samples", type=int, default=1500,
                    help="surface samples per protected link for the offline "
                         "true-clearance overlay on h_evolution")
    ap.add_argument("--real-robot", choices=("pickplace", "feeding"),
                    default="pickplace",
                    help="robot geometry used by offline true clearance "
                         "(default: pickplace with live Panda fingers)")
    ap.add_argument("--no-real-distance", action="store_true",
                    help="skip the offline true-clearance overlay on h_evolution")
    ap.add_argument("--measured-velocity", choices=("fd", "field"),
                    default="fd",
                    help="source for the executed joint speed in the per-joint "
                         "figure: fd = differentiate recorded positions "
                         "(default; the Gazebo JointState velocity field "
                         "reports phantom speed on stationary joints), "
                         "field = trust the JointState velocity field")
    ap.add_argument("--grad-lipschitz", type=float, default=5.0,
                    help="reflex ~grad_lipschitz L actually configured; drawn "
                         "as the assumed bound against the online empirical "
                         "estimate (/cbf_reflex/status[4])")
    ap.add_argument("--no-timing", action="store_true",
                    help="skip the per-stage /pipeline/timing/* time-series "
                         "figure (it needs one extra bag pass)")
    ap.add_argument("--no-summary", action="store_true",
                    help="skip the printed summary of the other recorded topics "
                         "(joint_states, persistent_obstacles, pipeline timing)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt

    t, s, t0 = load_bag(args.bag)
    if not args.no_summary:
        print_bag_summary(args.bag)
    # Measured joint velocities + reflex-layer timing (one shared bag pass).
    # The measured velocity is bin-averaged onto the diagnostics timebase:
    # the /joint_states stream is one to two orders of magnitude denser than
    # the (possibly decimated) diagnostics, and point-sampling it aliases.
    t_v, dq_meas, reflex_ms, reflex_period_ms = load_extras(
        args.bag, t0, velocity_source=args.measured_velocity)
    dq_meas_at_t = None
    if t_v is not None and len(t_v) > 2:
        dq_meas_at_t = bin_average(t, t_v, dq_meas)
    else:
        print("[per_joint_velocity] /joint_states not in bag — measured "
              "velocity overlay skipped.")
    general_dir = "run_diag"
    outdir = args.outdir or os.path.splitext(args.bag)[0] + "_diag"
    outdir = os.path.join(general_dir, outdir)
    os.makedirs(outdir, exist_ok=True)

    # Before the first preprocessing cycle the node feeds the barrier dummy
    # obstacles at 100 m, which makes h (and kappa*h in the constraint) huge.
    # Mask every barrier-derived series while no real obstacle is selected so
    # the plots show the true operating range.
    dummy = np.nan_to_num(s["selected_count"], nan=0.0) <= 0.0
    # Fallback: the dummy placeholder sits at 100 m, so any h above 5 m means
    # the barrier was evaluated against it regardless of the bookkeeping.
    dummy |= np.nan_to_num(s["h"], nan=0.0) > 5.0
    if dummy.any():
        for k in ("h", "h_corr", "lam", "constr_pre", "constr_final",
                  "grad_h_norm", "cap_active", "recovery_used",
                  "grad_degenerate", "repair_applied", "min_obs_dist",
                  "cmd_hdot", "real_hdot", "real_constr", "h_rate",
                  "alignment"):
            if k not in s:
                continue
            s[k] = s[k].copy()
            s[k][dummy] = np.nan
        # Finite-difference fields also spike on the first real cycle after a
        # dummy stretch (the difference straddles the 100 m placeholder).
        transition = np.zeros_like(dummy)
        transition[1:] = dummy[:-1] & ~dummy[1:]
        if "h_rate" in s and transition.any():
            s["h_rate"][transition] = np.nan
        print(f"Masked {int(dummy.sum())} cycles with dummy obstacle "
              f"selection ({100.0 * dummy.mean():.1f}% of the run)")

    cbf_active = norm(s["dq_cbf_delta"]) > 1e-6
    # One intervention criterion for ALL figures: the filter touched the
    # command this cycle, by projection or by post-filter repair.
    repaired = (np.nan_to_num(s["repair_applied"]) > 0.5
                if "repair_applied" in s else np.zeros_like(cbf_active))
    intervened = cbf_active | repaired
    repair_only = repaired & ~cbf_active
    escape = s["escape_active"] > 0.5

    # The pure output-filter command is consumed in place by the node (it is
    # overwritten by the repaired command at cbf node L1612-1648), so it is the
    # ONE signal in the chain that is not logged. Reconstruct it by replaying
    # the output low-pass F_tau_safe on the logged solver output. The node
    # filters the CLAMPED solver output (cbf node L1601), so clamp dq_pre_filter
    # to +-max_joint_velocity first to match it faithfully.
    #   dq_post_filter = F_tau_safe[ sat(dq_pre_filter) ]   (pure filter output)
    #   dq_rep         = dq_final - dq_post_filter          (repair + saturation)
    # Validated on run3: residual where repair_applied == 0 has median ~1e-4.
    vmax = args.max_joint_velocity
    dq_post_filter = None
    dq_rep_norm = np.zeros_like(t)
    if "dq_post_filter" in s:
        # Logged directly by the node (clamped solver output through F_tau_safe,
        # before the final-constraint repair). Exact — no replay, so it is
        # independent of --tau-safe and overlaps dq_pre_filter when tau == 0.
        dq_post_filter = s["dq_post_filter"]
        if "repair_applied" in s:
            dq_rep_norm = np.linalg.norm(s["dq_final"] - dq_post_filter, axis=1)
            dq_rep_norm[~repaired] = 0.0
    elif "dt" in s:
        # Backward compatibility for bags recorded before dq_post_filter was
        # logged: reconstruct by replaying the output low-pass on dq_pre_filter.
        # This replay uses --tau-safe and MUST match the node's
        # cbf_velocity_filter_tau or the curve will not overlap dq_pre.
        dts = np.nan_to_num(s["dt"], nan=0.0).copy()
        good = dts > 0
        dts[~good] = np.median(dts[good]) if good.any() else 1e-2
        u = np.clip(s["dq_pre_filter"], -vmax, vmax)
        state = u[0].copy()
        dq_post_filter = np.empty_like(u)
        for k in range(len(t)):
            g = dts[k] / (dts[k] + args.tau_safe)
            state = (1.0 - g) * state + g * u[k]
            dq_post_filter[k] = state
        if "repair_applied" in s:
            dq_rep_norm = np.linalg.norm(s["dq_final"] - dq_post_filter, axis=1)
            dq_rep_norm[~repaired] = 0.0

    # Offline ground-truth barrier value: true min distance of the protected
    # meshes to the FULL obstacle cloud, shifted by d_safe so it lives on the
    # same axis as h. The gap to the learned barrier combines basis-SDF error,
    # surface sampling and runtime obstacle pruning (plus soft-min bias in soft
    # mode).
    h_real = None
    real_details = None
    if not args.no_real_distance:
        real_result = real_clearance_series(
            args.bag, t0, t, dummy, args.real_samples, args.real_robot)
        if real_result is not None:
            d_real, real_details = real_result
            h_real = d_real - args.d_safe

    # 1. Barrier value -----------------------------------------------------
    hard_value_mode = (
        "h_hard" in s
        and np.isfinite(s["h_hard"]).any()
        and np.nanmax(np.abs(s["h"] - s["h_hard"])) < 1e-6
    )
    learned_h_label = r"$h_{\min}$" if hard_value_mode else r"$h_{\mathrm{soft}}$"
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
    ax.plot(t, s["h"], color=ts.BLUE, label=learned_h_label, lw=1.0)
    if h_real is not None:
        ax.plot(t, h_real, color=ts.ORANGE, lw=1.0,
                label=r"$h_{\mathrm{réel}}$ (maillages, nuage complet)")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    shade_active(ax, t, cbf_active, ts.GREEN, "projection CBF")
    shade_active(ax, t, repair_only, ts.YELLOW, "réparation seule")
    shade_active(ax, t, escape, ts.PURPLE, "échappement")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("h [m]")
    ts.legend_top(ax)
    fig.tight_layout()
    ts.save(fig, outdir, "h_evolution")

    # 1b. Per-group barriers (robot/fork vs grasped target) ----------------
    # Two CBF groups, each with its own d_safe: the "h" field is the robot/fork
    # group (d_safe); "h_food" is the grasped-target group (grasp_d_safe, already
    # folded in). Each crosses 0 at its own margin, so they share the h axis.
    # h_food is NaN before the grasp, so the target trace appears only once the
    # separate constraint is active.
    if "h_food" in s and np.isfinite(s["h_food"]).any():
        fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
        ax.plot(t, s["h"], color=ts.BLUE, lw=1.0,
                label=r"$h_{\mathrm{robot+fork}}$ ($d_{\mathrm{safe}}=%.0f$ mm)"
                % (args.d_safe * 1e3))
        ax.plot(t, s["h_food"], color=ts.ORANGE, lw=1.0,
                label=r"$h_{\mathrm{cible}}$ ($d_{\mathrm{safe}}=%.0f$ mm)"
                % (args.grasp_d_safe * 1e3))
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        shade_active(ax, t, cbf_active, ts.GREEN, "projection CBF")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("h [m]")
        ts.legend_top(ax)
        fig.tight_layout()
        ts.save(fig, outdir, "h_groups")
    else:
        print("[h_groups] no finite h_food in bag — skipping per-group barrier "
              "figure (record a run with the grasp separate constraint active, "
              "using the updated CBF node).")

    # 2. Velocity decomposition -------------------------------------------
    def disp_smooth(x):
        if args.smooth <= 0:
            return x
        w = max(1, int(round(args.smooth / max(np.median(np.diff(t)), 1e-4))))
        return np.convolve(x, np.ones(w) / w, mode="same")

    # Two panels, one chain: (a) the nominal command is feedforward plus
    # tracking feedback; (b) the CBF correction turns it into the final
    # command. dq_base repeats in (b) so the two stages share a reference.
    fig, axes = plt.subplots(2, 1, figsize=(ts.TEXTWIDTH, 4.4), sharex=True)

    ax = axes[0]
    series = [("dq_ff", r"$|\dot{q}_{\mathrm{ff}}|$ (avance)", ts.SKY),
              ("dq_fb", r"$|\dot{q}_{\mathrm{fb}}|$ (rétroaction)", ts.GREEN),
              ("dq_base", r"$|\dot{q}_{\mathrm{base}}|$ (nominale)", ts.ORANGE)]
    for name, label, color in series:
        v = norm(s[name])
        if np.nanmax(np.abs(v)) < 1e-9:
            continue
        ax.plot(t, disp_smooth(v), color=color, label=label, lw=1.2)
    ff_level = float(np.nanmedian(norm(s["dq_ff"])))
    # ax.annotate("retiming : consigne constante",
    #             xy=(0.62 * t[-1], ff_level), xytext=(0.62 * t[-1], ff_level * 1.18),
    #             color=ts.SKY, fontsize=8, ha="center",
    #             arrowprops=dict(arrowstyle="-", color=ts.SKY, lw=0.7))
    ax.set_ylabel("[rad/s]")
    ts.legend_top(ax, ncol=3)
    ts.panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(t, disp_smooth(norm(s["dq_base"])), color=ts.ORANGE, lw=1.0,
            alpha=0.6, label=r"$|\dot{q}_{\mathrm{base}}|$ (nominale)")
    ax.plot(t, disp_smooth(norm(s["dq_final"])), color=ts.BLUE, lw=1.2,
            label=r"$|\dot{q}_{\mathrm{final}}|$ (commandée)")
    ax.plot(t, disp_smooth(norm(s["dq_cbf_delta"])), color=ts.PURPLE, lw=1.2,
            label=r"$|\Delta\dot{q}_{\mathrm{CBF}}|$ (correction)")
    if np.nanmax(norm(s["dq_escape"])) > 1e-9:
        ax.plot(t, disp_smooth(norm(s["dq_escape"])), color=ts.YELLOW, lw=1.2,
                label=r"$|\dot{q}_{\mathrm{esc}}|$ (échappement)")
    if np.nanmax(dq_rep_norm) > 1e-9:
        ax.plot(t, disp_smooth(dq_rep_norm), color=ts.YELLOW, lw=1.2,
                label=r"$|\Delta\dot{q}_{\mathrm{rep}}|$ (réparation)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("[rad/s]")
    ts.legend_top(ax, ncol=4)
    ts.panel_label(ax, "(b)")

    fig.tight_layout()
    ts.save(fig, outdir, "velocity_decomposition")

    # 2b. CBF action: nominal vs safe velocity --------------------------------
    # One panel, one message: the filter brakes (orange) or pushes (green) the
    # nominal command. Shade ONLY where the filter actually intervened
    # (projection or repair); elsewhere the small |final|<|base| gap is just
    # the output low-pass and must not be painted as a safety action.
    nb = norm(s["dq_base"])
    nsafe = norm(s["dq_final"])
    nb_s, nsafe_s = disp_smooth(nb), disp_smooth(nsafe)
    shade = intervened.copy()
    if args.smooth > 0:  # dilate to the display-smoothing width
        w = max(1, int(round(args.smooth / max(np.median(np.diff(t)), 1e-4))))
        shade = np.convolve(shade.astype(float),
                            np.ones(w), mode="same") > 0
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.8))
    ax.plot(t, nb_s, color=ts.ORANGE, lw=1.3,
            label=r"$|\dot{q}_{\mathrm{base}}|$ (nominale)")
    ax.plot(t, nsafe_s, color=ts.BLUE, lw=1.3,
            label=r"$|\dot{q}_{\mathrm{final}}|$ (commandée)")
    ax.fill_between(t, nb_s, nsafe_s, where=(nsafe_s < nb_s) & shade,
                    color=ts.ORANGE, alpha=0.25, label="freinage CBF")
    ax.fill_between(t, nb_s, nsafe_s, where=(nsafe_s > nb_s) & shade,
                    color=ts.GREEN, alpha=0.30, label="poussée CBF")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("vitesse articulaire [rad/s]")
    ax.set_ylim(bottom=0.0)
    ts.legend_top(ax)
    fig.tight_layout()
    ts.save(fig, outdir, "cbf_action")

    # 3. Filter effects -----------------------------------------------------
    # The command passes through TWO low-pass filters at different stages.
    # (a) Input filter F_tau_nom, on the nominal command BEFORE the QP: both its
    #     input (dq_ff + dq_fb) and output (dq_base) are logged, so the panel is
    #     exact. It removes the velocity steps of the piecewise-linear waypoint
    #     interpolation so the solver tracks a smooth reference.
    # (b) Output filter F_tau_safe, on the safe command AFTER the QP: its input
    #     (dq_pre_filter) and its pure output (dq_post_filter) are both logged by
    #     the node, so the panel is exact. (Older bags lack dq_post_filter and
    #     reconstruct it by replaying the filter with --tau-safe.) It smooths the
    #     discontinuous CBF correction that jumps as the active obstacle set
    #     switches. dq_final (filter + repair + saturation) is shown faintly so
    #     the filter's effect is not conflated with the post-filter repair.
    dq_nom_raw = s["dq_ff"] + s["dq_fb"]
    fig, axes = plt.subplots(2, 1, figsize=(ts.TEXTWIDTH, 4.4), sharex=True)

    ax = axes[0]
    ax.plot(t, norm(dq_nom_raw), color=ts.ORANGE, lw=0.9,
            label=r"avant : $|\dot{q}_{\mathrm{ff}}+\dot{q}_{\mathrm{fb}}|$")
    ax.plot(t, norm(s["dq_base"]), color=ts.BLUE, lw=1.1,
            label=r"après : $|\dot{q}_{\mathrm{base}}|$")
    ax.set_ylabel("[rad/s]")
    ax.set_ylim(bottom=0.0)
    ts.legend_top(ax)
    ts.panel_label(
        ax, rf"(a)")

    ax = axes[1]
    ax.plot(t, norm(s["dq_pre_filter"]), color=ts.ORANGE, lw=0.9,
            label=r"avant : $|\dot{q}_{\mathrm{pre}}|$ (sortie solveur)")
    if dq_post_filter is not None:
        ax.plot(t, norm(dq_post_filter), color=ts.BLUE, lw=1.1,
                label=r"après : $|F_{\tau_{\mathrm{safe}}}[\dot{q}_{\mathrm{pre}}]|$")
    ax.plot(t, norm(s["dq_final"]), color=ts.GREEN, lw=0.9, zorder=0,
            label=r"$|\dot{q}_{\mathrm{final}}|$ (+ réparation, saturation)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("[rad/s]")
    ax.set_ylim(bottom=0.0)
    ts.legend_top(ax)
    ts.panel_label(
        ax, rf"(b)")

    fig.tight_layout()
    ts.save(fig, outdir, "filter_effect")

    # 3b. Per-joint velocity ------------------------------------------------
    # Every dq_* field is logged as a full 7-vector, so each joint can be shown
    # on its own axis instead of collapsed to a norm. Signed values matter here:
    # the CBF can brake, stop or REVERSE an individual joint, which the (always
    # positive) norm hides. One panel per joint shows the chain
    #   nominal  ->  after CBF projection  ->  commanded (filter + repair)
    # together with the per-joint correction Delta_q_CBF that produced it and,
    # when /joint_states was recorded, the MEASURED velocity actually executed
    # by the controller — the commanded-vs-real gap is the controller tracking
    # error plus, with the reflex layer on, its alpha-braking.
    # Wider than the text column (landscape figure): 4 panels across the text
    # width are too narrow on the time axis, so use ~1.8x the width.
    fig, axes = plt.subplots(2, 4, figsize=(ts.TEXTWIDTH * 1.8, 4.6), sharex=True)
    axes_flat = axes.ravel()
    for j in range(7):
        ax = axes_flat[j]
        ax.axhline(0.0, color="k", lw=0.5, ls="-", alpha=0.25)
        # Nominal: faint thin reference, sits in the background.
        ax.plot(t, s["dq_base"][:, j], color=ts.ORANGE, lw=0.8, alpha=0.55,
                label=r"$\dot{q}_{\mathrm{base}}$ (nominale)")
        # Correction: a soft filled band to zero rather than a fourth hard line.
        ax.fill_between(t, 0.0, s["dq_cbf_delta"][:, j], color=ts.GREEN,
                        alpha=0.40, lw=0,
                        label=r"$\Delta\dot{q}_{\mathrm{CBF}}$ (correction)")
        # Commanded: the single clean hero line.
        ax.plot(t, s["dq_final"][:, j], color=ts.BLUE, lw=1.3, alpha=0.95,
                solid_capstyle="round", solid_joinstyle="round",
                label=r"$\dot{q}_{\mathrm{final}}$ (commandée)")
        # Executed: what the robot actually did.
        if dq_meas_at_t is not None:
            ax.plot(t, dq_meas_at_t[:, j], color=ts.PURPLE, lw=0.9,
                    alpha=0.85, label=r"$\dot{q}$ mesurée (réelle)")
        ax.set_title(f"Articulation {j + 1}", fontsize=9)
        if j % 4 == 0:
            ax.set_ylabel("[rad/s]")
    axes_flat[7].axis("off")
    # sharex hides x ticks on non-bottom rows; the bottom of the rightmost
    # column is the disabled spare cell, so re-enable ticks/label on the panel
    # above it (joint 4) as well as the rest of the bottom row.
    for ax in (axes_flat[4], axes_flat[5], axes_flat[6], axes_flat[3]):
        ax.set_xlabel("t [s]")
        ax.tick_params(labelbottom=True)
    ts.fig_legend_top(fig, axes_flat[0], ncol=4)
    fig.tight_layout()
    ts.save(fig, outdir, "per_joint_velocity")

    # 3c. Filter effect per joint -------------------------------------------
    # Plot the raw commanded velocity (dq_ff + dq_fb) before the input low-pass
    # filter against the filtered nominal velocity (dq_base) fed to the solver,
    # for each joint individually. This reveals how the filter dampens high-frequency
    # waypoint chatter and step command jumps per-joint.
    fig, axes = plt.subplots(2, 4, figsize=(ts.TEXTWIDTH * 1.8, 4.6), sharex=True)
    axes_flat = axes.ravel()
    for j in range(7):
        ax = axes_flat[j]
        ax.axhline(0.0, color="k", lw=0.5, ls="-", alpha=0.25)
        # Raw: before filter
        ax.plot(t, dq_nom_raw[:, j], color=ts.ORANGE, lw=0.8, alpha=0.55,
                label=r"$\dot{q}_{\mathrm{ff}}+\dot{q}_{\mathrm{fb}}$ (brute)")
        # Filtered: after filter
        ax.plot(t, s["dq_base"][:, j], color=ts.BLUE, lw=1.3, alpha=0.95,
                solid_capstyle="round", solid_joinstyle="round",
                label=r"$\dot{q}_{\mathrm{base}}$ (filtrée)")
        ax.set_title(f"Articulation {j + 1}", fontsize=9)
        if j % 4 == 0:
            ax.set_ylabel("[rad/s]")
    axes_flat[7].axis("off")
    for ax in (axes_flat[4], axes_flat[5], axes_flat[6], axes_flat[3]):
        ax.set_xlabel("t [s]")
        ax.tick_params(labelbottom=True)
    ts.fig_legend_top(fig, axes_flat[0], ncol=2)
    fig.tight_layout()
    ts.save(fig, outdir, "per_joint_filter_effect")

    # 3d. Output filter effect per joint -------------------------------------
    # Plot the raw QP solver output velocity (dq_pre_filter) before the output
    # low-pass filter against the filtered safe velocity (dq_post_filter) for
    # each joint individually. This shows how the output filter smooths high-frequency
    # correction chatter caused by active constraint switches.
    if dq_post_filter is not None:
        fig, axes = plt.subplots(2, 4, figsize=(ts.TEXTWIDTH * 1.8, 4.6), sharex=True)
        axes_flat = axes.ravel()
        for j in range(7):
            ax = axes_flat[j]
            ax.axhline(0.0, color="k", lw=0.5, ls="-", alpha=0.25)
            # Before: solver output (raw / pre-filter)
            ax.plot(t, s["dq_pre_filter"][:, j], color=ts.ORANGE, lw=0.8, alpha=0.55,
                    label=r"$\dot{q}_{\mathrm{pre}}$ (brute)")
            # After: filtered output (post-filter)
            ax.plot(t, dq_post_filter[:, j], color=ts.BLUE, lw=1.3, alpha=0.95,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=r"$\dot{q}_{\mathrm{post}}$ (filtrée)")
            ax.set_title(f"Articulation {j + 1}", fontsize=9)
            if j % 4 == 0:
                ax.set_ylabel("[rad/s]")
        axes_flat[7].axis("off")
        for ax in (axes_flat[4], axes_flat[5], axes_flat[6], axes_flat[3]):
            ax.set_xlabel("t [s]")
            ax.tick_params(labelbottom=True)
        ts.fig_legend_top(fig, axes_flat[0], ncol=2)
        fig.tight_layout()
        ts.save(fig, outdir, "per_joint_output_filter_effect")

    # 4. Constraint residuals ----------------------------------------------
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
    ax.plot(t, s["constr_pre"], color=ts.ORANGE, label="avant projection (brute)", lw=0.9)
    ax.plot(t, s["constr_final"], color=ts.BLUE, label="après réparation", lw=0.9)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\nabla h^\top \dot{q} + \kappa h - m$")
    ts.legend_top(ax)
    fig.tight_layout()
    ts.save(fig, outdir, "constraint_residual")

    # 4b. Sampled-data QP margin m(dq) ---------------------------------------
    # State-dependent ZOH tightening (~cbf_sampled_data_margin in the multicbf
    # node): every constraint row is raised by m(dq) = (L/2)||dq||^2 T, the
    # sampled-data CBF term that guarantees h >= 0 over the WHOLE hold, not
    # just at sample instants. Because it scales with the commanded speed, its
    # magnitude and variability ARE the story: (a) time series on the shared
    # timebase with the same activity shading as the other figures, (b) its
    # distribution over the run. Skipped for bags recorded before the field
    # existed, or reported as inactive when the toggle was off (all-zero).
    if "sampled_margin" in s:
        m_sd = np.asarray(s["sampled_margin"], dtype=float)
        finite_m = np.isfinite(m_sd)
        if finite_m.any() and np.nanmax(m_sd) > 1e-9:
            mm = m_sd * 1e3  # [mm/s] — readable axis for 0..80 mm/s values
            m_mean = float(np.nanmean(mm))
            m_std = float(np.nanstd(mm))
            m_p95 = float(np.nanpercentile(mm[finite_m], 95))
            m_max = float(np.nanmax(mm))
            fig, axes = plt.subplots(
                2, 1, figsize=(ts.TEXTWIDTH, 4.4),
                gridspec_kw={"height_ratios": [2.0, 1.0]})

            ax = axes[0]
            ax.plot(t, mm, color=ts.BLUE, lw=1.0,
                    label=r"$m(\dot{q})$ (marge échantillonnée)")
            ax.axhline(m_mean, color=ts.ORANGE, lw=1.0, ls="--",
                       label=rf"moyenne $= {m_mean:.1f}$ mm/s")
            ax.axhline(m_max, color=ts.GREY, lw=0.9, ls=":",
                       label=rf"max $= {m_max:.1f}$ mm/s")
            shade_active(ax, t, cbf_active, ts.GREEN, "projection CBF")
            shade_active(ax, t, escape, ts.PURPLE, "échappement")
            ax.set_xlabel("t [s]")
            ax.set_ylabel(r"$m(\dot{q})$ [mm/s]")
            ax.set_ylim(bottom=0.0)
            ts.legend_top(ax, ncol=2)
            ts.panel_label(ax, "(a)")

            ax = axes[1]
            ax.hist(mm[finite_m], bins=40, color=ts.BLUE, alpha=0.75)
            ax.axvline(m_mean, color=ts.ORANGE, lw=1.0, ls="--")
            ax.axvline(m_p95, color=ts.GREY, lw=0.9, ls=":")
            ax.axvline(m_max, color=ts.GREY, lw=0.9, ls="-")
            ax.set_xlabel(r"$m(\dot{q})$ [mm/s]")
            ax.set_ylabel("cycles")
            ts.stats_box(ax, "moyenne %.1f | é.-t. %.1f | p95 %.1f | "
                             "max %.1f mm/s"
                         % (m_mean, m_std, m_p95, m_max))
            ts.panel_label(ax, "(b)")

            fig.tight_layout()
            ts.save(fig, outdir, "sampled_margin")
        else:
            print("[sampled_margin] field present but all-zero — "
                  "cbf_sampled_data_margin was off for this run.")
    else:
        print("[sampled_margin] not in bag layout (older node) — skipping "
              "the sampled-data margin figure.")

    # 5. Timing boxplot ------------------------------------------------------
    # When the 1 kHz reflex layer ran, its per-tick compute time is appended
    # as a fourth box (sampled on /pipeline/timing/cbf_reflex_compute). Same
    # linear axis on purpose: the visual point is that the brake costs
    # near-nothing next to the QP cycle ("heavy decision slow, cheap brake
    # fast").
    timing_data = [s[k][~np.isnan(s[k])] for k in
                   ("cbf_ms", "q_cmd_ms", "total_ms")]
    timing_labels = ["CBF", "commande", "total"]
    if len(reflex_ms):
        timing_data.append(reflex_ms)
        timing_labels.append("réflexe\n(1 kHz)")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    bp = ax.boxplot(timing_data,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=ts.ORANGE))
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor=ts.BLUE)
    ax.set_xticklabels(timing_labels)
    ax.set_ylabel("temps de cycle [ms]")
    fig.tight_layout()
    ts.save(fig, outdir, "timing_boxplot")

    # 5b. Per-stage timing over time -----------------------------------------
    # The boxplot above collapses each stage to a distribution; this figure keeps
    # the time axis, so WHEN a stage is slow is visible (warm-up, a burst while
    # the obstacle set switches, a stage that drifts as the cloud grows) and not
    # just how often. One panel per /pipeline/timing/* topic, stages ordered and
    # colored by pipeline group (perception / planning / safety), sharing the
    # time axis with every other figure. Each y axis is independent — the panels
    # answer "how much does THIS stage vary", and the annotated median/p95/max
    # carry the magnitude needed to compare across panels; a shared axis would
    # flatten every cheap stage into a line at zero next to the FM generation.
    if not args.no_timing:
        timing_series = load_pipeline_timing(args.bag, t0)
    else:
        timing_series = {}
    if timing_series:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from decode_pipeline_timing import ordered_stages
        group_color = {"perception": ts.GREEN, "planning": ts.BLUE,
                       "safety": ts.ORANGE, "other": ts.GREY}
        # A stage that published nothing but zeros was compiled out of the run
        # (e.g. fm_correction / casf_warp on a non-CASF pipeline); it has no
        # variation to show, so it gets a line of text instead of a panel.
        off = sorted(k for k, (_, v) in timing_series.items()
                     if np.allclose(v, 0.0))
        active = {k: v for k, v in timing_series.items() if k not in off}
        if off:
            print(f"[timing_per_stage] stages inactive (all-zero), not "
                  f"plotted: {', '.join(off)}")
        pairs = ordered_stages(active)
        ncol = 3
        nrow = int(np.ceil(len(pairs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, sharex=True,
                                 figsize=(ts.TEXTWIDTH * 1.8, 1.6 * nrow + 0.9),
                                 squeeze=False)
        axes_flat = axes.ravel()
        for k, (group, stage) in enumerate(pairs):
            ax = axes_flat[k]
            t_s, v_s = active[stage]
            color = group_color[group]
            # Raw samples faint, rolling median on top: the 1 kHz reflex stage
            # has ~3 orders of magnitude more samples than the 8 Hz planner, so
            # the raw trace alone is an unreadable band for the dense stages.
            ax.plot(t_s, v_s, color=color, lw=0.6, alpha=0.35)
            med = rolling_median(v_s, max(3, len(v_s) // 200))
            if med is not None:
                ax.plot(t_s, med, color=color, lw=1.2)
            p95 = float(np.percentile(v_s, 95))
            ax.axhline(p95, color=ts.GREY, lw=0.8, ls=":")
            ax.set_title(stage, fontsize=9, color=color)
            ax.set_ylim(bottom=0.0)
            ts.stats_box(ax, "méd %.2f | p95 %.2f | max %.2f ms"
                         % (np.median(v_s), p95, v_s.max()))
            if k % ncol == 0:
                ax.set_ylabel("[ms]")
        for k in range(len(pairs), len(axes_flat)):
            axes_flat[k].axis("off")
        # sharex hides the ticks on every non-bottom row, but the last row can be
        # partly disabled, so re-enable them on the lowest live panel per column.
        for c in range(ncol):
            col = [k for k in range(len(pairs)) if k % ncol == c]
            if col:
                axes_flat[col[-1]].set_xlabel("t [s]")
                axes_flat[col[-1]].tick_params(labelbottom=True)
        from matplotlib.lines import Line2D
        present = {g for g, _ in pairs}
        handles = [Line2D([], [], color=group_color[g], lw=1.4, label=g)
                   for g in ("perception", "planning", "safety", "other")
                   if g in present]
        handles.append(Line2D([], [], color=ts.GREY, lw=0.8, ls=":", label="p95"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                   ncol=len(handles), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        ts.save(fig, outdir, "timing_per_stage")
    elif not args.no_timing:
        print("[timing_per_stage] no /pipeline/timing/* topics in bag — record "
              "with: rosbag record -e '/pipeline/timing/.*'")

    # 6. Mode flags ----------------------------------------------------------
    flags = ["cap_active", "recovery_used", "grad_degenerate",
             "escape_active", "repair_applied", "ema_applied"]
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.2))
    for i, f in enumerate(flags):
        v = np.nan_to_num(s[f]) > 0.5
        ax.fill_between(t, i, i + v.astype(float) * 0.8, step="pre", color=ts.BLUE)
    ax.set_yticks([i + 0.4 for i in range(len(flags))])
    ax.set_yticklabels(flags, fontsize=8)
    ax.set_xlabel("t [s]")
    fig.tight_layout()
    ts.save(fig, outdir, "mode_flags")

    # 6b. Reflex layer: Lipschitz assumption + braking activity ---------------
    # Panel (a) validates certificate assumption (ii) of Section 4.x: the
    # configured L must upper-bound the empirical gradient-Lipschitz estimate
    # on the near-barrier rows. Transient spikes are critical-point switches
    # (expected, and outside the smoothness assumption); the bulk must sit
    # below the line. Panel (b) shows what the brake actually did.
    (t_rfx, rfx_alpha, _rfx_hpred, _rfx_nbrake,
     _rfx_age, rfx_L) = load_reflex_status(args.bag, t0)
    if len(t_rfx) > 2:
        L_cfg = args.grad_lipschitz
        finite = np.isfinite(rfx_L) & (rfx_L > 0)
        fig, axes = plt.subplots(2, 1, figsize=(ts.TEXTWIDTH, 4.4), sharex=True)

        ax = axes[0]
        ax.plot(t_rfx, rfx_L, color=ts.BLUE, lw=0.9,
                label=r"$\hat{L}$ empirique (en ligne)")
        ax.axhline(L_cfg, color=ts.ORANGE, lw=1.2, ls="--",
                   label=rf"$L$ retenu $= {L_cfg:g}$")
        if finite.any():
            p99 = float(np.percentile(rfx_L[finite], 99))
            ax.axhline(p99, color=ts.GREY, lw=0.9, ls=":",
                       label=rf"p99 $= {p99:.2f}$")
            above = 100.0 * float(np.mean(rfx_L[finite] > L_cfg))
            ts.stats_box(ax, f"dépassements : {above:.2f} % des cycles")
        ax.set_ylabel(r"[m/rad$^2$]")
        ax.set_yscale("log")
        ts.legend_top(ax, ncol=3)
        ts.panel_label(ax, "(a)")

        ax = axes[1]
        ax.plot(t_rfx, rfx_alpha, color=ts.BLUE, lw=0.9)
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$\alpha$ du frein")
        ts.panel_label(ax, "(b)")

        fig.tight_layout()
        ts.save(fig, outdir, "reflex_lipschitz")
    else:
        print("[reflex_lipschitz] /cbf_reflex/status not in bag — record it "
              "to validate the Lipschitz constant L (reflex must be on).")

    # 7. Commanded vs executed h-rate (multicbf nodes only) -------------------
    if "real_hdot" in s and not np.all(np.isnan(s["real_hdot"])):
        fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
        ax.plot(t, s["cmd_hdot"], color=ts.BLUE, label=r"$\dot{h}$ commandé", lw=0.9)
        ax.plot(t, s["real_hdot"], color=ts.ORANGE,
                label=r"$\dot{h}$ exécuté (vitesse mesurée)", lw=0.9)
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        shade_active(ax, t, cbf_active, ts.GREEN, "projection CBF")
        shade_active(ax, t, repair_only, ts.YELLOW, "réparation seule")
        shade_active(ax, t, escape, ts.PURPLE, "échappement")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$\dot{h}$ [m/s]")
        ts.legend_top(ax)
        fig.tight_layout()
        ts.save(fig, outdir, "hdot_cmd_vs_real")

    # Summary ----------------------------------------------------------------
    valid = ~dummy

    def pct(mask):
        if not valid.any():
            return float('nan')
        return 100.0 * float(np.mean(np.asarray(mask)[valid]))

    summary = {
        "duration_s": float(t[-1]),
        "cycles": int(len(t)),
        "pct_dummy_selection": 100.0 * float(np.mean(dummy)),
        "min_h": float(np.nanmin(s["h"])),
        "min_h_corr": float(np.nanmin(s["h_corr"])),
        "pct_cbf_active": pct(cbf_active),
        "pct_escape": pct(escape),
        "pct_cap_active": pct(np.nan_to_num(s["cap_active"]) > 0.5),
        "pct_recovery": pct(np.nan_to_num(s["recovery_used"]) > 0.5),
        "pct_repair": pct(np.nan_to_num(s["repair_applied"]) > 0.5),
        "worst_final_constraint": float(np.nanmin(s["constr_final"])),
        "mean_total_ms": float(np.nanmean(s["total_ms"])),
        "p99_total_ms": float(np.nanpercentile(
            s["total_ms"][~np.isnan(s["total_ms"])], 99)),
    }
    if "sampled_margin" in s:
        m_sd = np.asarray(s["sampled_margin"], dtype=float)
        fin_m = np.isfinite(m_sd)
        if fin_m.any() and np.nanmax(m_sd) > 1e-9:
            summary["sampled_margin_mean"] = float(np.nanmean(m_sd))
            summary["sampled_margin_std"] = float(np.nanstd(m_sd))
            summary["sampled_margin_p95"] = float(
                np.nanpercentile(m_sd[fin_m], 95))
            summary["sampled_margin_max"] = float(np.nanmax(m_sd))
    if len(t_rfx) > 2:
        fin = np.isfinite(rfx_L) & (rfx_L > 0)
        if fin.any():
            summary["reflex_L_configured"] = float(args.grad_lipschitz)
            summary["reflex_L_emp_median"] = float(np.median(rfx_L[fin]))
            summary["reflex_L_emp_p95"] = float(np.percentile(rfx_L[fin], 95))
            summary["reflex_L_emp_p99"] = float(np.percentile(rfx_L[fin], 99))
            summary["reflex_L_emp_max"] = float(np.max(rfx_L[fin]))
            summary["reflex_L_exceed_pct"] = 100.0 * float(
                np.mean(rfx_L[fin] > args.grad_lipschitz))
        summary["reflex_alpha_min"] = float(np.min(rfx_alpha))
        summary["reflex_alpha_braking_pct"] = 100.0 * float(
            np.mean(rfx_alpha < 0.999))
    if len(reflex_ms):
        summary["reflex_mean_compute_ms"] = float(np.mean(reflex_ms))
        summary["reflex_p99_compute_ms"] = float(np.percentile(reflex_ms, 99))
    if len(reflex_period_ms):
        # Achieved reflex loop rate: 1 kHz is the target; Python+rospy jitter
        # shows up here (thesis honesty number for the two-rate design).
        summary["reflex_median_period_ms"] = float(np.median(reflex_period_ms))
        summary["reflex_achieved_rate_hz"] = float(
            1000.0 / np.median(reflex_period_ms))
    if dq_meas_at_t is not None:
        # Command-tracking gap: RMS of measured minus commanded velocity.
        summary["rms_meas_vs_cmd_radps"] = float(np.sqrt(np.nanmean(
            (dq_meas_at_t - s["dq_final"]) ** 2)))
    # Necessity of each low-pass: ratio of commanded-acceleration RMS before vs
    # after the filter (a chatter/jerk proxy). >1 means the filter smooths the
    # command; the larger the ratio the more the unfiltered command would have
    # jerked the controller.
    def accel_rms(v):
        dtt = np.diff(t)
        pos = dtt > 0
        dtt[~pos] = np.median(dtt[pos]) if pos.any() else 1e-2
        a = np.linalg.norm(np.diff(v, axis=0), axis=1) / dtt
        return float(np.sqrt(np.nanmean(a ** 2)))

    a_in_before, a_in_after = accel_rms(dq_nom_raw), accel_rms(s["dq_base"])
    summary["accel_rms_in_before"] = a_in_before
    summary["accel_rms_in_after"] = a_in_after
    summary["accel_rms_in_reduction"] = (
        a_in_before / a_in_after if a_in_after > 1e-9 else float('nan'))
    if dq_post_filter is not None:
        a_out_before = accel_rms(np.clip(s["dq_pre_filter"], -vmax, vmax))
        a_out_after = accel_rms(dq_post_filter)
        summary["accel_rms_out_before"] = a_out_before
        summary["accel_rms_out_after"] = a_out_after
        summary["accel_rms_out_reduction"] = (
            a_out_before / a_out_after if a_out_after > 1e-9 else float('nan'))

    # Validation of the SDF barrier against the offline true clearance: the gap
    # justifies using h alone (fast, logged) everywhere else. Computed on valid
    # cycles where both are finite.
    if h_real is not None:
        both = valid & np.isfinite(h_real) & np.isfinite(s["h"])
        if both.any():
            gap = np.abs(s["h"][both] - h_real[both])
            summary["min_h_real"] = float(np.nanmin(h_real[both]))
            summary["median_abs_gap_h_real_soft"] = float(np.median(gap))
            summary["p95_abs_gap_h_real_soft"] = float(np.percentile(gap, 95))
            summary["max_abs_gap_h_real_soft"] = float(np.max(gap))
            valid_indices = np.flatnonzero(both)
            min_index = valid_indices[np.argmin(h_real[both])]
            if real_details is not None:
                summary["min_h_real_link"] = str(
                    real_details["link"][min_index])
    if "real_hdot" in s and not np.all(np.isnan(s["real_hdot"])):
        act = cbf_active & valid
        if act.any():
            cmd = s["cmd_hdot"][act]
            real = s["real_hdot"][act]
            ok = np.abs(cmd) > 1e-4
            summary["hdot_execution_ratio"] = (
                float(np.nanmedian(real[ok] / cmd[ok])) if ok.any() else float('nan'))
            summary["mean_alignment_active"] = float(np.nanmean(s["alignment"][act]))
        summary["mean_active_constraints"] = float(
            np.nanmean(s["active_constraints"][valid]))
    if "contact_stop" in s:
        stopped = np.nan_to_num(s["contact_stop"]) > 0.5
        summary["monitor_only"] = bool(
            np.nanmedian(np.nan_to_num(s.get("monitor_only", np.zeros_like(t)))) > 0.5)
        summary["contact_time_s"] = (
            float(t[np.argmax(stopped)]) if stopped.any() else None)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if args.csv:
        import csv
        with open(os.path.join(outdir, "diagnostics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            scalar_keys = [k for k in s if s[k].ndim == 1]
            vec_keys = [k for k in s if s[k].ndim == 2]
            header = (["t"] + scalar_keys
                      + [f"{k}_{j}" for k in vec_keys for j in range(7)])
            w.writerow(header)
            for i in range(len(t)):
                row = [t[i]] + [s[k][i] for k in scalar_keys]
                for k in vec_keys:
                    row.extend(s[k][i].tolist())
                w.writerow(row)

    print(f"Figures written to {outdir}/")


if __name__ == "__main__":
    main()
