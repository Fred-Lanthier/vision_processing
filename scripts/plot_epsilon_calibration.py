#!/usr/bin/env python3
"""Thesis figure: ISSf epsilon calibration campaign (mémoire §5.4.6).

Reads every trajectory bag of an epsilon-calibration batch produced by
calibrate_issf_epsilon.py (run_* AND failed_attempt_* directories: task
timeouts are valid safety-calibration trajectories) and produces one
two-panel figure:

  (a) ECDF of the per-window worst adverse projection, split by regime
      (execution alpha>0.9 vs braking), with the deployed bound eps_p;
  (b) the 26 per-trajectory maxima per regime with the distribution-free
      order-statistic bounds (max of N exchangeable maxima, coverage
      N/(N+1)).

Usage:
    source /opt/ros/noetic/setup.bash
    python3 plot_epsilon_calibration.py \
        --batch-dir ../epsilon_calibration_pp \
        --outdir ~/Documents/Memoire/memoire/images/Results
"""

import argparse
import glob
import os

import numpy as np

import thesis_style as ts

EPS_DEPLOYED = 0.025
EPSILON_TOPIC = "/cbf_reflex/epsilon_sample"


def read_batch(batch_dir):
    import rosbag

    trajectories = []
    bags = sorted(glob.glob(os.path.join(batch_dir, "run_*", "epsilon.bag")))
    bags += sorted(glob.glob(
        os.path.join(batch_dir, "failed_attempt_*", "epsilon.bag")))
    for bag_path in bags:
        normal, braking = [], []
        states = []
        with rosbag.Bag(bag_path) as bag:
            for topic, message, _ in bag.read_messages(
                    topics=[EPSILON_TOPIC, "/pp_grasp/state"]):
                if topic == "/pp_grasp/state":
                    states.append(str(message.data))
                    continue
                if message.has_normal:
                    normal.append(float(message.normal_adverse_projection))
                if message.has_braking:
                    braking.append(float(message.braking_adverse_projection))
        trajectories.append({
            "normal": np.asarray(normal),
            "braking": np.asarray(braking),
            "complete": "RELEASED" in states,
        })
    return trajectories


def ecdf(values):
    ordered = np.sort(values)
    return ordered, np.arange(1, ordered.size + 1) / float(ordered.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "epsilon_calibration_pp"))
    parser.add_argument("--outdir", default=os.path.expanduser(
        "~/Documents/Memoire/memoire/images/Results"))
    args = parser.parse_args()

    trajectories = read_batch(args.batch_dir)
    count = len(trajectories)
    normal_all = np.concatenate([t["normal"] for t in trajectories])
    braking_all = np.concatenate([t["braking"] for t in trajectories])
    normal_max = np.array([t["normal"].max() for t in trajectories])
    braking_max = np.array([t["braking"].max() for t in trajectories])
    complete = np.array([t["complete"] for t in trajectories])

    bound_normal = normal_max.max()
    bound_braking = braking_max.max()
    coverage_normal = float(np.mean(normal_all <= EPS_DEPLOYED))

    ts.apply()
    import matplotlib.pyplot as plt
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(ts.width(1.0), 3.1),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.32})

    # (a) ECDF per regime, log-x. The clip only affects where the curve
    # enters the axes: the zero-valued mass is the flat start of the curve.
    floor = 1e-4
    for values, color, label in (
            (normal_all, ts.BLUE, "exécution ($\\alpha>0{,}9$)"),
            (braking_all, ts.ORANGE, "freinage")):
        ordered, fraction = ecdf(np.clip(values, floor, None))
        ax_a.step(ordered, fraction, where="post", color=color, label=label)
    ax_a.axvline(EPS_DEPLOYED, color=ts.GREY, ls="--", lw=1.1,
                 label="$\\varepsilon_{\\mathrm{p}}=0{,}025$")
    ax_a.plot([EPS_DEPLOYED], [coverage_normal], "o", ms=4.5,
              color=ts.BLUE, zorder=5)
    ax_a.annotate(f"{100 * coverage_normal:.1f} %",
                  (EPS_DEPLOYED, coverage_normal),
                  textcoords="offset points", xytext=(-6, 5),
                  ha="right", fontsize=10, color=ts.BLUE)
    ax_a.set_xscale("log")
    ax_a.set_xlim(floor, 0.3)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xlabel("perturbation projetée adverse [rad/s]")
    ax_a.set_ylabel("fraction des fenêtres $\\leq x$")
    ts.fig_legend_top(fig, ax_a, ncol=3)
    ts.panel_label(ax_a, "(a)", x=-0.16)

    # (b) per-trajectory maxima and the order-statistic bounds.
    rng = np.random.default_rng(0)
    for column, maxima, color in (
            (0.0, normal_max, ts.BLUE), (1.0, braking_max, ts.ORANGE)):
        jitter = column + rng.uniform(-0.13, 0.13, size=count)
        filled = complete
        ax_b.plot(jitter[filled], maxima[filled], "o", ms=4.5, mew=0,
                  color=color, alpha=0.75)
        ax_b.plot(jitter[~filled], maxima[~filled], "o", ms=4.5,
                  mfc="none", mec=color, mew=1.1, alpha=0.9)
    ax_b.axhline(EPS_DEPLOYED, color=ts.GREY, ls="--", lw=1.1)
    ax_b.axhline(bound_normal, color=ts.BLUE, ls=":", lw=1.3)
    ax_b.axhline(bound_braking, color=ts.ORANGE, ls=":", lw=1.3)
    for bound, color in ((bound_normal, ts.BLUE), (bound_braking, ts.ORANGE)):
        ax_b.annotate(f"{bound:.3f}".replace(".", ","), (-0.40, bound),
                      ha="left", va="bottom", fontsize=9, color=color)
    ax_b.annotate("$\\varepsilon_{\\mathrm{p}}$", (1.40, EPS_DEPLOYED),
                  ha="right", va="bottom", fontsize=10, color=ts.GREY)
    ax_b.set_xlim(-0.45, 1.45)
    ax_b.set_xticks([0.0, 1.0])
    ax_b.set_xticklabels(["exécution", "freinage"])
    ax_b.set_ylim(0.0, 0.2)
    ax_b.set_ylabel("maximum par trajectoire [rad/s]")
    ts.panel_label(ax_b, "(b)", x=-0.44)

    ts.save(fig, os.path.expanduser(args.outdir), "epsilon_calibration")

    print(f"N={count} trajectories ({int(complete.sum())} complete), "
          f"normal windows={normal_all.size}, braking={braking_all.size}")
    print(f"coverage of eps_p in execution regime: {coverage_normal:.4f}")
    print(f"order-statistic bounds (coverage {count}/{count + 1}"
          f"={count / (count + 1.0):.4f}): "
          f"execution {bound_normal:.4f}, braking {bound_braking:.4f}")
    for quantile in (0.5, 0.99, 0.999):
        print(f"execution window quantile {quantile}: "
              f"{np.quantile(normal_all, quantile):.4f}")


if __name__ == "__main__":
    main()
