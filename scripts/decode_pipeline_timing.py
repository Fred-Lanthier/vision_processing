#!/usr/bin/env python3
"""Aggregate the per-stage pipeline timings recorded in a rosbag (thesis 5.6).

Every pipeline node publishes its stage durations [ms] on /pipeline/timing/<stage>
(see pipeline_timing.py). Record them with, e.g.::

    rosbag record -e '/pipeline/timing/.*' /joint_states ...
    # or simply: rosbag record -a

This script decodes every /pipeline/timing/* topic from the bag and produces:
  pipeline_timing.json       per-stage n / mean / median / p95 / p99 / std / min / max
  pipeline_timing.tex/.csv   per-stage table (median + p95), grouped
  pipeline_timing_boxplot.*  thesis-styled boxplot, stages colored by group

End-to-end budget: the per-frame stages do not carry a single propagated camera
timestamp through the planner (the planner restamps its output with now()), so a
true per-frame end-to-end latency cannot be reconstructed from header stamps.
Instead this reports an honest pipeline BUDGET = sum of the median per-pass stage
times (perception + planning + safety), which is what 'temps total du pipeline'
means in practice. Stages that are themselves aggregates (planner_total,
cbf_total) are used for the budget; their sub-stages are shown for the breakdown
but not re-summed.

Usage:
    python3 decode_pipeline_timing.py run.bag [-o outdir] [--csv]
"""

import argparse
import json
import os
import sys

import numpy as np
import rosbag

TIMING_NS = "/pipeline/timing/"

# Display order + grouping. Stages absent from the bag are silently skipped.
GROUPS = [
    ("perception", ["sam3_detect", "sam2_track", "projection",
                    "persistent_evict", "persistent_integrate"]),
    ("planning",   ["fm_encode", "fm_generation", "fm_correction", "ik",
                    "ik_correction", "casf_warp", "trajectory_build",
                    "planner_total"]),
    ("safety",     ["critical_point_selection", "cbf_correction",
                    "cbf_command", "cbf_total"]),
]
# Stages already aggregating their group (excluded from the breakdown sum but
# used as the per-pass contribution in the end-to-end budget).
TOTALS = {"planner_total", "cbf_total"}
# One representative per-pass contribution per pipeline pass for the budget.
BUDGET_STAGES = ["sam2_track", "projection", "persistent_integrate",
                 "planner_total", "cbf_total"]


def load_timings(path):
    """stage -> np.array(ms) for every /pipeline/timing/<stage> topic in the bag."""
    series = {}
    with rosbag.Bag(path) as bag:
        topics = bag.get_type_and_topic_info().topics
        timing_topics = [t for t in topics if t.startswith(TIMING_NS)]
        if not timing_topics:
            sys.exit(f"No {TIMING_NS}* topics in {path}. Was timing recorded "
                     f"(rosbag record -e '/pipeline/timing/.*')?")
        for topic, msg, _ in bag.read_messages(topics=timing_topics):
            stage = topic[len(TIMING_NS):]
            series.setdefault(stage, []).append(float(msg.data))
    return {k: np.asarray(v, dtype=np.float64) for k, v in series.items()}


def stats(arr):
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ordered_stages(series):
    """(group, stage) pairs in display order, only for stages present in the bag."""
    out = []
    seen = set()
    for group, stages in GROUPS:
        for s in stages:
            if s in series:
                out.append((group, s))
                seen.add(s)
    for s in sorted(series):  # any stage not in GROUPS, appended at the end
        if s not in seen:
            out.append(("other", s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output dir (default: <bagname>_timing)")
    ap.add_argument("--csv", action="store_true", help="also dump raw per-sample CSV")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt

    series = load_timings(args.bag)
    outdir = args.outdir or os.path.splitext(args.bag)[0] + "_timing"
    os.makedirs(outdir, exist_ok=True)

    pairs = ordered_stages(series)
    summary = {"per_stage": {s: stats(series[s]) for _, s in pairs}}

    # End-to-end budget = sum of median per-pass contributions.
    budget = sum(summary["per_stage"][s]["median"]
                 for s in BUDGET_STAGES if s in summary["per_stage"])
    summary["end_to_end_budget_ms"] = float(budget)
    summary["end_to_end_budget_stages"] = [s for s in BUDGET_STAGES
                                           if s in summary["per_stage"]]

    with open(os.path.join(outdir, "pipeline_timing.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- Boxplot (thesis style), stages colored by group --------------------
    group_color = {"perception": ts.GREEN, "planning": ts.BLUE,
                   "safety": ts.ORANGE, "other": ts.GREY}
    labels = [s for _, s in pairs]
    data = [series[s] for _, s in pairs]
    colors = [group_color[g] for g, _ in pairs]

    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 0.42 * len(labels) + 1.2))
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2),
                    widths=0.6)
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.65, edgecolor=c)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()                      # first stage on top
    ax.set_xscale("log")
    ax.set_xlabel("temps par cycle [ms]")
    # legend by group
    from matplotlib.patches import Patch
    present = [g for g, _ in pairs]
    handles = [Patch(facecolor=group_color[g], alpha=0.65, label=g)
               for g in ["perception", "planning", "safety", "other"]
               if g in present]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    ts.save(fig, outdir, "pipeline_timing_boxplot")

    # ---- Table (.tex + .csv) ------------------------------------------------
    with open(os.path.join(outdir, "pipeline_timing.csv"), "w") as f:
        f.write("group,stage,n,median_ms,mean_ms,p95_ms,p99_ms\n")
        for g, s in pairs:
            st = summary["per_stage"][s]
            f.write(f"{g},{s},{st['n']},{st['median']:.3f},{st['mean']:.3f},"
                    f"{st['p95']:.3f},{st['p99']:.3f}\n")
    with open(os.path.join(outdir, "pipeline_timing.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Groupe & \\'Etape & $N$ & m\\'ediane [ms] & moyenne [ms] "
                "& p95 [ms] \\\\\n\\midrule\n")
        last_g = None
        for g, s in pairs:
            st = summary["per_stage"][s]
            gcol = g if g != last_g else ""
            last_g = g
            name = s.replace("_", r"\_")
            f.write(f"{gcol} & {name} & {st['n']} & {st['median']:.2f} & "
                    f"{st['mean']:.2f} & {st['p95']:.2f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\multicolumn{{3}}{{l}}{{budget bout-à-bout (somme des médianes)}}"
                f" & \\multicolumn{{3}}{{r}}{{{budget:.2f} ms}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    if args.csv:
        import csv
        with open(os.path.join(outdir, "pipeline_timing_raw.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "ms"])
            for _, s in pairs:
                for v in series[s]:
                    w.writerow([s, v])

    print(f"\nWrote summary, boxplot and table to {outdir}/")


if __name__ == "__main__":
    main()
