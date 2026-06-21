#!/usr/bin/env python3
"""Attribute h(t) spikes to a protected link and a cloud point.

For every diagnostics sample whose barrier value dips below --h-threshold, this
recomputes the TRUE mesh-to-cloud clearance (real_distance) with details on and
prints which protected link owns the closest robot surface point, where that
point is, and which obstacle point it matched to. Use it to tell a real wall
approach (link is an arm link / obs_pt on the obstacle) apart from a self or
swept-point artifact (link == 'fork_tip' and obs_pt sitting on the fork body).

Usage:
    python3 scripts/diagnose_h_spike.py run22.bag [--h-threshold -0.002]
                                                  [--d-safe 0.015] [--samples 800]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_cbf_diagnostics as P
import real_distance as rdist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--h-threshold", type=float, default=-0.002,
                    help="report samples with h below this (barrier value, m)")
    ap.add_argument("--d-safe", type=float, default=0.015)
    ap.add_argument("--samples", type=int, default=800,
                    help="surface samples per protected link")
    args = ap.parse_args()

    t, s, t0 = P.load_bag(args.bag)
    h = s["h"]
    dummy = np.nan_to_num(s["selected_count"], nan=0.0) <= 0.0
    dummy |= np.nan_to_num(h, nan=0.0) > 5.0

    t_q, q, obs_t, obs = P.load_motion(args.bag, t0)
    if len(t_q) < 2 or len(obs_t) == 0:
        sys.exit("joint_states / persistent_obstacles missing from bag.")
    q_at_t = np.stack([np.interp(t, t_q, q[:, j]) for j in range(7)], axis=1)
    obs_index = np.abs(obs_t[None, :] - t[:, None]).argmin(axis=1)

    rc = rdist.RealClearance(samples_per_link=args.samples)
    print(f"[diagnose] protected meshes: {rc.found_links}")
    d_real, det = rc.clearance(q_at_t, obs, obs_index, return_details=True)
    d_real[dummy] = np.nan
    h_real = d_real - args.d_safe

    spike = (~dummy) & (h < args.h_threshold)
    idx = np.where(spike)[0]
    if len(idx) == 0:
        print(f"No samples with h < {args.h_threshold}.")
    else:
        print(f"\n{len(idx)} spike sample(s) with h < {args.h_threshold}:\n")
        print(f"{'t':>6}{'h':>9}{'h_real':>9}{'min_obs':>9}  {'link':<12}"
              f"{'robot_pt (xyz)':>26}{'obs_pt (xyz)':>26}")
        for i in idx:
            rp = det["robot_pt"][i]
            op = det["obs_pt"][i]
            link = det["link"][i] or "-"
            print(f"{t[i]:6.2f}{h[i]:9.4f}{h_real[i]:9.4f}{s['min_obs_dist'][i]:9.4f}  "
                  f"{link:<12}"
                  f"[{rp[0]:6.3f} {rp[1]:6.3f} {rp[2]:6.3f}]"
                  f"  [{op[0]:6.3f} {op[1]:6.3f} {op[2]:6.3f}]")

    # Global minimum, regardless of threshold.
    vi = np.isfinite(d_real)
    g = np.where(vi)[0][np.nanargmin(d_real[vi])]
    print(f"\nglobal min d_real = {d_real[g]:.4f} m at t={t[g]:.2f}  "
          f"(link={det['link'][g]}, h_real={h_real[g]:.4f})")
    # Which links own the spikes?
    if len(idx):
        links, counts = np.unique(det["link"][idx].astype(str), return_counts=True)
        print("spike culprit links: " +
              ", ".join(f"{l}×{c}" for l, c in zip(links, counts)))


if __name__ == "__main__":
    main()
