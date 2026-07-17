#!/usr/bin/env python3
"""Offline study of the CBF-QP inner solvers (thesis, ch. 4/5).

Compares, on randomized multi-constraint projection instances that mimic the
runtime regime of cbf_safety_node_Bernstein_multicbf.py:

  - POCS   : cyclic projections (legacy), the node's exact elementary update
  - Dykstra: corrected projections (deployed), same elementary update + memory
  - exact  : active-set enumeration of the KKT conditions (ground truth)

Produces (thesis_style, SVG + PNG):
  1. excess_vs_sweeps   — median/p90 excess W-deviation vs number of sweeps,
                          POCS vs Dykstra, with the deployed N_it marked.
                          Justifies N_it and the POCS->Dykstra decision.
  2. residual_vs_sweeps — worst residual constraint violation vs sweeps
                          (feasibility side; covered at runtime by the final
                          re-evaluation either way).
  3. example_2d         — didactic 2D instance: POCS parks at a feasible but
                          suboptimal point; Dykstra converges to the true
                          projection.
  4. solver_study.json  — all headline numbers.

Usage:
    python3 solver_study.py [-o outdir] [--n 2000] [--metric task|identity]
                            [--nit 6] [--seed 0]

No ROS required.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

EPS0 = 1e-8  # exact-projection denominator floor (node: cbf_projection_*)


# --------------------------------------------------------------------------
# Instance generation
# --------------------------------------------------------------------------
def make_metric(rng, kind):
    """Task metric W = Jp^T Jp + w_rot Jo^T Jo + lam I of a random pose
    Jacobian (deployed config: lam = w_rot = 0.01), or identity."""
    if kind == "identity":
        return np.eye(7)
    J = rng.standard_normal((6, 7)) * 0.5
    W = J[:3].T @ J[:3] + 0.01 * (J[3:].T @ J[3:]) + 0.01 * np.eye(7)
    return W


def make_instance(rng, metric_kind):
    """One projection instance (W, dq_base, G, b) with 1-4 violated rows.

    Rows are angularly correlated around a common direction (the wall-graze /
    pocket signature: several links share the local obstacle normal), with
    gradient norms and violation depths in the ranges observed at runtime.
    With probability 0.5 two slack rows are appended to mimic the fixed-size
    active set (they must not change any solver's answer).
    """
    W = make_metric(rng, metric_kind)
    dq_base = rng.standard_normal(7)
    dq_base *= rng.uniform(0.1, 0.6) / np.linalg.norm(dq_base)

    m = rng.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.1])
    u = rng.standard_normal(7)
    u /= np.linalg.norm(u)
    rows, rhs = [], []
    for _ in range(m):
        c = rng.uniform(0.3, 0.9)  # angular correlation with the cluster axis
        g = c * u + (1.0 - c) * rng.standard_normal(7)
        g *= rng.uniform(0.5, 1.5) / np.linalg.norm(g)
        rows.append(g)
        rhs.append(float(g @ dq_base) + rng.uniform(0.01, 0.25))  # violated
    n_slack = 2 if rng.random() < 0.5 else 0
    for _ in range(n_slack):
        g = rng.standard_normal(7)
        g /= np.linalg.norm(g)
        rows.append(g)
        rhs.append(float(g @ dq_base) - rng.uniform(0.05, 0.3))  # satisfied
    return W, dq_base, np.asarray(rows), np.asarray(rhs)


# --------------------------------------------------------------------------
# Solvers (mirror the node's elementary update)
# --------------------------------------------------------------------------
def _project_halfspace(dq, g, b, Winv_g, gWg):
    """Exact metric projection of dq onto {g^T x >= b} (node update, w=1)."""
    viol = b - float(g @ dq)
    if viol <= 0.0:
        return dq
    return dq + (viol / max(gWg, EPS0)) * Winv_g


def pocs(dq_base, G, b, Winv, sweeps):
    """Cyclic projections; yields the iterate after each sweep."""
    Winv_G = [Winv @ g for g in G]
    gWg = [float(g @ wg) for g, wg in zip(G, Winv_G)]
    dq = dq_base.copy()
    for _ in range(sweeps):
        for l in range(len(G)):
            dq = _project_halfspace(dq, G[l], b[l], Winv_G[l], gWg[l])
        yield dq.copy()


def dykstra(dq_base, G, b, Winv, sweeps):
    """Dykstra's corrected projections (= Hildreth for half-spaces)."""
    Winv_G = [Winv @ g for g in G]
    gWg = [float(g @ wg) for g, wg in zip(G, Winv_G)]
    dq = dq_base.copy()
    z = [np.zeros(7) for _ in range(len(G))]
    for _ in range(sweeps):
        for l in range(len(G)):
            dq_prime = dq + z[l]
            dq_new = _project_halfspace(dq_prime, G[l], b[l], Winv_G[l], gWg[l])
            z[l] = dq_prime - dq_new
            dq = dq_new
        yield dq.copy()


def dykstra_state(dq_base, G, b, Winv, sweeps, z=None):
    """Dykstra with injectable memories: returns (dq, z) after `sweeps`.

    Cross-cycle warm-start prototype: at 100 Hz the physical constellation
    persists across cycles, so carrying z (the dual state) accumulates sweeps
    across cycles instead of restarting the convergence from scratch."""
    Winv_G = [Winv @ g for g in G]
    gWg = [float(g @ wg) for g, wg in zip(G, Winv_G)]
    dq = dq_base.copy()
    if z is None:
        z = [np.zeros(len(dq_base)) for _ in range(len(G))]
    else:
        z = [zi.copy() for zi in z]
    for _ in range(sweeps):
        for l in range(len(G)):
            dq_prime = dq + z[l]
            dq_new = _project_halfspace(dq_prime, G[l], b[l], Winv_G[l], gWg[l])
            z[l] = dq_prime - dq_new
            dq = dq_new
    return dq, z


def hildreth_state(dq_base, G, b, Winv, sweeps, lam=None):
    """Hildreth dual coordinate descent (identical iterates to Dykstra for
    half-spaces) with WARM-STARTABLE multipliers: lam are scalars attached to
    the constraint identity, re-injected into the CURRENT geometry."""
    Winv_G = np.array([Winv @ g for g in G])
    gWg = np.array([max(float(g @ wg), 1e-12) for g, wg in zip(G, Winv_G)])
    lam = np.zeros(len(G)) if lam is None else lam.copy()
    dq = dq_base + Winv_G.T @ lam
    for _ in range(sweeps):
        for l in range(len(G)):
            delta = (b[l] - float(G[l] @ dq)) / gWg[l]
            new = max(0.0, lam[l] + delta)
            dq = dq + Winv_G[l] * (new - lam[l])
            lam[l] = new
    return dq, lam


def warmstart_scenario(rng, n_cycles, nit, drift=0.01):
    """One slowly drifting ill-conditioned episode (near-parallel gradients,
    the tail regime of the main study), solved cycle by cycle.

    Three arms, per-cycle distance-to-exact [% of correction]:
      cold    — memories reset every cycle (current node behaviour);
      warm_z  — raw z vectors carried over (naive warm start: DIVERGES,
                the memories encode the stale geometry);
      warm_l  — dual multipliers carried by constraint identity and
                re-injected into the current geometry (correct warm start).
    """
    W = make_metric(rng, "task")
    Winv = np.linalg.inv(W)
    u = rng.standard_normal(7)
    u /= np.linalg.norm(u)
    G0, off = [], []
    for _ in range(3):
        c = rng.uniform(0.9, 0.98)          # near-parallel: the tail regime
        g = c * u + (1.0 - c) * rng.standard_normal(7)
        G0.append(g / np.linalg.norm(g) * rng.uniform(0.5, 1.5))
        off.append(rng.uniform(0.05, 0.25))
    G0 = np.asarray(G0)
    dq_base = rng.standard_normal(7)
    dq_base *= 0.4 / np.linalg.norm(dq_base)
    drift_dir = rng.standard_normal(7)
    drift_dir *= drift / np.linalg.norm(drift_dir)

    cold, warm_z, warm_l = [], [], []
    z, lam = None, None
    for t in range(n_cycles):
        base_t = dq_base + t * drift_dir
        Gt = G0 + drift * 0.1 * rng.standard_normal(G0.shape)
        bt = Gt @ base_t + np.asarray(off)
        star = exact_projection(base_t, Gt, bt, W, Winv)
        if star is None:
            return None, None, None
        c_star = wnorm(star - base_t, W)
        if c_star < 1e-9:
            return None, None, None
        dq_c, _ = dykstra_state(base_t, Gt, bt, Winv, nit, z=None)
        dq_z, z = dykstra_state(base_t, Gt, bt, Winv, nit, z=z)
        dq_l, lam = hildreth_state(base_t, Gt, bt, Winv, nit, lam=lam)
        cold.append(100.0 * wnorm(dq_c - star, W) / c_star)
        warm_z.append(100.0 * wnorm(dq_z - star, W) / c_star)
        warm_l.append(100.0 * wnorm(dq_l - star, W) / c_star)
    return np.asarray(cold), np.asarray(warm_z), np.asarray(warm_l)


def exact_projection(dq_base, G, b, W, Winv):
    """Ground truth by enumeration of KKT active sets.

    dq* = dq_base + Winv G_S^T lam,  lam = (G_S Winv G_S^T)^-1 (b_S - G_S base),
    valid iff lam >= 0 (dual feasibility) and G dq* >= b (primal feasibility).
    """
    m = len(G)
    best, best_cost = None, np.inf
    for r in range(m + 1):
        for S in itertools.combinations(range(m), r):
            if r == 0:
                cand = dq_base.copy()
            else:
                GS = G[list(S)]
                A = GS @ Winv @ GS.T
                try:
                    lam = np.linalg.solve(A, b[list(S)] - GS @ dq_base)
                except np.linalg.LinAlgError:
                    continue
                if (lam < -1e-10).any():
                    continue
                cand = dq_base + Winv @ GS.T @ lam
            if (G @ cand - b < -1e-9).any():
                continue
            d = cand - dq_base
            cost = float(d @ W @ d)
            if cost < best_cost - 1e-15:
                best, best_cost = cand, cost
    return best


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------
def wnorm(v, W):
    return float(np.sqrt(max(v @ W @ v, 0.0)))


def run_study(n, metric_kind, max_sweeps, seed):
    rng = np.random.default_rng(seed)
    n_active = []                       # active-set size at the optimum
    exc = {"POCS": [], "Dykstra": []}   # [n, max_sweeps] excess deviation [%]
    res = {"POCS": [], "Dykstra": []}   # [n, max_sweeps] worst violation
    dist = {"POCS": [], "Dykstra": []}  # [n, max_sweeps] dist to optimum [%]
    skipped = 0
    for _ in range(n):
        W, dq_base, G, b = make_instance(rng, metric_kind)
        Winv = np.linalg.inv(W)
        star = exact_projection(dq_base, G, b, W, Winv)
        if star is None:
            skipped += 1
            continue
        c_star = wnorm(star - dq_base, W)
        if c_star < 1e-9:
            skipped += 1
            continue
        n_active.append(int(np.sum(np.abs(G @ star - b) < 1e-7)))
        for name, solver in (("POCS", pocs), ("Dykstra", dykstra)):
            e_row, r_row, d_row = [], [], []
            for dq in solver(dq_base, G, b, Winv, max_sweeps):
                e_row.append(100.0 * (wnorm(dq - dq_base, W) - c_star) / c_star)
                r_row.append(float(np.max(np.maximum(b - G @ dq, 0.0))))
                d_row.append(100.0 * wnorm(dq - star, W) / c_star)
            exc[name].append(e_row)
            res[name].append(r_row)
            dist[name].append(d_row)
    for d in (exc, res, dist):
        for k in d:
            d[k] = np.asarray(d[k])
    return exc, res, dist, np.asarray(n_active), skipped


def example_2d():
    """The didactic instance: base at origin, H1: x>=1, H2: x+y>=2."""
    W = np.eye(2)
    Winv = W
    dq_base = np.zeros(2)
    G = np.array([[1.0, 0.0], [1.0, 1.0] / np.sqrt(2.0)])
    b = np.array([1.0, 2.0 / np.sqrt(2.0)])

    def path(solver):
        pts = [dq_base.copy()]
        dq = dq_base.copy()
        z = [np.zeros(2), np.zeros(2)]
        for _ in range(6):
            for l in range(2):
                if solver == "dykstra":
                    dq_prime = dq + z[l]
                else:
                    dq_prime = dq
                gWg = float(G[l] @ Winv @ G[l])
                new = _project_halfspace(dq_prime, G[l], b[l], Winv @ G[l], gWg)
                if solver == "dykstra":
                    z[l] = dq_prime - new
                dq = new
                pts.append(dq.copy())
        return np.asarray(pts)

    return path("pocs"), path("dykstra")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", default="solver_study_out")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--metric", choices=("task", "identity"), default="task")
    ap.add_argument("--nit", type=int, default=6,
                    help="deployed sweep count to mark and report")
    ap.add_argument("--max-sweeps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-episodes", type=int, default=300,
                    help="drifting ill-conditioned episodes for the "
                         "cross-cycle warm-start study")
    ap.add_argument("--warm-cycles", type=int, default=50)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    exc, res, dist, n_active, skipped = run_study(args.n, args.metric,
                                                  args.max_sweeps, args.seed)
    sweeps = np.arange(1, args.max_sweeps + 1)
    k = args.nit - 1
    multi = n_active >= 2  # cf. run_study: excess is zero for both when m=1

    # 1. Distance to the exact projection vs sweeps (multi-active instances;
    # defined and positive at EVERY sweep, unlike the excess deviation which
    # is negative while the iterate is still infeasible). POCS plateaus: its
    # limit is a feasible point that is NOT the projection; Dykstra decays
    # geometrically (linear convergence, Iusem & De Pierro 1990). ------------
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.8))
    for name, color in (("POCS", ts.ORANGE), ("Dykstra", ts.BLUE)):
        med = np.median(dist[name][multi], axis=0)
        p90 = np.percentile(dist[name][multi], 90, axis=0)
        ax.plot(sweeps, med, color=color, marker="o", ms=3,
                label=f"{name} (médiane)")
        ax.plot(sweeps, p90, color=color, ls="--", lw=1.0,
                label=f"{name} (p90)")
    ax.axvline(args.nit, color="k", lw=0.8, ls=":",
               label=rf"$N_{{\mathrm{{it}}}}={args.nit}$ (déployé)")
    ax.set_xlabel("balayages")
    ax.set_ylabel(r"distance à $\dot{q}^\star$ [% corr.]")
    ax.set_yscale("log")
    from matplotlib.ticker import LogLocator
    ax.set_ylim(1e-5, 1e2)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=99))
    ts.legend_top(ax, ncol=3)
    fig.tight_layout()
    ts.save(fig, args.outdir, "dist_vs_sweeps")

    # 2. Feasibility residual vs sweeps ------------------------------------
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.8))
    for name, color in (("POCS", ts.ORANGE), ("Dykstra", ts.BLUE)):
        med = np.median(res[name], axis=0)
        p99 = np.percentile(res[name], 99, axis=0)
        ax.plot(sweeps, med, color=color, marker="o", ms=3,
                label=f"{name} (médiane)")
        ax.plot(sweeps, p99, color=color, ls="--", lw=1.0,
                label=f"{name} (p99)")
    ax.axvline(args.nit, color="k", lw=0.8, ls=":")
    ax.set_xlabel("balayages")
    ax.set_ylabel("violation résiduelle [m/s]")
    ax.set_yscale("log")
    ts.legend_top(ax, ncol=2)
    fig.tight_layout()
    ts.save(fig, args.outdir, "residual_vs_sweeps")

    # 3. 2D didactic example ------------------------------------------------
    p_pocs, p_dyk = example_2d()
    fig, ax = plt.subplots(figsize=(ts.width(0.62), 3.2))
    xx = np.linspace(-0.2, 2.2, 2)
    ax.fill_between(xx, np.maximum(2.0 - xx, -1), 2.6, where=xx >= 1.0,
                    color=ts.GREEN, alpha=0.10, lw=0,
                    label="ensemble admissible")
    ax.axvline(1.0, color=ts.GREY, lw=1.0)
    ax.plot(xx, 2.0 - xx, color=ts.GREY, lw=1.0)
    ax.plot(*p_dyk.T, color=ts.BLUE, marker="o", ms=3.5, lw=1.2,
            label="Dykstra")
    ax.plot(*p_pocs.T, color=ts.ORANGE, marker="o", ms=3.0, lw=1.6,
            ls="--", label="POCS (premier balayage, puis figé)")
    ax.plot(0, 0, marker="s", color="k", ms=5)
    ax.annotate(r"$\dot{q}_{\mathrm{base}}$", (0, 0), (0.05, -0.18),
                fontsize=10)
    ax.plot(1, 1, marker="*", color=ts.BLUE, ms=11)
    ax.annotate("projection exacte $(1,1)$", (1, 1), (1.08, 0.9), fontsize=10)
    ax.plot(*p_pocs[-1], marker="X", color=ts.ORANGE, ms=8)
    ax.annotate(r"POCS $(1{,}5,\ 0{,}5)$", (1.5, 0.5), (1.45, 0.25),
                fontsize=10)
    ax.set_xlim(-0.25, 2.3)
    ax.set_ylim(-0.35, 1.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\dot{q}_1$")
    ax.set_ylabel(r"$\dot{q}_2$")
    ts.legend_top(ax, ncol=3)
    fig.tight_layout()
    ts.save(fig, args.outdir, "example_2d")

    # 4. Cross-cycle warm start on drifting ill-conditioned episodes --------
    rng_w = np.random.default_rng(args.seed + 1)
    C, Z, L = [], [], []
    for _ in range(args.warm_episodes):
        c, z, l = warmstart_scenario(rng_w, args.warm_cycles, args.nit)
        if c is not None:
            C.append(c)
            Z.append(z)
            L.append(l)
    C, Z, L = np.asarray(C), np.asarray(Z), np.asarray(L)
    cyc = np.arange(1, args.warm_cycles + 1)
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 3.2))
    first = True
    for arr, color, lbl in ((C, ts.ORANGE, "froid (actuel)"),
                            (Z, ts.PURPLE, "chaud naïf ($\\mathbf{z}$)"),
                            (L, ts.BLUE, "chaud dual ($\\lambda$)")):
        ax.plot(cyc, np.maximum(np.median(arr, axis=0), 1e-6), color=color,
                lw=1.3, label=lbl)
        ax.plot(cyc, np.maximum(np.percentile(arr, 90, axis=0), 1e-6),
                color=color, ls="--", lw=1.0,
                label="p90 (tireté)" if first else None)
        first = False
    ax.set_xlabel("cycle de commande")
    ax.set_ylabel(r"distance à $\dot{q}^\star$ [% corr.]")
    ax.set_yscale("log")
    from matplotlib.ticker import LogLocator as _LL
    ax.yaxis.set_major_locator(_LL(base=10.0, numticks=99))
    ts.legend_top(ax, ncol=3)
    fig.tight_layout()
    ts.save(fig, args.outdir, "warmstart_vs_cycles")

    # Headline numbers ------------------------------------------------------
    summary = {
        # Population (all statistics below on the multi-active subset)
        "n_instances": int(dist["POCS"].shape[0]),
        "n_multi_active": int(multi.sum()),
        "skipped": skipped,
        "metric": args.metric,
        "seed": args.seed,
        "nit_deployed": args.nit,
        # Figure metric: distance to the exact projection [% of correction]
        "dykstra_dist_median_at_nit_pct": float(
            np.median(dist["Dykstra"][multi, k])),
        "dykstra_dist_p90_at_nit_pct": float(
            np.percentile(dist["Dykstra"][multi, k], 90)),
        "pocs_dist_median_asymptotic_pct": float(
            np.median(dist["POCS"][multi, -1])),
        "pocs_dist_p90_asymptotic_pct": float(
            np.percentile(dist["POCS"][multi, -1], 90)),
        # Feasibility side (why the final re-evaluation exists)
        "residual_median_at_nit": float(
            np.median(res["Dykstra"][multi, k])),
        "residual_p99_at_nit": float(
            np.percentile(res["Dykstra"][multi, k], 99)),
        # Warm-start study (drifting ill-conditioned episodes)
        "warm_episodes": int(C.shape[0]),
        "warm_cycles": int(args.warm_cycles),
        "cold_dist_p90_last_cycle_pct": float(np.percentile(C[:, -1], 90)),
        "warm_lambda_dist_p90_cycle3_pct": float(np.percentile(L[:, 2], 90)),
        "warm_lambda_dist_p90_last_cycle_pct": float(
            np.percentile(L[:, -1], 90)),
        "warm_naive_z_dist_median_last_cycle_pct": float(
            np.median(Z[:, -1])),
    }
    with open(os.path.join(args.outdir, "solver_study.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
