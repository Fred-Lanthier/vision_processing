#!/usr/bin/env python3
"""Decode and plot /cbf_safety/diagnostics from a rosbag.

Produces the thesis result figures from one experiment bag:
  1. h(t) and h_corr(t) with zero line and CBF-active / escape shading
  2. Joint-velocity decomposition |dq_ff|, |dq_fb|, |dq_base|, |dq_cbf_delta|,
     |dq_escape|, |dq_safe|
  3. Pre- vs post-filter safe velocity (low-pass effect)
  4. Post-repair constraint residual (negative = CBF inequality violated
     after filtering/clamping)
  5. Pipeline timing boxplot
  6. Mode-flag timeline (cap_active, recovery, escape, repair, EMA)

Usage:
    python3 plot_cbf_diagnostics.py run.bag [-o outdir] [--csv]

Record with:
    rosbag record /cbf_safety/diagnostics /cbf_safety/diagnostics_layout
"""

import argparse
import json
import os
import sys

import numpy as np

import rosbag

DIAG_TOPIC = "/cbf_safety/diagnostics"
LAYOUT_TOPIC = "/cbf_safety/diagnostics_layout"


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
    t = np.asarray(stamps) - stamps[0]

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
    return t, series


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
    ap.add_argument("--smooth", type=float, default=0.25,
                    help="display smoothing window in seconds for the velocity "
                         "decomposition (raw shown faintly behind; 0 = off)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt

    t, s = load_bag(args.bag)
    outdir = args.outdir or os.path.splitext(args.bag)[0] + "_diag"
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

    # The repair correction is not logged. Replay the output low-pass on the
    # logged solver output (dq_pre_filter) and subtract from the published
    # velocity: dq_rep = dq_final - LP(dq_pre_filter). Validated on run3:
    # residual where repair_applied == 0 has median ~1e-4 rad/s.
    dq_rep_norm = np.zeros_like(t)
    if "dt" in s and "repair_applied" in s:
        dts = np.nan_to_num(s["dt"], nan=0.0).copy()
        good = dts > 0
        dts[~good] = np.median(dts[good]) if good.any() else 1e-2
        state = s["dq_pre_filter"][0].copy()
        lp = np.empty_like(s["dq_pre_filter"])
        for k in range(len(t)):
            g = dts[k] / (dts[k] + args.tau_safe)
            state = (1.0 - g) * state + g * s["dq_pre_filter"][k]
            lp[k] = state
        dq_rep_norm = np.linalg.norm(s["dq_final"] - lp, axis=1)
        dq_rep_norm[~repaired] = 0.0

    # 1. Barrier value -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
    ax.plot(t, s["h"], color=ts.GREY, label=r"$h_{\mathrm{soft}}$ (brute)", lw=1.0)
    ax.plot(t, s["h_corr"], color=ts.BLUE, label=r"$h_{\mathrm{corr}}$", lw=1.0)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    shade_active(ax, t, cbf_active, ts.GREEN, "projection CBF")
    shade_active(ax, t, repair_only, ts.YELLOW, "réparation seule")
    shade_active(ax, t, escape, ts.PURPLE, "échappement")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("h [m]")
    ts.legend_top(ax)
    fig.tight_layout()
    ts.save(fig, outdir, "h_evolution")

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
    ax.annotate("retiming : consigne constante",
                xy=(0.62 * t[-1], ff_level), xytext=(0.62 * t[-1], ff_level * 1.18),
                color=ts.SKY, fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", color=ts.SKY, lw=0.7))
    ax.set_ylabel("[rad/s]")
    ax.set_ylim(bottom=0.0, top=ff_level * 1.45)
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
    ax.set_ylim(bottom=0.0)
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

    # 3. Filter effect ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
    ax.plot(t, norm(s["dq_pre_filter"]), color=ts.GREY, label="avant filtre (brute)", lw=0.9)
    ax.plot(t, norm(s["dq_final"]), color=ts.BLUE, label="après filtre", lw=0.9)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("[rad/s]")
    ts.legend_top(ax)
    fig.tight_layout()
    ts.save(fig, outdir, "filter_effect")

    # 4. Constraint residuals ----------------------------------------------
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
    ax.plot(t, s["constr_pre"], color=ts.GREY, label="avant projection (brute)", lw=0.9)
    ax.plot(t, s["constr_final"], color=ts.BLUE, label="après réparation", lw=0.9)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\nabla h^\top \dot{q} + \kappa h - m$")
    ts.legend_top(ax)
    fig.tight_layout()
    ts.save(fig, outdir, "constraint_residual")

    # 5. Timing boxplot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    keys = ["cbf_ms", "q_cmd_ms", "total_ms"]
    bp = ax.boxplot([s[k][~np.isnan(s[k])] for k in keys],
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=ts.ORANGE))
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor=ts.GREY)
    ax.set_xticklabels(["CBF", "commande", "total"])
    ax.set_ylabel("temps de cycle [ms]")
    fig.tight_layout()
    ts.save(fig, outdir, "timing_boxplot")

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

    # 7. Commanded vs executed h-rate (multicbf nodes only) -------------------
    if "real_hdot" in s and not np.all(np.isnan(s["real_hdot"])):
        fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH, 2.5))
        ax.plot(t, s["cmd_hdot"], color=ts.BLUE, label=r"$\dot{h}$ commandé", lw=0.9)
        ax.plot(t, s["real_hdot"], color=ts.ORANGE,
                label=r"$\dot{h}$ exécuté (vitesse mesurée)", lw=0.9)
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        shade_active(ax, t, intervened, ts.ORANGE, "CBF actif")
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
