#!/usr/bin/env python3
"""Generate publication-quality thesis figures + tables for the production Bernstein SDF.

Apples-to-apples: the SDF is measured against the mesh union of EXACTLY its own links.
Outputs (in ./thesis_results/):
  fig1_distance_accuracy.png   - predicted vs exact + signed error histogram
  fig2_gradient_accuracy.png   - angular error histogram + ||grad|| histogram
  fig3_speed.png               - Bernstein vs mesh timing across batch sizes (+speedup)
  table_per_link.tex / .csv    - per-link distance & gradient error
Reads production models read-only; writes nothing outside this folder.
"""

import os, sys, time
import numpy as np
import torch
import trimesh

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thesis_results")
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", "..")); sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V

sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import thesis_style as ts
import matplotlib.pyplot as plt

ts.apply()
BLUE, RED, GREEN, GREY = ts.BLUE, ts.ORANGE, ts.GREEN, ts.GREY
D_SAFE_MM = 15.0
RAYS_PER_LINK = 40
label = ts.panel_label


def domain_valid_mask(core, rl, pts, pose, q9, dev):
    """True where the query point lies INSIDE the [-1,1] Bernstein training cube of
    its nearest link. Outside that cube the model returns a clamped boundary value
    plus a Euclidean bound (bernstein_core.py), i.e. it is no longer the learned SDF,
    so those samples must not enter the accuracy benchmark."""
    x = torch.as_tensor(pts, dtype=torch.float32, device=dev)
    with torch.no_grad():
        _, sdf_pl = core.get_whole_body_sdf_batch(x, pose, q9, return_per_link=True)
        nearest = sdf_pl[0].argmin(dim=0)
        trans_list = rl.get_transformations_each_link(pose, q9)
        matched = []
        for target_link in core.used_links:
            t_name = target_link.replace('panda_', '').replace('_w', '').replace('.pt', '')
            idx = None
            for i, info in enumerate(rl.meshes_info):
                i_name = info['link_name'].replace('panda_', '')
                if t_name in i_name or i_name in t_name:
                    idx = i
                    break
            matched.append(trans_list[idx])
        trans = torch.stack(matched, dim=1)[0]                      # (K,4,4)
        diff = x.unsqueeze(0) - trans[:, :3, 3].unsqueeze(1)        # (K,M,3)
        x_local = torch.einsum('kmi,kij->kmj', diff, trans[:, :3, :3])
        x_scaled = (x_local - core.offsets.unsqueeze(1)) / core.scales.view(-1, 1, 1)
        clamped = (x_scaled.abs() > (1.0 - 1e-2)).any(dim=-1)       # (K,M)
        clamped_nearest = clamped[nearest, torch.arange(len(pts), device=dev)]
    return (~clamped_nearest).cpu().numpy()


def collect(core, rl, meshes, links, pose, q9, dev):
    # Clean ray sampling, apples-to-apples: rays are launched PER LINK (every link
    # represented + labelled by source), exact distance/gradient measured against the
    # full union of the SDF's own links. The (ray, offset) structure is kept so we can
    # plot distance / gradient error as a function of distance from the surface.
    combined = trimesh.util.concatenate(meshes)
    pq = trimesh.proximity.ProximityQuery(combined)
    offsets = np.linspace(0.001, 0.05, 25, dtype=np.float32)
    rays_per_link = RAYS_PER_LINK

    P, lab = [], []
    bern_RS, exact_RS, errRS, angRS, gnRS, okRS = [], [], [], [], [], []
    flat_pred, flat_exact, flat_ang, flat_gn, flat_ok = [], [], [], [], []
    for name, m in zip(links, meshes):
        # launch generously from this link, then KEEP ONLY rays that stay clean w.r.t.
        # the whole robot: the marched offset must equal the true union distance at every
        # sample, so a ray drifting near another link (offset != distance) is discarded.
        rp, gd, gg, _ = V.make_external_benchmark_rays([m], meshes, rays_per_link * 3, offsets, 123)
        clean = np.all(np.abs(gd - offsets[None, :]) < 0.003, axis=1)
        rp, gd, gg = rp[clean][:rays_per_link], gd[clean][:rays_per_link], gg[clean][:rays_per_link]
        R, S, _ = rp.shape
        pts = rp.reshape(-1, 3).astype(np.float32)
        gex = gg.reshape(-1, 3)
        gex /= np.clip(np.linalg.norm(gex, axis=1, keepdims=True), 1e-9, None)
        sdf, gb = V.evaluate_sdf_and_grad(core, pts, pose, q9, 32768, dev)
        gn = np.linalg.norm(gb, axis=1)
        gu = gb / np.clip(gn[:, None], 1e-9, None)
        ang = np.degrees(np.arccos(np.clip(np.sum(gu * gex, axis=1), -1, 1)))
        ok = domain_valid_mask(core, rl, pts, pose, q9, dev)
        bern_RS.append(sdf.reshape(R, S)); exact_RS.append(gd)
        errRS.append((sdf - gd.reshape(-1)).reshape(R, S)); angRS.append(ang.reshape(R, S))
        gnRS.append(gn.reshape(R, S)); okRS.append(ok.reshape(R, S))
        P.append(pts); lab += [name] * len(pts)
        flat_pred.append(sdf); flat_exact.append(gd.reshape(-1)); flat_ang.append(ang)
        flat_gn.append(gn); flat_ok.append(ok)
    pred = np.concatenate(flat_pred); exact = np.concatenate(flat_exact)
    return dict(
        P=np.vstack(P), lab=np.array(lab), pq=pq, combined=combined,
        pred=pred, exact=exact, err=pred - exact,
        ang=np.concatenate(flat_ang), gn=np.concatenate(flat_gn),
        ok=np.concatenate(flat_ok),
        off_mm=offsets * 1000.0,
        bern_RS=np.vstack(bern_RS), exact_RS=np.vstack(exact_RS),
        errRS=np.vstack(errRS), angRS=np.vstack(angRS),
        gnRS=np.vstack(gnRS), okRS=np.vstack(okRS),
    )


def fig_distance(R):
    ok = R["ok"]
    err_mm = R["err"][ok] * 1000
    mae, bias = np.mean(np.abs(err_mm)), np.mean(err_mm)
    p99 = np.percentile(np.abs(err_mm), 99)
    r = np.corrcoef(R["exact"][ok], R["pred"][ok])[0, 1]
    off = R["off_mm"]
    bern_mm = np.where(R["okRS"], R["bern_RS"], np.nan) * 1000.0
    exact_mm_mean = R["exact_RS"].mean(0) * 1000.0
    fig, ax = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 2.9))

    # (a) distance-along-normal profile
    for ray in bern_mm:
        ax[0].plot(off, ray, color=BLUE, alpha=0.06, lw=0.7, zorder=1)
    ax[0].plot(off, np.nanmean(bern_mm, 0), color=BLUE, lw=2.0, label="SDF Bernstein (moyenne)", zorder=4)
    ax[0].plot(off, exact_mm_mean, color="black", lw=1.5, ls="--", label="distance exacte (moyenne)", zorder=5)
    ax[0].set_xlabel("Distance à la surface du robot [mm]"); ax[0].set_ylabel("Distance SDF [mm]")
    ts.legend_top(ax[0], ncol=1)
    label(ax[0], "(a)", x=-0.22)
    ax[1].hist(err_mm, bins=60, color=BLUE, alpha=0.85)
    ax[1].axvline(0, color=GREY, ls="--", lw=1.2)
    ax[1].axvline(bias, color=RED, lw=1.6, label=f"biais = {bias:+.2f} mm")
    ax[1].set_xlabel("Erreur signée (Bernstein $-$ exacte) [mm]"); ax[1].set_ylabel("Nombre de points")
    ts.stats_box(ax[1],
                 f"MAE = {mae:.2f} mm\nRMSE = {np.sqrt(np.mean(err_mm**2)):.2f} mm"
                 f"\np99 $|$err$|$ = {p99:.1f} mm\n$R$ = {r:.4f}",
                 loc="upper right")
    ts.legend_top(ax[1], ncol=1)
    label(ax[1], "(b)", x=-0.22)
    fig.tight_layout(); ts.save(fig, OUT, "fig1_distance_accuracy")
    return mae, bias, r


def fig_gradient(R):
    off = R["off_mm"]
    ang = np.where(R["okRS"], R["angRS"], np.nan)    # (rays, offsets)
    gn = np.where(R["okRS"], R["gnRS"], np.nan)
    med_by = np.nanmedian(ang, 0); mean_by = np.nanmean(ang, 0)
    lo, hi = np.nanpercentile(ang, 25, 0), np.nanpercentile(ang, 75, 0)
    p10, p90 = np.nanpercentile(ang, 10, 0), np.nanpercentile(ang, 90, 0)
    ok = R["ok"]
    gmed, gmean = np.median(R["ang"][ok]), np.mean(R["ang"][ok])
    fig, ax = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 2.9))

    # (a) direction error
    ax[0].fill_between(off, p10, p90, color=BLUE, alpha=0.12, label="percentiles 10–90", zorder=1)
    ax[0].fill_between(off, lo, hi, color=BLUE, alpha=0.28, label="écart interquartile", zorder=2)
    ax[0].plot(off, med_by, color=RED, lw=2.0, label=f"médiane (globale {gmed:.1f}$^\\circ$)", zorder=4)
    ax[0].plot(off, mean_by, color=GREEN, lw=1.6, ls="--", label=f"moyenne (globale {gmean:.1f}$^\\circ$)", zorder=3)
    ax[0].axvline(D_SAFE_MM, color=GREY, ls=":", lw=1.4, zorder=2)
    ax[0].text(D_SAFE_MM + 0.8, 0.97, "$d_{safe}$", color=GREY, fontsize=8.5,
               transform=ax[0].get_xaxis_transform(), va="top")
    ax[0].set_xlabel("Distance à la surface du robot [mm]")
    ax[0].set_ylabel("Erreur de direction [$^\\circ$]")
    ax[0].set_xlim(off.min(), off.max())
    ax[0].set_ylim(0, max(18.0, np.nanmax(p90) * 1.15))
    ts.legend_top(ax[0], ncol=2)
    label(ax[0], "(a)", x=-0.22)

    # (b) gradient norm (Eikonal property)
    gn_med = np.nanmedian(gn, 0)
    gn_lo, gn_hi = np.nanpercentile(gn, 25, 0), np.nanpercentile(gn, 75, 0)
    ax[1].fill_between(off, gn_lo, gn_hi, color=BLUE, alpha=0.28, label="écart interquartile", zorder=2)
    ax[1].plot(off, gn_med, color=RED, lw=2.0, label="médiane", zorder=4)
    ax[1].axhline(1.0, color="black", ls="--", lw=1.2, label="idéal ($\\|\\nabla h\\| = 1$)", zorder=3)
    ax[1].axvline(D_SAFE_MM, color=GREY, ls=":", lw=1.4, zorder=2)
    ax[1].set_xlabel("Distance à la surface du robot [mm]")
    ax[1].set_ylabel("Norme du gradient $\\|\\nabla h\\|$")
    ax[1].set_xlim(off.min(), off.max())
    ts.legend_top(ax[1], ncol=2)
    label(ax[1], "(b)", x=-0.22)
    fig.tight_layout(); ts.save(fig, OUT, "fig2_gradient_accuracy")
    return gmed, gmean, R["gn"][ok].mean()


def time_ms(fn, reps=40, warm=5, cuda=False):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        if cuda: torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if cuda: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return np.median(ts)


def fig_speed(R, core, pose, q9, dev):
    # Extend well past the production operating point (K = 100 critical points) so the
    # plateau is shown for what it is: fixed per-call cost (FK ~2.5 ms + kernel launch +
    # autograd + transfers) dominating until the GPU saturates around ~4k points.
    sizes = [128, 256, 512, 1024, 4096, 16384, 65536]
    P = R["P"]; pq = R["pq"]
    rng = np.random.default_rng(7)
    bern, mesh = [], []
    for n in sizes:
        batch = P[rng.integers(0, len(P), size=n)]
        bern.append(time_ms(lambda: V.evaluate_sdf_and_grad(core, batch, pose, q9, 2 ** 17, dev),
                            cuda=(dev.type == "cuda")))
        # the mesh query is O(n) and slow: fewer reps at large n keep the run tractable
        reps, warm = (3, 1) if n >= 4096 else (20, 3)
        mesh.append(time_ms(lambda: pq.on_surface(batch), reps=reps, warm=warm))
    bern, mesh = np.array(bern), np.array(mesh)
    speed = mesh / bern
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(sizes, mesh, "-o", color=RED, lw=1.8, ms=4.5, label="vérité terrain maillage (trimesh, CPU)")
    ax.plot(sizes, bern, "-o", color=BLUE, lw=1.8, ms=4.5, label="SDF Bernstein (analytique, GPU)")
    ax.axhline(1000 / 80, color=GREY, ls="--", lw=1.2)
    ax.text(sizes[0] * 0.92, 1000 / 80 * 1.25, "budget 80 Hz (12,5 ms)",
            color=GREY, ha="left", va="bottom", fontsize=8)
    ax.axvline(100, color=GREEN, ls=":", lw=1.4)
    ax.text(100 * 1.1, 0.58, "$K = 100$", color=GREEN, fontsize=8,
            ha="left", va="bottom", transform=ax.get_xaxis_transform())
    for n, mv, s in zip(sizes, mesh, speed):
        ax.annotate(f"{s:.0f}$\\times$", (n, mv), textcoords="offset points",
                    xytext=(0, 6), ha="center", va="bottom", color=RED, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], rotation=30)
    ax.minorticks_off()
    ax.set_xlabel("Nombre de points de requête (échelle log)")
    ax.set_ylabel("Temps distance + gradient [ms]")
    ts.legend_top(ax, ncol=1)
    fig.tight_layout(); ts.save(fig, OUT, "fig3_speed")
    return sizes, bern, mesh, speed


def make_table(R):
    ok = R["ok"]
    rows = []
    for name in list(dict.fromkeys(R["lab"])):
        m = (R["lab"] == name) & ok
        rows.append((name.replace("panda_", "").replace("_", r"\_"),
                     int(m.sum()), np.mean(np.abs(R["err"][m])) * 1000, np.mean(R["err"][m]) * 1000,
                     np.median(R["ang"][m]), R["gn"][m].mean()))
    allrow = ("\\textbf{all}", int(ok.sum()), np.mean(np.abs(R["err"][ok])) * 1000, np.mean(R["err"][ok]) * 1000,
              np.median(R["ang"][ok]), R["gn"][ok].mean())
    # CSV
    with open(os.path.join(OUT, "table_per_link.csv"), "w") as f:
        f.write("link,n_points,dist_MAE_mm,dist_bias_mm,grad_median_deg,grad_norm_mean\n")
        for r in rows + [("all",) + allrow[1:]]:
            f.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f}\n")
    # LaTeX
    with open(os.path.join(OUT, "table_per_link.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Link & $N$ & MAE [mm] & Bias [mm] & Grad.\\ median [$^\\circ$] & $\\overline{\\|\\nabla\\|}$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]} & {r[2]:.2f} & {r[3]:+.2f} & {r[4]:.1f} & {r[5]:.2f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"{allrow[0]} & {allrow[1]} & {allrow[2]:.2f} & {allrow[3]:+.2f} & {allrow[4]:.1f} & {allrow[5]:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rows, allrow


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    links = V.CBF_PROTECTED_LINKS
    rl, w, core = V.build_sdf_stack(dev, links)
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)
    meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)
    R = collect(core, rl, meshes, links, pose, q9, dev)
    n_excl = int((~R["ok"]).sum())
    print(f"Collected {len(R['P'])} genuine exterior points across {len(links)} links")
    print(f"Excluded {n_excl} samples ({100.0 * n_excl / len(R['P']):.1f}%) outside the "
          f"per-link Bernstein training domain (model returns a bound there, not the learned SDF)")
    mae, bias, r = fig_distance(R)
    gmed, gmean, gnorm = fig_gradient(R)
    sizes, bern, mesh, speed = fig_speed(R, core, pose, q9, dev)
    rows, allrow = make_table(R)
    print("\n==== THESIS SUMMARY ====")
    print(f"Distance : MAE={mae:.2f} mm  bias={bias:+.2f} mm  R={r:.4f}")
    print(f"Gradient : median={gmed:.1f} deg  mean={gmean:.1f} deg  ||grad|| mean={gnorm:.2f}")
    print(f"Speed    : sizes={sizes}")
    print(f"           bernstein_ms={np.round(bern,3).tolist()}")
    print(f"           mesh_ms     ={np.round(mesh,1).tolist()}")
    print(f"           speedup     ={np.round(speed,0).tolist()}")
    print(f"\nFigures + tables in: {OUT}")


if __name__ == "__main__":
    main()
