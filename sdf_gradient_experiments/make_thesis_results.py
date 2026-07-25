#!/usr/bin/env python3
"""Generate publication-quality thesis figures + tables for the production Bernstein SDF.

Apples-to-apples: the SDF is measured against the mesh union of EXACTLY its own links,
and those links are the 9 bodies the production CBF actually protects
(cbf_safety_node_Bernstein_multicbf.py, param ~cbf_link_names).

One figure per plot (nothing is packed into subplots, so no (a)/(b) labels).
Outputs (in ./thesis_results/):
  fig1_distance_profile.png     - SDF vs exact distance along outward normals
  fig2_distance_error.png       - signed error histogram (mean + median)
  fig3_gradient_direction.png   - angular error vs distance to the surface
  fig4_gradient_norm.png        - ||grad h|| vs distance (Eikonal property)
  fig5_speed.png                - Bernstein vs mesh timing, 20 trials -> error bars
  fig6_per_link_distance.png    - per-link boxplot of |error| + MAE + p99
  fig7_per_link_gradient.png    - per-link boxplot of the gradient angular error
  fig8_surface_error_map.png    - 3D robot mesh coloured by the local SDF error
  table_per_link.tex / .csv     - per-link distance & gradient error

The expensive parts (ray benchmark, timing sweep, surface error field) are cached in
./thesis_results/cache/ so figure styling can be iterated without recomputing.
Reads production models read-only; writes nothing outside this folder.

Usage:
    python3 make_thesis_results.py                 # all figures (uses cache if present)
    python3 make_thesis_results.py --only 1,2,6    # restyle a few figures only
    python3 make_thesis_results.py --recompute all # force every measurement again
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import trimesh

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thesis_results")
CACHE = os.path.join(OUT, "cache")
PKG = os.path.join(SDF_BERNSTEIN, "..", "..")
sys.path.insert(0, PKG); sys.path.insert(0, SDF_BERNSTEIN)
sys.path.insert(0, os.path.join(PKG, "src", "vision_processing"))
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V

sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import thesis_style as ts
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ts.apply()
BLUE, ORANGE, GREEN, GREY = ts.BLUE, ts.ORANGE, ts.GREEN, ts.GREY
SKY, PURPLE, LIGHTGREY = ts.SKY, ts.PURPLE, ts.LIGHTGREY

# Semantic roles, identical in EVERY figure of this set:
#   BLUE   = the model under test (Bernstein SDF) / distribution body
#   GREEN  = mean
#   ORANGE = median  (and the mesh ground-truth baseline in the timing figure)
#   black  = exact / ideal reference
C_MEAN, C_MEDIAN = GREEN, ORANGE

# The 9 bodies the production CBF protects; see
# test_novelty/cbf_safety_node_Bernstein_multicbf.py (~cbf_link_names default).
CBF_LINKS = [
    "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7",
    "panda_hand", "panda_leftfinger", "panda_rightfinger", "fork_tip",
]
SHORT = {
    "panda_link3": "L3", "panda_link4": "L4", "panda_link5": "L5",
    "panda_link6": "L6", "panda_link7": "L7", "panda_hand": "Main",
    "panda_leftfinger": "Doigt G", "panda_rightfinger": "Doigt D",
    "fork_tip": "Fourchette",
}

D_SAFE_MM = 15.0
D_SAFE = D_SAFE_MM / 1000.0
RAYS_PER_LINK = 40
RIDGE_MM = 1.0             # medial-surface guard for the gradient-direction benchmark
SPEED_TRIALS = 20          # repetitions of the whole timing test -> error bars

# Error colour scale of the 3D surface map: green < 1 mm, yellow ~ 3 mm, red >= 6 mm.
ERR_VMAX_MM = 6.0
ERR_CMAP = LinearSegmentedColormap.from_list("sdf_err", [
    (0.00, "#1a9850"),   # 0 mm   deep green
    (1.0 / ERR_VMAX_MM, "#66bd63"),   # 1 mm   green
    (2.0 / ERR_VMAX_MM, "#a6d96a"),   # 2 mm
    (3.0 / ERR_VMAX_MM, "#fee08b"),   # 3 mm   yellow
    (4.5 / ERR_VMAX_MM, "#fdae61"),   # 4.5 mm orange
    (1.00, "#d73027"),   # 6 mm   red
])
ERR_CMAP.set_over("#7f0000")
ERR_CMAP.set_bad(LIGHTGREY)

SINGLE = (5.0, 3.4)                 # standard single-plot size
WIDE = (ts.TEXTWIDTH, 3.5)          # 9 categories need the full text width


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
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


class Union:
    """Exact distance / gradient / outside-test against the mesh union of the links.

    Same quantities as V.ground_truth_distance_and_gradient, but on an Embree BVH per
    link instead of trimesh's pure-python proximity: 0.004 ms per point against 15 ms,
    i.e. the difference between a 40-minute benchmark and a 10-second one. Distances
    agree with the trimesh path to 0.03 mm over the sample geometry used here.

    Caveat on `outside`: three of the Panda meshes (link6, link7, hand) are not
    watertight, so the nearest-triangle normal test that both libraries use is
    genuinely ambiguous at edge/vertex hits and the two disagree on ~11% of points.
    It is therefore used only where there is no better signal (the surface map);
    ray acceptance relies on the unambiguous distance-along-the-ray test instead."""

    def __init__(self, meshes):
        import open3d as o3d
        self.o3d = o3d
        self.meshes = meshes
        self.scenes = []
        for m in meshes:
            s = o3d.t.geometry.RaycastingScene()
            s.add_triangles(o3d.core.Tensor(np.asarray(m.vertices), o3d.core.float32),
                            o3d.core.Tensor(np.asarray(m.faces), o3d.core.uint32))
            self.scenes.append(s)

    def query(self, P):
        P = np.ascontiguousarray(P, dtype=np.float32)
        K, N = len(self.meshes), len(P)
        D = np.empty((K, N), np.float64)
        G = np.empty((K, N, 3), np.float64)
        inside = np.zeros((K, N), bool)
        q = self.o3d.core.Tensor(P, self.o3d.core.float32)
        for k, (m, scene) in enumerate(zip(self.meshes, self.scenes)):
            hit = scene.compute_closest_points(q)
            disp = P - hit["points"].numpy()
            nrm = m.face_normals[hit["primitive_ids"].numpy()]
            inside[k] = np.einsum("ij,ij->i", disp, nrm) < 0.0
            norm = np.linalg.norm(disp, axis=1)
            ok = norm > 1e-10
            D[k] = norm
            G[k] = np.where(ok[:, None], disp / np.maximum(norm, 1e-12)[:, None], nrm)
        near = D.argmin(0)
        idx = np.arange(N)
        # Gap to the SECOND nearest body. Where it vanishes the point sits on the medial
        # surface between two links and the EXACT gradient is discontinuous there (an
        # arbitrarily small tie flips the reference direction by up to 180 deg), so the
        # direction benchmark has to drop those samples — see RIDGE_MM.
        gap = (np.sort(D, axis=0)[1] - D[near, idx]) if K > 1 else np.full(N, np.inf)
        return (D[near, idx].astype(np.float32), G[near, idx].astype(np.float32),
                ~inside.any(0), gap.astype(np.float32))


def rays_for_link(union, mesh, offsets, want, seed=123):
    """Verified outward normal rays launched from one link.

    Acceptance test: the marched offset must equal the true union distance (<3 mm) at
    EVERY one of the 25 samples, so the ray probes a clean free-space corridor. Sampling
    the whole ray at 2 mm makes this test alone sufficient — a ray that entered a body
    would have to cross its surface, where the union distance collapses to zero — so we
    do not lean on the ambiguous inside/outside flag (see Union).

    Candidates are validated in batches, in two stages, rather than one proximity query
    each. That matters here: the thin bodies (fingers, fork) sit inside the 5 cm envelope
    of their neighbours, so only a small fraction of their outward normals survive and
    they need heavy oversampling."""
    rng = np.random.default_rng(seed)
    S = len(offsets)
    probe = np.array([S // 2, S - 1])

    def sample(n, s):
        pts, fidx = trimesh.sample.sample_surface(mesh, int(n), seed=s)
        return pts[:, None, :] + offsets[None, :, None] * mesh.face_normals[fidx][:, None, :]

    # Stage 1 — prefilter on two samples only (mid and far). A ray that is clean over
    # its whole length is necessarily clean at those two, so no valid ray is lost, at
    # 1/12 of the proximity cost. Escalate the oversampling only if a link is starved.
    rays, cand = None, np.array([], int)
    for attempt, mult in enumerate((20, 80, 320)):
        rays = sample(want * mult, seed + attempt)
        d, _, _, _ = union.query(rays[:, probe, :].reshape(-1, 3))
        good = (np.abs(d - np.tile(offsets[probe], len(rays))) < 0.003
                ).reshape(-1, len(probe)).all(1)
        cand = np.flatnonzero(good)
        if len(cand) >= 2 * want:
            break
    if len(cand) == 0:
        raise RuntimeError("no candidate ray escapes the robot for this link")
    cand = rng.permutation(cand)[:max(4 * want, 40)]

    # Stage 2 — full 25-sample verification on the survivors only. Everything below is
    # indexed in the survivor array, so rays/d/g stay aligned without any remapping.
    sub = rays[cand]
    d, g, _, gap = union.query(sub.reshape(-1, 3))
    d, g, gap = d.reshape(-1, S), g.reshape(-1, S, 3), gap.reshape(-1, S)
    keep = np.flatnonzero((np.abs(d - offsets[None, :]) < 0.003).all(1))
    if len(keep) < 8:
        raise RuntimeError("fewer than 8 verified rays available for this link")
    sel = rng.permutation(keep)[:want]
    return sub[sel].astype(np.float32), d[sel], g[sel], gap[sel]


def collect(core, rl, meshes, links, pose, q9, dev):
    # Clean ray sampling, apples-to-apples: rays are launched PER LINK (every link
    # represented + labelled by source), exact distance/gradient measured against the
    # full union of the SDF's own links. The (ray, offset) structure is kept so we can
    # plot distance / gradient error as a function of distance from the surface.
    offsets = np.linspace(0.001, 0.05, 25, dtype=np.float32)
    union = Union(meshes)

    P, lab = [], []
    bern_RS, exact_RS, errRS, angRS, gnRS, okRS, gapRS = [], [], [], [], [], [], []
    flat_pred, flat_exact, flat_ang, flat_gn, flat_ok, flat_gap = [], [], [], [], [], []
    for name, m in zip(links, meshes):
        rp, gd, gg, gap = rays_for_link(union, m, offsets, RAYS_PER_LINK)
        R, S, _ = rp.shape
        print(f"  {name:<18s} {R:3d} rayons vérifiés")
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
        gnRS.append(gn.reshape(R, S)); okRS.append(ok.reshape(R, S)); gapRS.append(gap)
        P.append(pts); lab += [name] * len(pts)
        flat_pred.append(sdf); flat_exact.append(gd.reshape(-1)); flat_ang.append(ang)
        flat_gn.append(gn); flat_ok.append(ok); flat_gap.append(gap.reshape(-1))
    pred = np.concatenate(flat_pred); exact = np.concatenate(flat_exact)
    return dict(
        P=np.vstack(P), lab=np.array(lab),
        pred=pred, exact=exact, err=pred - exact,
        ang=np.concatenate(flat_ang), gn=np.concatenate(flat_gn),
        ok=np.concatenate(flat_ok), gap=np.concatenate(flat_gap),
        off_mm=offsets * 1000.0,
        bern_RS=np.vstack(bern_RS), exact_RS=np.vstack(exact_RS),
        errRS=np.vstack(errRS), angRS=np.vstack(angRS),
        gnRS=np.vstack(gnRS), okRS=np.vstack(okRS), gapRS=np.vstack(gapRS),
    )


def grad_masks(R):
    """Where the DIRECTION benchmark is meaningful.

    Two extra exclusions on top of the Bernstein training domain:
      * the medial surface between two links, where the exact gradient is
        discontinuous and the reference direction is therefore undefined;
      * the rare points where autograd through the basis returns a non-finite
        gradient (reported, not silently dropped).
    The DISTANCE benchmark keeps all in-domain samples: the union distance stays
    continuous across a medial ridge, only its direction does not."""
    ridge = R["gap"] * 1000.0 < RIDGE_MM
    finite = np.isfinite(R["ang"]) & np.isfinite(R["gn"])
    flat = R["ok"] & ~ridge & finite
    rs = (R["okRS"] & (R["gapRS"] * 1000.0 >= RIDGE_MM)
          & np.isfinite(R["angRS"]) & np.isfinite(R["gnRS"]))
    return flat, rs, int((R["ok"] & ridge).sum()), int((R["ok"] & ~finite).sum())


def cached(name, recompute, build):
    """np.savez cache of a dict of arrays, so styling can be iterated for free."""
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path) and not recompute:
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    data = build()
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


# --------------------------------------------------------------------------- #
# figures 1-2 : distance accuracy
# --------------------------------------------------------------------------- #
def fig_distance_profile(R):
    off = R["off_mm"]
    bern_mm = np.where(R["okRS"], R["bern_RS"], np.nan) * 1000.0
    exact_mm = R["exact_RS"].mean(0) * 1000.0
    fig, ax = plt.subplots(figsize=SINGLE)
    for ray in bern_mm:
        ax.plot(off, ray, color=SKY, alpha=0.07, lw=0.7, zorder=1)
    ax.plot([], [], color=SKY, lw=2.0, alpha=0.55, label="rayons")
    ax.plot(off, np.nanmean(bern_mm, 0), color=C_MEAN, lw=2.2,
            label="moyenne", zorder=4)
    ax.plot(off, np.nanmedian(bern_mm, 0), color=C_MEDIAN, lw=1.8, ls=(0, (5, 2)),
            label="médiane", zorder=5)
    ax.plot(off, exact_mm, color="black", lw=1.4, ls=(0, (1.5, 1.5)),
            label="distance exacte", zorder=6)
    ax.set_xlabel("Distance à la surface du robot [mm]")
    ax.set_ylabel("Distance SDF prédite [mm]")
    ax.set_xlim(off.min(), off.max())
    ts.legend_top(ax, ncol=2)
    fig.tight_layout(); ts.save(fig, OUT, "fig1_distance_profile")


def fig_distance_error(R):
    ok = R["ok"]
    err_mm = R["err"][ok] * 1000
    mae = np.mean(np.abs(err_mm)); bias = np.mean(err_mm); med = np.median(err_mm)
    p99 = np.percentile(np.abs(err_mm), 99)
    r = np.corrcoef(R["exact"][ok], R["pred"][ok])[0, 1]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.hist(err_mm, bins=60, color=BLUE, alpha=0.80, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=GREY, ls=":", lw=1.3)
    ax.axvline(bias, color=C_MEAN, lw=2.0, label=f"moyenne = {bias:+.2f} mm")
    ax.axvline(med, color=C_MEDIAN, lw=1.8, ls=(0, (5, 2)), label=f"médiane = {med:+.2f} mm")
    ax.set_xlabel("Erreur signée (Bernstein $-$ exacte) [mm]")
    ax.set_ylabel("Nombre de points")
    ts.stats_box(ax,
                 f"MAE = {mae:.2f} mm\nRMSE = {np.sqrt(np.mean(err_mm**2)):.2f} mm"
                 f"\np99 $|$err$|$ = {p99:.1f} mm\n$R$ = {r:.4f}",
                 loc="upper right")
    ts.legend_top(ax, ncol=2)
    fig.tight_layout(); ts.save(fig, OUT, "fig2_distance_error")
    return mae, bias, r


# --------------------------------------------------------------------------- #
# figures 3-4 : gradient accuracy
# --------------------------------------------------------------------------- #
def fig_gradient_direction(R):
    off = R["off_mm"]
    gok, gokRS, _, _ = grad_masks(R)
    ang = np.where(gokRS, R["angRS"], np.nan)
    med_by, mean_by = np.nanmedian(ang, 0), np.nanmean(ang, 0)
    lo, hi = np.nanpercentile(ang, 25, 0), np.nanpercentile(ang, 75, 0)
    p10, p90 = np.nanpercentile(ang, 10, 0), np.nanpercentile(ang, 90, 0)
    gmed, gmean = np.median(R["ang"][gok]), np.mean(R["ang"][gok])
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.fill_between(off, p10, p90, color=BLUE, alpha=0.13, lw=0,
                    label="percentiles 10–90", zorder=1)
    ax.fill_between(off, lo, hi, color=BLUE, alpha=0.30, lw=0,
                    label="écart interquartile", zorder=2)
    ax.plot(off, mean_by, color=C_MEAN, lw=2.0,
            label=f"moyenne (globale {gmean:.1f}$^\\circ$)", zorder=4)
    ax.plot(off, med_by, color=C_MEDIAN, lw=1.8, ls=(0, (5, 2)),
            label=f"médiane (globale {gmed:.1f}$^\\circ$)", zorder=5)
    ax.axvline(D_SAFE_MM, color=GREY, ls=":", lw=1.3, zorder=2)
    ax.text(D_SAFE_MM + 0.9, 0.97, "$d_{\\mathrm{safe}}$", color=GREY, fontsize=10,
            transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlabel("Distance à la surface du robot [mm]")
    ax.set_ylabel("Erreur de direction [$^\\circ$]")
    ax.set_xlim(off.min(), off.max())
    ax.set_ylim(0, max(18.0, np.nanmax(p90) * 1.12))
    ts.legend_top(ax, ncol=2)
    fig.tight_layout(); ts.save(fig, OUT, "fig3_gradient_direction")
    return gmed, gmean


def fig_gradient_norm(R):
    off = R["off_mm"]
    gok, gokRS, _, _ = grad_masks(R)
    gn = np.where(gokRS, R["gnRS"], np.nan)
    lo, hi = np.nanpercentile(gn, 25, 0), np.nanpercentile(gn, 75, 0)
    p10, p90 = np.nanpercentile(gn, 10, 0), np.nanpercentile(gn, 90, 0)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.fill_between(off, p10, p90, color=BLUE, alpha=0.13, lw=0,
                    label="percentiles 10–90", zorder=1)
    ax.fill_between(off, lo, hi, color=BLUE, alpha=0.30, lw=0,
                    label="écart interquartile", zorder=2)
    ax.plot(off, np.nanmean(gn, 0), color=C_MEAN, lw=2.0, label="moyenne", zorder=4)
    ax.plot(off, np.nanmedian(gn, 0), color=C_MEDIAN, lw=1.8, ls=(0, (5, 2)),
            label="médiane", zorder=5)
    ax.axhline(1.0, color="black", ls=(0, (1.5, 1.5)), lw=1.4,
               label="idéal ($\\|\\nabla h\\| = 1$)", zorder=6)
    ax.axvline(D_SAFE_MM, color=GREY, ls=":", lw=1.3, zorder=2)
    ax.text(D_SAFE_MM + 0.9, 0.03, "$d_{\\mathrm{safe}}$", color=GREY, fontsize=10,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_xlabel("Distance à la surface du robot [mm]")
    ax.set_ylabel("Norme du gradient $\\|\\nabla h\\|$")
    ax.set_xlim(off.min(), off.max())
    ts.legend_top(ax, ncol=2)
    fig.tight_layout(); ts.save(fig, OUT, "fig4_gradient_norm")
    return R["gn"][gok].mean()


# --------------------------------------------------------------------------- #
# figure 5 : speed, 20 trials
# --------------------------------------------------------------------------- #
def time_trials(fn, trials, warm=3, cuda=False):
    """`trials` independent timings of one call -> a distribution, not a point."""
    for _ in range(warm):
        fn()
    out = []
    for _ in range(trials):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000.0)
    return np.asarray(out)


class AnalyticalSDF:
    """Distance + world-frame spatial gradient with NO autograd, FK hoisted, on-GPU.

    This is what the SDF actually costs to query: the analytical Bernstein derivative
    (analytical_bernstein.py, the production no-autograd path) instead of an autograd
    backward pass, forward kinematics computed ONCE (q is fixed for the whole sweep),
    and the query batch kept resident on the GPU so no host<->device copy is timed.

    Validated against the autograd BernsteinCore on the benchmark ray geometry: the
    distances agree to 1e-4 mm and the gradient directions to <0.04 deg (norm ratio
    1.00000). It is the same function, only cheaper to evaluate."""

    def __init__(self, core, dev):
        import analytical_bernstein as AB
        self.core = core
        self.dev = dev
        self.a = AB.AnalyticalBernsteinSoftmin(core, temperature=1.0, d_safe=0.0)
        dof = int(core.robot.dof)
        if dof > 7:   # match the finger position BernsteinCore uses (q9 fingers = 0.001)
            self.a.q_extra = torch.full((dof - 7,), 0.001, dtype=torch.float32, device=dev)
        self._group_idx = [(g["indices_tensor"].to(dev), g) for g in core.groups.values()]

    def prepare(self, q7, pose):
        """Hoisted forward kinematics: q and pose are constant across the whole sweep."""
        tr, _, _, _ = self.a._visual_kinematics(q7, pose)
        self.R = tr[..., :3, :3].contiguous()          # [B,K,3,3]
        self.t = tr[..., :3, 3].contiguous()           # [B,K,3]

    def query(self, points):
        """points: [M,3] tensor already on the GPU. Returns distance [M], grad [M,3]."""
        diff = points[None, None] - self.t[:, :, None, :]     # [B,K,M,3]
        local = torch.matmul(diff, self.R)                    # R^T @ diff -> link frame
        B, K, M = 1, self.R.shape[1], points.shape[0]
        sdf = points.new_empty((B, K, M))
        gloc = points.new_empty((B, K, M, 3))
        for idx, group in self._group_idx:
            s, g = self.a._evaluate_rdf_group(torch.index_select(local, 1, idx), group)
            sdf.index_copy_(1, idx, s)
            gloc.index_copy_(1, idx, g)
        gworld = torch.einsum("bkij,bkmj->bkmi", self.R, gloc)   # link frame -> world
        dist, arg = sdf.min(dim=1)                              # min over links, per point
        gmin = gworld.gather(1, arg[:, None, :, None].expand(-1, 1, -1, 3)).squeeze(1)
        return dist[0], gmin[0]


class BernsteinDistance:
    """Distance ONLY — no gradient — with FK hoisted and points GPU-resident.

    Many uses of the SDF need only the clearance value (candidate pruning, an obstacle
    within d_safe test, the barrier h itself), never its direction. Dropping the gradient
    removes the whole basis-derivative + world-rotation + per-link gather, leaving just
    the value-only forward BernsteinCore already exposes. FK is passed pre-computed via
    the link_poses kwarg so the sequential kinematic-tree walk is skipped every call."""

    def __init__(self, core, dev):
        self.core = core
        self.dev = dev

    def prepare(self, q9, pose):
        self.pose, self.theta = pose, q9
        self.link_poses = self.core.robot._native_forward_kinematics(
            self.core._pad_theta(q9))

    def query(self, points):
        """points: [M,3] tensor on the GPU. Returns distance [M] (no gradient)."""
        return self.core.get_whole_body_sdf_batch(
            points, self.pose, self.theta, link_poses=self.link_poses)[0]


def measure_speed(R, meshes, core, pose, q9, dev, prior=None):
    # Extend well past the production operating point (K = 100 critical points) so the
    # plateau is shown for what it is: fixed per-call cost (FK ~2.5 ms + kernel launch +
    # autograd + transfers) dominating until the GPU saturates around ~4k points.
    #
    # TWO mesh baselines, because they answer different questions. trimesh is the
    # reference implementation this work was compared against, but it is pure
    # python/numpy; Embree (via open3d) is what an optimised CPU mesh query actually
    # costs. Quoting only the first would credit the SDF for someone else's slow code.
    sizes = np.array([128, 256, 512, 1024, 4096, 16384, 65536])
    P = R["P"]
    rng = np.random.default_rng(7)

    # The mesh baselines are expensive (trimesh alone is ~25 s at the largest size) but
    # they never change, so reuse a matching cached measurement and re-time only the SDF.
    reuse = (prior is not None and "mesh" in prior and "embree" in prior
             and np.array_equal(prior.get("sizes"), sizes))
    if reuse:
        print("  (reutilise les courbes maillage/Embree en cache; recalcul du SDF seul)")
        mesh, embree = list(prior["mesh"]), list(prior["embree"])
    else:
        combined = trimesh.util.concatenate(meshes)
        pq = trimesh.proximity.ProximityQuery(combined)
        union = Union([combined])
        mesh, embree = [], []

    sdf = AnalyticalSDF(core, dev)
    q7 = torch.as_tensor(V.DEFAULT_Q, dtype=torch.float32, device=dev).reshape(1, 7)
    sdf.prepare(q7, pose)
    sdf_d = BernsteinDistance(core, dev)
    sdf_d.prepare(q9, pose)

    bern, bern_d = [], []
    for i, n in enumerate(sizes):
        idx = rng.integers(0, len(P), size=n)
        batch = P[idx]
        batch_gpu = torch.as_tensor(batch, dtype=torch.float32, device=dev)  # resident
        t0 = time.perf_counter()
        bern.append(time_trials(lambda: sdf.query(batch_gpu),
                                SPEED_TRIALS, cuda=(dev.type == "cuda")))
        bern_d.append(time_trials(lambda: sdf_d.query(batch_gpu),
                                  SPEED_TRIALS, cuda=(dev.type == "cuda")))
        if not reuse:
            embree.append(time_trials(lambda: union.query(batch), SPEED_TRIALS))
            # trimesh is O(n) and slow, but the 20 trials are the point of the figure
            mesh.append(time_trials(lambda: pq.on_surface(batch), SPEED_TRIALS, warm=1))
        me = mesh[i].mean() if hasattr(mesh[i], "mean") else float(np.mean(mesh[i]))
        print(f"  n={n:6d}  bern d+grad {bern[-1].mean():6.2f}   dist-only "
              f"{bern_d[-1].mean():6.2f}   embree {np.mean(embree[i]):6.2f}   "
              f"trimesh {me:8.1f} ms   ({time.perf_counter()-t0:.0f} s)")
    return dict(sizes=sizes, bern=np.stack(bern), bern_d=np.stack(bern_d),
                mesh=np.stack(mesh), embree=np.stack(embree))


def fig_speed(S):
    sizes, bern, mesh, embree = S["sizes"], S["bern"], S["mesh"], S["embree"]
    bern_d = S["bern_d"]
    bm, bd = bern.mean(1), bern_d.mean(1)
    mm, em = mesh.mean(1), embree.mean(1)
    speed, speed_de = mm / bd, em / bd     # both against the distance-only SDF
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    series = ((mesh, ORANGE, "maillage, trimesh (numpy)", "-o"),
              (embree, GREY, "maillage, Embree (C++)", "-s"),
              (bern_d, BLUE, "SDF Bernstein (GPU)", "-o"))
    for raw, c, lbl, fmt in series:
        y, sd = raw.mean(1), raw.std(1)
        ax.fill_between(sizes, raw.min(1), raw.max(1), color=c, alpha=0.16, lw=0)
        ax.errorbar(sizes, y, yerr=np.vstack([np.minimum(sd, y * 0.95), sd]),
                    fmt=fmt, color=c, lw=1.8, ms=4.0, capsize=3, elinewidth=1.2,
                    label=lbl)
    # trimesh speedup on its curve (orange); the headline "beats C++" factor of the
    # distance-only SDF against Embree, printed on the Embree curve (grey).
    for n, mv, s in zip(sizes, mm, speed):
        ax.annotate(f"{s:.0f}$\\times$", (n, mv), textcoords="offset points",
                    xytext=(0, 8), ha="center", va="bottom", color=ORANGE, fontsize=8.5)
    for n, ev, s in zip(sizes, em, speed_de):
        ax.annotate(f"{s:.1f}$\\times$", (n, ev), textcoords="offset points",
                    xytext=(0, -13), ha="center", va="bottom", color=GREY, fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], rotation=30)
    ax.minorticks_off()
    ax.set_xlabel("Nombre de points de requête")
    ax.set_ylabel("Temps de calcul [ms]")
    ts.legend_top(ax, ncol=2)
    fig.tight_layout(); ts.save(fig, OUT, "fig5_speed")
    return sizes, bm, bd, mm, em, speed, speed_de


# --------------------------------------------------------------------------- #
# figures 6-7 : per-link boxplots
# --------------------------------------------------------------------------- #
def _style_boxes(bp, face, edge):
    for b in bp["boxes"]:
        b.set(facecolor=face, edgecolor=edge, linewidth=1.0, alpha=0.85)
    for k in ("whiskers", "caps"):
        for a in bp[k]:
            a.set(color=edge, linewidth=1.0)
    for m in bp["medians"]:
        m.set(color="white", linewidth=1.6)


def _draw_outliers(ax, data, positions, edge, whis=(5, 95)):
    """Show the beyond-whisker points as a single vertical line of larger,
    half-transparent markers centred on the box: no horizontal jitter (which reads as
    noise), and overlapping markers darken where outliers are dense."""
    for x, d in zip(positions, data):
        lo, hi = np.percentile(d, whis)
        out = d[(d < lo) | (d > hi)]
        if len(out) == 0:
            continue
        ax.scatter(np.full(len(out), x), out, s=13, facecolor=edge,
                   edgecolor="none", alpha=0.4, zorder=3)


def _link_groups(R, key, links, mask=None):
    m = R["ok"] if mask is None else mask
    return [R[key][(R["lab"] == n) & m] for n in links]


def fig_per_link_distance(R, links):
    data = [np.abs(d) * 1000.0 for d in _link_groups(R, "err", links)]
    mae = [d.mean() for d in data]
    p99 = [np.percentile(d, 99) for d in data]
    x = np.arange(1, len(links) + 1)
    fig, ax = plt.subplots(figsize=WIDE)
    bp = ax.boxplot(data, positions=x, widths=0.58, patch_artist=True, showfliers=False,
                    whis=(5, 95))
    _style_boxes(bp, BLUE, "#00537F")
    _draw_outliers(ax, data, x, "#00537F")
    ax.plot(x, mae, "D", color=C_MEAN, ms=6.5, mec="white", mew=0.8, zorder=5,
            label="MAE (moyenne)")
    ax.plot(x, p99, "^", color=C_MEDIAN, ms=7.5, mec="white", mew=0.8, zorder=5,
            label="centile 99")
    ax.plot([], [], "s", color=BLUE, alpha=0.85, ms=7, label="boîte : quartiles ; moustaches : 5–95 %")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in links], rotation=25, ha="right")
    ax.set_xlim(0.4, len(links) + 0.6)
    ax.set_ylabel("Erreur absolue de distance [mm]")
    ax.set_xlabel("Corps protégé")
    ts.legend_top(ax, ncol=3)
    fig.tight_layout(); ts.save(fig, OUT, "fig6_per_link_distance")


def fig_per_link_gradient(R, links):
    data = _link_groups(R, "ang", links, grad_masks(R)[0])
    mean = [d.mean() for d in data]
    p99 = [np.percentile(d, 99) for d in data]
    x = np.arange(1, len(links) + 1)
    fig, ax = plt.subplots(figsize=WIDE)
    bp = ax.boxplot(data, positions=x, widths=0.58, patch_artist=True, showfliers=False,
                    whis=(5, 95))
    _style_boxes(bp, PURPLE, "#8C4470")
    _draw_outliers(ax, data, x, "#8C4470")
    ax.plot(x, mean, "D", color=C_MEAN, ms=6.5, mec="white", mew=0.8, zorder=5,
            label="moyenne")
    ax.plot(x, p99, "^", color=C_MEDIAN, ms=7.5, mec="white", mew=0.8, zorder=5,
            label="centile 99")
    ax.plot([], [], "s", color=PURPLE, alpha=0.85, ms=7, label="boîte : quartiles ; moustaches : 5–95 %")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in links], rotation=25, ha="right")
    ax.set_xlim(0.4, len(links) + 0.6)
    ax.set_ylabel("Erreur angulaire du gradient [$^\\circ$]")
    ax.set_xlabel("Corps protégé")
    ts.legend_top(ax, ncol=3)
    fig.tight_layout(); ts.save(fig, OUT, "fig7_per_link_gradient")


# --------------------------------------------------------------------------- #
# figure 8 : 3D surface error map
# --------------------------------------------------------------------------- #
def decimate(mesh, target_faces):
    """Quadric decimation via open3d; the full 131k-triangle robot is far too heavy
    for a matplotlib 3D collection and the error field is smooth anyway."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        m = m.simplify_quadric_decimation(int(target_faces))
        return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                               process=False)
    except Exception as exc:                                     # pragma: no cover
        print(f"  [warn] decimation unavailable ({exc}); rendering the full mesh")
        return mesh


# Every moving body: the surface map shows the whole arm, not just the 9 the CBF
# protects. panda_link0 is left out — it is the bolted-down base, it never moves, so its
# SDF fidelity cannot affect a collision and its flat mounting plate would otherwise be
# the loudest thing in the figure.
SURFACE_LINKS = [n for n in V.ALL_LINKS if n != "panda_link0"]


def build_full_robot(dev):
    """SDF stack + meshes for the surface map (every link except the fixed base)."""
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)
    rl, w, core = V.build_sdf_stack(dev, SURFACE_LINKS)
    return core, rl, V.build_ground_truth_meshes(w, rl, pose, q9, SURFACE_LINKS)


def measure_surface_error(core, rl, meshes, pose, q9, dev, target_faces=6000,
                          probe_offset=0.0):
    """Local SDF error ON the robot skin.

    Each triangle of the (decimated) robot is probed at its centroid and the error there
    is painted back onto that triangle. The probe sits ON the surface by default, where
    the reference is unambiguous — the exact union distance is zero, so the colour is
    literally how far the learned zero-level-set has drifted from the real skin.

    Pushing the probe outward instead (probe_offset = D_SAFE, the distance the CBF
    actually reads) is defensible but much noisier to look at: at every joint the outward
    normal of one link points straight into its neighbour, so the probe lands a
    millimetre from a different body and the panel fills with junction bands that say
    more about the mesh seams than about the model."""
    tris, probes = [], []
    for mesh in meshes:
        d = decimate(mesh, target_faces)
        tris.append(d.triangles.astype(np.float32))
        probes.append((d.triangles_center + probe_offset * d.face_normals).astype(np.float32))
    tris = np.concatenate(tris)
    probe = np.concatenate(probes)

    # ONE ground-truth pass over all triangles: the union query is O(points x links),
    # so calling it per link would cost 12x more for exactly the same answer.
    print(f"  {len(probe)} triangles sondés à {probe_offset * 1000:.0f} mm de la surface")
    exact, _, _, _ = Union(meshes).query(probe)
    pred = V.evaluate_sdf(core, probe, pose, q9, 32768, dev)
    valid = domain_valid_mask(core, rl, probe, pose, q9, dev)
    return dict(tris=tris, err=np.abs(pred - exact) * 1000.0, valid=valid)


def _view_dir(elev, azim):
    """Unit vector from the scene toward an orthographic matplotlib 3D camera."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def fig_surface_error(H, views=((26.0, -60.0), (26.0, 120.0))):
    tris, err, valid = H["tris"], H["err"], H["valid"].astype(bool)
    norm = Normalize(vmin=0.0, vmax=ERR_VMAX_MM)
    base = ERR_CMAP(norm(np.where(valid, err, np.nan)))

    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12, None)

    fig = plt.figure(figsize=(ts.TEXTWIDTH, 3.4))
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.02 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        # Back-face culling. matplotlib 3D has no depth buffer — it painter-sorts whole
        # polygons — so without this the inside of every hollow link bleeds through the
        # outer skin as spurious dark patches. Dropping away-facing triangles also halves
        # what has to be drawn.
        camera = _view_dir(elev, azim)
        front = (nrm @ camera) > 0.0
        # Mild Lambert shading from a head-light (0.78-1.0) so the 3D form reads without
        # shifting the hue enough to mislead the colour scale.
        light = camera + np.array([0.0, 0.0, 0.45])
        light /= np.linalg.norm(light)
        rgba = base[front].copy()
        rgba[:, :3] *= (0.78 + 0.22 * np.clip(nrm[front] @ light, 0.0, 1.0))[:, None]
        pc = Poly3DCollection(tris[front], facecolors=rgba, edgecolor="none", shade=False)
        pc.set_rasterized(True)
        ax.add_collection3d(pc)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        # Box aspect = the true extents (not a cube): the arm is a diagonal band, and a
        # cube would waste most of the panel on empty space.
        try:
            ax.set_box_aspect(hi - lo, zoom=1.45)
        except TypeError:
            ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=ERR_CMAP)
    cax = fig.add_axes([0.32, 0.10, 0.36, 0.040])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Erreur locale du SDF $|h - d_{\\mathrm{exacte}}|$ [mm]", labelpad=3)
    cb.set_ticks([0, 1, 2, 3, 4, 5, 6])
    cb.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.02, bottom=0.16, wspace=0.0)
    ts.save(fig, OUT, "fig8_surface_error_map")
    m = err[valid]
    return m.mean(), np.percentile(m, 99), float(valid.mean())


# --------------------------------------------------------------------------- #
def make_table(R, links):
    ok = R["ok"]
    gok = grad_masks(R)[0]

    def row(title, md, mg):
        # La MÉDIANE de |err| est ajoutée à côté de la MAE : c'est elle qui
        # décrit le corps « typique », la MAE étant tirée vers le haut par la
        # queue que chiffre déjà p99.
        e = np.abs(R["err"][md]) * 1000
        return (title, int(md.sum()), np.median(e), e.mean(),
                np.mean(R["err"][md]) * 1000,
                np.percentile(e, 99), np.median(R["ang"][mg]),
                np.percentile(R["ang"][mg], 99), R["gn"][mg].mean())

    rows = [row(SHORT[n], (R["lab"] == n) & ok, (R["lab"] == n) & gok) for n in links]
    allrow = row("Tous", ok, gok)
    hdr = ("link,n_points,dist_median_mm,dist_MAE_mm,dist_bias_mm,dist_p99_mm,"
           "grad_median_deg,grad_p99_deg,grad_norm_mean")
    with open(os.path.join(OUT, "table_per_link.csv"), "w") as f:
        f.write(hdr + "\n")
        for r in rows + [allrow]:
            f.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},"
                    f"{r[6]:.3f},{r[7]:.3f},{r[8]:.3f}\n")
    with open(os.path.join(OUT, "table_per_link.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        f.write("Corps & $N$ & Méd.\\ [mm] & MAE [mm] & Biais [mm] & p99 [mm] & "
                "$\\nabla$ méd.\\ [$^\\circ$] & $\\nabla$ p99 [$^\\circ$] & "
                "$\\overline{\\|\\nabla\\|}$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]} & {r[2]:.2f} & {r[3]:.2f} & {r[4]:+.2f} & "
                    f"{r[5]:.2f} & {r[6]:.1f} & {r[7]:.1f} & {r[8]:.2f} \\\\\n")
        f.write("\\midrule\n\\textbf{%s}" % allrow[0])
        f.write(f" & {allrow[1]} & {allrow[2]:.2f} & {allrow[3]:.2f} & {allrow[4]:+.2f} & "
                f"{allrow[5]:.2f} & {allrow[6]:.1f} & {allrow[7]:.1f} & {allrow[8]:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="all",
                    help="comma-separated figure numbers to (re)draw, e.g. 1,2,6")
    ap.add_argument("--recompute", default="",
                    help="'all', or any of bench,speed,surface — force re-measurement")
    args = ap.parse_args()
    want = (set(range(1, 9)) if args.only == "all"
            else {int(v) for v in args.only.split(",")})
    rec = args.recompute.split(",") if args.recompute else []
    redo = lambda k: "all" in rec or k in rec

    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    links = CBF_LINKS
    rl, w, core = V.build_sdf_stack(dev, links)
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)
    meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)

    print(f"\n[bench] {len(links)} corps protégés par le CBF")
    R = cached("bench", redo("bench"),
               lambda: collect(core, rl, meshes, links, pose, q9, dev))
    R["lab"] = R["lab"].astype(str)
    n = len(R["P"])
    n_excl = int((~R["ok"]).sum())
    _, _, n_ridge, n_nan = grad_masks(R)
    print(f"{n} points extérieurs; {n_excl} exclus ({100.0 * n_excl / n:.1f}%) hors du "
          f"domaine d'entraînement Bernstein")
    print(f"direction du gradient: {n_ridge} échantillons de plus écartés "
          f"({100.0 * n_ridge / n:.1f}%) sur la surface médiane entre deux corps "
          f"(< {RIDGE_MM:.0f} mm d'écart, gradient exact discontinu), "
          f"{n_nan} ({100.0 * n_nan / n:.2f}%) à gradient non fini")

    if 1 in want:
        fig_distance_profile(R)
    mae = bias = r = gmed = gmean = gnorm = None
    if 2 in want:
        mae, bias, r = fig_distance_error(R)
    if 3 in want:
        gmed, gmean = fig_gradient_direction(R)
    if 4 in want:
        gnorm = fig_gradient_norm(R)
    if 5 in want:
        print(f"\n[speed] {SPEED_TRIALS} essais par taille de lot")
        # The SDF curve is cheap and always recomputed; the mesh baselines are slow and
        # reused from cache when present (pass --recompute speed to force them too).
        path = os.path.join(CACHE, "speed.npz")
        prior = None
        if os.path.exists(path) and not redo("speed"):
            z = np.load(path)
            prior = {k: z[k] for k in z.files}
        S = measure_speed(R, meshes, core, pose, q9, dev, prior=prior)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, **S)
        sizes, bm, bd, mm, em, speed, speed_de = fig_speed(S)
    if 6 in want:
        fig_per_link_distance(R, links)
    if 7 in want:
        fig_per_link_gradient(R, links)
    if 8 in want:
        # The surface map is a picture of the ROBOT, so it covers every body the SDF
        # can represent, not just the 9 the CBF happens to protect.
        print(f"\n[surface] carte d'erreur sur la peau du robot "
              f"({len(SURFACE_LINKS)} corps mobiles)")
        H = cached("surface", redo("surface"),
                   lambda: measure_surface_error(*build_full_robot(dev), pose, q9, dev))
        smean, sp99, sfrac = fig_surface_error(H)
    make_table(R, links)

    print("\n==== THESIS SUMMARY ====")
    if mae is not None:
        print(f"Distance : MAE={mae:.2f} mm  biais={bias:+.2f} mm  R={r:.4f}")
    if gmed is not None:
        print(f"Gradient : médiane={gmed:.1f} deg  moyenne={gmean:.1f} deg")
    if gnorm is not None:
        print(f"||grad|| : moyenne={gnorm:.3f}")
    if 5 in want:
        print(f"Vitesse  : tailles          ={sizes.tolist()}")
        print(f"           bern d+grad_ms   ={np.round(bm,3).tolist()}")
        print(f"           bern dist-only_ms={np.round(bd,3).tolist()}")
        print(f"           trimesh_ms       ={np.round(mm,1).tolist()}")
        print(f"           embree_ms        ={np.round(em,3).tolist()}")
        print(f"           (d+grad) vs trimesh   ={np.round(speed,0).tolist()}")
        print(f"           (dist-only) vs Embree ={np.round(speed_de,1).tolist()}")
    if 8 in want:
        print(f"Surface  : erreur moyenne={smean:.2f} mm  p99={sp99:.2f} mm  "
              f"({100*sfrac:.0f}% des triangles valides)")
    print(f"\nFigures + tables dans : {OUT}")


if __name__ == "__main__":
    main()
