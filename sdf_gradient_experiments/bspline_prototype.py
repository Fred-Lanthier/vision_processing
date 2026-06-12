#!/usr/bin/env python3
"""ISOLATED experiment (writes nothing outside this folder, modifies no production code).

Prototype replacing the single global degree-11 Bernstein patch for ONE link with a
uniform cubic B-spline control grid: degree 3, local 4^3 support, C2 continuous.
This is still piecewise-Bernstein -- just many low-degree local patches instead of one
high-degree global one. Goal: test whether local+low-degree fixes the gradient direction
oscillation that loss-weight tuning could not.

All math is in the link's normalized frame ([-1,1] after centroid/scale), where a true
SDF has ||grad||=1, so numbers are directly comparable to the production grad_diag output
(gradient direction is invariant to the isotropic normalize transform).
"""

import argparse, os, sys
import numpy as np
import torch
import trimesh

# Read-only access to the existing package (imports only; nothing is written there).
SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
PANDA_TEST = os.path.join(SDF_BERNSTEIN, "panda_test")
sys.path.insert(0, SDF_BERNSTEIN)
from src.core.train.weight_train import build_normal_ray_training_data

DMIN, DMAX = -1.15, 1.15


def bspline_basis(t):
    t2, t3 = t * t, t * t * t
    b0 = (1 - 3 * t + 3 * t2 - t3) / 6.0
    b1 = (3 * t3 - 6 * t2 + 4) / 6.0
    b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
    b3 = t3 / 6.0
    return torch.stack([b0, b1, b2, b3], dim=-1)


def bspline_eval(P, x, G):
    h = (DMAX - DMIN) / (G - 3)
    u = (x - DMIN) / h
    i = torch.floor(u).long().clamp(0, G - 4)
    t = u - i.float()
    B = bspline_basis(t)
    ar = torch.arange(4, device=x.device)
    ix, iy, iz = i[:, 0:1] + ar, i[:, 1:2] + ar, i[:, 2:3] + ar
    Pn = P[ix[:, :, None, None], iy[:, None, :, None], iz[:, None, None, :]]
    return torch.einsum('ni,nj,nk,nijk->n', B[:, 0], B[:, 1], B[:, 2], Pn)


def grad_of(P, x, G):
    x = x.clone().detach().requires_grad_(True)
    v = bspline_eval(P, x, G)
    g = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    return v, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", default="panda_link6")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Value-only epochs before gradient losses switch on.")
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--lambda-eik", type=float, default=0.1)
    ap.add_argument("--lambda-dir", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = args.grid
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = np.load(os.path.join(PANDA_TEST, "Dataset", f"{args.link}.npy"), allow_pickle=True).item()
    nm = np.linalg.norm(ds["near_points"], axis=1) <= 1.0
    near_p = torch.tensor(ds["near_points"][nm], dtype=torch.float32, device=dev)
    near_s = torch.tensor(ds["near_sdf"][nm], dtype=torch.float32, device=dev)
    qry_p = torch.tensor(ds["query_points"], dtype=torch.float32, device=dev)
    qry_s = torch.tensor(ds["query_sdf"], dtype=torch.float32, device=dev)
    mesh = trimesh.load(os.path.join(PANDA_TEST, "Meshes", f"{args.link}.stl"), force="mesh")
    rp, rs, rd = build_normal_ray_training_data(
        mesh, ds["mesh_centroid_offset"], ds["mesh_scale_factor"],
        number_of_points=40000, max_distance_m=0.05, seed=args.seed)
    ray_p = torch.tensor(rp, dtype=torch.float32, device=dev)
    ray_d = torch.tensor(rd, dtype=torch.float32, device=dev)

    P = torch.zeros(G, G, G, device=dev, requires_grad=True)
    opt = torch.optim.Adam([P], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"Link {args.link}: B-spline grid {G}^3 = {G**3} control pts "
          f"(production = 1728 for one degree-11 patch)")

    for ep in range(args.epochs):
        opt.zero_grad()
        warm = ep < args.warmup_epochs           # value-only phase
        l_eik = 0.0 if warm else args.lambda_eik
        l_dir = 0.0 if warm else args.lambda_dir
        ni = torch.randint(0, len(near_p), (4096,), device=dev)
        qi = torch.randint(0, len(qry_p), (2048,), device=dev)
        ri = torch.randint(0, len(ray_p), (2048,), device=dev)
        v_n = bspline_eval(P, near_p[ni], G)
        v_q = bspline_eval(P, qry_p[qi], G)
        loss_val = torch.nn.functional.mse_loss(v_n, near_s[ni]) + \
                   torch.nn.functional.mse_loss(v_q, qry_s[qi])
        if warm:
            loss_eik = torch.zeros((), device=dev)
            loss_dir = torch.zeros((), device=dev)
        else:
            _, g_r = grad_of(P, ray_p[ri], G)
            _, g_e = grad_of(P, torch.cat([near_p[ni[:1024]], ray_p[ri[:1024]]], 0), G)
            loss_eik = ((g_e.norm(dim=1) - 1.0) ** 2).mean()
            ru = torch.nn.functional.normalize(g_r, dim=1)
            du = torch.nn.functional.normalize(ray_d[ri], dim=1)
            loss_dir = (1.0 - (ru * du).sum(dim=1)).mean()
        loss = loss_val + l_eik * loss_eik + l_dir * loss_dir
        loss.backward(); opt.step(); sched.step()
        if ep % 250 == 0 or ep == args.epochs - 1 or ep == args.warmup_epochs:
            ph = "warmup" if warm else "refine"
            print(f"  ep {ep:4d} [{ph}] val={loss_val.item():.5f} eik={loss_eik.item():.4f} dir={loss_dir.item():.4f}")

    # ---- diagnostic in normalized frame ----
    mn = mesh.copy()
    mn.vertices = (np.asarray(mn.vertices) - np.asarray(ds["mesh_centroid_offset"])) / float(ds["mesh_scale_factor"])
    pts, fi = trimesh.sample.sample_surface(mn, 6000, seed=3)
    nrm = mn.face_normals[fi]
    off = np.random.default_rng(3).uniform(0.005, 0.03, size=(len(pts), 1)) / float(ds["mesh_scale_factor"])
    Pext = (pts + nrm * off).astype(np.float32)
    closest, _, _ = trimesh.proximity.closest_point(mn, Pext)
    gexact = Pext - closest
    gexact /= np.clip(np.linalg.norm(gexact, axis=1, keepdims=True), 1e-9, None)

    _, g = grad_of(P, torch.tensor(Pext, device=dev), G)
    g = g.detach().cpu().numpy()
    gn = np.linalg.norm(g, axis=1)
    gu = g / np.clip(gn[:, None], 1e-9, None)
    ang = np.degrees(np.arccos(np.clip(np.sum(gu * gexact, axis=1), -1, 1)))
    print(f"\nBSPLINE {args.link}: ||grad|| mean={gn.mean():.3f} std={gn.std():.3f} "
          f"[{gn.min():.3f},{gn.max():.3f}] %<0.8={np.mean(gn<0.8)*100:.1f} %>1.2={np.mean(gn>1.2)*100:.1f} "
          f"| dir err mean={ang.mean():.1f} med={np.median(ang):.1f} p95={np.percentile(ang,95):.1f} deg")
    print(f"PROD    {args.link}: ||grad|| mean=0.841 std=0.269 %<0.8=38.0 "
          f"| dir err mean=41.4 med=9.9 p95=166.8 deg   (n_func=12, world frame reference)")


if __name__ == "__main__":
    main()
