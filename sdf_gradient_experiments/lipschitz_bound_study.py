#!/usr/bin/env python3
"""Certified vs sampled joint-space gradient-Lipschitz constants per link.

Question answered (reflex brake, certified mode): what L bounds
||grad_q f(q) - grad_q f(q')|| / ||q - q'|| for the PER-POINT barrier
f_i(q) = RDF_link(T_wl(q)^{-1} p_w) as deployed by the analytical CBF path?

Two independent estimates, which must bracket each other:

  1. ANALYTIC (certified, in-domain): Bernstein coefficient arithmetic.
     Derivatives of Bernstein polynomials are Bernstein polynomials of the
     finite differences of the coefficients, and |P| <= max|coeff| on the box
     (convex-hull property). With the deployed normalization
     u=(p_l-c)/s, t=(u+1)/2, f=s*P(t):
         sup||grad_p f||   <= G_p = ||(n*max|D_a w|)_a|| / 2        [-]
         sup||hess_p f||_2 <= H_p = ||(sup|d2P/dt_a dt_b|)_ab||_F / (4s) [1/m]
     Kinematic factors (sampled sups, reported separately):
         A     = sup||d p_l / d q||_2                        [m/rad]
         B_kin = sup sqrt(sum_c ||d2 p_l,c / dq2||_2^2)      [m/rad^2]
     Assembled joint-space bound per link:
         L_asm = A^2 * H_p + G_p * B_kin                     [m/rad^2]

  2. SAMPLED (ground truth): central finite differences of the ANALYTIC
     joint gradient the runtime actually publishes (M=1 point per link via a
     diagonal point_mask, so the softmin is exact), giving the true
     ||hess_q f||_2 at random (q, p_w). In-domain samples must satisfy
     sampled <= L_asm; the gap is the conservatism of the assembled bound.
     Out-of-domain samples exercise the Pythagorean extension branch, whose
     curvature is NOT covered by the polynomial bound (reported separately).

Run with the venv python (needs torch + pytorch_kinematics + xacro):
    venv_sam3/bin/python3 sdf_gradient_experiments/lipschitz_bound_study.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, 'src'))

from third_party.RDF.urdf_layer import URDFLayer
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore

# Load by file path: the package __init__ imports ROS srv bindings that the
# source tree shadows when it sits ahead of devel on sys.path.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'analytical_bernstein',
    os.path.join(PKG, 'src', 'vision_processing', 'analytical_bernstein.py'))
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)
AnalyticalBernsteinSoftmin = _ab.AnalyticalBernsteinSoftmin

LINKS = ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link5',
         'panda_link6', 'panda_link7', 'panda_hand',
         'panda_rightfinger', 'panda_leftfinger']
ARM_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]


def build_adapter(device, dtype, override_dir=None, override_links=()):
    urdf = os.path.join(PKG, 'urdf', 'panda_pickplace.xacro')
    voxel = os.path.join(PKG, 'third_party/RDF/panda_layer/meshes/voxel_128')
    robot = URDFLayer(urdf_path=urdf, device=device, package_dir=PKG,
                      voxel_dir=voxel)
    wh = RDF_Weights(device=device, dtype=torch.float32)
    wh.init_robot_folder(
        os.path.join(PKG, 'third_party/SDF_Bernstein_Basis/panda_test'),
        robot_name='panda')
    wh.add_models(LINKS, robot_name='panda')
    if override_dir:
        from src.core.assets.load_model_wrapper import load_link_weight_model
        for link in override_links:
            path = os.path.join(override_dir, f'{link}_w.pt')
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            candidate = load_link_weight_model(
                torch.load(path, map_location=device, weights_only=False),
                device=device, dtype=torch.float32)
            setattr(wh, link + wh.model_extension, candidate)
            print(f'override: {link} <- {path} (n_func={int(candidate.n_func)})')
    core = BernsteinCore(wh, robot, device, LINKS)
    adapter = AnalyticalBernsteinSoftmin(core, temperature=0.002, d_safe=0.0)
    # The study runs in float64 for clean finite differences; the FK path
    # casts per-call from q.dtype, only the cached core tensors need moving.
    for group in core.groups.values():
        for key in ('weights', 'offsets', 'scales'):
            group[key] = group[key].to(dtype)
    adapter._basis_cache = {
        n: (b.to(dtype), None if lb is None else lb.to(dtype))
        for n, (b, lb) in adapter._basis_cache.items()}
    return robot, core, adapter


def coefficient_bounds(core, dtype):
    """Exact sup bounds on ||grad_t P|| and ||hess_t P||_F per link."""
    out = {}
    for group in core.groups.values():
        n_func = int(group['n_func'])
        n = n_func - 1
        w = group['weights'].reshape(-1, n_func, n_func, n_func).to(dtype)
        for i, link in enumerate(group['link_names']):
            wi = w[i]
            d1 = [n * torch.diff(wi, dim=a) for a in range(3)]
            g_sup = torch.tensor([d.abs().max() for d in d1])
            hess_sup = torch.zeros(3, 3)
            for a in range(3):
                for b in range(a, 3):
                    if a == b:
                        coeff = n * (n - 1) * torch.diff(wi, n=2, dim=a) \
                            if n >= 2 else torch.zeros(1)
                    else:
                        coeff = n * torch.diff(n * torch.diff(wi, dim=a),
                                               dim=b)
                    hess_sup[a, b] = hess_sup[b, a] = coeff.abs().max()
            scale = float(group['scales'][i])
            out[link] = {
                'n_func': n_func,
                'scale': scale,
                'G_p': float(torch.linalg.vector_norm(g_sup)) / 2.0,
                'H_p': float(torch.linalg.matrix_norm(hess_sup)) / (4.0 * scale),
            }
    return out


def sample_points(core, transforms, ood_mask, clamp_margin, rng):
    """One query point per (config, link): world position + domain label."""
    B, K = transforms.shape[:2]
    device, dtype = transforms.device, transforms.dtype
    lim = 1.0 - clamp_margin - 0.01
    u = (torch.rand(B, K, 3, device=device, dtype=dtype) * 2.0 - 1.0) * lim
    # OOD rows: push radially past the box, physical excursion 5 mm - 25 cm.
    scales = core.scales.to(dtype)                                    # [K]
    inf_norm = u.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-6)
    exc = 0.005 + 0.245 * torch.rand(B, K, 1, device=device, dtype=dtype)
    u_ood = u / inf_norm * (1.0 + exc / scales[None, :, None])
    u = torch.where(ood_mask[:, None, None], u_ood, u)
    offsets = core.offsets.to(dtype)                                  # [K,3]
    p_l = offsets[None] + scales[None, :, None] * u                   # [B,K,3]
    R = transforms[..., :3, :3]
    o = transforms[..., :3, 3]
    return torch.einsum('bkij,bkj->bki', R, p_l) + o                  # [B,K,3]


def local_jacobian(adapter, q, points):
    """p_l(q), J_p = dp_l/dq [B,K,3,7] from one FK pass (analytic)."""
    eye = torch.eye(4, device=q.device, dtype=q.dtype).expand(q.shape[0], 4, 4)
    transforms, jv, jw, _ = adapter._visual_kinematics(q, eye)
    R = transforms[..., :3, :3]
    o = transforms[..., :3, 3]
    r = points - o                                                    # [B,K,3]
    p_l = torch.einsum('bkij,bki->bkj', R, r)
    # dp_l/dq_j = -R^T (J_v[:,j] + J_w[:,j] x r)
    cross = torch.cross(jw.transpose(-1, -2), r[:, :, None, :], dim=-1)
    cols = jv.transpose(-1, -2) + cross                               # [B,K,7,3]
    J_p = -torch.einsum('bkij,bkqi->bkjq', R, cols)                   # [B,K,3,7]
    return p_l, J_p, transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=3072)
    ap.add_argument('--ood-frac', type=float, default=0.33)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--band', type=float, default=0.12,
                    help='activation band upper edge on f [m]')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model-override-dir', default=None,
                    help='directory with candidate <link>_w.pt models')
    ap.add_argument('--override-links', nargs='*',
                    default=['panda_leftfinger', 'panda_rightfinger'])
    ap.add_argument('--json', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'lipschitz_bound_study.json'))
    args = ap.parse_args()

    dtype = torch.float64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    robot, core, adapter = build_adapter(
        device, dtype, args.model_override_dir, args.override_links)
    bounds = coefficient_bounds(core, dtype)

    # Arm joint limits by name (the pickplace URDF also has finger joints).
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
    arm_idx = [name_to_idx[j] for j in ARM_JOINTS]
    lo = robot.theta_min[arm_idx].to(dtype=dtype, device=device)
    hi = robot.theta_max[arm_idx].to(dtype=dtype, device=device)

    B, K = args.samples, len(LINKS)
    q = lo + (hi - lo) * torch.rand(B, 7, device=device, dtype=dtype)
    ood_mask = torch.rand(B, device=device) < args.ood_frac

    eye4 = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4)
    with torch.no_grad():
        transforms0, _, _, _ = adapter._visual_kinematics(q, eye4)
        points = sample_points(core, transforms0, ood_mask,
                               adapter.clamp_margin, rng)             # [B,K,3]
        mask = torch.eye(K, dtype=torch.bool, device=device)          # link k sees point k

        def grads_at(qq):
            res = adapter.evaluate(qq, points, point_mask=mask)
            return res.h.clone(), res.grad_q.clone()                  # [B,K],[B,K,7]

        f0, g0 = grads_at(q)
        H = torch.zeros(B, K, 7, 7, device=device, dtype=dtype)
        T = torch.zeros(B, K, 3, 7, 7, device=device, dtype=dtype)    # d2 p_l/dq2
        p_l0, J_p0, _ = local_jacobian(adapter, q, points)
        for j in range(7):
            dq = torch.zeros_like(q)
            dq[:, j] = args.eps
            _, gp = grads_at(q + dq)
            _, gm = grads_at(q - dq)
            H[:, :, j, :] = (gp - gm) / (2 * args.eps)
            _, Jp, _ = local_jacobian(adapter, q + dq, points)
            _, Jm, _ = local_jacobian(adapter, q - dq, points)
            T[:, :, :, j, :] = (Jp - Jm) / (2 * args.eps)
        H = 0.5 * (H + H.transpose(-1, -2))
        bad = ~torch.isfinite(H).all(dim=-1).all(dim=-1)              # [B,K]
        if bad.any():
            print(f"note: dropping {int(bad.sum())} non-finite Hessian "
                  f"samples of {B * K}")
            H = torch.where(bad[..., None, None], torch.zeros_like(H), H)
            f0 = torch.where(bad, torch.full_like(f0, float('inf')), f0)

        H_norm = torch.linalg.matrix_norm(H, ord=2)                   # [B,K]
        # The sampled-data certificate h(q0+D) >= h + g.D - (L/2)||D||^2 only
        # needs H >= -L*I: the binding constant is the NEGATIVE curvature
        # sup (-lambda_min)_+, not ||H||_2. Convex link surfaces (fingers,
        # hand edges) produce large POSITIVE exterior-SDF curvature that
        # ||H||_2 counts but the certificate never pays for.
        # CPU eigvalsh: the batched cusolver path rejects very large batches.
        lam_min = torch.linalg.eigvalsh(H.cpu())[..., 0].to(device)   # [B,K]
        L_need = (-lam_min).clamp_min(0.0)                            # [B,K]
        A_all = torch.linalg.matrix_norm(J_p0, ord=2)                 # [B,K]
        B_all = torch.sqrt(
            torch.linalg.matrix_norm(T, ord=2).square().sum(dim=2))   # [B,K]

    in_dom = ~ood_mask
    report = {}
    print(f"\n{'link':<18}{'n':>3}{'s[m]':>7}{'G_p':>6}{'H_p':>8}{'A':>6}"
          f"{'B_kin':>7}{'L_asm':>8} | {'|H| max':>8} |"
          f"{'Lneed med':>10}{'p99':>8}{'max_in':>8}{'band':>8}{'max_ood':>9}")
    for k, link in enumerate(LINKS):
        cb = bounds[link]
        A_k = float(A_all[:, k].max())
        B_k = float(B_all[:, k].max())
        L_asm = A_k ** 2 * cb['H_p'] + cb['G_p'] * B_k
        l_in = L_need[in_dom, k]
        l_ood = L_need[ood_mask, k]
        band_sel = in_dom & (f0[:, k] >= 0.0) & (f0[:, k] <= args.band)
        band_max = float(L_need[band_sel, k].max()) if band_sel.any() else float('nan')
        row = {
            **cb, 'A': A_k, 'B_kin': B_k, 'L_assembled': L_asm,
            'specnorm_max_in': float(H_norm[in_dom, k].max()),
            'Lneed_med_in': float(l_in.median()),
            'Lneed_p99_in': float(l_in.quantile(0.99)),
            'Lneed_max_in': float(l_in.max()),
            'Lneed_max_band': band_max,
            'Lneed_max_ood': float(l_ood.max()),
            'n_band': int(band_sel.sum()),
        }
        report[link] = row
        flag = '' if row['Lneed_max_in'] <= L_asm else '  << VIOLATED'
        print(f"{link:<18}{cb['n_func']:>3}{cb['scale']:>7.3f}{cb['G_p']:>6.2f}"
              f"{cb['H_p']:>8.1f}{A_k:>6.2f}{B_k:>7.2f}{L_asm:>8.1f} |"
              f"{row['specnorm_max_in']:>8.2f} |"
              f"{row['Lneed_med_in']:>10.2f}{row['Lneed_p99_in']:>8.2f}"
              f"{row['Lneed_max_in']:>8.2f}{band_max:>8.2f}"
              f"{row['Lneed_max_ood']:>9.2f}{flag}")

    worst = max(r['L_assembled'] for r in report.values())
    worst_in = max(r['Lneed_max_in'] for r in report.values())
    worst_band = np.nanmax([r['Lneed_max_band'] for r in report.values()])
    worst_ood = max(r['Lneed_max_ood'] for r in report.values())
    print(f"\nconfigured reflex L               : 20.0  (run empirical median ~2.0)")
    print(f"assembled certified bound (two-sided): {worst:.1f}  (max over links)")
    print(f"sampled sup -lambda_min (in-domain)  : {worst_in:.2f}")
    print(f"sampled sup -lambda_min (f in band)  : {worst_band:.2f}")
    print(f"sampled sup -lambda_min (OOD branch) : {worst_ood:.2f}")

    # Where does the binding curvature live relative to the surface?
    print("\n-lambda_min by clearance bin (p99 / max, in-domain then OOD):")
    edges = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
    header = ''.join(f"{f'[{a * 1e3:.0f},{b * 1e3:.0f})mm':>16}"
                     for a, b in zip(edges[:-1], edges[1:]))
    print(f"{'link':<18}{header}")
    for k, link in enumerate(LINKS):
        for dom, sel0 in (('in', in_dom), ('ood', ood_mask)):
            cells = []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = sel0 & (f0[:, k] >= a) & (f0[:, k] < b)
                if sel.sum() > 3:
                    cells.append(f"{float(L_need[sel, k].quantile(0.99)):>7.1f}"
                                 f"/{float(L_need[sel, k].max()):<7.1f}")
                else:
                    cells.append(f"{'-':>15} ")
            print(f"{link + ' ' + dom:<18}{''.join(cells)}")

    np.savez(args.json.replace('.json', '.npz'),
             f=f0.cpu().numpy(), L_need=L_need.cpu().numpy(),
             H_norm=H_norm.cpu().numpy(), ood=ood_mask.cpu().numpy(),
             links=np.array(LINKS))
    with open(args.json, 'w') as fh:
        json.dump({'args': vars(args), 'links': report}, fh, indent=2)
    print(f"\nwrote {args.json} (+ .npz with per-sample data)")


if __name__ == '__main__':
    main()
