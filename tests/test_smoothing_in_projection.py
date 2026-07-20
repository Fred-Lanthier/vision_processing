#!/usr/bin/env python3
"""Offline A/B of the output-smoothing seam (~cbf_smoothing_mode).

Math-level harness (solver_study.py style, no ROS): both pipelines are
implemented with the node's exact formulas — Dykstra projection onto the
active half-spaces in metric W, band bound b(h), exponential memory with
gamma = dt/(dt+tau), all-rows Dykstra repair — and compared on a kinematic
plant h_{k+1} = h_k + g^T dq dt with the launch constants.

  legacy (output_filter):  dq = repair(EMA(project(nominal)))
  in_projection:           dq = project((1-gamma)*prev + gamma*nominal)

Claims verified:
  1. Free space: the two pipelines are IDENTICAL (the in-projection blend is
     the same exponential recursion when the projection is inactive).
  2. Graze: legacy transiently violates the certified rows BEFORE its repair
     (the filter-then-repair seam); in_projection is feasible pre-repair by
     construction, with comparable smoothness and equilibrium standoff.
  3. Recovery (h<0): the legacy memory dilutes the commanded retreat before
     repair (the "low-pass can eat the retreat" caveat); in_projection
     enforces the recovery bound on the output directly.

Run:  python3 tests/test_smoothing_in_projection.py
"""

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# Launch constants (green_cube_feeding_casf_pp.launch).
DT = 0.01
TAU = 0.1
GAMMA = DT / (DT + TAU)
MJV = 0.7
H_ACT = 0.04
MAX_INWARD = 0.5
RECOVERY_SPEED = 0.05
RECOVERY_DEPTH = 0.015
MARGIN = 0.001
SOLVE_SWEEPS = 6
REPAIR_SWEEPS = 24


def make_metric():
    J = torch.randn(3, 7) * 0.4
    W = J.T @ J + 0.01 * torch.eye(7)
    return W, torch.linalg.inv(W)


def bound(h):
    outward = -MAX_INWARD * torch.clamp(h / H_ACT, min=0.0, max=1.0)
    recovery = RECOVERY_SPEED * torch.clamp(
        -h / RECOVERY_DEPTH, min=0.0, max=1.0)
    return torch.where(h > 0.0, outward, recovery) + MARGIN


def project_rows(dq0, G, h, Winv, sweeps):
    """Dykstra projection onto {dq : G dq >= bound(h)} for active rows."""
    active = h <= H_ACT
    if not bool(active.any()):
        return dq0.clone()
    b = bound(h)
    wg = G @ Winv
    denom = torch.clamp((G * wg).sum(dim=1), min=1e-8)
    dq = dq0.clone()
    z = torch.zeros_like(G)
    for _ in range(sweeps):
        for k in range(G.shape[0]):
            if not bool(active[k]):
                continue
            tmp = dq + z[k]
            corr = torch.clamp(b[k] - (G[k] * tmp).sum(), min=0.0)
            dq = tmp + (corr / denom[k]) * wg[k]
            z[k] = tmp - dq
    return dq


def worst_residual(dq, G, h):
    active = h <= H_ACT
    if not bool(active.any()):
        return float("inf")
    res = G @ dq - bound(h)
    return float(res[active].min())


class LegacyPipeline:
    """project -> clamp -> EMA -> all-rows repair -> clamp."""

    def __init__(self, Winv):
        self.Winv = Winv
        # Zero-seeded memory, matching the graphed node (_graph_dq_filtered).
        self.filt = torch.zeros(7)
        self.pre_repair_residuals = []

    def step(self, dq_nom, G, h):
        dq = project_rows(dq_nom, G, h, self.Winv, SOLVE_SWEEPS)
        dq = torch.clamp(dq, -MJV, MJV)
        self.filt = (1.0 - GAMMA) * self.filt + GAMMA * dq
        dq_f = self.filt.clone()
        self.pre_repair_residuals.append(worst_residual(dq_f, G, h))
        if self.pre_repair_residuals[-1] < -1e-9:
            dq_f = project_rows(dq_f, G, h, self.Winv, REPAIR_SWEEPS)
        return torch.clamp(dq_f, -MJV, MJV)


class InProjectionPipeline:
    """project((1-gamma)*prev_published + gamma*nominal) -> clamp.

    The repair stage is kept as verification (it also covers the clamp);
    the pre-repair residual is recorded to show it is already feasible.
    """

    def __init__(self, Winv):
        self.Winv = Winv
        self.prev = torch.zeros(7)
        self.pre_repair_residuals = []

    def step(self, dq_nom, G, h):
        target = dq_nom + (1.0 - GAMMA) * (self.prev - dq_nom)
        dq = project_rows(target, G, h, self.Winv, SOLVE_SWEEPS)
        dq = torch.clamp(dq, -MJV, MJV)
        self.pre_repair_residuals.append(worst_residual(dq, G, h))
        if self.pre_repair_residuals[-1] < -1e-9:
            dq = project_rows(dq, G, h, self.Winv, REPAIR_SWEEPS)
        dq = torch.clamp(dq, -MJV, MJV)
        self.prev = dq.clone()
        return dq


def rollout(pipeline, dq_nom, G, h0, cycles):
    h = h0.clone()
    published, h_hist = [], []
    for _ in range(cycles):
        dq = pipeline.step(dq_nom, G, h)
        h = h + (G @ dq) * DT
        published.append(dq.clone())
        h_hist.append(h.clone())
    return torch.stack(published), torch.stack(h_hist)


def smoothness(published):
    return float((published[1:] - published[:-1]).norm(dim=1).max())


def check(name, ok):
    print(("PASS" if ok else "FAIL") + f"  {name}")
    assert ok, name


def main():
    W, Winv = make_metric()

    # ── 1. Free space: no active rows, pipelines must be identical. ──
    G = torch.randn(2, 7) * 0.3
    h_free = torch.tensor([0.5, 0.6])          # far outside the band
    dq_nom = torch.randn(7).clamp(-0.3, 0.3)
    a, _ = rollout(LegacyPipeline(Winv), dq_nom, G, h_free, 200)
    b, _ = rollout(InProjectionPipeline(Winv), dq_nom, G, h_free, 200)
    diff = float((a - b).norm(dim=1).max())
    print(f"[free space] max command difference over 200 cycles: {diff:.2e}")
    check("free space: pipelines identical", diff < 1e-12)

    # ── 2. Graze: single row, nominal pushes inward. ──
    g = torch.randn(7)
    g = 0.5 * g / g.norm()
    G1 = g.view(1, 7)
    dq_in = -0.3 * g / g.norm()                # straight at the wall
    dq_tan = torch.randn(7)
    dq_tan -= (dq_tan @ g) / (g @ g) * g       # tangential slide
    dq_nom = dq_in + 0.2 * dq_tan / dq_tan.norm()
    h0 = torch.tensor([0.06])
    legacy = LegacyPipeline(Winv)
    inproj = InProjectionPipeline(Winv)
    pa, ha = rollout(legacy, dq_nom, G1, h0, 600)
    pb, hb = rollout(inproj, dq_nom, G1, h0, 600)
    res_a = min(legacy.pre_repair_residuals)
    res_b = min(inproj.pre_repair_residuals)
    print(f"[graze] worst PRE-repair residual  legacy {res_a:.5f}  "
          f"in_projection {res_b:.2e}")
    print(f"[graze] equilibrium h [mm]         legacy {1e3*float(ha[-1]):.3f}  "
          f"in_projection {1e3*float(hb[-1]):.3f}")
    print(f"[graze] max per-tick command delta legacy {smoothness(pa):.4f}  "
          f"in_projection {smoothness(pb):.4f}")
    check("graze: legacy violates pre-repair (the seam exists)",
          res_a < -1e-4)
    check("graze: in_projection feasible pre-repair", res_b > -1e-9)
    check("graze: same equilibrium standoff (2% of band)",
          abs(float(ha[-1]) - float(hb[-1])) < 0.02 * H_ACT)
    check("graze: smoothness comparable (<=1.5x)",
          smoothness(pb) <= 1.5 * smoothness(pa) + 1e-9)

    # ── 3. Pinch: two opposing rows (runPP1022 geometry, cos ~ -0.3). ──
    g1 = torch.randn(7); g1 = 0.4 * g1 / g1.norm()
    g2 = -0.3 * g1 / g1.norm() + 0.5 * torch.randn(7)
    g2 = 0.4 * g2 / g2.norm()
    G2 = torch.stack([g1, g2])
    cosang = float((g1 @ g2) / (g1.norm() * g2.norm()))
    dq_nom = -0.25 * (g1 + g2) / (g1 + g2).norm()
    h0 = torch.tensor([0.02, 0.02])
    legacy = LegacyPipeline(Winv)
    inproj = InProjectionPipeline(Winv)
    pa, ha = rollout(legacy, dq_nom, G2, h0, 600)
    pb, hb = rollout(inproj, dq_nom, G2, h0, 600)
    res_a = min(legacy.pre_repair_residuals)
    res_b = min(inproj.pre_repair_residuals)
    print(f"[pinch cos={cosang:.2f}] worst PRE-repair residual  "
          f"legacy {res_a:.5f}  in_projection {res_b:.2e}")
    print(f"[pinch] final h [mm]  legacy {1e3*ha[-1].min():.3f}  "
          f"in_projection {1e3*hb[-1].min():.3f}")
    check("pinch: in_projection feasible pre-repair", res_b > -1e-9)
    check("pinch: both stay out of violation",
          float(ha[-1].min()) > 0.0 and float(hb[-1].min()) > 0.0)

    # ── 4. Recovery: start in violation, nominal still pushes inward. ──
    # The legacy memory is sloppy in BOTH directions around the recovery
    # bound: it dilutes the retreat in the transient (negative pre-repair
    # residuals, rescued by the repair), then OVER-retreats once the bound
    # shrinks (the EMA carries earlier, larger outward commands — an
    # uncertified bonus that surfaces faster). in_projection rides the
    # certified recovery law exactly (residual ~ 0 at equality every
    # cycle), which is the designed minimum outward rate; recovery is
    # slower but exactly the certified trajectory.
    h0 = torch.tensor([-0.005])
    dq_nom = -0.3 * g / g.norm()
    legacy = LegacyPipeline(Winv)
    inproj = InProjectionPipeline(Winv)
    pa, ha = rollout(legacy, dq_nom, G1, h0, 400)
    pb, hb = rollout(inproj, dq_nom, G1, h0, 400)

    def cycles_to_surface(h_hist):
        above = (h_hist.squeeze(-1) >= 0.0).nonzero()
        return int(above[0]) if above.numel() else 10**9

    ta, tb = cycles_to_surface(ha), cycles_to_surface(hb)
    res_a = min(legacy.pre_repair_residuals)
    res_b = min(inproj.pre_repair_residuals)
    res_b_max = max(inproj.pre_repair_residuals[:tb])
    print(f"[recovery] worst PRE-repair residual  legacy {res_a:.5f}  "
          f"in_projection {res_b:.2e}")
    print(f"[recovery] cycles to h>=0             legacy {ta}  "
          f"in_projection {tb}")
    print(f"[recovery] in_projection rides the bound at equality: "
          f"max residual while h<0 = {res_b_max:.2e}")
    check("recovery: legacy memory dilutes the retreat pre-repair "
          "(line-1939 caveat)", res_a < -1e-4)
    check("recovery: in_projection retreat certified on the output",
          res_b > -1e-9)
    check("recovery: in_projection rides the certified recovery law "
          "at equality", res_b_max < 1e-9)
    check("recovery: both surface in finite time",
          ta < 10**9 and tb < 10**9)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
