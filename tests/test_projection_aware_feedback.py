#!/usr/bin/env python3
"""Property checks for the projection_aware tracking-feedback shaping.

Replicates the formula of _shape_tracking_feedback(mode="projection_aware")
in cbf_safety_node_Bernstein_multicbf.py:

    ramp   = clip((h_act - h_prev)/h_act, 0, 1)
    inward = clip(<fb, g>, max=0) / ||g||^2
    fb'    = fb - ramp * inward * g          (g = previous critical gradient)

Verified properties:
  1. Identity outside the band (h_prev >= h_act) and for degenerate g.
  2. Continuity through h_act and in fb/g (no switch, unlike tangent_near).
  3. At full ramp the shaped feedback has NO inward component along g,
     while outward and cross-track components are untouched.
  4. Safety-neutrality: the shaping never increases the inward component,
     for random fb/g/h (the QP constraint never sees fb anyway; this checks
     the objective-level claim).
  5. Well-defined at ||dq_ff|| = 0 (needs no tangent direction).

Run:  python3 tests/test_projection_aware_feedback.py
"""

import torch

torch.manual_seed(1)
torch.set_default_dtype(torch.float64)

H_ACT = 0.04


def shape(fb, g, h_prev):
    ramp = (H_ACT - h_prev) / H_ACT
    if not (ramp > 0.0):
        return fb.clone()
    ramp = min(ramp, 1.0)
    g_norm_sq = (g * g).sum()
    g_ok = 1.0 if float(g_norm_sq) > 1e-12 else 0.0
    inward = torch.clamp((fb * g).sum(), max=0.0) / (g_norm_sq + 1e-12)
    return fb - ramp * g_ok * inward * g


def check(name, ok):
    print(("PASS" if ok else "FAIL") + f"  {name}")
    assert ok, name


def main():
    fb = torch.randn(7)
    g = torch.randn(7) * 0.4

    # 1. Identity outside the band / degenerate gradient.
    check("identity for h_prev >= h_act",
          torch.equal(shape(fb, g, H_ACT + 1e-9), fb))
    check("identity for h_prev = inf", torch.equal(
        shape(fb, g, float("inf")), fb))
    check("identity for nan h_prev (startup)", torch.equal(
        shape(fb, g, float("nan")), fb))
    check("identity for g = 0", torch.equal(
        shape(fb, torch.zeros(7), 0.0), fb))

    # 2. Continuity through h_act and inside the band.
    eps = 1e-9
    jump = (shape(fb, g, H_ACT - eps) - shape(fb, g, H_ACT + eps)).norm()
    check("continuous through h_act", float(jump) < 1e-6)
    h_grid = torch.linspace(-0.02, 0.06, 400)
    outs = torch.stack([shape(fb, g, float(h)) for h in h_grid])
    step = (outs[1:] - outs[:-1]).norm(dim=1).max()
    check("no switch anywhere in the band (max step ~ grid resolution)",
          float(step) < 1e-2 * fb.norm())

    # 3. Full ramp: inward component gone, orthogonal part untouched.
    fb_in = fb.clone()
    if float((fb_in * g).sum()) > 0:
        fb_in = -fb_in                      # make it inward
    shaped = shape(fb_in, g, 0.0)
    check("no inward component at full ramp",
          float((shaped * g).sum()) > -1e-12)
    g_hat = g / g.norm()
    ortho_before = fb_in - (fb_in @ g_hat) * g_hat
    ortho_after = shaped - (shaped @ g_hat) * g_hat
    check("cross-track component untouched",
          float((ortho_before - ortho_after).norm()) < 1e-12)
    fb_out = fb_in - 2.0 * (fb_in @ g_hat) * g_hat   # outward mirror
    check("outward feedback untouched at full ramp",
          torch.equal(shape(fb_out, g, 0.0), fb_out))

    # 4. Monotone: shaping never makes the feedback MORE inward.
    worst = 0.0
    for _ in range(2000):
        fb_r = torch.randn(7)
        g_r = torch.randn(7) * torch.rand(1)
        h_r = float(torch.empty(1).uniform_(-0.02, 0.08))
        delta = float(((shape(fb_r, g_r, h_r) - fb_r) * g_r).sum())
        worst = min(worst, delta)
    check("shaping only ever ADDS along +grad_h (2000 random cases)",
          worst > -1e-12)

    # 5. No tangent direction needed: dq_ff = 0 is a non-event.
    check("well-defined with zero feedforward",
          torch.isfinite(shape(fb, g, 0.0)).all().item())

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
