#!/usr/bin/env python3
"""Offline replay of the reflex certificate on a recorded run.

For every /cbf_safety/reflex_state in the bag, pair it with the latest
/joint_states and /cbf_safety/pre_reflex_command, re-run the certified-mode
alpha computation under several (L, horizon) configurations, and attribute
the braking: which link row binds, and how much certificate each remainder
term consumes (measured drift Gp, drift remainder (L/2)||p||^2, future
quadratic c, environment charge).

    source /opt/ros/noetic/setup.bash
    python3 sdf_gradient_experiments/replay_reflex_alpha.py runPP1009.bag
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import rosbag

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'cbf_reflex_node', os.path.join(PKG, 'scripts', 'cbf_reflex_node.py'))
_rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rx)

LINKS = ['link2', 'link3', 'link4', 'link5', 'link6', 'link7',
         'hand', 'rfinger', 'lfinger']
JOINTS = [f'panda_joint{i}' for i in range(1, 8)]

SUP = np.array([5, 35, 25, 30, 85, 85, 50, 600, 600.0])
P99 = np.array([2, 10, 6, 11, 25, 35, 22, 120, 120.0])

def _fingers(x):
    v = SUP.copy(); v[7] = v[8] = x; return v

# 2nd field: horizon [s]. This is now an OFFLINE what-if sweep: the node no
# longer takes ~horizon, it measures its own decision period (~1-2 ms) per
# tick and publishes it as /cbf_reflex/status[5]. 0.020 reproduces the bags
# recorded BEFORE that change (runPP1009 and earlier); for a newer bag, read
# the horizon actually used from status[5] instead of assuming this value.
# 4th field: h-row offset emulating cbf_barrier_value_mode:=hard (the soft
# value under-reports clearance by alpha*log(M_eff) ~= alpha*log(N) = 9.2 mm
# on the dense synthetic PP cloud; bag: min_h -0.5 mm vs min_h_corr +8.7 mm).
# Offsets emulate the closed-loop equilibrium shift h* = m/kappa from
# raising cbf_constraint_margin m (kappa=5): m=0.025 -> h* = 5 mm, i.e.
# +4.8 mm vs the recorded m=0.001 equilibrium.
CONFIGS = {
    'as recorded (m=0.001)     ': (SUP, 0.020, 0.0, 0.0),
    'm=0.025 (h*=5mm)          ': (SUP, 0.020, 0.0, 0.0048),
    'm=0.025 + fingers=150     ': (_fingers(150.0), 0.020, 0.0, 0.0048),
    'm=0.040 (h*=8mm)          ': (SUP, 0.020, 0.0, 0.0078),
}


def make_node(L, horizon, h_stop=0.0):
    o = object.__new__(_rx.CbfReflexNode)
    o.grad_lipschitz = L
    o.horizon = horizon
    o.h_stop = h_stop
    o.h_activate = 0.08
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    args = ap.parse_args()

    q_t, q_v, cmd_t, cmd_v, states = [], [], [], [], []
    with rosbag.Bag(args.bag) as bag:
        for topic, msg, t in bag.read_messages(
                ['/joint_states', '/cbf_safety/pre_reflex_command',
                 '/cbf_safety/reflex_state']):
            ts = t.to_sec()
            if topic == '/joint_states':
                pos = dict(zip(msg.name, msg.position))
                if all(j in pos for j in JOINTS):
                    q_t.append(ts)
                    q_v.append([pos[j] for j in JOINTS])
            elif topic == '/cbf_safety/pre_reflex_command':
                if len(msg.data) >= 7:
                    cmd_t.append(ts)
                    cmd_v.append(list(msg.data[:7]))
            else:
                states.append((ts, np.asarray(msg.data, dtype=np.float64)))
    q_t = np.asarray(q_t); q_v = np.asarray(q_v)
    cmd_t = np.asarray(cmd_t); cmd_v = np.asarray(cmd_v)
    print(f"{len(states)} reflex states, {len(q_t)} joint states, "
          f"{len(cmd_t)} commands")

    rows = []
    for ts, d in states:
        n = int(round(d[1]))
        if d.shape[0] < 9 + 9 * n:
            continue
        qi = np.searchsorted(q_t, ts) - 1
        ci = np.searchsorted(cmd_t, ts) - 1
        if qi < 0 or ci < 0:
            continue
        rows.append((d[0], ts, d[2:9], d[9:9 + n], d[9 + n:9 + 2 * n],
                     d[9 + 2 * n:9 + 9 * n].reshape(n, 7).copy(),
                     q_v[qi], cmd_v[ci]))

    print(f"\n{'config':<24}{'brake%':>7}{'a_med':>7}{'a_p05':>7}"
          f"{'a=0%':>6}  binding row histogram (of braked)")
    for name, (L, T, stop, hoff) in CONFIGS.items():
        node = make_node(L, T, stop)
        alphas, bind = [], np.zeros(9, dtype=int)
        for t_solve, ts, q0, h0, env, G, q, dq in rows:
            h = h0 + hoff
            p = q - q0
            dq = np.asarray(dq)
            alpha, _, nb, _ = node._alpha_certified(
                dq, ts, t_solve, p, h, env, G, 0.0)
            alphas.append(alpha)
            if alpha < 1.0:
                # re-derive the binding row for attribution
                Lv = L if np.ndim(L) else np.full(9, L)
                c = 0.5 * Lv * float(dq @ dq) * T * T
                h_now = h + G @ p - 0.5 * Lv * float(p @ p)
                a = h_now - env * (max(ts - t_solve, 0.0) + T) - stop
                b = (G @ dq - Lv * float(p @ dq)) * T
                m = a + b * alpha - c * alpha ** 2
                m[(h_now >= 0.08) | ((a <= 0) & (G @ dq >= 0))] = np.inf
                bind[int(np.argmin(m))] += 1
        alphas = np.asarray(alphas)
        braked = alphas < 1.0
        hist = ' '.join(f"{LINKS[i]}:{bind[i]}"
                        for i in np.argsort(-bind)[:3] if bind[i])
        print(f"{name:<24}{100 * braked.mean():>6.1f}%"
              f"{np.median(alphas):>7.2f}{np.percentile(alphas, 5):>7.2f}"
              f"{100 * (alphas == 0).mean():>5.1f}%  {hist}")

    # Term attribution under the deployed config, braked cycles only.
    L, T, stop, _hoff = CONFIGS['as recorded (m=0.001)     ']
    node = make_node(L, T, stop)
    terms = []
    for t_solve, ts, q0, h, env, G, q, dq in rows:
        p = q - q0
        dq = np.asarray(dq)
        alpha, _, _, _ = node._alpha_certified(dq, ts, t_solve, p, h, env, G, 0.0)
        if alpha >= 1.0:
            continue
        c = 0.5 * L * float(dq @ dq) * T * T
        h_now = h + G @ p - 0.5 * L * float(p @ p)
        a = h_now - env * (max(ts - t_solve, 0.0) + T)
        b = (G @ dq - L * float(p @ dq)) * T
        m = a + b * alpha - c * alpha ** 2
        m[(h_now >= 0.08) | ((a <= 0) & (G @ dq >= 0))] = np.inf
        k = int(np.argmin(m))
        terms.append([h[k], (G @ p)[k], 0.5 * L[k] * float(p @ p),
                      c[k], (G @ dq)[k] * T, L[k] * float(p @ dq) * T,
                      float(np.linalg.norm(p)), k])
    terms = np.asarray(terms)
    print(f"\nDeployed config, binding-row terms over {len(terms)} braked "
          f"cycles (medians, mm):")
    med = np.median(terms[:, :7], axis=0) * 1000.0
    print(f"  h={med[0]:.1f}  Gp_drift={med[1]:+.1f}  (L/2)|p|^2={med[2]:.1f}  "
          f"c={med[3]:.1f}  Gdq*T={med[4]:+.1f}  L*pdq*T={med[5]:+.1f}  "
          f"|p|={med[6]:.1f} mrad")
    fingers = np.isin(terms[:, 7].astype(int), [7, 8])
    print(f"  binding row is a finger: {100 * fingers.mean():.1f}% of braked cycles")


if __name__ == '__main__':
    main()
