#!/usr/bin/env python3
"""1 kHz reflex brake layer under the 100 Hz CBF-QP.

The CBF-QP is a discrete-time safety filter: it certifies hdot >= -alpha(h)
only at solve instants, and the velocity command is zero-order-held for ~10 ms
between solves, during which h can dip unmonitored. This node closes that gap
without touching the QP budget ("heavy decision slow, cheap brake fast"):

  * The CBF node publishes /cbf_safety/reflex_state at the solver rate:
    per-link barrier values h_l, dynamic-scene tightening env_hdot_l, and
    joint-space gradients grad_h_l, plus the measured q0 the solve used.
  * This node, at ~1 kHz, evaluates each barrier under the currently
    commanded velocity in one of two modes (~mode):

    'linear' (legacy default): first-order extrapolation with the frozen
    gradient — an ESTIMATE, not a bound:
        h_now_l  = h_l + grad_h_l . p                 (drift since the solve)
        hdot_l   = grad_h_l . dq_cmd - env_hdot_l     (commanded h-rate)
        h_pred_l = h_now_l + hdot_l * T
    and if any h_pred_l heads below ~h_stop it scales the WHOLE velocity by
    a single alpha in [0, 1]:
        alpha_l = (h_now_l - h_stop) / (-hdot_l * T)
        alpha   = min over violating rows.

THE HORIZON T IS MEASURED, NOT CONFIGURED. It is this node's OWN decision
period (a leaky max of the achieved loop period, ~1-2 ms), not the CBF's
solve period: the reflex publishes a fresh alpha every tick, so the only
interval it must guarantee is the one until it decides again. Safety over
the whole run then follows by induction over ticks — each tick certifies the
segment its own command holds, and the segments chain. Nothing in that
argument depends on WHEN the next CBF command arrives, which is the point:
a slow or stalled QP no longer breaks the guarantee, it only ages the state
(bigger drift p, bigger remainder, smaller alpha), so the brake tightens on
its own instead of silently exceeding a hand-set horizon. The unmeasured
tail since the last /joint_states sample (30 Hz in sim) is dead-reckoned
into p at the velocity actually applied then: past motion is history and is
charged unconditionally, never scaled by the alpha now being solved for.
The knob that used to be ~horizon is now ~h_stop — a distance, which is
measurable, rather than a latency guess.

    'certified': sampled-data CBF bound in the sense of Breeden, Garg &
    Panagou [2]. Assume each row's gradient is Lipschitz in q with constant
    L = ~grad_lipschitz. Then for any joint displacement D from the solve
    state q0 the Taylor remainder is bounded:
        h(q0 + D) >= h_l + grad_h_l . D - (L/2) ||D||^2        (a BOUND).
    A velocity-controlled arm under zero-order hold is a single integrator,
    so the inter-sample path is exactly a straight joint-space segment; with
    the drift p from the solve state and the future segment alpha*dq*T, the
    barrier lower bound along the hold interval is concave in path time,
    hence certifying the ENDPOINT certifies the whole interval [2]. The
    end-of-horizon certificate per row l is the scalar quadratic
        a_l + b_l*alpha - c*alpha^2 >= 0, with
        a_l = h_l + g_l.p - (L/2)||p||^2
              - env_l*(t_elapsed + T) - h_stop
        b_l = (g_l.dq - L*(p.dq)) * T
        c   = (L/2) ||dq||^2 T^2
    (env_l is charged over the full time since the solve, not just the
    horizon), and the largest admissible per-row scale stays closed-form:
        alpha_l = (b_l + sqrt(b_l^2 + 4*c*a_l)) / (2c)
    which reduces to the linear rule as L -> 0. alpha = min over rows.
    ~grad_lipschitz may be one constant PER ROW (cbf_link_names order): the
    barrier curvature is dominated by the distal links, so per-link values
    (from sdf_gradient_experiments/lipschitz_bound_study.py) avoid a global
    worst-case compromise.
    Unlike the linear mode this brakes OUTWARD-commanded rows too when
    their certificate fails (under curvature, motion along +grad can still
    lower true h); the only exemption is a row already at/below h_stop
    with an outward command — that is recovery, and a pure brake must not
    zero a recovery velocity (stays the QP / recovery stack's job).
    Verified numerically (20k random trials, worst-case Hessians with
    ||H|| = L): every alpha the certificate issues keeps the true h >=
    h_stop over the entire hold interval; the only excursions occur in the
    two exempt already-below-stop cases, which no brake can prevent.

    UNIFORM EXPONENTIAL CONDITION (~exp_kappa > 0, certified mode only):
    the certified requirement per row changes from the invariance floor
    "h_end >= h_stop" to the discrete exponential CBF condition
        h_end >= (1 - kappa*T) * h_now
    (Agrawal & Sreenath [1], the sampled analogue of hdot >= -kappa*h),
    evaluated on the certified lower bounds. This is the SAME condition
    the QP enforces, checked at the reflex rate with the Lipschitz
    remainder — and it restores the half of the zeroing-CBF theorem the
    floor check drops: for h < 0 the condition REQUIRES certified
    increase (attractivity of the safe set, Xu, Tabuada, Grizzle & Ames,
    "Robustness of Control Barrier Functions for Safety Critical
    Control", ADHS 2015), so a QP recovery command passes at the largest
    certifiable alpha instead of hitting the floor mode's a<=0 dead-end
    (runPP1022: alpha = 0 for 5 s at h = -0.2 mm while the QP commanded
    a certified outward recovery). With h < 0 rows the per-row feasible
    alpha-set is an INTERVAL excluding 0; if the intervals of all rows
    have no common point, no scalar brake is certifiable and the node
    publishes zeros (a jointly-recovering direction is the QP's job —
    see ~cbf_repair_all_rows in the CBF node). Floor semantics (h_stop)
    are unchanged when ~exp_kappa is 0. Match ~exp_kappa to the QP's
    cbf_kappa: near the boundary the QP band is tighter than kappa*h, so
    a fresh QP-certified command always passes.

    ISSf DISTURBANCE CHARGE (~issf_epsilon > 0): certificate assumption
    (i) — perfect velocity tracking — is replaced by the identified bound
    ||dq_real - dq_cmd|| <= epsilon (Kolathaya & Ames, "Input-to-State
    Safety with Control Barrier Functions", IEEE L-CSS 2019, matching the
    QP's cbf_issf_epsilon). Two charges per row, both proportional to
    ||g_l||*epsilon: the dead-reckoned drift tail since the last
    /joint_states sample (position uncertainty of h_now: the ZOH odometry
    assumed the command was executed) and the disturbance over the coming
    hold T. 0 = off (legacy: tracking assumed exact).

Scaling toward zero along the QP-certified direction is a pure brake: for
state-only barriers it cannot create a new violation of any other constraint,
so no QP, feasibility reasoning, or constraint deconfliction is needed here —
a few numpy flops per tick. Braking applies instantly; release is low-passed
(~release_tau) to avoid 1 kHz chatter at the margin.

The linear mode prevents inter-sample constraint excursion under a
first-order model of h; the certified mode upgrades that to a guarantee over
the whole hold interval under the assumptions below. Neither recovers from
h < 0 (that stays the QP / escape stack's job), and the node never scales up
or steers. If the COMMAND is stale the node publishes zeros (a strict safety
improvement over the controller holding the last velocity forever).

STALE REFLEX STATE splits the two modes, and it is the reason to prefer
'certified' on anything that matters. The linear mode has no remainder term
to lean on, so a stale gradient is simply wrong and it stands down to
passthrough after ~state_timeout — i.e. exactly when the QP is too slow, the
brake is gone. The certified mode has no such hole: staleness enters as a
larger drift p, whose remainder -(L/2)||p||^2 eats the certificate, so it
brakes harder the longer it waits. Tick simulation of a nominal driving
relentlessly at a wall (30 Hz joint_states, h_stop = 0):

    CBF rate      linear            certified (L=2)
    100 Hz        h_min = 0.000     h_min = 0.000
     10 Hz        h_min = 0.000     h_min = 0.000
      1 Hz        h_min = -0.629    stops at 0.000, holds
    one solve     h_min = -0.900    stops at 0.058, holds

so the certified mode holds the fort at ANY solver rate, including none.

CERTIFICATE ASSUMPTIONS (certified mode; each violation degrades the proof
gracefully back to "conservative estimate"):
  (i)   the velocity controller tracks the commanded dq exactly between
        samples (single-integrator ZOH path; controller lag is unmodeled).
        This carries more weight now that T is one tick: the brake is
        certified to act within ~1-2 ms of the decision, so any actuation
        lag beyond that is uncovered. It also makes the brake late and
        sharp by construction (a 1 ms horizon only sees a violation ~1 ms
        of travel away); buy standoff with ~h_stop, a distance, not with a
        padded horizon;
  (ii)  each row's gradient is L-Lipschitz along the traversed segment.
        The Softmin barrier is C^1 so a finite local L exists, but its
        curvature grows with the softmin temperature and spikes when the
        critical obstacle point switches; the node publishes a per-update
        empirical estimate as /cbf_reflex/status[4] — keep ~grad_lipschitz
        above the bulk of that signal (transient switch spikes excepted);
  (iii) h_l and grad_h_l were exact at the solve state. The Softmin VALUE
        crowding bias under dense clouds violates this — run the CBF with
        cbf_barrier_value_mode:=hard for an unbiased h;
  (iv)  env_hdot_l upper-bounds the environment-induced barrier decay.

REFERENCES (verify exact bibliographic details before citing in print):
  [1] A. Agrawal and K. Sreenath, "Discrete Control Barrier Functions for
      Safety-Critical Control of Discrete Systems with Application to
      Bipedal Robot Navigation," RSS 2017.  (discrete-time CBF condition)
  [2] J. Breeden, K. Garg, and D. Panagou, "Control Barrier Functions in
      Sampled-Data Systems," IEEE Control Systems Letters (L-CSS), vol. 6,
      2022.  (margin-tightened sampled-data CBF; the quadratic-remainder
      certificate above specializes their inter-sample bound to
      single-integrator ZOH segments)
  [3] A. J. Taylor et al., "Safety of Sampled-Data Systems with Control
      Barrier Functions via Approximate Discrete Time Models," IEEE CDC
      2022.  (sampled-data CBF via discrete-time approximation)
  [4] U. Rosolia, A. Singletary, and A. D. Ames, "Unified Multirate
      Control: From Low-Level Actuation to High-Level Planning," IEEE
      Trans. Automatic Control, 2022.  (multi-rate layered safety:
      "heavy decision slow, cheap brake fast")
  [5] A. Singletary, S. Kolathaya, and A. D. Ames, "Safety-Critical
      Kinematic Control of Robotic Systems," IEEE L-CSS, 2022.
      (velocity-level CBFs on manipulators)
  [6] A. D. Ames et al., "Control Barrier Functions: Theory and
      Applications," ECC 2019.  (survey; the class-K condition the QP
      layer enforces)
  [7] ISO/TS 15066:2016, "Robots and robotic devices — Collaborative
      robots."  (speed-and-separation monitoring: the industrial analogue
      of distance-scaled velocity braking)

Python + rospy will not hold a hard 1 kHz (expect ~0.7-1 kHz with jitter);
that is fine for the simulation study. On real hardware this loop is the
libfranka 1 kHz callback.

Wiring (see enable_cbf_reflex in the CASF launches): the CBF node's velocity
output is remapped to ~in_velocity_topic and this node publishes to the real
controller topic.
"""
import os
import sys
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray

try:
    import rospkg
    _pkg = rospkg.RosPack().get_path('vision_processing')
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..'))
from pipeline_timing import TimingPublisher

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

# Leaky-max decay for the measured loop period, per tick (~0.2 s of memory at
# 1 kHz): covers scheduling jitter without keeping one stall spike charged
# into the horizon forever.
TICK_DECAY = 0.98


def certified_alpha_intervals(a, b, c, tol=1e-12):
    """Largest jointly-certifiable brake scale for rows a + b*al - c*al^2 >= 0.

    The quadratic is concave (c >= 0), so each row's feasible set within
    [0, 1] is an interval [lo, hi]. Rows with a >= 0 contain al = 0 (full
    stop certifiable); rows with a < 0 — already below the required decay
    curve — need a MINIMUM authority lo > 0: only letting their (outward)
    command act is certifiable, not braking it away. The single published
    scale must lie in every row's interval.

    Returns (alpha, feasible): alpha = min(hi) over rows, feasible = True
    iff every row is non-empty and min(hi) >= max(lo) - tol. On conflict
    (feasible False) no scalar alpha satisfies all certificates and the
    caller must fall back to a full stop.
    """
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    c = np.broadcast_to(np.asarray(c, dtype=np.float64), a.shape)
    lo = np.zeros_like(a)
    hi = np.ones_like(a)
    quad = c > tol
    # Linear rows: a + b*al >= 0.
    linm = ~quad
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = linm & (b > tol)          # increasing: al >= -a/b
        lo[pos] = np.maximum(0.0, -a[pos] / b[pos])
        neg = linm & (b < -tol)         # decreasing: al <= a/(-b)
        hi[neg] = np.minimum(1.0, a[neg] / (-b[neg]))
        flat = linm & (np.abs(b) <= tol) & (a < -tol)
        lo[flat], hi[flat] = 1.0, -1.0  # empty
        # Quadratic rows: feasible between the roots of c*al^2 - b*al - a.
        disc = b * b + 4.0 * c * a
        bad = quad & (disc < 0.0)
        lo[bad], hi[bad] = 1.0, -1.0    # empty
        okq = quad & (disc >= 0.0)
        sq = np.sqrt(np.maximum(disc, 0.0))
        lo[okq] = np.maximum(0.0, (b[okq] - sq[okq]) / (2.0 * c[okq]))
        hi[okq] = np.minimum(1.0, (b[okq] + sq[okq]) / (2.0 * c[okq]))
    if np.any(lo > hi + tol):
        return 0.0, False
    alpha = float(np.clip(hi.min(), 0.0, 1.0))
    feasible = alpha >= float(lo.max()) - tol
    return (alpha, True) if feasible else (0.0, False)


class CbfReflexNode:
    def __init__(self):
        rospy.init_node('cbf_reflex')

        self.joint_names = list(rospy.get_param('~joint_names', JOINT_NAMES))
        self.rate_hz = float(rospy.get_param('~rate_hz', 1000.0))
        # Certificate horizon [s]: MEASURED, not a parameter (see module
        # docstring). The reflex re-decides every tick, so the interval it
        # must guarantee is its own loop period — tracked as a leaky max of
        # the achieved period so jitter is covered. Published as
        # /cbf_reflex/status[5]. Nothing here waits on the CBF's period.
        self.horizon = 1.0 / self.rate_hz
        # Extrapolated h below this -> brake. 0 = the QP's own margin surface.
        self.h_stop = float(rospy.get_param('~h_stop', 0.0))
        # Rows with h_now above this are ignored (cheap prefilter; also keeps
        # the far-init h=1 rows of untouched links inert).
        self.h_activate = float(rospy.get_param('~h_activate', 0.15))
        # Reflex state older than this -> passthrough. LINEAR MODE ONLY: its
        # first-order estimate has no remainder term, so a stale gradient is
        # simply wrong and the node stands down. The certified mode needs no
        # such escape hatch and ignores this: its drift remainder
        # (L/2)||p||^2 grows with the un-refreshed excursion, so a stale
        # state brakes on its own (verified in sim: a 1 Hz CBF stops the arm
        # ~10 cm out and holds it there until a fresh state arrives, instead
        # of passing the command through). That IS the point of the layer —
        # it keeps the fort until the QP speaks again.
        self.state_timeout = float(rospy.get_param('~state_timeout', 0.25))
        # No fresh velocity command within this -> command zeros.
        self.command_timeout = float(rospy.get_param('~command_timeout', 0.3))
        # Brake engages instantly; release recovers with this time constant.
        self.release_tau = float(rospy.get_param('~release_tau', 0.05))
        # 'linear' = legacy first-order estimate; 'certified' = sampled-data
        # CBF bound with the Lipschitz remainder (see module docstring, [2]).
        self.mode = str(rospy.get_param('~mode', 'linear')).strip().lower()
        if self.mode not in ('linear', 'certified'):
            rospy.logwarn("cbf_reflex: unknown ~mode '%s', using 'linear'",
                          self.mode)
            self.mode = 'linear'
        # Certified mode: Lipschitz constant of each gradient row [m/rad^2],
        # i.e. ||g(q) - g(q')|| <= L ||q - q'||. Scalar, or one value PER ROW
        # (space/comma-separated string or list, in cbf_link_names order):
        # the deployed barrier's curvature is dominated by the distal links
        # (finger polynomial ringing), so a per-link vector keeps arm-body
        # braking tight without a global worst-case compromise. Values from
        # sdf_gradient_experiments/lipschitz_bound_study.py (one-sided
        # curvature sups of the exact analytic gradient); validate against
        # the empirical estimate published as /cbf_reflex/status[4].
        # Uniform exponential condition (certified mode; module docstring).
        # 0 = legacy invariance floor h_end >= h_stop. > 0 = enforce
        # h_end >= (1 - exp_kappa*T) * h_now per row: same class-K condition
        # as the QP, so it also certifies RECOVERY (h < 0 demands increase)
        # instead of freezing on it. Match the QP's cbf_kappa.
        self.exp_kappa = float(rospy.get_param('~exp_kappa', 0.0))
        # h < 0 branch of the exp-mode class-K: demanded recovery rate is a
        # ~recovery_slack fraction of the QP's own SATURATING recovery law
        # rec_speed*min(-h/rec_depth, 1), NOT exp_kappa*|h|. Extending
        # kappa*h below zero demands kappa*|h| of certified outward speed -
        # at kappa 25 and h = -2.2 mm that is 55 mm/s against a QP recovery
        # stack that delivers ~12 mm/s BY DESIGN (recovery_speed cap), so
        # the per-row alpha interval goes empty and the reflex vetoes the
        # very retreat that would fix the violation (runPP1023 v5: grasp-
        # weld yank drove hand/link4 to -3 mm, then 10 s frozen at -2.2 mm
        # with the QP commanding certified recovery the whole time). Same
        # nested-class-K rule as exp_kappa vs the band slope, applied to
        # the other side of zero: the backstop's demand must be strictly
        # weaker than what the shaping layer can certify. alpha(h) stays a
        # valid extended class-K (increasing through 0, saturating), so
        # invariance AND attractivity are preserved. Match ~recovery_speed
        # / ~recovery_depth to the QP's cbf_recovery_speed / _depth.
        self.recovery_speed = float(rospy.get_param('~recovery_speed', 0.05))
        self.recovery_depth = float(rospy.get_param('~recovery_depth', 0.015))
        self.recovery_slack = float(rospy.get_param('~recovery_slack', 0.5))
        # ISSf tracking-disturbance bound [rad/s] (module docstring); match
        # the QP's cbf_issf_epsilon. 0 = assume exact tracking (legacy).
        # With ~issf_rho > 0 the bound is speed-dependent, eps(v) = eps0 +
        # rho * v, evaluated at the speed of the command being certified
        # (matching the QP's cbf_issf_rho identification): the alpha-
        # proportional part folds into the linear coefficient b, so braking
        # earns back disturbance headroom.
        self.issf_epsilon = float(rospy.get_param('~issf_epsilon', 0.0))
        self.issf_rho = float(rospy.get_param('~issf_rho', 0.0))
        gl = rospy.get_param('~grad_lipschitz', 2.0)
        if isinstance(gl, str):
            gl = [float(x) for x in gl.replace(',', ' ').split()]
        if isinstance(gl, (list, tuple)):
            self.grad_lipschitz = np.asarray(gl, dtype=np.float64)
        else:
            self.grad_lipschitz = float(gl)
        self.in_topic = rospy.get_param(
            '~in_velocity_topic', '/cbf_safety/pre_reflex_command')
        self.out_topic = rospy.get_param(
            '~out_velocity_topic', '/joint_group_velocity_controller/command')
        self.status_divisor = int(rospy.get_param('~status_divisor', 20))
        # Per-tick compute time / achieved loop period on /pipeline/timing/*,
        # systematically sampled every Nth tick so the 1 kHz loop is not taxed
        # by its own instrumentation (an unbiased sample for the boxplot).
        self.timing_divisor = int(rospy.get_param('~timing_divisor', 10))
        self.timing = TimingPublisher(
            enabled=rospy.get_param('~publish_timing', True))
        self._last_tick_pc = None

        self._lock = threading.Lock()
        self._cmd = None          # (dq[7], t_recv)
        self._state = None        # (t_solve, q0[7], h[n], env[n], G[n,7], t_recv)
        self._state_gen = -1      # selection generation of the last state
        self._q = None            # (q[7], t_recv, odom[7] at t_recv)
        self._alpha = 1.0
        self._dq_applied = np.zeros(7)   # last velocity actually published
        # Running integral of the applied velocity ("ZOH odometry"): under
        # assumption (i) this is exactly the joint path, so differencing it
        # against its value at the last /joint_states sample closes the gap
        # between that sample and now. Free-running; never reset (a race-free
        # snapshot beats a shared accumulator the callback would clear).
        self._odom = np.zeros(7)
        self._last_tick_t = None         # ROS time of the previous tick
        self._tick = 0
        self._brake_ticks = 0
        self._had_cmd = False
        # Empirical gradient-Lipschitz estimate from consecutive reflex
        # states (assumption (ii) monitor); updated at the solver rate.
        self._L_emp = 0.0
        # Monitored-assumption veto: the CBF node's ISSf monitor reports when
        # the realized plant disturbance exceeded the assumed epsilon on an
        # active barrier row. The certificate is void for such ticks, so the
        # reflex brakes fully for a short hold. 0 disables the veto.
        self.issf_veto_hold = float(
            rospy.get_param('~issf_veto_hold', 0.05))
        self._issf_veto_until = 0.0

        rospy.Subscriber(self.in_topic, Float64MultiArray,
                         self._cmd_cb, queue_size=1)
        rospy.Subscriber('/cbf_safety/reflex_state', Float32MultiArray,
                         self._state_cb, queue_size=1)
        rospy.Subscriber('/joint_states', JointState,
                         self._joint_cb, queue_size=1)
        if self.issf_veto_hold > 0.0:
            rospy.Subscriber('/cbf_safety/issf_monitor', Float32MultiArray,
                             self._issf_monitor_cb, queue_size=1)

        self.pub = rospy.Publisher(self.out_topic, Float64MultiArray,
                                   queue_size=1)
        # [alpha, min h_pred, n rows braking, state age (s),
        #  empirical gradient-Lipschitz estimate, measured horizon (s)]
        self.status_pub = rospy.Publisher('/cbf_reflex/status',
                                          Float32MultiArray, queue_size=1)

        gl_str = (np.array2string(self.grad_lipschitz, precision=1)
                  if isinstance(self.grad_lipschitz, np.ndarray)
                  else '%.2f' % self.grad_lipschitz)
        rospy.loginfo(
            'cbf_reflex ready: %s -> %s at %.0f Hz | mode=%s '
            'grad_lipschitz=%s horizon=measured (per-tick, ~%.1fms) '
            'h_stop=%.3fm h_activate=%.3fm release_tau=%.3fs '
            'exp_kappa=%.1f issf_eps=%.3f+%.3f*v',
            self.in_topic, self.out_topic, self.rate_hz, self.mode,
            gl_str, 1e3 / self.rate_hz, self.h_stop,
            self.h_activate, self.release_tau,
            self.exp_kappa, self.issf_epsilon, self.issf_rho)

    # ── callbacks (numpy only, atomic single-ref swaps) ────────────────────

    def _cmd_cb(self, msg):
        if len(msg.data) >= 7:
            self._cmd = (np.asarray(msg.data[:7], dtype=np.float64),
                         rospy.get_time())
            self._had_cmd = True

    def _state_cb(self, msg):
        d = np.asarray(msg.data, dtype=np.float64)
        if d.shape[0] < 2:
            return
        n = int(round(d[1]))
        if d.shape[0] < 2 + 7 + n * 9:
            return
        q0 = d[2:9]
        h = d[9:9 + n]
        env = d[9 + n:9 + 2 * n]
        G = d[9 + 2 * n:9 + 9 * n].reshape(n, 7)
        # Optional trailing selection-generation counter (newer CBF nodes):
        # differencing gradients ACROSS a critical-point selection update
        # measures a function change, not curvature, so those pairs are
        # excluded from the Lipschitz monitor. -1 = not published (legacy).
        gen = int(round(d[9 + 9 * n])) if d.shape[0] > 9 + 9 * n else -1
        prev = self._state
        prev_gen = self._state_gen
        self._state = (d[0], q0, h, env, G, rospy.get_time())
        self._state_gen = gen
        # Empirical Lipschitz monitor for certificate assumption (ii): the
        # rows are in fixed per-link order, so consecutive states give
        # ||dg|| / ||dq0|| per row. Only near-barrier rows carry real
        # gradients (far links are inert placeholders). Remaining spikes on
        # a gated (same-generation) pair are REAL curvature, not switches;
        # the bulk should sit below ~grad_lipschitz.
        if prev is not None and prev[4].shape == G.shape \
                and (gen < 0 or gen == prev_gen):
            dq0 = float(np.linalg.norm(q0 - prev[1]))
            if dq0 > 1e-4:
                rows = (h < self.h_activate) & (prev[2] < self.h_activate)
                if np.any(rows):
                    emp = np.linalg.norm(
                        G[rows] - prev[4][rows], axis=1) / dq0
                    self._L_emp = float(emp.max())
                    L = self.grad_lipschitz
                    L_rows = L[rows] if isinstance(L, np.ndarray) \
                        and L.shape[0] == n else np.max(L)
                    if self.mode == 'certified' and \
                            np.any(emp > 2.0 * L_rows):
                        rospy.logwarn_throttle(
                            5.0,
                            'cbf_reflex: empirical gradient Lipschitz %.2f '
                            '>> assumed ~grad_lipschitz (same-generation '
                            'pair: real curvature, L set too low?)',
                            self._L_emp)

    def _issf_monitor_cb(self, msg):
        # [t, eps_proj, ||d||, exceeded, running max] from the CBF node.
        if len(msg.data) >= 4 and msg.data[3] > 0.5:
            self._issf_veto_until = rospy.get_time() + self.issf_veto_hold

    def _joint_cb(self, msg):
        pos = {nm: p for nm, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in self.joint_names]
        if None not in q:
            self._q = (np.asarray(q, dtype=np.float64), rospy.get_time(),
                       self._odom)

    # ── 1 kHz loop ─────────────────────────────────────────────────────────

    def _compute_alpha(self, dq, now):
        """Worst-case brake scale over the active barrier rows."""
        state = self._state
        qs = self._q
        if state is None or qs is None:
            return 1.0, float('inf'), 0, float('inf')
        t_solve, q0, h, env, G, t_recv = state
        age = now - t_recv
        if age > self.state_timeout and self.mode != 'certified':
            return 1.0, float('inf'), 0, age

        # Drift from the solve state. /joint_states is slower than this loop
        # (30 Hz in sim), so the tail since the last sample is closed with
        # the ZOH odometry: the exact integral of the velocities actually
        # applied over it, not the latest one held constant (which
        # under-counts travel while braking — worth ~1 mm at 0.3 m/s). That
        # motion is history and is charged unconditionally, never scaled by
        # the alpha solved for here. It also degrades safely: a stalled
        # /joint_states inflates p until the certificate brakes to zero.
        q, _t_q, odom_q = qs
        p = (q - q0) + (self._odom - odom_q)
        if self.mode == 'certified':
            return self._alpha_certified(
                dq, now, t_solve, p, h, env, G, age,
                meas_gap=max(now - _t_q, 0.0))

        h_now = h + G @ p
        hdot = G @ dq - env
        active = (h_now < self.h_activate) & (hdot < 0.0)
        if not np.any(active):
            return 1.0, float('inf'), 0, age

        h_a = h_now[active]
        hd_a = hdot[active]
        h_pred = h_a + hd_a * self.horizon
        min_pred = float(h_pred.min())
        viol = h_pred < self.h_stop
        if not np.any(viol):
            return 1.0, min_pred, 0, age
        # Largest alpha such that h_now + alpha*hdot*horizon >= h_stop,
        # per violating row; 0 if already at/below the stop margin.
        alphas = (h_a[viol] - self.h_stop) / (-hd_a[viol] * self.horizon)
        alpha = float(np.clip(alphas.min(), 0.0, 1.0))
        return alpha, min_pred, int(viol.sum()), age

    def _alpha_certified(self, dq, now, t_solve, p, h, env, G, age,
                         meas_gap=0.0):
        """Sampled-data CBF certificate (module docstring, [2]).

        Solves, per inward-commanded row, the largest alpha in [0, 1] with
            a + b*alpha - c*alpha^2 >= 0
        i.e. the Lipschitz-remainder lower bound on h at the end of the
        horizon stays >= h_stop (floor mode) or >= (1 - exp_kappa*T)*h_now
        (uniform exponential mode, ~exp_kappa); concavity along the
        straight ZOH segment makes the endpoint check sufficient for the
        whole interval.
        """
        # L: scalar, or one constant per row (per-link curvature). A vector
        # whose length does not match the received rows falls back to its
        # max (conservative) rather than guessing an alignment.
        L = self.grad_lipschitz
        if isinstance(L, np.ndarray) and L.shape[0] != h.shape[0]:
            L = float(L.max())
        T = self.horizon
        # Row-independent scalars of the quadratic (c is per-row iff L is).
        p2 = float(p @ p)
        pdq = float(p @ dq)
        c = np.broadcast_to(
            np.asarray(0.5 * L * float(dq @ dq) * T * T), h.shape)
        t_elapsed = max(now - t_solve, 0.0)

        Gp = G @ p
        Gdq = G @ dq
        # Certified CURRENT value: measured drift with its remainder bound.
        h_now = h + Gp - 0.5 * L * p2
        b = (Gdq - L * pdq) * T
        if self.issf_epsilon > 0.0 or self.issf_rho > 0.0:
            # ISSf charges (module docstring): the dead-reckoned drift tail
            # is position uncertainty of h_now (the ZOH odometry assumed
            # exact tracking over the unmeasured gap); the hold charge is
            # the disturbance's worst h-decay over the coming horizon. The
            # constant (eps0) parts and the past-gap charge do not scale
            # with alpha (history and floor disturbance are not braked);
            # the rho part of the FUTURE hold scales with the commanded
            # speed alpha*||dq|| and therefore folds into b.
            g_norm = np.linalg.norm(G, axis=1)
            dq_norm = float(np.linalg.norm(dq))
            # The gap charge is HISTORY: it must be priced at the velocity
            # that was actually APPLIED over the unmeasured tail (already
            # alpha-scaled; ~0 when parked), never at the candidate command
            # being certified now. Charging eps(||dq_cand||) here made the
            # parked state infeasible on its own history (runPP1023 v3:
            # alpha=0 lock at h=+4 mm with the arm stationary).
            dq_applied_norm = float(np.linalg.norm(self._dq_applied))
            eps_gap = self.issf_epsilon + self.issf_rho * dq_applied_norm
            h_now = h_now - g_norm * eps_gap * meas_gap
            issf_T = g_norm * self.issf_epsilon * T
            b = b - g_norm * self.issf_rho * dq_norm * T
        else:
            issf_T = 0.0

        if self.exp_kappa > 0.0:
            # Uniform exponential condition h_end >= h_now - alpha_K(h_now)*T
            # on the certified bounds: allows decay at exp_kappa*h when
            # h_now > 0, REQUIRES certified increase when h_now < 0
            # (attractivity — a recovery command passes at the largest
            # certifiable alpha; a full stop is no longer certifiable).
            # The h<0 demand follows the slack-scaled QP recovery law (see
            # ~recovery_slack): saturating extended class-K, feasible by
            # construction against the QP's own recovery commands.
            rec_demand = (self.recovery_slack * self.recovery_speed
                          * np.minimum(np.maximum(-h_now, 0.0)
                                       / max(self.recovery_depth, 1e-6),
                                       1.0))
            allow = np.where(h_now >= 0.0,
                             self.exp_kappa * h_now, -rec_demand)
            a = allow * T - env * (t_elapsed + T) - issf_T
            active = h_now < self.h_activate
            if not np.any(active):
                return 1.0, float('inf'), 0, age
            a_a, b_a, c_a = a[active], b[active], c[active]
            end_full = a_a + b_a - c_a    # certificate margin at alpha = 1
            min_pred = float(end_full.min())
            if np.all(end_full >= 0.0):
                return 1.0, min_pred, 0, age
            alpha, feasible = certified_alpha_intervals(a_a, b_a, c_a)
            if not feasible:
                # Per-row feasible alpha-intervals have no common point: a
                # scalar brake cannot serve rows demanding authority and
                # rows demanding a stop at once. Safest reachable action is
                # a full stop; a jointly-recovering DIRECTION is the QP's
                # job (cbf_repair_all_rows).
                rospy.logwarn_throttle(
                    1.0, 'cbf_reflex: conflicting per-row certificates '
                         '(no scalar alpha feasible), commanding stop')
                alpha = 0.0
            n_brake = int(np.sum(end_full < 0.0))
            return alpha, min_pred, n_brake, age

        a = h_now - env * (t_elapsed + T) - issf_T - self.h_stop
        # Unlike the linear mode, OUTWARD-commanded rows are braked too when
        # their endpoint certificate fails (under curvature, motion along
        # +grad can still lower true h; the root formula is valid for any
        # sign of b when a > 0). The one exemption: rows ALREADY at/below
        # the stop bound with an outward command — that is recovery, and a
        # pure brake must never zero a recovery velocity (QP's job).
        recovery = (a <= 0.0) & (Gdq >= 0.0)
        active = (h_now < self.h_activate) & ~recovery
        if not np.any(active):
            return 1.0, float('inf'), 0, age

        a_a = a[active]
        b_a = b[active]
        c_a = c[active]
        end_full = a_a + b_a - c_a        # certificate margin at alpha = 1
        min_pred = float(end_full.min()) + self.h_stop
        viol = end_full < 0.0
        if not np.any(viol):
            return 1.0, min_pred, 0, age
        a_v = a_a[viol]
        b_v = b_a[viol]
        c_v = c_a[viol]
        # Per row: quadratic root, or the linear rule where L -> 0.
        lin = np.where(b_v < 0.0, a_v / np.maximum(-b_v, 1e-12), 1.0)
        disc = np.maximum(b_v * b_v + 4.0 * c_v * a_v, 0.0)
        quad = (b_v + np.sqrt(disc)) / np.maximum(2.0 * c_v, 1e-12)
        alphas = np.where(c_v < 1e-12, lin, quad)
        # Already at/below the stop bound now (inward-commanded, since
        # outward ones were exempted above) -> full brake for that row.
        alphas = np.where(a_v <= 0.0, 0.0, alphas)
        alpha = float(np.clip(alphas.min(), 0.0, 1.0))
        return alpha, min_pred, int(viol.sum()), age

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        dt_nom = 1.0 / self.rate_hz
        release_gamma = dt_nom / (dt_nom + max(self.release_tau, 1e-4))
        zeros = [0.0] * 7
        while not rospy.is_shutdown():
            now = rospy.get_time()
            # The horizon IS the achieved decision period (leaky max over
            # recent ticks, floored at nominal): what this tick's command
            # holds for before the next one replaces it.
            if self._last_tick_t is not None:
                dt_tick = now - self._last_tick_t
                self.horizon = max(dt_tick, dt_nom, TICK_DECAY * self.horizon)
                # The previous tick's command held for exactly dt_tick.
                # Rebind, never mutate: _joint_cb snapshots this reference.
                self._odom = self._odom + self._dq_applied * dt_tick
            self._last_tick_t = now
            cmd = self._cmd
            if cmd is None:
                # Never command before the CBF has: preserves startup order.
                rate.sleep()
                continue
            dq, t_cmd = cmd
            if now - t_cmd > self.command_timeout:
                # Upstream died mid-motion: do not let the velocity
                # controller hold the last command forever.
                self.pub.publish(Float64MultiArray(data=zeros))
                self._dq_applied = np.zeros(7)
                rate.sleep()
                continue

            t_tick = time.perf_counter()
            alpha_raw, min_pred, n_brake, age = self._compute_alpha(dq, now)
            if now < self._issf_veto_until:
                # The upstream ISSf monitor reported a disturbance beyond the
                # assumed epsilon on an active row: the certificate this brake
                # relies on is void, so hold until the monitor clears.
                # Recovery exemption (same philosophy as _compute_alpha): a
                # brake must never zero an outward recovery command, or the
                # veto pins the arm inside the margin it is meant to protect.
                state = self._state
                in_recovery = False
                if state is not None:
                    h_rows = state[2]
                    G_rows = state[4]
                    below = h_rows < 0.0
                    if np.any(below):
                        in_recovery = bool(
                            np.all((G_rows[below] @ dq) >= 0.0))
                if not in_recovery:
                    rospy.logwarn_throttle(
                        1.0, 'cbf_reflex: ISSf disturbance bound exceeded '
                             'upstream -- monitored-assumption veto (brake).')
                    alpha_raw = 0.0
            if alpha_raw < self._alpha:
                self._alpha = alpha_raw          # brake instantly
            else:
                self._alpha += release_gamma * (alpha_raw - self._alpha)
            self._dq_applied = self._alpha * dq
            self.pub.publish(Float64MultiArray(
                data=self._dq_applied.tolist()))
            if self.timing_divisor > 0 and \
                    self._tick % self.timing_divisor == 0:
                self.timing.publish(
                    'cbf_reflex_compute',
                    (time.perf_counter() - t_tick) * 1000.0)
                if self._last_tick_pc is not None:
                    self.timing.publish(
                        'cbf_reflex_period',
                        (t_tick - self._last_tick_pc) * 1000.0)
            self._last_tick_pc = t_tick

            if n_brake > 0:
                self._brake_ticks += 1
                rospy.logwarn_throttle(
                    1.0,
                    'REFLEX BRAKE: alpha=%.2f h_pred_min=%.4fm rows=%d '
                    'state_age=%.0fms' % (
                        self._alpha, min_pred, n_brake, age * 1e3))
            self._tick += 1
            if self.status_divisor > 0 and \
                    self._tick % self.status_divisor == 0:
                self.status_pub.publish(Float32MultiArray(
                    data=[self._alpha, min_pred, float(n_brake), age,
                          self._L_emp, self.horizon]))
            rate.sleep()


if __name__ == '__main__':
    CbfReflexNode().run()
