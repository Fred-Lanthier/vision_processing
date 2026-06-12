#!/usr/bin/env python3
"""
Standalone Bernstein CBF gradient debugger.

Runs WITHOUT Gazebo.  Requires only the venv and the package on PYTHONPATH.
If roscore is running, also publishes to RViz.

Usage:
    python3 scripts/test_cbf_gradient.py            # matplotlib only
    python3 scripts/test_cbf_gradient.py --rviz     # matplotlib + RViz arrows

What is tested for each (q, obstacles) scenario:
  1. Finite-diff gradient matches autograd grad_h  → checks BernsteinBarrier correctness
  2. h(q + eps * grad_h_unit) > h(q)              → checks that moving along grad_h is safe
  3. Cartesian grad_h points AWAY from obstacle    → catches gradient direction problems

Key fix vs. original: obstacles are placed relative to the fork MESH CENTER in world
frame (fork_origin + R_fork @ centroid_offset), NOT the fork_tip link origin.
The fork_tip link origin is the mount base; the tines extend ~91 mm along fork-local Z
above it.  Placing obstacles at the link origin puts them BELOW the bounding box, so
the barrier always returns the bottom-face normal (= fork-local Z direction), masking
the real per-direction behaviour.
"""

import os, sys, argparse
import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
if os.environ.get("DISPLAY"):
    matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

# ── Path setup ─────────────────────────────────────────────────────────────────
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, PKG_DIR)

import xacro, tempfile

try:
    import rospkg
    PKG_DIR = rospkg.RosPack().get_path('vision_processing')
except Exception:
    pass   # use path inferred from file location

from third_party.RDF.urdf_layer import URDFLayer
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.bernstein_barrier import BernsteinBarrier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LINK_NAMES = [
    'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
    'panda_hand', 'fork_tip',
]
VOXEL_DIR = os.path.join(PKG_DIR, 'third_party/RDF/panda_layer/meshes/voxel_128')


# ── Model loading ──────────────────────────────────────────────────────────────

def build_barrier(d_safe=0.015, alpha=0.001):
    """Exact replica of cbf_safety_node_Bernstein.__init__ model setup."""
    xacro_path = os.path.join(PKG_DIR, 'urdf/panda_camera.xacro')
    doc = xacro.process_file(xacro_path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(doc.toxml())
        urdf_path = f.name

    robot_layer = URDFLayer(
        urdf_path=urdf_path, device=DEVICE,
        package_dir=PKG_DIR, voxel_dir=VOXEL_DIR,
    )
    weights = RDF_Weights(device=DEVICE, dtype=torch.float32)
    weights.init_robot_folder(
        os.path.join(PKG_DIR, 'third_party/SDF_Bernstein_Basis/panda_test'),
        robot_name='panda',
    )
    weights.add_models(LINK_NAMES, robot_name='panda')
    core    = BernsteinCore(weights, robot_layer, DEVICE, LINK_NAMES)
    barrier = BernsteinBarrier(core, d_safe=d_safe, alpha=alpha)

    # Fork-tip Bernstein model geometry in fork-local frame.
    fork_centroid_local = weights.fork_tip_pt.centroid_offset.detach().cpu().numpy()
    fork_scale          = float(weights.fork_tip_pt.scale_factor)

    return barrier, robot_layer, fork_centroid_local, fork_scale


# ── FK helpers ─────────────────────────────────────────────────────────────────

def _q9(q7):
    q7t = torch.as_tensor(q7, dtype=torch.float32, device=DEVICE)
    if q7t.dim() == 1:
        q7t = q7t.unsqueeze(0)
    return torch.cat([q7t, torch.zeros(q7t.shape[0], 2, device=DEVICE)], dim=-1)

def fk_all(robot_layer, q7):
    """Return {link: (4,4) numpy} for all protected links."""
    poses = robot_layer._native_forward_kinematics(_q9(q7))
    return {k: poses[k][0].detach().cpu().numpy() for k in LINK_NAMES if k in poses}

def fork_transform_world(robot_layer, q7):
    """Full 4×4 world transform of fork_tip link frame."""
    return fk_all(robot_layer, q7)['fork_tip']

def fork_mesh_center_world(robot_layer, q7, centroid_local):
    """
    Approximate fork mesh center in world frame.

    The Bernstein model centroid (centroid_local) is expressed in fork_tip link-local
    coordinates.  Mapping it to world gives the geometric centre of the fork mesh —
    which is where obstacles should be placed to exercise the barrier correctly.
    """
    T   = fork_transform_world(robot_layer, q7)
    R   = T[:3, :3]
    t   = T[:3, 3]
    return t + R @ centroid_local

def link_transform_world(robot_layer, q7, link_name):
    """World transform of any protected link."""
    return fk_all(robot_layer, q7)[link_name]

def point_world_to_link(robot_layer, q7, link_name, p_world):
    T = link_transform_world(robot_layer, q7, link_name)
    R = T[:3, :3]
    t = T[:3, 3]
    return R.T @ (np.asarray(p_world, dtype=np.float64) - t)

def point_link_to_world(robot_layer, q7, link_name, p_link):
    T = link_transform_world(robot_layer, q7, link_name)
    R = T[:3, :3]
    t = T[:3, 3]
    return t + R @ np.asarray(p_link, dtype=np.float64)

def fork_local_axes_world(robot_layer, q7):
    """
    Columns of the fork_tip rotation matrix = fork-local X, Y, Z axes in world frame.
    fork-local Z is the tine direction (the fork extends along +Z in fork-local).
    """
    T = fork_transform_world(robot_layer, q7)
    return T[:3, :3]   # shape (3, 3); col i = fork-local axis i in world


# ── Gradient tools ─────────────────────────────────────────────────────────────

def grad_h_to_cartesian(robot_layer, q7, grad_h_joint, centroid_local, eps=0.02):
    """
    Map the 7-D joint-space grad_h to a Cartesian direction at the fork mesh centre
    via finite-difference FK.

    Using the mesh centre (not the link origin) is crucial: the link origin is the
    mount base; the fork extends ~91 mm above it.  Using the link origin gives the
    wrong Cartesian direction when the obstacle is near the tines.

    Returns (origin_xyz, cartesian_direction_unit).
    """
    g      = np.array(grad_h_joint, dtype=np.float64)
    g_norm = g / (np.linalg.norm(g) + 1e-8)
    p0 = fork_mesh_center_world(robot_layer, q7,                                centroid_local)
    p1 = fork_mesh_center_world(robot_layer, np.array(q7) + eps * g_norm, centroid_local)
    direction = p1 - p0
    n = np.linalg.norm(direction)
    return p0, direction / (n + 1e-8)


def _q_tensor(q_np):
    """Create a (1, 7) LEAF tensor — avoids the unsqueeze non-leaf problem."""
    return torch.tensor(q_np.reshape(1, 7), dtype=torch.float32, device=DEVICE)

def finite_diff_grad(barrier, q_np, obs_pt, eps=1e-4):
    """Central-difference gradient of h w.r.t. q (7-D).
    BernsteinBarrier.forward always calls torch.autograd.grad internally,
    so we must never use torch.no_grad() — barrier sets requires_grad_ internally.
    """
    obs  = torch.tensor(obs_pt.reshape(-1, 3), dtype=torch.float32, device=DEVICE)
    grad = np.zeros(7)
    for i in range(7):
        qp_np = q_np.copy(); qp_np[i] += eps
        qm_np = q_np.copy(); qm_np[i] -= eps
        hp, _, _ = barrier(_q_tensor(qp_np), obs)
        hm, _, _ = barrier(_q_tensor(qm_np), obs)
        grad[i] = (float(hp.detach().item()) - float(hm.detach().item())) / (2 * eps)
    return grad


def barrier_details(barrier, q_np, obs_pts):
    """
    Report the active link/point for the same scalar h used by the CBF.

    This is critical for debugging bowl/ring cases: the "nearest obstacle" in
    world space is not always the point/link that dominates the softmin barrier.
    """
    q_t = _q_tensor(q_np)
    obs_t = torch.tensor(obs_pts.reshape(-1, 3), dtype=torch.float32, device=DEVICE)
    eye4 = torch.eye(4, device=DEVICE).unsqueeze(0)
    _, sdf_per_link = barrier.core.get_whole_body_sdf_batch(
        obs_t, eye4, q_t, return_per_link=True)

    sdf_min, _ = sdf_per_link.min(dim=-1, keepdim=True)
    shifted_sdf = sdf_per_link - sdf_min
    exp_terms = torch.exp(-shifted_sdf / barrier.alpha)
    weights = exp_terms / (exp_terms.sum(dim=-1, keepdim=True) + 1e-12)
    h_per_link = (
        sdf_min.squeeze(-1)
        - barrier.alpha * torch.log(exp_terms.sum(dim=-1))
        - barrier.d_safe
    )

    active_link_idx = int(h_per_link[0].argmin().detach().cpu().item())
    active_link = LINK_NAMES[active_link_idx]
    active_point_idx = int(sdf_per_link[0, active_link_idx].argmin().detach().cpu().item())
    active_sdf = float(sdf_per_link[0, active_link_idx, active_point_idx].detach().cpu().item())

    flat_idx = int(sdf_per_link[0].reshape(-1).argmin().detach().cpu().item())
    n_obs = int(sdf_per_link.shape[-1])
    global_link_idx = flat_idx // n_obs
    global_point_idx = flat_idx % n_obs
    global_sdf = float(sdf_per_link[0, global_link_idx, global_point_idx].detach().cpu().item())

    link_weights = weights[0, active_link_idx].detach().cpu().numpy()
    top_count = min(5, link_weights.shape[0])
    top_idx = np.argsort(link_weights)[-top_count:][::-1]
    top_weights = [(int(i), float(link_weights[i])) for i in top_idx]

    return {
        "active_link_idx": active_link_idx,
        "active_link": active_link,
        "active_point_idx": active_point_idx,
        "active_sdf": active_sdf,
        "global_link": LINK_NAMES[global_link_idx],
        "global_point_idx": int(global_point_idx),
        "global_sdf": global_sdf,
        "top_weights": top_weights,
    }


def line_search_along_grad(barrier, q7, obs_pts, grad_h, h0, steps):
    g_unit = grad_h / (np.linalg.norm(grad_h) + 1e-8)
    rows = []
    obs_t = torch.tensor(obs_pts.reshape(-1, 3), dtype=torch.float32, device=DEVICE)
    for step in steps:
        q_step = q7 + step * g_unit
        h_step, _, _ = barrier(_q_tensor(q_step), obs_t)
        details = barrier_details(barrier, q_step, obs_pts)
        rows.append((step, float(h_step.detach().item()), details))
    return rows


def active_sdf_normal_world(barrier, q7, obs_pts, active_link_idx, active_point_idx):
    """
    Gradient of the active link SDF w.r.t. the active obstacle point in world.

    For a positive outside SDF, this normal points from the robot surface toward
    the obstacle point. A robot surface point should move along -normal to
    increase clearance.
    """
    q_t = _q_tensor(q7)
    obs_t = torch.tensor(
        obs_pts.reshape(-1, 3),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )
    eye4 = torch.eye(4, device=DEVICE).unsqueeze(0)
    _, sdf_per_link = barrier.core.get_whole_body_sdf_batch(
        obs_t, eye4, q_t, return_per_link=True)
    active_sdf = sdf_per_link[0, active_link_idx, active_point_idx]
    grad_x = torch.autograd.grad(
        outputs=active_sdf,
        inputs=obs_t,
        grad_outputs=torch.ones_like(active_sdf),
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0][active_point_idx]

    normal = grad_x.detach().cpu().numpy().astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-8
    return float(active_sdf.detach().cpu().item()), normal


def active_contact_motion(
        barrier, robot_layer, q7, obs_pts, grad_h, details, centroid_local,
        eps=1e-3):
    """
    Compare +grad_h against the motion of the active link contact surface.

    This is the more meaningful Cartesian debug than fork-centre motion. On a
    table or bowl, +grad_h can rotate the fork so the active surface moves away
    even if the fork centroid moves sideways or toward the obstacle.
    """
    obs_pts = obs_pts.reshape(-1, 3)
    link_name = details["active_link"]
    point_idx = details["active_point_idx"]
    obs_world = obs_pts[point_idx]

    sdf, normal_world = active_sdf_normal_world(
        barrier, q7, obs_pts, details["active_link_idx"], point_idx)
    expected_surface_motion = -normal_world
    surface_world = obs_world - sdf * normal_world
    surface_link = point_world_to_link(robot_layer, q7, link_name, surface_world)

    g_unit = grad_h / (np.linalg.norm(grad_h) + 1e-8)
    q_step = np.asarray(q7, dtype=np.float64) + eps * g_unit

    surface_after = point_link_to_world(robot_layer, q_step, link_name, surface_link)
    surface_dir = surface_after - surface_world
    surface_dir /= np.linalg.norm(surface_dir) + 1e-8

    centre_before = fork_mesh_center_world(robot_layer, q7, centroid_local)
    centre_after = fork_mesh_center_world(robot_layer, q_step, centroid_local)
    centre_dir = centre_after - centre_before
    centre_dir /= np.linalg.norm(centre_dir) + 1e-8

    return {
        "link": link_name,
        "point_idx": point_idx,
        "obs_world": obs_world,
        "sdf": sdf,
        "normal_world": normal_world,
        "surface_world": surface_world,
        "surface_dir": surface_dir,
        "centre_dir": centre_dir,
        "expected_surface_motion": expected_surface_motion,
        "surface_alignment": float(np.dot(surface_dir, expected_surface_motion)),
        "centre_alignment_to_normal": float(np.dot(centre_dir, expected_surface_motion)),
    }


# ── Single-scenario runner ─────────────────────────────────────────────────────

def run_scenario(name, q7, obs_pts, barrier, robot_layer, d_safe, centroid_local):
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")

    q7      = np.array(q7,      dtype=np.float64)
    obs_pts = np.array(obs_pts, dtype=np.float64)

    # Analytical h and grad_h
    q_t   = _q_tensor(q7)
    obs_t = torch.tensor(obs_pts.reshape(-1, 3), dtype=torch.float32, device=DEVICE)
    h_t, grad_h_t, min_idx = barrier(q_t, obs_t)
    h_val     = float(h_t.item())
    grad_h    = grad_h_t.squeeze(0).detach().cpu().numpy()
    nearest_i = int(min_idx.cpu().item())
    nearest_obs = obs_pts.reshape(-1, 3)[nearest_i]
    details = barrier_details(barrier, q7, obs_pts)

    print(f"  h          = {h_val:+.5f}   (negative → barrier violated)")
    print(f"  SDF_min    = {h_val + d_safe:.5f} m  (actual clearance ≈ this)")
    print(f"  ||grad_h|| = {np.linalg.norm(grad_h):.4f}")
    print(f"  Active h   = link {details['active_link']}  point {details['active_point_idx']}  "
          f"sdf={details['active_sdf']:+.5f}")
    print(f"  Global min = link {details['global_link']}  point {details['global_point_idx']}  "
          f"sdf={details['global_sdf']:+.5f}")
    print(f"  Softmin top weights on active link: {details['top_weights']}")

    # ── Check 1: finite-diff vs autograd ──────────────────────────────────────
    g_fd  = finite_diff_grad(barrier, q7, obs_pts, eps=1e-4)
    err   = np.linalg.norm(grad_h - g_fd)
    print(f"\n  [CHECK 1] Finite-diff error: {err:.6f}  "
          f"{'✅ match' if err < 0.05 else '❌ MISMATCH — autograd is wrong'}")
    if err >= 0.05:
        print(f"            autograd  : {np.round(grad_h, 4)}")
        print(f"            finite-diff: {np.round(g_fd, 4)}")

    # ── Check 2: h increases after a local step along grad_h ───────────────────
    # The old 0.03 rad test is intentionally too large for ring/cavity cases:
    # it can switch the active obstacle/link and make h decrease even when the
    # local gradient sign is correct. Keep it in the line search below, but do
    # not use it as the sign test.
    local_step = 1e-3
    large_step = 0.03
    g_unit = grad_h / (np.linalg.norm(grad_h) + 1e-8)
    h_local, _, _ = barrier(_q_tensor(q7 + local_step * g_unit), obs_t)
    h_large, _, _ = barrier(_q_tensor(q7 + large_step * g_unit), obs_t)
    h_local_val = float(h_local.detach().item())
    h_large_val = float(h_large.detach().item())
    delta_h_local = h_local_val - h_val
    delta_h_large = h_large_val - h_val
    print(f"\n  [CHECK 2] h after local step {local_step:g} rad along grad_h: "
          f"{h_local_val:+.5f}  (Δh = {delta_h_local:+.5f})  "
          f"{'✅ local h increases' if delta_h_local > -1e-5 else '❌ local h decreases'}")
    print(f"            h after large step {large_step:g} rad: {h_large_val:+.5f}  "
          f"(Δh = {delta_h_large:+.5f})")

    print("\n  [LINE SEARCH] step along +grad_h")
    for step_i, h_i, det_i in line_search_along_grad(
            barrier, q7, obs_pts, grad_h, h_val,
            steps=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)):
        print(f"            step={step_i:>7.4g}  h={h_i:+.5f}  "
              f"Δh={h_i - h_val:+.5f}  active={det_i['active_link']}[{det_i['active_point_idx']}]")

    # ── Check 3: Cartesian direction of fork mesh centre points away from obs ──
    # Using fork_mesh_center_world (not the link origin) so the reference point
    # is at the geometric centre of the fork, where the nearest obstacle lives.
    origin, cart_dir = grad_h_to_cartesian(robot_layer, q7, grad_h, centroid_local)
    to_obs     = nearest_obs - origin
    away_unit  = -to_obs / (np.linalg.norm(to_obs) + 1e-8)
    alignment  = float(np.dot(cart_dir, away_unit))
    print(f"\n  [CHECK 3] Cartesian alignment (fork-mesh-centre motion vs nearest-obs away dir): "
          f"{alignment:+.3f}")
    print(f"            {'✅ intuitive at fork centre'  if alignment > 0.0 else '⚠️ fork centre moves toward nearest point'}")
    print(f"            Fork mesh centre: {np.round(origin, 3)}")
    print(f"            Nearest obs:      {np.round(nearest_obs, 3)}")
    print(f"            grad_h Cartesian: {np.round(cart_dir, 3)}")
    print(f"            Expected dir:     {np.round(away_unit, 3)}")

    contact = active_contact_motion(
        barrier, robot_layer, q7, obs_pts, grad_h, details, centroid_local)
    print(f"\n  [CHECK 4] Active contact-surface motion under +grad_h")
    print(f"            Active:           {contact['link']}[{contact['point_idx']}]")
    print(f"            Obs point:        {np.round(contact['obs_world'], 3)}")
    print(f"            Surface point:    {np.round(contact['surface_world'], 3)}")
    print(f"            SDF normal:       {np.round(contact['normal_world'], 3)}")
    print(f"            Expected motion:  {np.round(contact['expected_surface_motion'], 3)}")
    print(f"            Surface motion:   {np.round(contact['surface_dir'], 3)}")
    print(f"            Centre motion:    {np.round(contact['centre_dir'], 3)}")
    print(f"            Surface align:    {contact['surface_alignment']:+.3f}  "
          f"{'✅ contact moves away' if contact['surface_alignment'] > 0.0 else '❌ contact moves toward obstacle'}")
    print(f"            Centre align:     {contact['centre_alignment_to_normal']:+.3f}")

    return dict(
        name=name, q=q7, obs=obs_pts.reshape(-1, 3), h=h_val, grad_h=grad_h,
        origin=origin, cart_dir=cart_dir, nearest_obs=nearest_obs,
        alignment=alignment, delta_h=delta_h_local,
        delta_h_local=delta_h_local,
        delta_h_large=delta_h_large,
        surface_alignment=contact["surface_alignment"],
        centre_alignment_to_normal=contact["centre_alignment_to_normal"],
        active_link=details["active_link"],
        active_point_idx=details["active_point_idx"],
        link_positions={k: v[:3, 3] for k, v in fk_all(robot_layer, q7).items()},
    )


# ── Matplotlib visualizer ──────────────────────────────────────────────────────

def _arrow3d(ax, origin, direction, scale, color, label=None):
    d = np.array(direction) * scale
    ax.quiver(origin[0], origin[1], origin[2],
              d[0], d[1], d[2],
              color=color, linewidth=2.5, arrow_length_ratio=0.25, label=label)

def plot_all(results):
    n   = len(results)
    fig = plt.figure(figsize=(7 * n, 7))
    for i, r in enumerate(results):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        ok = r['alignment'] > 0 and r['delta_h_local'] > -1e-5
        ax.set_title(
            f"{r['name']}\n"
            f"h={r['h']:+.4f}  align={r['alignment']:+.2f}  "
            f"Δh_local={r['delta_h_local']:+.4f}",
            color='darkgreen' if ok else 'red', fontsize=9,
        )

        lp = r['link_positions']
        if lp:
            xs = [v[0] for v in lp.values()]
            ys = [v[1] for v in lp.values()]
            zs = [v[2] for v in lp.values()]
            ax.plot(xs, ys, zs, 'k.-', markersize=8, linewidth=2, label='Links')

        obs = r['obs']
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2],
                   c='tomato', s=40, alpha=0.6, label='Obstacles')

        no = r['nearest_obs']
        ax.scatter([no[0]], [no[1]], [no[2]],
                   c='darkred', s=300, marker='*', zorder=6, label='Nearest obs')

        o = r['origin']   # fork mesh centre
        ax.scatter([o[0]], [o[1]], [o[2]],
                   c='gold', s=200, marker='^', zorder=7, label='Fork mesh centre')

        scale = 0.10
        _arrow3d(ax, o, r['cart_dir'], scale, 'lime',       'grad_h (Cartesian)')
        away = -(r['nearest_obs'] - o)
        away /= np.linalg.norm(away) + 1e-8
        _arrow3d(ax, o, away,          scale, 'deepskyblue', 'Expected (away from obs)')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.show()


# ── Optional RViz publisher ────────────────────────────────────────────────────

def publish_rviz(results):
    try:
        import rospy
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        rospy.init_node('cbf_gradient_test', anonymous=True)
        pub = rospy.Publisher('/debug/grad_h_test', MarkerArray, queue_size=1, latch=True)
        rospy.sleep(0.5)

        ma  = MarkerArray()
        mid = 0
        for r in results:
            for dname, direction, r_col, g_col, b_col in [
                ('grad_h',   r['cart_dir'],                    0.0, 1.0, 0.0),
                ('away_obs', -(r['nearest_obs'] - r['origin']), 0.0, 0.8, 1.0),
            ]:
                d = np.array(direction)
                d /= np.linalg.norm(d) + 1e-8
                scale = 0.12
                tip   = r['origin'] + d * scale
                m     = Marker()
                m.header.frame_id = 'world'
                m.header.stamp    = rospy.Time.now()
                m.ns = dname
                m.id = mid; mid += 1
                m.type   = Marker.ARROW
                m.action = Marker.ADD
                m.scale.x, m.scale.y, m.scale.z = 0.008, 0.016, 0.025
                m.color.a = 0.9
                m.color.r, m.color.g, m.color.b = r_col, g_col, b_col
                tail = Point(x=float(r['origin'][0]), y=float(r['origin'][1]), z=float(r['origin'][2]))
                head = Point(x=float(tip[0]),         y=float(tip[1]),         z=float(tip[2]))
                m.points = [tail, head]
                ma.markers.append(m)

            m2 = Marker()
            m2.header.frame_id = 'world'
            m2.header.stamp    = rospy.Time.now()
            m2.ns = 'nearest_obs'; m2.id = mid; mid += 1
            m2.type = Marker.SPHERE; m2.action = Marker.ADD
            no = r['nearest_obs']
            m2.pose.position.x, m2.pose.position.y, m2.pose.position.z = float(no[0]), float(no[1]), float(no[2])
            m2.pose.orientation.w = 1.0
            m2.scale.x = m2.scale.y = m2.scale.z = 0.04
            m2.color.a, m2.color.r = 0.8, 1.0
            ma.markers.append(m2)

        pub.publish(ma)
        rospy.loginfo("Published %d markers to /debug/grad_h_test", len(ma.markers))
        rospy.sleep(1.0)
    except Exception as e:
        print(f"  [RViz] skipped: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rviz',   action='store_true', help='Also publish to RViz')
    parser.add_argument('--no-plot', action='store_true', help='Skip matplotlib window')
    parser.add_argument('--d_safe', type=float, default=0.015)
    parser.add_argument('--alpha',  type=float, default=0.001)
    args = parser.parse_args()

    print(f"Loading Bernstein barrier  d_safe={args.d_safe}  alpha={args.alpha}  device={DEVICE}")
    barrier, robot_layer, fork_centroid_local, fork_scale = build_barrier(
        d_safe=args.d_safe, alpha=args.alpha)
    print("✅ Ready\n")

    # Homing configuration from feeding_simulation_position_controller.launch
    q_home = np.array([-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511])

    # Fork geometry in world frame at homing config
    T_fork         = fork_transform_world(robot_layer, q_home)
    fork_origin    = T_fork[:3, 3]             # link mount base
    R_fork         = T_fork[:3, :3]            # fork-local axes → world
    fork_x_world   = R_fork[:, 0]             # fork-local X in world
    fork_y_world   = R_fork[:, 1]             # fork-local Y in world
    fork_z_world   = R_fork[:, 2]             # fork-local Z = tine direction in world
    mesh_center    = fork_origin + R_fork @ fork_centroid_local  # Bernstein centroid in world

    print(f"Fork link origin (mount base):  {np.round(fork_origin, 3)}")
    print(f"Fork mesh centre (Bernstein):   {np.round(mesh_center, 3)}")
    print(f"  Bernstein centroid_local:     {np.round(fork_centroid_local, 3)}  (fork-local frame)")
    print(f"  Bernstein scale_factor:       {fork_scale:.4f} m")
    print(f"Fork-local Z in world (tines):  {np.round(fork_z_world, 3)}")
    print(f"Fork-local X in world:          {np.round(fork_x_world, 3)}")
    print(f"Fork-local Y in world:          {np.round(fork_y_world, 3)}")
    print()

    gap = args.d_safe * 0.5   # 50 % inside the safety margin → h < 0
    results = []

    # ── Scenario 1: obstacle along fork-local +Z (beyond tines tip) ───────────
    # Obstacle is slightly past the tip of the fork along the tine direction.
    # Expected centre motion: away from the obstacle, roughly -fork_z_world.
    obs_z_tip = (mesh_center + (fork_scale + gap) * fork_z_world).reshape(1, 3)
    results.append(run_scenario(
        "1 — Obs past tines tip (along fork-local +Z)",
        q_home, obs_z_tip, barrier, robot_layer, args.d_safe, fork_centroid_local,
    ))

    # ── Scenario 2: obstacle along fork-local +X (tines side, X direction) ────
    # Obstacle is to the SIDE of the fork, in the fork-local X direction.
    # Expected centre motion: away from the obstacle, roughly -fork_x_world.
    # If Bernstein always returns fork-local Z this will show an unintuitive direction.
    obs_x_side = (mesh_center + (fork_scale * 0.7 + gap) * fork_x_world).reshape(1, 3)
    results.append(run_scenario(
        "2 — Obs to the side in fork-local +X",
        q_home, obs_x_side, barrier, robot_layer, args.d_safe, fork_centroid_local,
    ))

    # ── Scenario 3: obstacle along fork-local -Z (below mount base / handle end)
    # Obstacle is behind the fork along its shaft (handle side).
    # Expected centre motion: away from the obstacle, roughly +fork_z_world.
    obs_z_base = (mesh_center - (fork_scale + gap) * fork_z_world).reshape(1, 3)
    results.append(run_scenario(
        "3 — Obs at handle end (along fork-local -Z)",
        q_home, obs_z_base, barrier, robot_layer, args.d_safe, fork_centroid_local,
    ))

    # ── Scenario 4: bowl cavity ring around fork mesh centre ───────────────────
    # Ring of obstacles encircling the fork tines in the fork-local XY plane,
    # placed at the fork mesh centre depth (not the link origin as before).
    angles   = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    bowl_r   = fork_scale * 0.8 + gap   # ring at 80 % scale + safety gap
    bowl_pts = np.stack([
        mesh_center[0] + bowl_r * (np.cos(angles) * fork_x_world[0] + np.sin(angles) * fork_y_world[0]),
        mesh_center[1] + bowl_r * (np.cos(angles) * fork_x_world[1] + np.sin(angles) * fork_y_world[1]),
        mesh_center[2] + bowl_r * (np.cos(angles) * fork_x_world[2] + np.sin(angles) * fork_y_world[2]),
    ], axis=-1)
    results.append(run_scenario(
        "4 — Bowl cavity ring (fork-local XY plane around mesh centre)",
        q_home, bowl_pts, barrier, robot_layer, args.d_safe, fork_centroid_local,
    ))

    # ── Scenario 5: table below arm (grid near mesh centre height) ─────────────
    xx, yy   = np.meshgrid(np.linspace(0.0, 0.6, 10), np.linspace(-0.3, 0.3, 8))
    table_z  = mesh_center[2] - args.d_safe * 0.6
    table_pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, table_z)], axis=-1)
    results.append(run_scenario(
        "5 — Table surface below arm (h<0 seen in simulation logs)",
        q_home, table_pts, barrier, robot_layer, args.d_safe, fork_centroid_local,
    ))

    print("\n\nSUMMARY")
    print(f"{'Scenario':<50} {'h':>8} {'align':>7} {'Δh local':>10} {'Δh 0.03':>10}  Checks")
    print('-' * 106)
    for r in results:
        checks = (
            ('✅' if r['surface_alignment'] > 0 else '❌')
            + ('✅' if r['delta_h_local'] > -1e-5 else '❌')
        )
        print(f"{r['name']:<50} {r['h']:>+8.4f} {r['alignment']:>+7.3f} "
              f"{r['delta_h_local']:>+10.5f} {r['delta_h_large']:>+10.5f}  {checks} "
              f"active={r['active_link']}[{r['active_point_idx']}] "
              f"surf_align={r['surface_alignment']:+.3f}")

    if args.rviz:
        publish_rviz(results)

    if not args.no_plot:
        plot_all(results)


if __name__ == '__main__':
    main()
