#!/usr/bin/env python3
"""
Visualize and evaluate the Bernstein SDF used by the CBF safety node.

Examples:
  python3 scripts/visualize_robot_sdf_layers.py
  python3 scripts/visualize_robot_sdf_layers.py --links all --levels 0 0.01 0.05 0.10
  python3 scripts/visualize_robot_sdf_layers.py --from-joint-states --grid-spacing 0.015
  python3 scripts/visualize_robot_sdf_layers.py --eval-point 0.45,0.0,0.55 --no-show

The arrows show +grad SDF in world coordinates, i.e. the direction of
increasing signed distance away from the closest robot link.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import rospkg
import torch
import trimesh
import xacro


PKG_PATH = rospkg.RosPack().get_path("vision_processing")
SDF_BERNSTEIN_PATH = os.path.join(PKG_PATH, "third_party", "SDF_Bernstein_Basis")

sys.path.insert(0, PKG_PATH)
sys.path.insert(0, SDF_BERNSTEIN_PATH)

from third_party.RDF.urdf_layer import URDFLayer
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights


ALL_LINKS = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
    "fork_tip",
]

CBF_PROTECTED_LINKS = [
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
    "fork_tip",
]

DEFAULT_Q = np.array(
    [-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511],
    dtype=np.float32,
)

SHELL_COLORS = [
    "#2f2f2f",
    "#d7191c",
    "#fdae61",
    "#2c7bb6",
    "#1a9641",
    "#7b3294",
    "#008080",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Bernstein SDF distance shells and world-frame SDF gradients."
    )
    parser.add_argument(
        "--q",
        nargs=7,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=DEFAULT_Q.tolist(),
        help="Seven Panda joint positions in radians. Defaults to the launch spawn pose.",
    )
    parser.add_argument(
        "--from-joint-states",
        action="store_true",
        help="Read one /joint_states message and use panda_joint1..7 instead of --q.",
    )
    parser.add_argument(
        "--joint-states-topic",
        default="/joint_states",
        help="JointState topic used with --from-joint-states.",
    )
    parser.add_argument(
        "--links",
        default="protected",
        help="Link set: 'protected', 'all', or a comma-separated list of SDF link names.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.05, 0.10],
        help="SDF shell distances in meters. Example: --levels 0 0.01 0.05 0.10",
    )
    parser.add_argument(
        "--gradient-levels",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10],
        help="Shell distances in meters where gradient arrows are sampled.",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.02,
        help="SDF sampling grid spacing in meters.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding around the selected robot mesh bounds in meters.",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Manual SDF grid bounds in world meters. Overrides automatic mesh bounds.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32768,
        help="Number of grid/query points evaluated per SDF batch.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device for SDF evaluation.",
    )
    parser.add_argument(
        "--finger-position",
        type=float,
        default=0.001,
        help="Finger joint value appended to q for URDF FK.",
    )
    parser.add_argument(
        "--eval-point",
        action="append",
        default=[],
        help="Evaluate one world point 'x,y,z'. May be passed multiple times.",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=400,
        help="Maximum total gradient arrows to draw across all gradient levels.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.04,
        help="PyVista arrow scale in meters.",
    )
    parser.add_argument(
        "--robot-opacity",
        type=float,
        default=0.35,
        help="Opacity of the transformed robot visual mesh.",
    )
    parser.add_argument(
        "--shell-opacity",
        type=float,
        default=0.30,
        help="Opacity of non-zero SDF distance shells.",
    )
    parser.add_argument(
        "--surface-opacity",
        type=float,
        default=0.55,
        help="Opacity of the zero SDF surface.",
    )
    parser.add_argument(
        "--save-screenshot",
        help="Optional PNG path. Enables off-screen rendering if --no-show is also used.",
    )
    parser.add_argument(
        "--xvfb",
        action="store_true",
        help="Start a virtual framebuffer before PyVista rendering. Useful on headless machines.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive PyVista window.",
    )
    return parser.parse_args()


def resolve_link_names(link_arg):
    if link_arg == "protected":
        return list(CBF_PROTECTED_LINKS)
    if link_arg == "all":
        return list(ALL_LINKS)
    links = [item.strip() for item in link_arg.split(",") if item.strip()]
    unknown = sorted(set(links) - set(ALL_LINKS))
    if unknown:
        raise ValueError("Unknown SDF link name(s): " + ", ".join(unknown))
    return links


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def parse_eval_points(values):
    points = []
    for value in values:
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) != 3:
            raise ValueError(f"Expected --eval-point x,y,z, got '{value}'")
        points.append([float(piece) for piece in pieces])
    return np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)


def read_joint_states(topic):
    import rospy
    from sensor_msgs.msg import JointState

    rospy.init_node("robot_sdf_layer_visualizer", anonymous=True, disable_signals=True)
    msg = rospy.wait_for_message(topic, JointState, timeout=5.0)
    positions = {name: value for name, value in zip(msg.name, msg.position)}
    missing = [f"panda_joint{i}" for i in range(1, 8) if f"panda_joint{i}" not in positions]
    if missing:
        raise RuntimeError(f"JointState is missing: {', '.join(missing)}")
    return np.array([positions[f"panda_joint{i}"] for i in range(1, 8)], dtype=np.float32)


def build_robot_layer(device):
    xacro_path = os.path.join(PKG_PATH, "urdf", "panda_camera.xacro")
    doc = xacro.process_file(xacro_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as handle:
        handle.write(doc.toxml())
        urdf_path = handle.name

    return URDFLayer(
        urdf_path=urdf_path,
        device=device,
        package_dir=PKG_PATH,
        voxel_dir=os.path.join(PKG_PATH, "third_party", "RDF", "panda_layer", "meshes", "voxel_128"),
    )


def build_sdf_stack(device, link_names):
    robot_layer = build_robot_layer(device)
    weights_dir = os.path.join(PKG_PATH, "third_party", "SDF_Bernstein_Basis", "panda_test")
    weights = RDF_Weights(device=device, dtype=torch.float32)
    weights.init_robot_folder(weights_dir, robot_name="panda")
    weights.add_models(link_names, robot_name="panda")
    core = BernsteinCore(weights, robot_layer, device, link_names)
    return robot_layer, core


def q7_to_q9(q7_np, finger_position, device):
    q7 = torch.as_tensor(q7_np, dtype=torch.float32, device=device).reshape(1, 7)
    fingers = torch.full((1, 2), float(finger_position), dtype=torch.float32, device=device)
    return torch.cat([q7, fingers], dim=1)


def link_matches(mesh_link_name, sdf_link_name):
    mesh_name = mesh_link_name.replace("panda_", "")
    sdf_name = sdf_link_name.replace("panda_", "")
    return mesh_name == sdf_name or mesh_name in sdf_name or sdf_name in mesh_name


def selected_robot_mesh(robot_layer, pose, q9, link_names):
    all_meshes = robot_layer.get_forward_robot_mesh(pose, q9)[0]
    selected = []
    for info, mesh in zip(robot_layer.meshes_info, all_meshes):
        if any(link_matches(info["link_name"], link_name) for link_name in link_names):
            selected.append(mesh)
    if not selected:
        raise RuntimeError("No URDF visual meshes matched the selected SDF link names.")
    if len(selected) == 1:
        return selected[0]
    return trimesh.util.concatenate(selected)


def automatic_bounds(mesh, padding):
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    bounds[0] -= padding
    bounds[1] += padding
    return bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1], bounds[0, 2], bounds[1, 2]


def make_grid(bounds, spacing):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    xs = np.arange(x_min, x_max + 0.5 * spacing, spacing, dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * spacing, spacing, dtype=np.float32)
    zs = np.arange(z_min, z_max + 0.5 * spacing, spacing, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack((gx, gy, gz), axis=-1)
    points = coords.reshape(-1, 3, order="F")
    dims = (len(xs), len(ys), len(zs))
    return points, dims, (float(xs[0]), float(ys[0]), float(zs[0]))


@torch.no_grad()
def evaluate_sdf(core, points_np, pose, q9, chunk_size, device):
    values = []
    for start in range(0, len(points_np), chunk_size):
        chunk_np = points_np[start:start + chunk_size]
        chunk = torch.as_tensor(chunk_np, dtype=torch.float32, device=device)
        sdf = core.get_whole_body_sdf_batch(chunk, pose, q9)
        values.append(sdf.reshape(-1).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def evaluate_sdf_and_grad(core, points_np, pose, q9, chunk_size, device):
    sdf_values = []
    grad_values = []
    for start in range(0, len(points_np), chunk_size):
        chunk_np = points_np[start:start + chunk_size]
        points = torch.as_tensor(chunk_np, dtype=torch.float32, device=device).clone().detach()
        points.requires_grad_(True)
        sdf = core.get_whole_body_sdf_batch(points, pose, q9).reshape(-1)
        grad = torch.autograd.grad(sdf.sum(), points, create_graph=False, retain_graph=False)[0]
        sdf_values.append(sdf.detach().cpu().numpy())
        grad_values.append(grad.detach().cpu().numpy())
    return np.concatenate(sdf_values).astype(np.float32), np.vstack(grad_values).astype(np.float32)


def trimesh_to_pyvista(mesh):
    import pyvista as pv

    faces = np.hstack((np.full((mesh.faces.shape[0], 1), 3), mesh.faces)).astype(np.int64)
    return pv.PolyData(mesh.vertices, faces.reshape(-1))


def new_image_data():
    import pyvista as pv

    if hasattr(pv, "ImageData"):
        return pv.ImageData()
    if hasattr(pv, "UniformGrid"):
        return pv.UniformGrid()
    from pyvista.core import UniformGrid

    return UniformGrid()


def maybe_start_xvfb(pv, should_start):
    if not should_start:
        return
    try:
        pv.start_xvfb()
    except Exception as exc:
        print(f"[warn] Could not start PyVista Xvfb: {exc}")


def build_pyvista_grid(sdf_values, dims, origin, spacing):
    grid = new_image_data()
    grid.dimensions = dims
    grid.origin = origin
    grid.spacing = (spacing, spacing, spacing)
    grid.point_data["sdf_m"] = sdf_values
    return grid


def contour_for_level(grid, level):
    contour = grid.contour([float(level)], scalars="sdf_m")
    if contour.n_points == 0:
        return None
    return contour


def sample_points(points, max_count):
    if len(points) <= max_count:
        return points
    indices = np.linspace(0, len(points) - 1, max_count, dtype=np.int64)
    return points[indices]


def add_distance_shells(plotter, grid, levels, shell_opacity, surface_opacity):
    contours = {}
    for index, level in enumerate(levels):
        contour = contour_for_level(grid, level)
        if contour is None:
            print(f"[warn] No contour for SDF level {level:.4f} m; level is outside the sampled range.")
            continue
        color = SHELL_COLORS[index % len(SHELL_COLORS)]
        opacity = surface_opacity if abs(level) < 1e-9 else shell_opacity
        label = "SDF 0 cm" if abs(level) < 1e-9 else f"SDF {level * 100.0:.1f} cm"
        plotter.add_mesh(
            contour,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            label=label,
        )
        contours[float(level)] = contour
    return contours


def add_gradient_arrows(plotter, core, contours, gradient_levels, pose, q9, args, device):
    available_levels = sorted(contours.keys())
    requested = []
    for level in gradient_levels:
        closest = min(available_levels, key=lambda candidate: abs(candidate - level), default=None)
        if closest is not None and abs(closest - level) < 1e-9:
            requested.append(closest)

    if not requested:
        return

    arrows_per_level = max(1, args.max_arrows // len(requested))
    for level in requested:
        contour = contours[level]
        points = sample_points(np.asarray(contour.points, dtype=np.float32), arrows_per_level)
        _, gradients = evaluate_sdf_and_grad(
            core,
            points,
            pose,
            q9,
            chunk_size=min(args.chunk_size, 4096),
            device=device,
        )
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        if not np.any(valid):
            continue
        directions = gradients[valid] / norms[valid]
        color = SHELL_COLORS[(list(contours.keys()).index(level)) % len(SHELL_COLORS)]
        plotter.add_arrows(points[valid], directions, mag=args.arrow_scale, color=color)


def print_point_evaluations(core, points_np, pose, q9, chunk_size, device):
    if len(points_np) == 0:
        return
    sdf, grad = evaluate_sdf_and_grad(core, points_np, pose, q9, chunk_size, device)
    print("\nPoint evaluations:")
    for point, value, gradient in zip(points_np, sdf, grad):
        norm = float(np.linalg.norm(gradient))
        direction = gradient / norm if norm > 1e-8 else gradient
        print(
            "  p=({:.4f}, {:.4f}, {:.4f})  sdf={:.5f} m  grad=({:.4f}, {:.4f}, {:.4f})".format(
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(value),
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            )
        )


def main():
    args = parse_args()
    link_names = resolve_link_names(args.links)
    eval_points = parse_eval_points(args.eval_point)
    device = choose_device(args.device)

    q7 = read_joint_states(args.joint_states_topic) if args.from_joint_states else np.asarray(args.q, dtype=np.float32)
    print(f"Using device: {device}")
    print("Using q:", np.array2string(q7, precision=6, separator=", "))
    print("SDF links:", ", ".join(link_names))

    robot_layer, core = build_sdf_stack(device, link_names)
    pose = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    q9 = q7_to_q9(q7, args.finger_position, device)

    robot_mesh = selected_robot_mesh(robot_layer, pose, q9, link_names)
    bounds = tuple(args.bounds) if args.bounds else automatic_bounds(robot_mesh, args.padding)
    grid_points, dims, origin = make_grid(bounds, args.grid_spacing)
    print(
        "Grid: {} x {} x {} = {:,} points, spacing={:.3f} m".format(
            dims[0], dims[1], dims[2], len(grid_points), args.grid_spacing
        )
    )
    print(
        "Bounds: x=[{:.3f}, {:.3f}] y=[{:.3f}, {:.3f}] z=[{:.3f}, {:.3f}]".format(
            bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
        )
    )

    sdf_values = evaluate_sdf(core, grid_points, pose, q9, args.chunk_size, device)
    print(f"SDF range on grid: [{float(np.min(sdf_values)):.5f}, {float(np.max(sdf_values)):.5f}] m")
    print_point_evaluations(core, eval_points, pose, q9, args.chunk_size, device)

    if args.no_show and not args.save_screenshot:
        return

    import pyvista as pv

    off_screen = bool(args.no_show and args.save_screenshot)
    maybe_start_xvfb(pv, args.xvfb or off_screen)
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    plotter.add_mesh(
        trimesh_to_pyvista(robot_mesh),
        color="#bdbdbd",
        opacity=args.robot_opacity,
        smooth_shading=True,
        label="URDF visual mesh",
    )

    grid = build_pyvista_grid(sdf_values, dims, origin, args.grid_spacing)
    contours = add_distance_shells(
        plotter,
        grid,
        sorted(set(float(level) for level in args.levels)),
        shell_opacity=args.shell_opacity,
        surface_opacity=args.surface_opacity,
    )
    add_gradient_arrows(plotter, core, contours, args.gradient_levels, pose, q9, args, device)

    plotter.add_axes()
    plotter.add_legend()
    plotter.camera_position = "xy"
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 25
    plotter.reset_camera()

    if args.save_screenshot:
        plotter.screenshot(args.save_screenshot)
        print(f"Saved screenshot: {args.save_screenshot}")
    if not args.no_show:
        plotter.show()


if __name__ == "__main__":
    main()
