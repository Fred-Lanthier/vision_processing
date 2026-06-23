import importlib.util
import pathlib

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "persistent_cloud_node.py"
SPEC = importlib.util.spec_from_file_location("persistent_cloud_node", MODULE_PATH)
persistent_cloud_node = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(persistent_cloud_node)


def test_fibonacci_sphere_points_lie_on_requested_surface():
    radius = 0.08
    offsets = persistent_cloud_node._fibonacci_sphere_offsets(radius, 1000)

    assert offsets.shape == (1000, 3)
    assert offsets.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(offsets, axis=1), radius, rtol=1e-6, atol=1e-7)


def test_linear_sphere_motion_uses_all_three_velocity_components():
    initial = [0.50, -0.50, 1.05]
    velocity = [0.0, 0.05, 0.0]

    position = persistent_cloud_node._linear_motion_position(
        initial, velocity, elapsed=4.0)

    np.testing.assert_allclose(position, [0.50, -0.30, 1.05], atol=1e-7)


def test_negative_elapsed_does_not_move_sphere_backwards():
    position = persistent_cloud_node._linear_motion_position(
        [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], elapsed=-1.0)

    np.testing.assert_array_equal(position, [1.0, 2.0, 3.0])


def test_colored_persistent_cloud_keeps_xyz_as_first_three_fields():
    points = np.array([[0.5, -0.5, 1.05], [0.5, -0.4, 1.05]], dtype=np.float32)
    colors = np.array([0xB4B4B4, 0xFF00FF], dtype=np.uint32)
    header = type("Header", (), {})()

    message = persistent_cloud_node._make_cloud_msg(header, points, colors)
    unpacked = np.frombuffer(message.data, dtype=np.float32).reshape(-1, 4)

    assert message.point_step == 16
    assert [field.name for field in message.fields] == ["x", "y", "z", "rgb"]
    np.testing.assert_array_equal(unpacked[:, :3], points)
    np.testing.assert_array_equal(unpacked[:, 3].view(np.uint32), colors)


def test_publish_appends_sphere_to_persistent_obstacle_message():
    class CloudStore:
        def __init__(self, points):
            self.points = points

        def get_points(self, min_confidence):
            return self.points

        def count(self):
            return 0 if self.points is None else len(self.points)

    class Publisher:
        message = None

        def publish(self, message):
            self.message = message

    node = persistent_cloud_node.PersistentCloudNode.__new__(
        persistent_cloud_node.PersistentCloudNode)
    measured = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32)
    node.world_frame = "world"
    node._tgt_world = CloudStore(None)
    node.tgt_commit_thresh = 1
    node._live_target = np.empty((0, 3), dtype=np.float32)
    node.obs_world = CloudStore(measured)
    node.obs_commit_thresh = 1
    node._live_obs = np.empty((0, 3), dtype=np.float32)
    node._carve_against_target = lambda points: points
    node._generate_floor_patch = lambda points: None
    node.dilate_published_cloud = False
    node.obs_max_pts = 50000
    node.moving_sphere_enabled = True
    node.moving_sphere_initial_position = np.array(
        [0.5, -0.5, 1.05], dtype=np.float32)
    node.moving_sphere_velocity = np.zeros(3, dtype=np.float32)
    node._moving_sphere_offsets = persistent_cloud_node._fibonacci_sphere_offsets(
        0.08, 1000)
    node._moving_sphere_start_time = 0.0
    sphere_points = (
        node._moving_sphere_offsets + node.moving_sphere_initial_position[None, :])
    node._generate_moving_sphere = lambda: sphere_points
    node.pub = Publisher()
    node.log_counts = False

    node._publish(type("Stamp", (), {})())

    message = node.pub.message
    unpacked = np.frombuffer(message.data, dtype=np.float32).reshape(-1, 4)
    assert message.width == 1002
    np.testing.assert_array_equal(unpacked[:2, :3], measured)
    np.testing.assert_allclose(
        np.linalg.norm(unpacked[2:, :3] - node.moving_sphere_initial_position, axis=1),
        0.08,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        unpacked[:2, 3].view(np.uint32), np.full(2, 0xB4B4B4, np.uint32))
    np.testing.assert_array_equal(
        unpacked[2:, 3].view(np.uint32), np.full(1000, 0xFF00FF, np.uint32))
