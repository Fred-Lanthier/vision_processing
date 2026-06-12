#!/usr/bin/env python3
"""
sdf_visualizer_node.py

Visualize the Bernstein SDF zero-level set (h=0) in RViz.
Each robot link is drawn with a distinct color so you can verify that
the fork/hand geometry in the SDF model matches the physical URDF.

Topics published:
  /viz/sdf_surface  (PointCloud2, frame: world)

Tuning params:
  ~grid_resolution   [m]   voxel spacing of the evaluation grid (default 0.025)
  ~surface_threshold [m]   half-thickness of the surface shell   (default 0.012)
  ~rate_hz                 publish rate in Hz                    (default 1.0)
  ~x_min/x_max            workspace bounds in x                 (default -0.9/0.9)
  ~y_min/y_max            workspace bounds in y                 (default -0.9/0.9)
  ~z_min/z_max            workspace bounds in z                 (default  0.0/1.3)
"""

import os
import sys
import struct
import tempfile

import numpy as np
import torch
import rospy
import rospkg
import xacro

from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from third_party.RDF.urdf_layer import URDFLayer
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore


# One distinct color per link (BGR packed as RViz XYZRGB float).
# fork_tip is bright red so it stands out immediately.
LINK_NAMES = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
    'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
    'panda_hand', 'fork_tip',
]

LINK_COLORS_RGB = [
    (160, 160, 160),  # link0  — grey
    (220,  80,  80),  # link1  — red
    (220, 140,  60),  # link2  — orange
    (200, 200,  60),  # link3  — yellow
    ( 80, 200,  80),  # link4  — green
    ( 60, 200, 200),  # link5  — cyan
    ( 80,  80, 220),  # link6  — blue
    (140,  80, 220),  # link7  — purple
    (220,  80, 220),  # hand   — magenta
    (255, 200,  80),  # left finger  — gold
    (255, 200,  80),  # right finger — gold
    (255,  30,  30),  # fork_tip — bright red  ← check this one carefully
]


def pack_rgb_float(r, g, b):
    """Pack R,G,B (0-255 ints) into the float32 RViz XYZRGB convention."""
    packed = struct.pack('BBBB', b, g, r, 0)
    return struct.unpack('f', packed)[0]


LINK_RGB_FLOATS = [pack_rgb_float(*c) for c in LINK_COLORS_RGB]


class SDFVisualizerNode:
    def __init__(self):
        rospy.init_node('sdf_visualizer')
        # Force CPU: the visualizer runs alongside the CBF which owns the GPU.
        # Sharing CUDA compute causes CBF timing delays that look like spurious triggers.
        self.device = torch.device('cpu')
        rospy.loginfo('SDF visualizer using CPU (GPU reserved for CBF)')

        # --- Load robot model (same URDF as CBF) ---
        xacro_file = os.path.join(pkg_path, 'urdf', 'panda_camera.xacro')
        doc = xacro.process_file(xacro_file, mappings={'arm_id': 'panda'})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(doc.toxml())
            urdf_path = f.name

        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=pkg_path,
            voxel_dir=os.path.join(pkg_path, 'third_party', 'RDF',
                                   'panda_layer', 'meshes', 'voxel_128'),
        )

        # --- Load Bernstein SDF weights ---
        weights_dir = os.path.join(pkg_path,
                                   'third_party', 'SDF_Bernstein_Basis', 'panda_test')
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(weights_dir, robot_name='panda')
        weight_handler.add_models(LINK_NAMES, robot_name='panda')
        self.bernstein_core = BernsteinCore(
            weight_handler, self.robot_layer, self.device, LINK_NAMES)
        rospy.loginfo('Bernstein SDF loaded.')

        # --- Build evaluation grid ---
        res   = float(rospy.get_param('~grid_resolution',   0.025))
        x_min = float(rospy.get_param('~x_min', -0.9));  x_max = float(rospy.get_param('~x_max',  0.9))
        y_min = float(rospy.get_param('~y_min', -0.9));  y_max = float(rospy.get_param('~y_max',  0.9))
        z_min = float(rospy.get_param('~z_min',  0.0));  z_max = float(rospy.get_param('~z_max',  1.3))

        xs = torch.arange(x_min, x_max, res, device=self.device)
        ys = torch.arange(y_min, y_max, res, device=self.device)
        zs = torch.arange(z_min, z_max, res, device=self.device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1).float()
        rospy.loginfo(f'Grid: {self.grid.shape[0]:,} points at {res*100:.1f}cm resolution')

        self.surface_threshold = float(rospy.get_param('~surface_threshold', 0.012))
        self.chunk_size = 50_000  # evaluate in chunks to avoid CUDA OOM

        # --- State ---
        self.current_q = None
        rospy.Subscriber('/joint_states', JointState, self._joint_cb, queue_size=1)
        self.pub = rospy.Publisher('/viz/sdf_surface', PointCloud2, queue_size=1)
        self.rate_hz = float(rospy.get_param('~rate_hz', 1.0))

    def _joint_cb(self, msg):
        pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos_dict.get(f'panda_joint{i}', 0.0) for i in range(1, 8)]
        if all(f'panda_joint{i}' in pos_dict for i in range(1, 8)):
            self.current_q = np.array(q, dtype=np.float32)

    @torch.no_grad()
    def _evaluate_sdf(self, q7_np):
        """Return sdf_per_link [K, N] evaluated over the full grid."""
        q7  = torch.tensor(q7_np, device=self.device).float().unsqueeze(0)
        q9  = torch.cat([q7, torch.zeros((1, 2), device=self.device)], dim=1)
        pose = torch.eye(4, device=self.device).unsqueeze(0)

        N = self.grid.shape[0]
        K = len(LINK_NAMES)
        sdf_all = torch.empty((K, N), device=self.device)

        for start in range(0, N, self.chunk_size):
            chunk = self.grid[start:start + self.chunk_size]
            _, sdf_chunk = self.bernstein_core.get_whole_body_sdf_batch(
                chunk, pose, q9, return_per_link=True)
            # sdf_chunk: [1, K, chunk]
            sdf_all[:, start:start + self.chunk_size] = sdf_chunk[0]

        return sdf_all  # [K, N]

    def _build_cloud(self, sdf_per_link):
        """Build XYZRGB PointCloud2 from per-link SDF tensor."""
        K, N = sdf_per_link.shape
        grid_np = self.grid.cpu().numpy()

        # Collect (x, y, z, rgb_float) for all near-surface points
        rows = []
        for k in range(K):
            mask = sdf_per_link[k].abs() < self.surface_threshold
            if not mask.any():
                continue
            pts  = grid_np[mask.cpu().numpy()]   # [M, 3]
            rgb  = LINK_RGB_FLOATS[k]
            col  = np.full((pts.shape[0], 1), rgb, dtype=np.float32)
            rows.append(np.hstack([pts.astype(np.float32), col]))

        if not rows:
            return None

        data = np.vstack(rows)  # [total, 4]
        header = Header(stamp=rospy.Time.now(), frame_id='world')
        fields = [
            PointField('x',   0, PointField.FLOAT32, 1),
            PointField('y',   4, PointField.FLOAT32, 1),
            PointField('z',   8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, data.tolist())

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.current_q is not None:
                t0 = rospy.get_time()
                sdf = self._evaluate_sdf(self.current_q)
                cloud = self._build_cloud(sdf)
                if cloud is not None:
                    self.pub.publish(cloud)
                    n_pts = sum(
                        int((sdf[k].abs() < self.surface_threshold).sum())
                        for k in range(len(LINK_NAMES))
                    )
                    rospy.loginfo_throttle(
                        5.0,
                        f'SDF surface: {n_pts:,} points published '
                        f'({(rospy.get_time()-t0)*1000:.0f}ms)')
            rate.sleep()


if __name__ == '__main__':
    SDFVisualizerNode().run()
