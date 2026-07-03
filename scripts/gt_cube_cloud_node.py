#!/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3
"""
gt_cube_cloud_node.py
=====================
Ground-truth cube conditioning cloud for the pick-and-place pipeline.

Publishes an analytic box cloud at the cube's live Gazebo pose. By default only
the faces visible from the static camera are kept (back-face culling), making
this a "perfect SAM": exact pose and segmentation with a realistic partial
view, matching the visibility-filtered training preprocess
(Data_preprocess_PickPlace.py, OBJECT_VISIBLE_ONLY). Set ~visible_only:=false
for the full surface. A/B alternative to the SAM cube cloud for diagnosing
conditioning bias:

  /gazebo/model_states (red_cube)  ->  ~topic (default /perception/target_cube_gt)

The pose is converted from the Gazebo world (origin on the floor) to the TF
`world` frame (robot base, on the table top at Gazebo z=0.75) by subtracting
~table_z_offset. Selected in green_cube_feeding_casf_pp.launch via use_gt_cube.

NOTE sizes: the Gazebo red_cube is 0.04 x 0.04 x 0.06 (half 0.02/0.02/0.03),
while the ManiSkill training cube was 0.04^3 (half 0.02^3). Default here is the
REAL Gazebo geometry; set ~half_x/y/z to 0.02 to reproduce the training size.
"""
import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R


def _sample_box_surface_with_normals(half, n, rng):
    # Same sampler as Data_preprocess_PickPlace.py, centered at the origin,
    # returning each point's outward face normal (box-local frame).
    half = np.asarray(half, dtype=np.float64)
    areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
    areas = areas / areas.sum()
    pts = np.empty((n, 3))
    nrm = np.zeros((n, 3))
    for i in range(n):
        ax = rng.choice(3, p=areas)
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        p = (rng.random(3) * 2 - 1) * half
        p[ax] = half[ax] * sgn
        pts[i] = p
        nrm[i, ax] = sgn
    return pts, nrm


def sample_box_visible_world(half, Rm, pos, n, rng, cam_pos, max_tries=10):
    """n visible-face points of a box at world pose (Rm, pos) seen from cam_pos."""
    kept = []
    total = 0
    for _ in range(max_tries):
        loc, nrm = _sample_box_surface_with_normals(half, 3 * n, rng)
        pw = loc @ Rm.T + pos
        nw = nrm @ Rm.T
        vis = np.einsum('ij,ij->i', nw, cam_pos[None, :] - pw) > 1e-12
        if vis.any():
            kept.append(pw[vis])
            total += int(vis.sum())
            if total >= n:
                break
    if total == 0:
        loc, _ = _sample_box_surface_with_normals(half, n, rng)
        return loc @ Rm.T + pos
    out = np.vstack(kept)
    return out[:n] if len(out) >= n else out[np.arange(n) % len(out)]


class GtCubeCloudNode:
    def __init__(self):
        rospy.init_node('gt_cube_cloud', anonymous=True)

        self.model_name = rospy.get_param("~model_name", "red_cube")
        self.topic = rospy.get_param("~topic", "/perception/target_cube_gt")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.table_z_offset = float(rospy.get_param("~table_z_offset", 0.75))
        self.half = np.array([float(rospy.get_param("~half_x", 0.02)),
                              float(rospy.get_param("~half_y", 0.02)),
                              float(rospy.get_param("~half_z", 0.03))])
        # 600 points matches the training preprocess cube count.
        self.num_points = int(rospy.get_param("~num_points", 600))
        rate_hz = float(rospy.get_param("~publish_rate_hz", 30.0))
        # Keep only faces visible from the static camera (back-face culling,
        # exact self-occlusion for a convex box) -> "perfect SAM": exact pose
        # and segmentation, realistic partial view, matching the visibility-
        # filtered training preprocess. false = full surface.
        self.visible_only = bool(rospy.get_param("~visible_only", True))
        self.cam_pos = np.array([float(rospy.get_param("~cam_x", 0.9)),
                                 float(rospy.get_param("~cam_y", -0.19)),
                                 float(rospy.get_param("~cam_z", 0.62))])

        self.rng = np.random.default_rng(0)
        self.pos = None
        self.quat_xyzw = None

        self.pub = rospy.Publisher(self.topic, PointCloud2, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._states_cb,
                         queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._publish)
        rospy.loginfo("gt_cube_cloud: '%s' -> %s (half=%s, %d pts, z_off=%.3f)",
                      self.model_name, self.topic, self.half.tolist(),
                      self.num_points, self.table_z_offset)

    def _states_cb(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            rospy.logwarn_throttle(10, "gt_cube_cloud: model '%s' not in "
                                   "/gazebo/model_states", self.model_name)
            return
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        self.pos = np.array([p.x, p.y, p.z - self.table_z_offset])
        self.quat_xyzw = np.array([q.x, q.y, q.z, q.w])

    def _publish(self, _event):
        if self.pos is None:
            return
        Rm = R.from_quat(self.quat_xyzw).as_matrix()
        if self.visible_only:
            world = sample_box_visible_world(self.half, Rm, self.pos,
                                             self.num_points, self.rng,
                                             self.cam_pos)
        else:
            loc, _ = _sample_box_surface_with_normals(self.half,
                                                      self.num_points, self.rng)
            world = loc @ Rm.T + self.pos
        header = Header(stamp=rospy.Time.now(), frame_id=self.world_frame)
        self.pub.publish(pc2.create_cloud_xyz32(header,
                                                world.astype(np.float32)))


if __name__ == '__main__':
    try:
        GtCubeCloudNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
