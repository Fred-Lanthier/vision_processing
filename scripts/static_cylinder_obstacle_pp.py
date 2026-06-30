#!/usr/bin/env python3
"""
Publish a deterministic point cloud for the pick-place CBF obstacle.

Gazebo collision geometry is not directly available to the CBF node, which expects
a world-frame PointCloud2 obstacle cloud. This node mirrors the spawned obstacle
geometry and publishes its surface to the obstacle topics used by the PP launch.
"""
import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def transform_local(points, center_x, center_y, base_z, yaw):
    points = np.asarray(points, dtype=np.float32).copy()
    c, s = math.cos(yaw), math.sin(yaw)
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 0] = center_x + c * x - s * y
    points[:, 1] = center_y + s * x + c * y
    points[:, 2] = base_z + points[:, 2]
    return points.astype(np.float32)


def _fibonacci_sphere_offsets(radius, num_points):
    """Return uniformly distributed surface offsets for a sphere."""
    radius = float(radius)
    num_points = int(num_points)
    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive.")
    if num_points < 4:
        raise ValueError("A sphere point cloud requires at least four points.")

    indices = np.arange(num_points, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / num_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = np.pi * (1.0 + np.sqrt(5.0)) * indices
    unit_sphere = np.column_stack([
        radial * np.cos(theta),
        radial * np.sin(theta),
        z,
    ])
    return np.ascontiguousarray(radius * unit_sphere, dtype=np.float32)


def _linear_motion_position(initial_position, velocity, elapsed):
    """Position of a constant-velocity obstacle after ``elapsed`` seconds."""
    initial = np.asarray(initial_position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    if initial.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Initial position and velocity must each contain x, y, z.")
    return initial + velocity * np.float32(max(0.0, float(elapsed)))


def sample_cylinder_surface(radius, height, num_points):
    num_points = max(16, int(num_points))
    radius = float(radius)
    height = float(height)

    side_count = max(8, int(num_points * 0.70))
    cap_count = max(4, (num_points - side_count) // 2)

    golden = math.pi * (3.0 - math.sqrt(5.0))

    i = np.arange(side_count, dtype=np.float32)
    theta = i * golden
    z = height * ((i + 0.5) / side_count)
    side = np.column_stack((
        radius * np.cos(theta),
        radius * np.sin(theta),
        z,
    ))

    j = np.arange(cap_count, dtype=np.float32)
    theta_cap = j * golden
    # sqrt gives approximately uniform disk density.
    r_cap = radius * np.sqrt((j + 0.5) / cap_count)
    cap_xy = np.column_stack((r_cap * np.cos(theta_cap),
                              r_cap * np.sin(theta_cap)))
    bottom = np.column_stack((cap_xy, np.zeros(cap_count, dtype=np.float32)))
    top = np.column_stack((cap_xy, np.full(cap_count, height, dtype=np.float32)))

    return np.vstack((side, bottom, top)).astype(np.float32)


def _low_discrepancy_2d(count):
    i = np.arange(count, dtype=np.float32)
    u = (0.5 + i * 0.754877666) % 1.0
    v = (0.5 + i * 0.569840291) % 1.0
    return u, v


def sample_box_surface(size_x, size_y, height, num_points):
    size_x = float(size_x)
    size_y = float(size_y)
    height = float(height)
    num_points = max(24, int(num_points))

    areas = np.array([
        size_y * height, size_y * height,
        size_x * height, size_x * height,
        size_x * size_y, size_x * size_y,
    ], dtype=np.float64)
    raw = areas / areas.sum() * num_points
    counts = np.maximum(2, np.floor(raw).astype(np.int64))
    while counts.sum() < num_points:
        counts[np.argmax(raw - counts)] += 1
    while counts.sum() > num_points:
        idx = np.argmax(counts)
        if counts[idx] > 2:
            counts[idx] -= 1
        else:
            break

    faces = []
    for face, n in enumerate(counts):
        u, v = _low_discrepancy_2d(int(n))
        u = u - 0.5
        v = v - 0.5
        pts = np.zeros((int(n), 3), dtype=np.float32)
        if face == 0:
            pts[:, 0] = -0.5 * size_x
            pts[:, 1] = u * size_y
            pts[:, 2] = (v + 0.5) * height
        elif face == 1:
            pts[:, 0] = 0.5 * size_x
            pts[:, 1] = u * size_y
            pts[:, 2] = (v + 0.5) * height
        elif face == 2:
            pts[:, 0] = u * size_x
            pts[:, 1] = -0.5 * size_y
            pts[:, 2] = (v + 0.5) * height
        elif face == 3:
            pts[:, 0] = u * size_x
            pts[:, 1] = 0.5 * size_y
            pts[:, 2] = (v + 0.5) * height
        elif face == 4:
            pts[:, 0] = u * size_x
            pts[:, 1] = v * size_y
            pts[:, 2] = 0.0
        else:
            pts[:, 0] = u * size_x
            pts[:, 1] = v * size_y
            pts[:, 2] = height
        faces.append(pts)
    return np.vstack(faces).astype(np.float32)


def _crescent_footprint_mask(x, y, outer_radius, inner_radius, bite_offset):
    outer = x * x + y * y <= outer_radius * outer_radius
    inner = x * x + (y - bite_offset) * (y - bite_offset) < inner_radius * inner_radius
    return outer & (~inner)


def _sample_crescent_cap(count, outer_radius, inner_radius, bite_offset, z):
    pts = []
    needed = int(count)
    batch = max(needed * 4, 64)
    cursor = 0
    while len(pts) < needed:
        i = np.arange(cursor, cursor + batch, dtype=np.float32)
        cursor += batch
        x = (2.0 * ((0.5 + i * 0.754877666) % 1.0) - 1.0) * outer_radius
        y_min = -outer_radius
        y_max = max(outer_radius, bite_offset + inner_radius)
        y = y_min + ((0.5 + i * 0.569840291) % 1.0) * (y_max - y_min)
        keep = _crescent_footprint_mask(x, y, outer_radius, inner_radius, bite_offset)
        for px, py in zip(x[keep], y[keep]):
            pts.append((px, py, z))
            if len(pts) >= needed:
                break
    return np.asarray(pts, dtype=np.float32)


def sample_crescent_surface(outer_radius, inner_radius, bite_offset, height, num_points):
    """Extruded 2-D crescent. Local concavity faces +Y."""
    outer_radius = float(outer_radius)
    inner_radius = float(inner_radius)
    bite_offset = float(bite_offset)
    height = float(height)
    num_points = max(64, int(num_points))

    side_count = max(32, int(num_points * 0.70))
    cap_count = max(8, (num_points - side_count) // 2)
    outer_count = side_count // 2
    inner_count = side_count - outer_count

    golden = math.pi * (3.0 - math.sqrt(5.0))

    outer_pts = []
    k = 0
    while len(outer_pts) < outer_count:
        theta = k * golden
        x = outer_radius * math.cos(theta)
        y = outer_radius * math.sin(theta)
        if x * x + (y - bite_offset) * (y - bite_offset) >= inner_radius * inner_radius:
            z = height * ((len(outer_pts) + 0.5) / outer_count)
            outer_pts.append((x, y, z))
        k += 1

    inner_pts = []
    k = 0
    while len(inner_pts) < inner_count:
        theta = k * golden
        x = inner_radius * math.cos(theta)
        y = bite_offset + inner_radius * math.sin(theta)
        if x * x + y * y <= outer_radius * outer_radius:
            z = height * ((len(inner_pts) + 0.5) / inner_count)
            inner_pts.append((x, y, z))
        k += 1

    bottom = _sample_crescent_cap(cap_count, outer_radius, inner_radius,
                                  bite_offset, 0.0)
    top = _sample_crescent_cap(cap_count, outer_radius, inner_radius,
                               bite_offset, height)
    return np.vstack((
        np.asarray(outer_pts, dtype=np.float32),
        np.asarray(inner_pts, dtype=np.float32),
        bottom,
        top,
    )).astype(np.float32)


class StaticObstaclePP:
    def __init__(self):
        rospy.init_node("static_obstacle_pp")

        self.frame_id = rospy.get_param("~frame_id", "world")
        self.topic = rospy.get_param("~topic", "/perception/cleaned_obstacles")
        self.extra_topic = rospy.get_param("~extra_topic", "/perception/persistent_obstacles")
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 30.0))
        self.shape = str(rospy.get_param("~shape", "cylinder")).strip().lower()

        self.center_x = float(rospy.get_param("~center_x", 0.535))
        self.center_y = float(rospy.get_param("~center_y", -0.190))
        self.base_z = float(rospy.get_param("~base_z", 0.0))
        self.yaw = float(rospy.get_param("~yaw", 0.0))
        self.radius = float(rospy.get_param("~radius", 0.04))
        self.height = float(rospy.get_param("~height", 0.18))
        self.prism_size_x = float(rospy.get_param("~prism_size_x", 0.10))
        self.prism_size_y = float(rospy.get_param("~prism_size_y", 0.06))
        self.crescent_outer_radius = float(rospy.get_param("~crescent_outer_radius", 0.055))
        self.crescent_inner_radius = float(rospy.get_param("~crescent_inner_radius", 0.055))
        self.crescent_bite_offset = float(rospy.get_param("~crescent_bite_offset", 0.030))
        self.num_points = int(rospy.get_param("~num_points", 1000))

        # Camera-independent moving obstacle. This mirrors persistent_cloud_node:
        # the sphere is overlaid only on the published obstacle cloud, so it does
        # not leave a stale point trail as it moves.
        self.moving_sphere_enabled = bool(rospy.get_param("~moving_sphere_enabled", False))
        self.moving_sphere_initial_position = np.array([
            float(rospy.get_param("~moving_sphere_initial_x", 0.30)),
            float(rospy.get_param("~moving_sphere_initial_y", -9.6)),
            float(rospy.get_param("~moving_sphere_initial_z", 0.5)),
        ], dtype=np.float32)
        self.moving_sphere_velocity = np.array([
            float(rospy.get_param("~moving_sphere_velocity_x", 0.0)),
            float(rospy.get_param("~moving_sphere_velocity_y", 0.30)),
            float(rospy.get_param("~moving_sphere_velocity_z", 0.0)),
        ], dtype=np.float32)
        self.moving_sphere_radius = float(rospy.get_param("~moving_sphere_radius", 0.08))
        self.moving_sphere_num_points = int(rospy.get_param("~moving_sphere_num_points", 1000))
        self._moving_sphere_start_time = None
        self._moving_sphere_offsets = None
        if self.moving_sphere_enabled:
            values = np.concatenate([
                self.moving_sphere_initial_position,
                self.moving_sphere_velocity,
                np.array([self.moving_sphere_radius], dtype=np.float32),
            ])
            if not np.isfinite(values).all():
                raise ValueError("Moving-sphere position, velocity, and radius must be finite.")
            self._moving_sphere_offsets = _fibonacci_sphere_offsets(
                self.moving_sphere_radius, self.moving_sphere_num_points)

        if self.shape == "cylinder":
            local = sample_cylinder_surface(self.radius, self.height, self.num_points)
        elif self.shape == "prism":
            local = sample_box_surface(self.prism_size_x, self.prism_size_y,
                                       self.height, self.num_points)
        elif self.shape == "crescent":
            local = sample_crescent_surface(self.crescent_outer_radius,
                                            self.crescent_inner_radius,
                                            self.crescent_bite_offset,
                                            self.height,
                                            self.num_points)
        else:
            raise ValueError("~shape must be one of: cylinder, crescent, prism")

        self.static_points = transform_local(local, self.center_x, self.center_y,
                                             self.base_z, self.yaw)
        self.points = self.static_points

        self.pub = rospy.Publisher(self.topic, PointCloud2, queue_size=1)
        self.extra_pub = None
        if self.extra_topic:
            self.extra_pub = rospy.Publisher(self.extra_topic, PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher("/viz/pp_cbf_obstacle", Marker, queue_size=1, latch=True)

        rospy.loginfo(
            "PP static %s obstacle: center=(%.3f, %.3f, %.3f) yaw=%.3f "
            "height=%.3f points=%d topic=%s extra=%s",
            self.shape,
            self.center_x,
            self.center_y,
            self.base_z + 0.5 * self.height,
            self.yaw,
            self.height,
            len(self.static_points),
            self.topic,
            self.extra_topic or "none",
        )
        if self.moving_sphere_enabled:
            rospy.loginfo(
                "PP moving sphere obstacle: initial=(%.3f, %.3f, %.3f) "
                "velocity=(%.3f, %.3f, %.3f) radius=%.3f points=%d",
                self.moving_sphere_initial_position[0],
                self.moving_sphere_initial_position[1],
                self.moving_sphere_initial_position[2],
                self.moving_sphere_velocity[0],
                self.moving_sphere_velocity[1],
                self.moving_sphere_velocity[2],
                self.moving_sphere_radius,
                len(self._moving_sphere_offsets),
            )

    def _moving_sphere_center(self, now_sec=None):
        if not self.moving_sphere_enabled:
            return None

        now_sec = rospy.get_time() if now_sec is None else float(now_sec)
        if self._moving_sphere_start_time is None or \
                now_sec < self._moving_sphere_start_time:
            self._moving_sphere_start_time = now_sec

        elapsed = now_sec - self._moving_sphere_start_time
        return _linear_motion_position(
            self.moving_sphere_initial_position,
            self.moving_sphere_velocity,
            elapsed)

    def _generate_moving_sphere(self, now_sec=None):
        center = self._moving_sphere_center(now_sec)
        if center is None:
            return None
        return self._moving_sphere_offsets + center[None, :]

    def _combined_points(self, sphere_points):
        if sphere_points is None or len(sphere_points) == 0:
            return self.static_points
        return np.vstack([self.static_points, sphere_points]).astype(np.float32)

    def _cloud_msg(self, points, stamp):
        header = Header(stamp=stamp, frame_id=self.frame_id)
        return pc2.create_cloud_xyz32(header, points)

    def _marker_msg(self, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = "pp_cbf_obstacle"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.08
        marker.color.g = 0.08
        marker.color.b = 0.08
        marker.color.a = 0.7

        if self.shape == "crescent":
            marker.type = Marker.POINTS
            marker.scale.x = 0.008
            marker.scale.y = 0.008
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                             for p in self.points]
        else:
            marker.pose.position.x = self.center_x
            marker.pose.position.y = self.center_y
            marker.pose.position.z = self.base_z + 0.5 * self.height
            qx, qy, qz, qw = yaw_quat(self.yaw)
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
            if self.shape == "cylinder":
                marker.type = Marker.CYLINDER
                marker.scale.x = 2.0 * self.radius
                marker.scale.y = 2.0 * self.radius
                marker.scale.z = self.height
            else:
                marker.type = Marker.CUBE
                marker.scale.x = self.prism_size_x
                marker.scale.y = self.prism_size_y
                marker.scale.z = self.height
            marker.points.append(Point())
        return marker

    def _moving_sphere_marker_msg(self, stamp, center):
        if center is None:
            return None

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = "pp_cbf_obstacle"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * self.moving_sphere_radius
        marker.scale.y = 2.0 * self.moving_sphere_radius
        marker.scale.z = 2.0 * self.moving_sphere_radius
        marker.color.r = 0.75
        marker.color.g = 0.05
        marker.color.b = 0.75
        marker.color.a = 0.55
        return marker

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate_hz))
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            now_sec = rospy.get_time()
            sphere_center = self._moving_sphere_center(now_sec)
            sphere_points = None
            if sphere_center is not None:
                sphere_points = self._moving_sphere_offsets + sphere_center[None, :]
            msg = self._cloud_msg(self._combined_points(sphere_points), stamp)
            self.pub.publish(msg)
            if self.extra_pub is not None:
                self.extra_pub.publish(msg)
            self.marker_pub.publish(self._marker_msg(stamp))
            sphere_marker = self._moving_sphere_marker_msg(stamp, sphere_center)
            if sphere_marker is not None:
                self.marker_pub.publish(sphere_marker)
            rate.sleep()


if __name__ == "__main__":
    StaticObstaclePP().run()
