"""
RobotSelfFilter — the fork SDF shell, generalised to every protected link.

ForkFilter carves a 5 mm shell around the TRUE fork surface: sample fork_tip.stl
once, build a KD-tree, drop any point closer than the margin. It can keep that
tree in the CAMERA frame forever because the fork and the wrist camera are both
rigid on panda_TCP.

This is the same filter for the rest of the robot. The only difference is that
the arm moves under the camera, so each link's tree lives in its own LINK frame
and the runtime step transforms the candidate points into that frame using TF
(which the projector already has) before querying.

Surfaces come from the URDF's own <collision> geometry — meshes and primitives
alike — so the shell always matches the robot actually being simulated.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation


def _origin_transform(elem):
    """4x4 from a URDF <origin> child, identity when absent."""
    T = np.eye(4)
    if elem is None:
        return T
    o = elem.find("origin")
    if o is None:
        return T
    xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


class RobotSelfFilter:
    """Drop cloud points lying within ``margin`` of any protected link surface.

    Parameters
    ----------
    robot_xml : str
        URDF text (typically the /robot_description parameter).
    link_names : list[str]
        Protected links — the same set the CBF is given.
    margin : float
        Shell thickness [m]; 0.005 mirrors the fork filter.
    resolution : float
        Target surface-sample spacing [m]. Sample COUNT is derived per link from
        its area, because a fixed count is what breaks on big links: the fork is
        tiny, so 2000 samples put them ~1.7 mm apart and the shell is tight, but
        the same 2000 spread over panda_link5 land ~8 mm apart — wider than the
        5 mm margin, so points sitting between samples measure as clearance and
        survive. Defaults to margin/4, which keeps the discretisation error well
        inside the shell on every link.
    """

    def __init__(self, robot_xml, link_names, margin=0.005,
                 resolution=None, max_samples_per_link=200000, logger=None):

        self.active = False
        self.margin = float(margin)
        self.resolution = float(resolution) if resolution else self.margin / 4.0
        self.max_samples = int(max_samples_per_link)
        self.links = []            # [(name, cKDTree, centre, radius, hulls)]
        self._log = logger or (lambda msg: print("[robot_self_filter] " + msg))

        try:
            import trimesh
            from scipy.spatial import cKDTree
            import xml.etree.ElementTree as ET
        except Exception as e:                              # pragma: no cover
            self._log(f"disabled (deps missing: {e!r}); cloud passes through.")
            return

        try:
            root = ET.fromstring(robot_xml)
        except Exception as e:                              # pragma: no cover
            self._log(f"disabled (URDF parse failed: {e!r}); cloud passes through.")
            return

        by_name = {l.get("name"): l for l in root.findall("link")}
        for name in link_names:
            link = by_name.get(name)
            if link is None:
                self._log(f"link {name} not in URDF, skipped.")
                continue
            built = self._build_link(link, trimesh)
            if built is None:
                self._log(f"link {name} has no usable collision geometry, skipped.")
                continue
            pts, hulls = built
            # Broad phase is the link-frame AABB, not a bounding ball: the arm
            # links are long and thin, so a ball around panda_link5 sweeps most
            # of the workspace into the KD query for nothing.
            lo = pts.min(axis=0) - self.margin
            hi = pts.max(axis=0) + self.margin
            self.links.append((name, cKDTree(pts), lo, hi, hulls))

        if not self.links:
            self._log("disabled (no link surfaces built); cloud passes through.")
            return

        self.active = True
        n_pts = sum(len(t.data) for _, t, _, _, _ in self.links)
        self._log(
            f"active: {self.margin * 1e3:.0f} mm shell around {len(self.links)} links "
            f"({', '.join(n for n, _, _, _, _ in self.links)}); "
            f"{n_pts} surface samples at {self.resolution * 1e3:.1f} mm."
        )

    # ── Startup: one surface sample cloud per link, in the LINK frame ─────────

    def _build_link(self, link, trimesh):
        """Return (surface samples, convex half-space sets) for one link.

        The half-spaces are the INTERIOR test. Distance to a sampled surface is
        unsigned, so a depth point that noise pushed a centimetre *into* a link
        reads as a centimetre of clearance and survives — it then enters the
        obstacle cloud as a phantom obstacle inside the robot. The fork never
        showed this (thin tines have no interior to speak of); an arm link very
        much does. Every Panda collision mesh is already convex (hull volume ==
        mesh volume), and the fingers are unions of boxes kept separately here,
        so testing "inside ANY of these convex pieces" is exact, not an
        approximation, and costs one matrix product per link.
        """
        import rospkg

        meshes = []
        for coll in link.findall("collision"):
            geom = coll.find("geometry")
            if geom is None:
                continue
            T = _origin_transform(coll)
            mesh = None
            if geom.find("mesh") is not None:
                m = geom.find("mesh")
                path = self._resolve_mesh_path(m.get("filename"), rospkg)
                if path is None:
                    continue
                mesh = trimesh.load(path, force="mesh")
                if m.get("scale"):
                    mesh.apply_scale([float(v) for v in m.get("scale").split()])
            elif geom.find("box") is not None:
                mesh = trimesh.creation.box(
                    extents=[float(v) for v in geom.find("box").get("size").split()])
            elif geom.find("cylinder") is not None:
                c = geom.find("cylinder")
                mesh = trimesh.creation.cylinder(
                    radius=float(c.get("radius")), height=float(c.get("length")))
            elif geom.find("sphere") is not None:
                mesh = trimesh.creation.icosphere(
                    radius=float(geom.find("sphere").get("radius")))
            if mesh is None:
                continue
            mesh.apply_transform(T)
            meshes.append(mesh)

        if not meshes:
            return None

        # Sample count from area, so spacing is uniform across links.
        merged = trimesh.util.concatenate(meshes)
        n = int(np.clip(merged.area / (self.resolution ** 2), 500, self.max_samples))
        pts, _ = trimesh.sample.sample_surface(merged, n)

        hulls = []
        for m in meshes:
            hull = m.convex_hull
            A = np.asarray(hull.face_normals, dtype=np.float64)
            b = -(A * np.asarray(hull.triangles)[:, 0, :]).sum(axis=1)
            hulls.append((A, b))
        return np.asarray(pts, dtype=np.float64), hulls

    @staticmethod
    def _resolve_mesh_path(filename, rospkg):
        if not filename:
            return None
        if filename.startswith("package://"):
            pkg, _, rel = filename[len("package://"):].partition("/")
            try:
                filename = os.path.join(rospkg.RosPack().get_path(pkg), rel)
            except Exception:
                return None
        elif filename.startswith("file://"):
            filename = filename[len("file://"):]
        return filename if os.path.isfile(filename) else None

    # ── Runtime ───────────────────────────────────────────────────────────────

    def filter_points(self, pts, lookup):
        """Remove points on the robot.

        pts    : (N, 3) in some frame F.
        lookup : callable(link_name) -> (R, t) placing the link in F, or None
                 when that TF is unavailable (the link is then skipped).

        Returns the surviving points; ``pts`` unchanged if the filter is off.
        """
        if not self.active or pts is None or len(pts) == 0:
            return pts

        pts = np.asarray(pts)
        keep = np.ones(len(pts), dtype=bool)
        for name, tree, lo, hi, hulls in self.links:
            pose = lookup(name)
            if pose is None:
                continue
            R, t = pose
            # Only points still alive and inside the link's AABB can be in the
            # shell — the KD query then runs on a handful of points, not the
            # whole frame.
            idx = np.flatnonzero(keep)
            local = (pts[idx] - t) @ R                      # F -> link frame
            sel = ((local >= lo) & (local <= hi)).all(axis=1)
            near, local_near = idx[sel], local[sel]
            if near.size == 0:
                continue
            # distance_upper_bound lets the tree stop descending as soon as it
            # is past the margin: we only ever care about the shell, not the
            # true nearest distance.
            dist, _ = tree.query(local_near, k=1,
                                 distance_upper_bound=self.margin, workers=-1)
            drop = np.isfinite(dist)                        # in the shell
            for A, b in hulls:                              # or inside the link
                drop |= ((local_near @ A.T + b) <= 0.0).all(axis=1)
            keep[near[drop]] = False
        return pts[keep]
