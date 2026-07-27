#!/usr/bin/env python3
"""Episode director: reset the world + pipeline in place, without reloading.

WHY THIS EXISTS
---------------
Bringing the pick-place pipeline up costs ~150 s (measured: Gazebo + controllers
12 s, the CBF node's URDFLayer + 9 Bernstein SDF models + graph capture 38 s,
SAM3 ~45 s, the FM checkpoint + torch.compile warmup on top). An ablation
campaign of ~290 episodes that relaunches per trial spends ~12 h reproducing
state that is IDENTICAL in every cell. This node makes a trial an episode inside
one long-lived process tree instead: the weights, the URDF layers, the CUDA
graphs and Gazebo all stay resident, and only the world and the per-episode node
state are rewound.

WHAT IT DOES NOT DO
-------------------
It never calls /gazebo/reset_simulation. That service rewinds /clock, and with
use_sim_time every rospy timer, every rospy.Rate and every bag timestamp in the
system is derived from it -- a backwards jump corrupts the recording and the
control loops alike. The reset here keeps the clock strictly monotonic: pause,
write model states, unpause.

RESET ORDER (it matters)
------------------------
  0. close the episode gate         -- the pipeline is a live closed loop; the
                                       planner replans at 3 Hz and would simply
                                       resume the task through the reset
  1. pause physics                  -- nothing moves while the world is rewritten
  2. node state resets              -- INCLUDING the grasp FSM, which detaches
                                       the weld; a red_cube still attached to
                                       panda_hand would fight step 4 and the
                                       scene reset would silently half-fail
  3. arm -> home configuration      -- set_model_configuration, then link
                                       twists explicitly zeroed
  4. objects -> episode poses       -- with optional per-episode randomization
  5. unpause, settle, verify home
  6. wait for the perception chain, then open the gate

Node resets are DISCOVERED, not hardcoded: any node advertising
``<node>/episode/reset`` is included. That keeps the director identical across
the three tier-4 groups (panda-feeding, panda-pickplace, xarm7), which advertise
different node sets.
"""

from __future__ import annotations

import math
import random
import threading
import time

import rospy
from controller_manager_msgs.srv import SwitchController
from gazebo_msgs.msg import LinkState, ModelState
from gazebo_msgs.srv import (GetJointProperties, GetLinkState, GetModelState,
                             SetLinkState, SetModelConfiguration, SetModelState)
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Empty, Trigger, TriggerResponse

# Home configuration, matching the -J spawn arguments in
# pickplace_simulation.launch. Kept here (not read from the launch) so a reset
# cannot drift from what the campaign believes it reset to.
PANDA_HOME = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.188029,
    "panda_joint3": 0.0,
    "panda_joint4": -1.767649,
    "panda_joint5": 0.0,
    "panda_joint6": 1.955677,
    "panda_joint7": 0.785399,
}
FINGER_HOME = {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}

# TCP offset along the hand frame's approach axis (condition_pcd_pickplace.py).
TCP_FROM_HAND_Z = 0.1034


class EpisodeDirector:
    # Links whose twist is zeroed after the teleport (_zero_link_velocities).
    ARM_LINKS = [f"panda_link{i}" for i in range(8)] + [
        "panda_hand", "panda_leftfinger", "panda_rightfinger"]

    def __init__(self):
        rospy.init_node("episode_director")

        self.robot_model = rospy.get_param("~robot_model", "panda")
        # Gazebo advertises its services only once the world + models exist;
        # the director launches alongside it, so this must cover startup.
        self.gazebo_service_timeout = float(rospy.get_param(
            "~gazebo_service_timeout", 300.0))
        self.settle_time = float(rospy.get_param("~settle_time", 1.0))
        self.ready_timeout = float(rospy.get_param("~ready_timeout", 60.0))
        # Topics whose freshness proves the perception chain re-converged after
        # the tracker memory was dropped. Empty list = skip the wait.
        self.ready_topics = rospy.get_param(
            "~ready_topics", ["/vision/sam2_mask_cube", "/vision/merged_cloud"])
        self.home = dict(PANDA_HOME)
        self.home.update(FINGER_HOME if rospy.get_param("~reset_fingers", True)
                         else {})
        # Re-cleared after the unpause, once post-reset joint states exist: any
        # node that latches onto /joint_states would otherwise still hold the
        # pre-reset pose and command the arm back out of home.
        self.post_settle_services = rospy.get_param("~post_settle_services", [
            "/trajectory_executor/episode/reset",
            "/cbf_safety_Bernstein/episode/reset",
        ])

        # --- home verification -------------------------------------------
        # 0 disables the home check entirely.
        self.home_tolerance = float(rospy.get_param("~home_tolerance", 0.05))
        # TCP tolerance [m]: the tight, task-relevant bound -- this is what the
        # policy and the task actually see. 2 cm is well under the cube (4 cm)
        # and the object-pose randomization (5 cm).
        self.tcp_tolerance = float(rospy.get_param("~tcp_tolerance", 0.02))
        # Joint tolerance [rad]: LOOSE, a sanity bound on gross posture only.
        # The CBF's null-space clearance deliberately swivels the elbow WITHOUT
        # moving the TCP, so a 0.07 rad joint error can leave the gripper
        # visibly in the same place. Measured residual after velocity zeroing is
        # 0.02-0.07, mostly that swivel, so a 0.05 bound here was rejecting
        # episodes that were fine. The barrier is whole-body, though, so a
        # grossly different posture is still a different episode -- hence a
        # bound rather than none.
        self.joint_tolerance = float(rospy.get_param("~joint_tolerance", 0.15))
        self.home_timeout = float(rospy.get_param("~home_timeout", 10.0))
        # Log Gazebo's own joint positions/rates around the teleport.
        self.probe_joints = bool(rospy.get_param("~probe_joints", True))
        # Teleport passes. DEFAULT 1, on measurement: a second pass was tried to
        # absorb the post-unpause controller kick, and made things WORSE
        # whenever the residual is not a passive transient. With an obstacle
        # inside the home clearance band the CBF drives j1/j3 apart and the
        # error GROWS with time (0.059 rad after pass 1, 0.098 after pass 2),
        # so extra passes just donate more pushing time.
        self.home_passes = max(1, int(rospy.get_param("~home_passes", 1)))
        # Controllers to stop/restart after homing. OFF by default: it was added
        # to clear the C++ controller's latched hold target, but the explicit
        # /ablation/hold_pose command now drives the arm home through the normal
        # feedback path, so the cycle is redundant AND harmful -- its restart
        # re-latches on whatever pose exists at that instant.
        self.cycle_controllers = rospy.get_param("~cycle_controllers", [])

        # Object poses. Defaults reproduce pickplace_simulation.launch exactly.
        self.objects = rospy.get_param("~objects", {
            "red_cube": {"x": 0.56, "y": -0.035, "z": 0.77},
            "brown_box": {"x": 0.515, "y": -0.35, "z": 0.755},
        })
        # Per-episode uniform randomization half-widths [m], applied to the
        # object poses above. Tier-1: costs nothing, changes every episode.
        self.randomize = rospy.get_param("~randomize", {})

        self._home_tcp = None       # TCP sampled at the teleport, per reset
        self._lock = threading.RLock()
        self.episode_index = 0
        self._rng = random.Random(int(rospy.get_param("~seed", 0)))

        self._pause = self._proxy("/gazebo/pause_physics", Empty)
        self._unpause = self._proxy("/gazebo/unpause_physics", Empty)
        self._set_state = self._proxy("/gazebo/set_model_state", SetModelState)
        self._set_config = self._proxy("/gazebo/set_model_configuration",
                                       SetModelConfiguration)
        self._get_state = self._proxy("/gazebo/get_model_state", GetModelState)
        # Reads joint position AND rate straight from Gazebo, so it works while
        # physics is paused (unlike /joint_states, which stops publishing).
        self._get_joint = self._proxy("/gazebo/get_joint_properties",
                                      GetJointProperties)
        self._get_link = self._proxy("/gazebo/get_link_state", GetLinkState)
        self._set_link = self._proxy("/gazebo/set_link_state", SetLinkState)

        self.episode_pub = rospy.Publisher("/ablation/episode", Bool,
                                           queue_size=1, latch=True)
        # Explicit hold target for the armed state. The executor commands this
        # while the gate is closed, so the position feedback actively holds the
        # home configuration instead of letting the arm settle wherever gravity
        # puts it (measured sag: 0.04-0.07 rad on every pitch joint).
        self.hold_pose_pub = rospy.Publisher("/ablation/hold_pose",
                                             Float64MultiArray,
                                             queue_size=1, latch=True)
        rospy.Service("~reset_episode", Trigger, self._srv_reset)
        rospy.Service("~reconfigure", Trigger, self._srv_reconfigure)

        rospy.loginfo("episode_director ready (robot=%s). Reset services "
                      "discovered dynamically at each reset.", self.robot_model)

    # ------------------------------------------------------------------ util

    def _proxy(self, name, srv_type):
        """Wait for a Gazebo service, patiently.

        The director starts with the rest of the launch, but Gazebo advertises
        these only after the world and all models are spawned (~12 s measured,
        longer under load). A short timeout just loses the race and kills the
        director, so wait out the whole pipeline startup budget instead."""
        deadline = time.time() + self.gazebo_service_timeout
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service(name, timeout=10.0)
                return rospy.ServiceProxy(name, srv_type)
            except rospy.ROSException:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"{name} never appeared within "
                        f"{self.gazebo_service_timeout:.0f}s -- is Gazebo "
                        f"running?")
                rospy.loginfo_throttle(20.0,
                                       "episode_director: waiting for %s", name)
        raise RuntimeError("shutdown while waiting for " + name)

    @staticmethod
    def _discover(suffix):
        """Every advertised service ending in ``suffix`` (e.g. episode/reset)."""
        import rosservice
        try:
            return sorted(s for s in rosservice.get_service_list()
                          if s.endswith(suffix))
        except Exception as exc:                       # noqa: BLE001
            rospy.logwarn("service discovery failed: %s", exc)
            return []

    def _call_all(self, suffix):
        """Call every discovered hook; report failures rather than swallow them.

        A silently failed reset is the worst outcome available here -- it
        produces a plausible-looking episode contaminated by the previous one --
        so failures propagate to the campaign runner as a failed trial."""
        results, failures = [], []
        for name in self._discover(suffix):
            try:
                response = rospy.ServiceProxy(name, Trigger)()
                if response.success:
                    results.append(f"{name}: {response.message}")
                else:
                    failures.append(f"{name}: {response.message}")
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"{name}: {exc}")
        for line in results:
            rospy.loginfo("  %s", line)
        if failures:
            raise RuntimeError("; ".join(failures))
        return results

    def _call_named(self, names):
        for name in names:
            try:
                response = rospy.ServiceProxy(name, Trigger)()
                if not response.success:
                    raise RuntimeError(f"{name}: {response.message}")
            except Exception as exc:                   # noqa: BLE001
                raise RuntimeError(f"post-settle reset {name}: {exc}")

    def _cycle_controllers(self):
        """Stop then restart the velocity controller so it re-latches on home."""
        if not self.cycle_controllers:
            return
        try:
            rospy.wait_for_service("/controller_manager/switch_controller",
                                   timeout=10.0)
            switch = rospy.ServiceProxy("/controller_manager/switch_controller",
                                        SwitchController)
            switch(start_controllers=[], stop_controllers=self.cycle_controllers,
                   strictness=2)
            switch(start_controllers=self.cycle_controllers, stop_controllers=[],
                   strictness=2)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(f"controller cycle failed: {exc}")

    # -------------------------------------------------------------- probing

    def _probe_joints(self, label):
        """Log Gazebo's own joint positions and rates. Works while paused.

        Probing before/after the teleport and after the unpause is what
        distinguishes 'the teleport did not land' from 'it landed and then
        something drove the arm back off home'."""
        if not self.probe_joints:
            return None
        pos, rate = [], []
        for i in range(1, 8):
            try:
                r = self._get_joint(f"panda_joint{i}")
                pos.append(r.position[0] if r.position else float("nan"))
                rate.append(r.rate[0] if r.rate else float("nan"))
            except Exception:                          # noqa: BLE001
                return None
        err = max(abs(p - PANDA_HOME[f"panda_joint{i + 1}"])
                  for i, p in enumerate(pos))
        rate_max = max(abs(r) for r in rate)
        rospy.loginfo("  [probe %-16s] max|q-home|=%.4f rad  max|rate|=%.4f "
                      "rad/s  q=[%s]", label, err, rate_max,
                      " ".join(f"{v:+.3f}" for v in pos))
        return err, rate_max

    def _tcp_position(self):
        """TCP in world, from the hand link pose."""
        try:
            r = self._get_link(f"{self.robot_model}::panda_hand", "")
            if not r.success:
                return None
        except Exception:                              # noqa: BLE001
            return None
        p, q = r.link_state.pose.position, r.link_state.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        return (p.x + TCP_FROM_HAND_Z * 2.0 * (x * z + y * w),
                p.y + TCP_FROM_HAND_Z * 2.0 * (y * z - x * w),
                p.z + TCP_FROM_HAND_Z * (1.0 - 2.0 * (x * x + y * y)))

    # --------------------------------------------------------------- world

    def _home_arm(self):
        """Arm -> home configuration, with the teleport's momentum removed."""
        state = ModelState()
        state.model_name = self.robot_model
        current = self._get_state(self.robot_model, "")
        state.pose = current.pose          # base is fixed to the table; keep it
        state.reference_frame = "world"    # zero twist by construction
        self._set_state(state)

        names = list(self.home.keys())
        self._set_config(model_name=self.robot_model,
                         urdf_param_name="robot_description",
                         joint_names=names,
                         joint_positions=[self.home[n] for n in names])
        self._zero_link_velocities()

    def _zero_link_velocities(self):
        """Kill the link twists the teleport creates.

        set_model_configuration zeroes JOINT rates, but Gazebo derives LINK
        velocities from the instantaneous position change, so the first physics
        tick after the unpause saw ~4.3 rad/s -- six times max_joint_velocity --
        and kicked the arm off home. Writing each link back at its current pose
        with an explicitly zero twist removes the kick at its source (measured:
        4.35 -> ~0.95 rad/s), which is simpler and more direct than trying to
        absorb it afterwards."""
        for link in self.ARM_LINKS:
            try:
                current = self._get_link(f"{self.robot_model}::{link}", "")
                if not current.success:
                    continue
                state = LinkState()
                state.link_name = f"{self.robot_model}::{link}"
                state.pose = current.link_state.pose
                state.reference_frame = "world"   # twist defaults to zero
                self._set_link(state)
            except Exception as exc:                   # noqa: BLE001
                rospy.logwarn_throttle(30.0, "could not zero %s twist: %s",
                                       link, exc)

    def _place_objects(self):
        placed = []
        for name, pose in self.objects.items():
            jitter = self.randomize.get(name, {})
            state = ModelState()
            state.model_name = name
            state.pose.position.x = pose["x"] + self._jitter(jitter.get("x", 0.0))
            state.pose.position.y = pose["y"] + self._jitter(jitter.get("y", 0.0))
            state.pose.position.z = pose["z"] + self._jitter(jitter.get("z", 0.0))
            state.pose.orientation.w = 1.0
            state.reference_frame = "world"
            self._set_state(state)
            placed.append(f"{name}@({state.pose.position.x:.3f},"
                          f"{state.pose.position.y:.3f})")
        return placed

    def _jitter(self, half_width):
        half_width = float(half_width)
        return self._rng.uniform(-half_width, half_width) if half_width else 0.0

    def _wait_fresh_joint_states(self, count=3, timeout=10.0):
        """Block until joint states published AFTER the unpause have flowed."""
        deadline = time.time() + timeout
        for _ in range(count):
            remaining = max(0.2, deadline - time.time())
            try:
                rospy.wait_for_message("/joint_states", JointState,
                                       timeout=remaining)
            except Exception:
                raise RuntimeError("no /joint_states after unpause -- is "
                                   "Gazebo still paused?")

    # -------------------------------------------------------- verification

    def _home_error(self):
        msg = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
        by_name = dict(zip(msg.name, msg.position))
        worst, worst_joint = 0.0, ""
        for joint, target in PANDA_HOME.items():
            if joint not in by_name:
                continue
            error = abs(by_name[joint] - target)
            if error > worst:
                worst, worst_joint = error, joint
        return worst, worst_joint

    def _verify_home(self):
        """Fail the episode only if the arm is not REALLY back at the start.

        Two bounds, because joint angle alone is misleading. The CBF's
        null-space clearance deliberately swivels the elbow WITHOUT moving the
        TCP, so a 0.07 rad joint error can leave the gripper visibly in the same
        place -- which is why this looked fine to the eye while a tight joint
        tolerance kept failing episodes.

          * TCP position (tight): what the task and the policy actually see.
          * joint angles (loose): still bounded, because the barrier is
            whole-body, so a grossly different posture is a different episode
            even at the same TCP.

        A failure here is loud on purpose: a trial that silently starts from the
        previous episode's pose looks valid and is not comparable, which is
        worse than a missing trial."""
        if self.home_tolerance <= 0.0:
            return
        deadline = time.time() + self.home_timeout
        worst_j, worst_joint, worst_tcp = float("inf"), "", float("inf")
        while time.time() < deadline:
            try:
                worst_j, worst_joint = self._home_error()
            except Exception:
                raise RuntimeError("cannot verify home: no /joint_states")
            tcp = self._tcp_position()
            worst_tcp = (float("inf") if (tcp is None or self._home_tcp is None)
                         else math.dist(tcp, self._home_tcp))
            tcp_ok = (self._home_tcp is None or worst_tcp <= self.tcp_tolerance)
            if tcp_ok and worst_j <= self.joint_tolerance:
                rospy.loginfo("  home reached (TCP %.4f m, worst joint %.4f rad "
                              "on %s)", worst_tcp, worst_j, worst_joint)
                return
            rospy.sleep(0.1)
        raise RuntimeError(
            f"arm never returned home: TCP off by {worst_tcp:.4f} m (max "
            f"{self.tcp_tolerance}), {worst_joint} off by {worst_j:.4f} rad "
            f"(max {self.joint_tolerance}) after {self.home_timeout:.1f}s")

    def _wait_ready(self):
        """Wait for the pipeline to re-converge after the tracker memory drop.

        The SAM2 banks were cleared, so the masks (and everything downstream)
        are stale until SAM3 re-prompts on the next frame. Recording before that
        would capture an episode whose first seconds ran on no conditioning
        cloud at all. Wall clock, not sim time: readiness is about how long the
        operator waits, and wait_for_message's own timeout is wall-based."""
        deadline = time.time() + self.ready_timeout
        for topic in self.ready_topics:
            msg_type = PointCloud2 if "cloud" in topic else Image
            remaining = max(0.5, deadline - time.time())
            try:
                rospy.wait_for_message(topic, msg_type, timeout=remaining)
            except Exception:
                raise RuntimeError(
                    f"timed out waiting for {topic} after the reset -- the "
                    f"perception chain did not re-converge, so this episode "
                    f"would start on stale conditioning")
        try:
            rospy.wait_for_message("/cbf_safety/ready", Bool,
                                   timeout=max(0.5, deadline - time.time()))
        except Exception:
            rospy.logwarn("no /cbf_safety/ready within timeout "
                          "(latched topic; continuing)")

    # ------------------------------------------------------------- services

    def _srv_reset(self, _req):
        with self._lock:
            try:
                message = self._reset()
            except Exception as exc:                   # noqa: BLE001
                # ALWAYS reopen the gate, even on failure. _reset closes it at
                # the start and only reopens it on its last line, so anything
                # that raises in between would otherwise leave the pipeline
                # permanently held: the planner stops planning, the executor
                # holds, and every later reset fails the same way. The trial is
                # still reported as failed -- the runner writes no bag -- but
                # the system stays usable and recoverable.
                rospy.logerr("episode reset FAILED: %s", exc)
                rospy.logwarn("reopening the episode gate so the pipeline "
                              "stays usable; this episode is NOT valid")
                self.episode_pub.publish(Bool(data=True))
                return TriggerResponse(success=False, message=str(exc))
            return TriggerResponse(success=True, message=message)

    def _reset(self):
        t0 = time.time()
        # CLOSE THE EPISODE GATE FIRST. The pipeline is a continuously running
        # closed loop: the FM planner replans at 3 Hz, the executor tracks, the
        # CBF drives the arm. Rewinding the world underneath a live policy just
        # teleports the arm home and lets the policy resume the task from
        # wherever it believes it is -- measured 0.69 rad AWAY from home within
        # 10 s, getting worse over time. There is no "homing mechanism" fix for
        # that; the loop has to be held. /ablation/episode is latched, and the
        # planner + executor gate on it (defaulting to True, so nothing changes
        # outside the campaign).
        self.episode_pub.publish(Bool(data=False))

        placed = []
        for attempt in range(1, self.home_passes + 1):
            self._pause()
            try:
                if attempt == 1:
                    # Node state first: this detaches the grasp weld, without
                    # which the cube cannot be repositioned.
                    rospy.loginfo("episode reset: clearing node state")
                    self._probe_joints("before teleport")
                    self._call_all("episode/reset")

                self._home_arm()
                self._probe_joints(f"after teleport {attempt}")
                # Joints are exactly home here, so this TCP is the reference.
                self._home_tcp = self._tcp_position()
                if attempt == 1:
                    placed = self._place_objects()
                # Published while still paused, so the executor's very first
                # tick after the unpause already commands home rather than the
                # pose the previous episode ended in.
                self.hold_pose_pub.publish(Float64MultiArray(
                    data=[PANDA_HOME[f"panda_joint{i}"] for i in range(1, 8)]))
            finally:
                self._unpause()

            self._probe_joints(f"after unpause {attempt}")
            # The control chain latches onto /joint_states, and while physics
            # was paused none were published -- so the executor's hold position
            # and the CBF's integrator state still describe the PRE-reset pose.
            # Wait for genuinely post-reset joint states before re-latching.
            self._wait_fresh_joint_states()
            if attempt == 1:
                self._cycle_controllers()
                self._call_named(self.post_settle_services)

            rospy.sleep(self.settle_time)
            probe = self._probe_joints(f"after settle {attempt}")
            if probe is not None and probe[0] <= self.joint_tolerance:
                break

        self._verify_home()
        self._wait_ready()

        self.episode_index += 1
        self.episode_pub.publish(Bool(data=True))
        return (f"episode {self.episode_index} armed in "
                f"{time.time() - t0:.1f}s; placed {', '.join(placed)}")

    def _srv_reconfigure(self, _req):
        """Push whatever the campaign wrote to the param server into the nodes.

        Tier 2: the CBF node REFUSES a live change to any parameter baked into
        its CUDA graph and asks for a restart instead, so a refusal here is the
        system working, not breaking."""
        with self._lock:
            try:
                results = self._call_all("episode/reconfigure")
            except Exception as exc:                   # noqa: BLE001
                rospy.logerr("reconfigure FAILED: %s", exc)
                return TriggerResponse(success=False, message=str(exc))
            return TriggerResponse(success=True,
                                   message=" | ".join(results) or "no nodes")


if __name__ == "__main__":
    EpisodeDirector()
    rospy.spin()
