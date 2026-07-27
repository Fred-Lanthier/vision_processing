"""Opt-in episode lifecycle hooks, so an ablation campaign never reloads models.

Startup of the full pick-place pipeline costs ~150 s (Gazebo 12 s, the CBF
node's URDFLayer + 9 Bernstein SDF models + graph capture 38 s, SAM3 ~45 s, the
FM checkpoint + torch.compile warmup on top). None of that depends on the
ablation cell: the weights, the URDF layer and the CUDA graph are identical in
every cell of the matrix. Relaunching per trial therefore pays a fixed ~2.5 min
to reproduce state that never changed.

This module lets a node expose two ``std_srvs/Trigger`` hooks so a trial becomes
an *episode* inside one long-lived process instead of a process lifecycle:

  ``~episode/reset``        clear PER-EPISODE state (tracker memory, voxel map,
                            filter memories, progress clocks, grasp state).
                            Must be cheap -- it runs ~290 times in a campaign.
  ``~episode/reconfigure``  re-read parameters from the param server and rebuild
                            whatever they are baked into (e.g. the CBF CUDA
                            graph, ~1-3 s). Runs ~25 times in a campaign.

The param server stays the single source of truth: the campaign director writes
rosparams and then calls ``reconfigure``, so a bag's recorded ``params.yaml``
still describes the cell exactly, and nothing needs a custom .srv (no rebuild).

INERT BY DEFAULT. ``EpisodeControl`` advertises nothing unless the node is
launched with ``~enable_episode_control`` true, so the live feeding launches
carry this code without being able to reach it. Handlers are also serialized
under one lock, because a reset racing a 100 Hz control loop is exactly the kind
of bug that would silently poison a campaign.

Usage in a node::

    import rospkg, sys
    _pkg = rospkg.RosPack().get_path('vision_processing')
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)
    from episode_control import EpisodeControl

    self.episode = EpisodeControl(
        on_reset=self._episode_reset,
        on_reconfigure=self._episode_reconfigure,   # optional
    )

where ``_episode_reset`` returns None (success) or a message string, and raises
to report failure. The director treats a failed reset as a failed trial rather
than silently recording a contaminated episode.
"""

import threading


class EpisodeControl:
    """Advertises ``~episode/reset`` / ``~episode/reconfigure`` when enabled.

    Fully inert when ``~enable_episode_control`` is false (the default): no
    services, no publishers, no behaviour change whatsoever. Degrades to a no-op
    if rospy is unavailable so importing it can never break a node."""

    def __init__(self, on_reset=None, on_reconfigure=None,
                 param="~enable_episode_control", enabled=None):
        self.on_reset = on_reset
        self.on_reconfigure = on_reconfigure
        self.lock = threading.RLock()
        self.episode_count = 0
        self.enabled = False
        self._rospy = None
        self._services = []

        try:
            import rospy
            from std_srvs.srv import Trigger, TriggerResponse
        except Exception:
            return
        self._rospy = rospy
        self._Trigger = Trigger
        self._TriggerResponse = TriggerResponse

        self.enabled = bool(rospy.get_param(param, False)) if enabled is None \
            else bool(enabled)
        if not self.enabled:
            return

        if on_reset is not None:
            self._advertise("~episode/reset", on_reset, "reset")
        if on_reconfigure is not None:
            self._advertise("~episode/reconfigure", on_reconfigure, "reconfigure")
        rospy.loginfo("Episode control enabled: %s",
                      ", ".join(s.resolved_name for s in self._services))

    def _advertise(self, name, handler, label):
        def _wrapped(_req):
            rospy = self._rospy
            try:
                # Serialized: a reset must never interleave with another hook
                # (or with a node's own use of `with self.episode.lock`).
                with self.lock:
                    message = handler()
                    if label == "reset":
                        self.episode_count += 1
            except Exception as exc:                  # noqa: BLE001 - reported
                rospy.logerr("episode %s failed: %s", label, exc)
                return self._TriggerResponse(success=False, message=str(exc))
            text = message if isinstance(message, str) else f"{label} ok"
            rospy.loginfo("episode %s: %s", label, text)
            return self._TriggerResponse(success=True, message=text)

        self._services.append(
            self._rospy.Service(name, self._Trigger, _wrapped))
