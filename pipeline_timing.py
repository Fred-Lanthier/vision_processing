"""Lightweight per-stage timing publisher, recorded into the rosbag.

Every pipeline node imports this and reports how long each stage takes, so the
full per-stage timing breakdown (thesis section 5.6) can be reconstructed
offline from one bag. Each stage publishes its duration in milliseconds on its
own topic ``<namespace>/<stage>`` (default ``/pipeline/timing/<stage>``) as a
``std_msgs/Float64``. Using one topic per stage avoids a custom message (no
rebuild) and lets ``rosbag record -e '/pipeline/timing/.*'`` grab them all.

Usage in a node::

    import rospkg, sys, os
    _pkg = rospkg.RosPack().get_path('vision_processing')
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)
    from pipeline_timing import TimingPublisher

    timing = TimingPublisher(enabled=rospy.get_param('~log_timing_topic', True))
    ...
    with timing.measure('sam2_track'):
        mask = track(frame)            # whatever the stage does
    # or, when you already have the number:
    timing.publish('fm_generation', dt_ms)

The decode script reads every ``/pipeline/timing/*`` topic from the bag and
turns it into a JSON summary plus a thesis-styled boxplot. End-to-end latency
(camera frame -> controller command) needs no publisher here; it is rebuilt
offline from the recorded message ``header.stamp`` timestamps.
"""

import time
from contextlib import contextmanager


class TimingPublisher:
    """Publishes stage durations [ms] on per-stage topics under ``namespace``.

    Publishers are created lazily on first use (one per stage name) so a node
    only advertises the stages it actually runs. Fully inert when ``enabled`` is
    False, and degrades to a no-op if rospy is unavailable, so importing it can
    never break a node."""

    def __init__(self, namespace="/pipeline/timing", enabled=True, queue_size=50):
        self.namespace = namespace.rstrip("/")
        self.enabled = bool(enabled)
        self._queue_size = queue_size
        self._pubs = {}
        self._rospy = None
        self._Float64 = None
        if self.enabled:
            try:
                import rospy
                from std_msgs.msg import Float64
                self._rospy = rospy
                self._Float64 = Float64
            except Exception:
                self.enabled = False

    def _pub_for(self, stage):
        pub = self._pubs.get(stage)
        if pub is None:
            topic = f"{self.namespace}/{stage}"
            pub = self._rospy.Publisher(topic, self._Float64,
                                        queue_size=self._queue_size)
            self._pubs[stage] = pub
        return pub

    def publish(self, stage, ms):
        """Publish a measured duration ``ms`` (float, milliseconds) for ``stage``."""
        if not self.enabled:
            return
        try:
            self._pub_for(stage).publish(self._Float64(float(ms)))
        except Exception:
            pass  # timing must never take a node down

    @contextmanager
    def measure(self, stage):
        """Context manager: times the wrapped block and publishes it for ``stage``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.publish(stage, (time.perf_counter() - t0) * 1000.0)
