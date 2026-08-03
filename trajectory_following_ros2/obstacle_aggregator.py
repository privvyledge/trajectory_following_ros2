"""Merge several obstacle feeds into one gated ``derived_object_msgs/ObjectArray``.

Two jobs, both of which the controller must not do itself:

1. **Merge N sources into one.** ``BaseTrajectoryTracker`` subscribes to exactly one
   obstacle topic and replaces its cache wholesale on every message, so two feeds into
   one controller callback would erase each other. Object ids are namespaced per source
   (``source_index * id_namespace_stride + id``) because the controller's go-around side
   hysteresis is keyed by id, and two independent publishers both emit low integers.
   The merged feed is republished on a fixed-rate timer with VOLATILE QoS, so a latched
   static source re-emits every cycle and the controller's silent-feed watchdog keeps
   working with no subscriber-side special case.

2. **Gate the feed to what can reach the vehicle.** rclpy deserializes every object of
   every message *in the executor, before the callback runs*: a 1235-object CARLA feed
   measured 137 ms of CPU per message, which saturated the controller process's GIL and
   dilated its control tick roughly 5x. The controller's own ``obstacle_ingest_radius``
   cannot help — it runs after deserialization. Running this node in its own process
   moves that cost off the control loop; ``ego_gate_radius`` then shrinks what the
   controller has to deserialize at all (1235 -> ~100 objects on Town01).

Objects are **not** TF-transformed here, exactly as in the controller: every source must
already publish in the controller's ``global_frame``, and the merged output carries the
frame of the newest contributing source. **Future work, and the first thing to add when a
second source is a sensor-frame clusterer:** look each source's ``header.frame_id`` up
against ``global_frame`` through TF and transform on ingest. Until then a sensor-frame
source silently becomes a phantom keep-out near the origin — the same failure the
controller has (see the GOTCHAS entry), just moved one node upstream. Mixing frames is
undetectable downstream because the merged message can only declare one frame.

Example::

    ros2 run trajectory_following_ros2 obstacle_aggregator --ros-args \
        -p input_topics:="['/carla/merged_obstacles', '/carla/static_obstacles']" \
        -p input_qos:="['volatile', 'transient_local']" \
        -p ego_gate_radius:=50.0 -p odom_topic:=/carla/ego_vehicle/odometry
"""
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry

from derived_object_msgs.msg import ObjectArray


class ObstacleAggregatorState:
    """Per-source cache, ego gate, id namespacing and staleness eviction.

    Deliberately ROS-free so it can be unit-tested as a pure state machine (the same
    treatment ``ProgressWatchdog`` gets). ``payload`` is opaque — the ROS layer passes
    the deserialized ``Object`` through it.
    """

    def __init__(self, num_sources: int, source_timeout: float = 1.0,
                 id_namespace_stride: int = 10000, gate_radius: float = 0.0,
                 expiring: Optional[Sequence[bool]] = None):
        if num_sources < 1:
            raise ValueError('num_sources must be >= 1')
        self.num_sources = int(num_sources)
        self.source_timeout = float(source_timeout)
        # A latched source publishes once and never again by design, so ageing it out
        # would delete a static map a second after it arrived. Per-source, not global,
        # because a live feed alongside it still must expire when its publisher dies.
        self.expiring = ([True] * self.num_sources if expiring is None
                         else [bool(flag) for flag in expiring])
        if len(self.expiring) != self.num_sources:
            raise ValueError('expiring must have one entry per source')
        self.id_namespace_stride = int(id_namespace_stride)
        self.gate_radius = max(0.0, float(gate_radius))
        # Per source: (last_update_monotonic, [(namespaced_id, x, y, payload), ...]).
        self._cache: List[Tuple[Optional[float], list]] = [
            (None, []) for _ in range(self.num_sources)]
        self._live = [False] * self.num_sources
        # None until the first odometry: the gate cannot know where the ego is, so it
        # passes everything through rather than silently dropping a real obstacle.
        self._ego_xy: Optional[Tuple[float, float]] = None

    def set_ego(self, x: float, y: float) -> None:
        """Update the gate centre. A tuple rebind is atomic, so no lock is needed."""
        self._ego_xy = (float(x), float(y))

    @property
    def ego_known(self) -> bool:
        return self._ego_xy is not None

    def namespaced_id(self, source_index: int, obj_id: int) -> int:
        return int(source_index) * self.id_namespace_stride + int(obj_id)

    def ingest(self, source_index: int, items: Sequence, now: float) -> list:
        """Cache one message from ``source_index``; return the stored records.

        ``items`` is a sequence of ``(obj_id, x, y, payload)``; the returned records are
        ``(namespaced_id, x, y, payload)`` in input order. Nothing is gated here — the
        gate is applied at ``collect`` so a latched source that publishes once is still
        re-evaluated against the ego pose as the vehicle drives.
        """
        if not 0 <= source_index < self.num_sources:
            raise IndexError(f'source_index {source_index} out of range')
        records = [(self.namespaced_id(source_index, obj_id), x, y, payload)
                   for obj_id, x, y, payload in items]
        self._cache[source_index] = (float(now), records)
        self._live[source_index] = True
        return records

    def collect(self, now: float) -> Tuple[list, List[int]]:
        """Return ``(gated_records, newly_evicted_source_indices)``.

        A source is evicted once ``source_timeout`` has passed without a message, so a
        publisher that dies stops contributing stale keep-outs. ``source_timeout <= 0``
        disables eviction globally; a source marked non-expiring (a latched static map)
        is exempt individually.
        """
        gate_xy = self._ego_xy if self.gate_radius > 0.0 else None
        gate_sq = self.gate_radius * self.gate_radius
        records = []
        evicted = []
        for index, (stamp, items) in enumerate(self._cache):
            if stamp is None:
                continue
            if (self.expiring[index] and self.source_timeout > 0.0
                    and (now - stamp) > self.source_timeout):
                if self._live[index]:
                    evicted.append(index)
                    self._live[index] = False
                self._cache[index] = (stamp, [])
                continue
            if gate_xy is None:
                records.extend(items)
                continue
            for record in items:
                dx = record[1] - gate_xy[0]
                dy = record[2] - gate_xy[1]
                if dx * dx + dy * dy <= gate_sq:
                    records.append(record)
        return records, evicted

    def source_counts(self) -> List[int]:
        """Cached object count per source (pre-gate), for diagnostics."""
        return [len(items) for _, items in self._cache]


def _qos_from_name(name: str, depth: int = 1) -> QoSProfile:
    """Build a subscription QoS from a ``'volatile'`` / ``'transient_local'`` name."""
    durability = (DurabilityPolicy.TRANSIENT_LOCAL
                  if str(name).strip().lower() == 'transient_local'
                  else DurabilityPolicy.VOLATILE)
    return QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST,
                      reliability=ReliabilityPolicy.RELIABLE, durability=durability)


class ObstacleAggregatorNode(Node):
    """Subscribe to N obstacle feeds, gate by ego distance, republish one merged feed."""

    def __init__(self):
        super().__init__('obstacle_aggregator')

        self.declare_parameter('input_topics', ['fake_obstacles/object_array'])
        self.declare_parameter('input_qos', [''])
        self.declare_parameter('output_topic', 'obstacles/object_array')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('source_timeout', 1.0)
        self.declare_parameter('id_namespace_stride', 10000)
        self.declare_parameter('ego_gate_radius', 0.0)
        self.declare_parameter('odom_topic', 'odometry/local')
        self.declare_parameter('frame_id', '')

        topics = [t for t in self.get_parameter('input_topics').value if t]
        if not topics:
            raise ValueError('input_topics must contain at least one topic name.')
        qos_names = list(self.get_parameter('input_qos').value or [])
        if len(qos_names) < len(topics):
            qos_names += [''] * (len(topics) - len(qos_names))

        self._topics = topics
        self._frame_id_override = str(self.get_parameter('frame_id').value)
        self._gate_radius = max(0.0, float(self.get_parameter('ego_gate_radius').value))
        stride = int(self.get_parameter('id_namespace_stride').value)
        # A transient_local source is latched static content: it publishes once, so it
        # must not be aged out. Everything else expires when its publisher goes quiet.
        expiring = [str(name).strip().lower() != 'transient_local'
                    for name in qos_names[:len(topics)]]
        self._state = ObstacleAggregatorState(
            num_sources=len(topics),
            source_timeout=float(self.get_parameter('source_timeout').value),
            id_namespace_stride=stride,
            gate_radius=self._gate_radius,
            expiring=expiring)
        self._stride = stride
        self._latest_frame_id = ''

        group = ReentrantCallbackGroup()
        self._subs = []
        for index, (topic, qos_name) in enumerate(zip(topics, qos_names)):
            self._subs.append(self.create_subscription(
                ObjectArray, topic,
                lambda msg, i=index: self._obstacle_callback(i, msg),
                _qos_from_name(qos_name), callback_group=group))

        odom_topic = str(self.get_parameter('odom_topic').value)
        self._odom_sub = None
        if self._gate_radius > 0.0 and odom_topic:
            self._odom_sub = self.create_subscription(
                Odometry, odom_topic, self._odom_callback, 1, callback_group=group)

        self._pub = self.create_publisher(
            ObjectArray, str(self.get_parameter('output_topic').value),
            _qos_from_name('volatile'))

        rate = float(self.get_parameter('publish_rate').value)
        if rate <= 0.0:
            raise ValueError('publish_rate must be > 0.')
        self._timer = self.create_timer(1.0 / rate, self._publish,
                                        callback_group=group)

        gate_note = (f'{self._gate_radius:.1f} m around {odom_topic}'
                     if self._gate_radius > 0.0 else 'disabled')
        self.get_logger().info(
            f'Aggregating {len(topics)} obstacle source(s) {topics} -> '
            f'{self._pub.topic_name} at {rate:.1f} Hz; ego gate: {gate_note}; '
            f'id stride {stride}.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry) -> None:
        self._state.set_ego(msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _obstacle_callback(self, source_index: int, msg: ObjectArray) -> None:
        objects = msg.objects
        items = [(obj.id, obj.pose.position.x, obj.pose.position.y, obj)
                 for obj in objects]
        records = self._state.ingest(source_index, items, self._now())
        # Stamp the namespaced id once, here, so republishing the same cached message on
        # later ticks cannot apply the stride twice.
        overflow = False
        for namespaced_id, _x, _y, obj in records:
            overflow = overflow or (namespaced_id - source_index * self._stride
                                    >= self._stride)
            obj.id = int(namespaced_id)
        if overflow:
            self.get_logger().warn(
                f'{self._topics[source_index]} reports ids >= id_namespace_stride '
                f'({self._stride}); namespaces overlap and side hysteresis will be '
                'applied to the wrong obstacle. Raise id_namespace_stride.',
                throttle_duration_sec=10.0)
        if msg.header.frame_id:
            self._latest_frame_id = msg.header.frame_id
        if self._gate_radius > 0.0 and not self._state.ego_known:
            self.get_logger().warn(
                f'ego_gate_radius={self._gate_radius:.1f} m is set but no odometry has '
                'arrived yet; forwarding the feed ungated.',
                throttle_duration_sec=5.0)

    def _publish(self) -> None:
        records, evicted = self._state.collect(self._now())
        for index in evicted:
            self.get_logger().warn(
                f'Obstacle source {self._topics[index]} went silent for more than '
                'source_timeout; dropping its objects.')

        msg = ObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id_override or self._latest_frame_id
        msg.objects = [payload for _id, _x, _y, payload in records]
        self._pub.publish(msg)

        self.get_logger().info(
            f'obstacles cached per source {self._state.source_counts()} -> '
            f'{len(msg.objects)} published', throttle_duration_sec=10.0)

    def _now(self) -> float:
        """Monotonic-ish seconds for staleness.

        Uses the node clock so a sim-time run (``use_sim_time``) ages sources on the same
        clock its messages are stamped with, instead of on wall time that may run many
        times faster than the simulation.
        """
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAggregatorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
