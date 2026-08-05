"""Replay captured ``derived_object_msgs/ObjectArray`` feeds from JSON snapshots.

``fake_obstacle_publisher`` emits spheres at hand-typed coordinates, which is enough
for bring-up but cannot stand in for a recorded field feed: the failures this node
exists to reproduce are driven by **box geometry with orientation** (the planar
diagonal sets the keep-out radius, and the yaw sets where box sub-discs land), by the
**raw object ids** (the controller's go-around side hysteresis is keyed by id), and by
the **object count** (obstacle slots are scarce and ranking decides who gets one).
Replaying the capture verbatim is the only way an offline run sees the same scene the
vehicle saw.

Each JSON file becomes one **source topic**, mirroring the live topology
(``/carla/static_obstacles`` + ``/carla/objects``) so the feeds pass through
``obstacle_aggregator`` exactly as they do on the vehicle — same merge, same ego
gate, same id namespacing, same side-hysteresis keys. Publishing one pre-merged topic
instead would quietly change all four.

Snapshot schema (as written by the capture scripts)::

    {"topic": ..., "frame_id": "map", "stamp": {"sec": ..., "nanosec": ...},
     "n_objects": N,
     "objects": [{"id": int, "classification": int, "position": [x, y, z],
                  "orientation": [qx, qy, qz, qw], "shape_type": int,
                  "dimensions": [...], "twist_linear": [x, y, z],
                  "twist_angular": [x, y, z]}, ...]}

Obstacles are **not** TF-transformed (same contract as the aggregator and the
controller): the snapshot frame must already be the controller's ``global_frame``.
``closed_loop_sim`` publishes an identity ``map -> odom`` TF, so a ``map``-frame
capture passes straight through.

Parameters (all runtime-declared):

- ``snapshot_files`` (string array) — capture JSONs, one per source.
- ``output_topics`` (string array, parallel) — topic per snapshot. Short entries fall
  back to ``replay/obstacles_<stem>``.
- ``publish_qos`` (string array, parallel) — ``'volatile'`` (default) or
  ``'transient_local'``.
- ``exclude_ids`` (integer array, default ``[-1]`` = exclude nothing) — ids to drop
  from every
  snapshot. A raw CARLA ``/carla/objects`` capture contains the **ego vehicle** itself
  (id 197 in the Town01 capture, sitting exactly on the recorded stall pose); the live
  bridge filtered it out, and replaying it would plant a phantom keep-out on the car.
- ``frame_id`` (str, default ``''``) — override the snapshot's own frame.
- ``publish_rate`` (double Hz, default ``10.0``) — one timer drives every source.
- ``ego_gate_radius`` (double m, default ``0.0`` = off) with ``odom_topic`` — an
  optional publish-time gate, the fallback for running without the aggregator. Left
  off in the standard topology, where the aggregator owns the gate.

Example::

    ros2 run trajectory_following_ros2 obstacle_replay_publisher --ros-args \
        -p snapshot_files:="['data/carla_static_obstacles_town01.json',
                             'data/carla_dynamic_objects_town01.json']" \
        -p output_topics:="['replay/static_obstacles', 'replay/dynamic_objects']"
"""
import json
import math
import os
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry

from derived_object_msgs.msg import Object, ObjectArray


def load_snapshot(path: str) -> Tuple[str, List[dict]]:
    """Return ``(frame_id, objects)`` from one capture JSON.

    Raises on a malformed file rather than publishing a partial scene: an obstacle
    feed that silently drops objects reads downstream as a clean run (the same
    failure mode the controller's silent-feed watchdog exists to catch).
    """
    with open(path, 'r') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or 'objects' not in payload:
        raise ValueError(f'{path}: expected a dict with an "objects" list.')
    objects = payload['objects']
    if not isinstance(objects, list):
        raise ValueError(f'{path}: "objects" must be a list.')
    declared = payload.get('n_objects')
    if declared is not None and int(declared) != len(objects):
        raise ValueError(
            f'{path}: n_objects={declared} but {len(objects)} objects present.')
    return str(payload.get('frame_id', '')), objects


def build_object(record: dict) -> Object:
    """Rebuild one ``Object`` from a snapshot record, verbatim.

    Every field the controller reads is restored: id (side hysteresis key), shape type
    and dimensions (keep-out radius), orientation (box sub-disc axis) and twist
    (horizon propagation under ``predict_obstacle_motion``).
    """
    obj = Object()
    obj.id = int(record['id'])
    position = record.get('position') or [0.0, 0.0, 0.0]
    obj.pose.position.x = float(position[0])
    obj.pose.position.y = float(position[1])
    obj.pose.position.z = float(position[2]) if len(position) > 2 else 0.0

    orientation = record.get('orientation') or [0.0, 0.0, 0.0, 1.0]
    obj.pose.orientation.x = float(orientation[0])
    obj.pose.orientation.y = float(orientation[1])
    obj.pose.orientation.z = float(orientation[2])
    obj.pose.orientation.w = float(orientation[3])

    twist = record.get('twist_linear') or [0.0, 0.0, 0.0]
    obj.twist.linear.x = float(twist[0])
    obj.twist.linear.y = float(twist[1])
    obj.twist.linear.z = float(twist[2]) if len(twist) > 2 else 0.0
    angular = record.get('twist_angular') or [0.0, 0.0, 0.0]
    obj.twist.angular.x = float(angular[0])
    obj.twist.angular.y = float(angular[1])
    obj.twist.angular.z = float(angular[2]) if len(angular) > 2 else 0.0

    obj.shape.type = int(record.get('shape_type', obj.shape.BOX))
    obj.shape.dimensions = [float(value) for value in record.get('dimensions', [])]
    obj.classification = int(record.get('classification', 0))
    return obj


def snapshot_yaw(record: dict) -> float:
    """Planar yaw of a snapshot record, matching the controller's own extraction."""
    qx, qy, qz, qw = (record.get('orientation') or [0.0, 0.0, 0.0, 1.0])[:4]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _qos(name: str, depth: int = 1) -> QoSProfile:
    durability = (DurabilityPolicy.TRANSIENT_LOCAL
                  if str(name).strip().lower() == 'transient_local'
                  else DurabilityPolicy.VOLATILE)
    return QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST,
                      reliability=ReliabilityPolicy.RELIABLE, durability=durability)


def _default_topic(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return f'replay/obstacles_{stem}'


class ObstacleReplayPublisher(Node):
    """Publish one captured ``ObjectArray`` per snapshot file at a fixed rate."""

    def __init__(self):
        super().__init__('obstacle_replay_publisher')

        self.declare_parameter('snapshot_files', [''])
        self.declare_parameter('output_topics', [''])
        self.declare_parameter('publish_qos', [''])
        # rclpy types an empty default as BYTE_ARRAY and a descriptor cannot override
        # that, so the "exclude nothing" default is a sentinel instead: object ids are
        # unsigned, so -1 can never match a real one.
        self.declare_parameter('exclude_ids', [-1])
        self.declare_parameter('frame_id', '')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('ego_gate_radius', 0.0)
        self.declare_parameter('odom_topic', 'odometry/local')

        files = [f for f in self.get_parameter('snapshot_files').value if f]
        if not files:
            raise ValueError('snapshot_files must name at least one capture JSON.')
        topics = list(self.get_parameter('output_topics').value or [])
        topics += [''] * (len(files) - len(topics))
        qos_names = list(self.get_parameter('publish_qos').value or [])
        qos_names += [''] * (len(files) - len(qos_names))

        self._frame_override = str(self.get_parameter('frame_id').value)
        self._gate_radius = max(0.0, float(self.get_parameter('ego_gate_radius').value))
        self._ego_xy: Optional[Tuple[float, float]] = None

        # (topic, frame_id, [(x, y, Object), ...], publisher). The messages are built
        # once at startup: the scene is static per snapshot, so rebuilding it every
        # tick would burn CPU on a 1229-object feed for an identical result.
        excluded = {int(value)
                    for value in (self.get_parameter('exclude_ids').value or [])}
        self._sources = []
        for path, topic, qos_name in zip(files, topics, qos_names):
            resolved = path if os.path.isabs(path) else os.path.abspath(path)
            frame_id, records = load_snapshot(resolved)
            kept = [r for r in records if int(r['id']) not in excluded]
            dropped = len(records) - len(kept)
            entries = [(float(r['position'][0]), float(r['position'][1]),
                        build_object(r)) for r in kept]
            topic = topic or _default_topic(resolved)
            publisher = self.create_publisher(ObjectArray, topic, _qos(qos_name))
            self._sources.append(
                (topic, self._frame_override or frame_id, entries, publisher))
            self.get_logger().info(
                f'Replaying {len(entries)} object(s) from {resolved} on {topic} '
                f'(frame={self._frame_override or frame_id}, '
                f'qos={qos_name or "volatile"}; {dropped} excluded by id).')

        self._odom_sub = None
        odom_topic = str(self.get_parameter('odom_topic').value)
        if self._gate_radius > 0.0 and odom_topic:
            self._odom_sub = self.create_subscription(
                Odometry, odom_topic, self._odom_callback, 1)
            self.get_logger().info(
                f'Publish-time ego gate: {self._gate_radius:.1f} m around {odom_topic}. '
                'Prefer the aggregator gate when the aggregator is in the loop.')

        rate = float(self.get_parameter('publish_rate').value)
        if rate <= 0.0:
            raise ValueError('publish_rate must be > 0.')
        self._timer = self.create_timer(1.0 / rate, self._publish)

    def _odom_callback(self, msg: Odometry) -> None:
        self._ego_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _gated(self, entries: Sequence) -> List[Object]:
        if self._gate_radius <= 0.0 or self._ego_xy is None:
            return [obj for _x, _y, obj in entries]
        ex, ey = self._ego_xy
        gate_sq = self._gate_radius * self._gate_radius
        return [obj for x, y, obj in entries
                if (x - ex) ** 2 + (y - ey) ** 2 <= gate_sq]

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for topic, frame_id, entries, publisher in self._sources:
            msg = ObjectArray()
            msg.header.stamp = stamp
            msg.header.frame_id = frame_id
            msg.objects = self._gated(entries)
            # The per-object header is what a live CARLA bridge stamps; the controller
            # ignores it, but a bag of this feed should not look different from a bag
            # of the real one.
            for obj in msg.objects:
                obj.header = msg.header
            publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleReplayPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
