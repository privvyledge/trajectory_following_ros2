#!/usr/bin/env python3
"""Phase 2 topology sanity check: replay publisher -> aggregator -> controller topic.

Holds the ego at the run-H stall pose, runs the real replay publisher and the real
``obstacle_aggregator`` as separate processes (the live topology), and counts what
lands on the controller's obstacle topic. The expected count is the 295 objects of
``data/carla_merged_obstacles_town01.json`` -- that file is the live gated feed frozen
at one ego pose, so it is a cross-check on the gate, not the source of the scene.

Usage::

    python3 phase2_topology_check.py [--domain 44] [--gate 80.0]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
# Run-H stall pose, independently corroborated by dynamic-capture object 197 (the ego
# itself, recorded where it wedged) at (-0.14, -53.75).
STALL_XY = (-0.14, -53.72)
STATIC_JSON = os.path.join(REPO, 'data', 'carla_static_obstacles_town01.json')
DYNAMIC_JSON = os.path.join(REPO, 'data', 'carla_dynamic_objects_town01.json')
MERGED_JSON = os.path.join(REPO, 'data', 'carla_merged_obstacles_town01.json')
EGO_OBJECT_ID = 197


def _env(domain):
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(domain)
    return env


def _spawn(cmd, env, log_path):
    handle = open(log_path, 'wb')
    return subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True), handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=int, default=44)
    parser.add_argument('--gate', type=float, default=80.0)
    parser.add_argument('--settle', type=float, default=12.0)
    args = parser.parse_args()

    env = _env(args.domain)
    logs = os.path.join(REPO, 'data', 'phase2_logs')
    os.makedirs(logs, exist_ok=True)

    procs = []
    try:
        publisher = [
            'ros2', 'run', 'trajectory_following_ros2', 'obstacle_replay_publisher',
            '--ros-args',
            # Dynamic FIRST: the aggregator namespaces ids as
            # source_index * stride + id, so source 0 passes ids through unchanged.
            # That keeps the F2 obstacle at its live id 208 (as run G/H logged it)
            # instead of 10208. The static capture's ids already diverge from the
            # live feed's, so nothing is lost by namespacing that source.
            '-p', f"snapshot_files:=['{DYNAMIC_JSON}','{STATIC_JSON}']",
            '-p', "output_topics:=['replay/dynamic_objects','replay/static_obstacles']",
            '-p', "publish_qos:=['volatile','volatile']",
            '-p', f'exclude_ids:=[{EGO_OBJECT_ID}]',
            '-p', 'publish_rate:=10.0',
        ]
        aggregator = [
            'ros2', 'run', 'trajectory_following_ros2', 'obstacle_aggregator',
            '--ros-args',
            '-p', "input_topics:=['replay/dynamic_objects','replay/static_obstacles']",
            '-p', "input_qos:=['volatile','volatile']",
            '-p', 'output_topic:=obstacles/object_array',
            '-p', f'ego_gate_radius:={args.gate}',
            '-p', 'odom_topic:=odometry/local',
            '-p', 'publish_rate:=10.0',
        ]
        odom_yaml = (
            '{header: {frame_id: odom}, pose: {pose: {position: '
            '{x: %r, y: %r, z: 0.0}}}}' % (STALL_XY[0], STALL_XY[1]))
        odom = ['ros2', 'topic', 'pub', '-r', '10', 'odometry/local',
                'nav_msgs/msg/Odometry', odom_yaml]
        for name, cmd in (('publisher', publisher), ('aggregator', aggregator),
                          ('odom', odom)):
            proc, handle = _spawn(cmd, env, os.path.join(logs, f'topology_{name}.log'))
            procs.append((name, proc, handle))
            time.sleep(1.0)

        time.sleep(args.settle)

        echo = subprocess.run(
            ['ros2', 'topic', 'echo', '--once', '--field', 'objects',
             'obstacles/object_array', '--csv'],
            env=env, capture_output=True, text=True, timeout=30)
        # --csv on a large array is unwieldy; count via a one-shot python subscriber
        # instead and keep the echo only as a liveness probe.
        alive = bool(echo.stdout.strip())

        count = subprocess.run(
            [sys.executable, '-c', _COUNTER], env=env, capture_output=True,
            text=True, timeout=60)
        print(count.stdout.strip() or count.stderr.strip())
        observed = json.loads(count.stdout.strip().splitlines()[-1])

        expected = len(json.load(open(MERGED_JSON))['objects'])
        print(f'\nechoed topic alive: {alive}')
        print(f'aggregator output : {observed["count"]} objects')
        print(f'merged capture    : {expected} objects')
        print(f'delta             : {observed["count"] - expected:+d}')
        ok = abs(observed['count'] - expected) <= 2
        print('TOPOLOGY CHECK:', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        for _name, proc, handle in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        time.sleep(2.0)
        for _name, proc, handle in procs:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            handle.close()


_COUNTER = '''
import json
import rclpy
from rclpy.node import Node
from derived_object_msgs.msg import ObjectArray

rclpy.init()
node = Node("phase2_topology_counter")
seen = {}


def cb(msg):
    seen["count"] = len(msg.objects)
    seen["ids"] = sorted({int(o.id) for o in msg.objects})
    seen["frame"] = msg.header.frame_id


node.create_subscription(ObjectArray, "obstacles/object_array", cb, 1)
for _ in range(200):
    rclpy.spin_once(node, timeout_sec=0.1)
    if "count" in seen:
        break
node.destroy_node()
rclpy.shutdown()
print(json.dumps({"count": seen.get("count", 0), "frame": seen.get("frame", ""),
                  "has_208": 208 in seen.get("ids", []),
                  "has_ego_197": 197 in seen.get("ids", [])}))
'''


if __name__ == '__main__':
    sys.exit(main())
