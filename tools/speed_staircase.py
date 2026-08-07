#!/usr/bin/env python3
"""Open-loop speed staircase: measure the drivetrain's breakaway and dropout speeds.

Publishes a zero-steering AckermannDriveStamped ladder on the controller's drive
topic and records nothing itself -- run `ros2 bag record` alongside it. The car
must be on stands: this commands wheel motion with no path and no localization.

The ladder climbs from rest so the *ascending* steps measure breakaway (the
lowest command that starts a stopped motor), then descends so the *descending*
steps measure dropout (the lowest command that keeps an already-rolling motor
turning). The two differ -- that hysteresis is what a release edge has to
straddle -- and only a ladder driven in both directions separates them.

Usage:
    ros2 run ... or: python3 speed_staircase.py [--reverse] [--dwell 1.5]

The operator must hold the deadman throughout; the command gate drops every
message otherwise and the run records a ladder the actuator never saw.
"""
import argparse
import sys

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy

LADDER = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.25, 0.30,
          0.35, 0.40]


class SpeedStaircase(Node):
    """Step the commanded speed up and back down, announcing each step."""

    def __init__(self, topic, dwell, rate, sign, settle, joy_topic, deadman):
        super().__init__('speed_staircase')
        self.pub = self.create_publisher(AckermannDriveStamped, topic,
                                         QoSProfile(depth=10))
        self.dwell = dwell
        self.sign = sign
        # The command gate drops everything unless the deadman is held, so a
        # step that elapses while it is released is recorded as a step the
        # actuator never saw. Gate the ladder clock on the button itself
        # rather than on the operator and the script starting together.
        self.deadman = deadman
        self.held = deadman < 0
        self.announced_wait = False
        if deadman >= 0:
            self.create_subscription(Joy, joy_topic, self._joy_cb,
                                     QoSProfile(depth=10))
        # zero at both ends and in the middle: the descending half must start
        # from a rolling motor, and the ascending half from a stopped one.
        self.plan = ([(0.0, settle)]
                     + [(v, dwell) for v in LADDER]
                     + [(v, dwell) for v in reversed(LADDER[:-1])]
                     + [(0.0, settle)])
        self.step = 0
        self.elapsed = 0.0
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'staircase on {topic}: {len(self.plan)} steps, '
            f'{sum(d for _, d in self.plan):.0f} s total, sign {sign:+d}. '
            'HOLD THE DEADMAN.')

    def _joy_cb(self, msg):
        held = (len(msg.buttons) > self.deadman
                and bool(msg.buttons[self.deadman]))
        if held != self.held:
            self.get_logger().info(
                f'deadman {"HELD - ladder running" if held else "RELEASED - ladder paused"}')
        self.held = held

    def _tick(self):
        if self.step >= len(self.plan):
            self._publish(0.0)
            self.get_logger().info('staircase complete')
            raise SystemExit(0)

        if not self.held:
            # hold this step's clock and command zero: resuming mid-step from a
            # rolling wheel would contaminate an ascending (breakaway) reading.
            if not self.announced_wait:
                self.get_logger().info('waiting for the deadman...')
                self.announced_wait = True
            self._publish(0.0)
            self.elapsed = 0.0
            return
        self.announced_wait = False

        speed, dwell = self.plan[self.step]
        if self.elapsed == 0.0:
            phase = 'up' if self.step <= len(LADDER) else 'down'
            self.get_logger().info(
                f'step {self.step:2d}/{len(self.plan) - 1} [{phase}] '
                f'{self.sign * speed:+.3f} m/s for {dwell:.1f} s')
        self._publish(self.sign * speed)
        self.elapsed += self.dt
        if self.elapsed >= dwell:
            self.step += 1
            self.elapsed = 0.0

    def _publish(self, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = 0.0
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/gosling1/drive')
    ap.add_argument('--dwell', type=float, default=1.5,
                    help='seconds per ladder step')
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds of zero command at each end')
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--reverse', action='store_true',
                    help='drive the ladder negative (reverse breakaway)')
    ap.add_argument('--joy-topic', default='/gosling1/joy')
    ap.add_argument('--deadman', type=int, default=10,
                    help='Joy button index of the deadman; -1 to not wait for it')
    args = ap.parse_args()

    rclpy.init()
    node = SpeedStaircase(args.topic, args.dwell, args.rate,
                          -1 if args.reverse else 1, args.settle,
                          args.joy_topic, args.deadman)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # leave the actuator with an explicit zero, not a stale last command
        for _ in range(10):
            node._publish(0.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
