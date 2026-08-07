"""Regression tests for Twist-to-Ackermann command conversion."""
import math

import pytest

from trajectory_following_ros2.twist_to_ackermann_drive import Twist2Ackermann


def _converter(wheelbase=0.256):
    converter = Twist2Ackermann.__new__(Twist2Ackermann)
    converter.WHEELBASE = wheelbase
    return converter


@pytest.mark.parametrize(
    'speed,yaw_rate,expected_sign',
    [
        (1.0, 0.5, 1.0),
        (1.0, -0.5, -1.0),
        (-1.0, 0.5, -1.0),
        (-1.0, -0.5, 1.0),
    ],
)
def test_yaw_rate_conversion_preserves_ackermann_sign(speed, yaw_rate, expected_sign):
    converter = _converter()

    steering = converter.yaw_rate_to_steering_angle(speed, yaw_rate)

    expected = math.atan(converter.WHEELBASE * yaw_rate / speed)
    assert steering == pytest.approx(expected)
    assert math.copysign(1.0, steering) == expected_sign


@pytest.mark.parametrize('speed,yaw_rate', [(0.0, 0.5), (1.0, 0.0)])
def test_yaw_rate_conversion_returns_zero_when_motion_is_undefined(speed, yaw_rate):
    converter = _converter()

    assert converter.yaw_rate_to_steering_angle(speed, yaw_rate) == 0.0


class _Logger:
    def info(self, _message):
        pass


class _Clock:
    class _Now:
        @staticmethod
        def to_msg():
            from builtin_interfaces.msg import Time

            return Time()

    @staticmethod
    def now():
        return _Clock._Now()


class _Publisher:
    def __init__(self):
        self.message = None

    def publish(self, message):
        self.message = message


@pytest.mark.parametrize('requested,expected', [(0.4, 0.25), (-0.4, -0.25)])
def test_twist_callback_uses_configured_steering_limit(requested, expected):
    from geometry_msgs.msg import Twist

    converter = _converter()
    converter.MAX_STEERING_ANGLE = 0.25
    converter.cmd_angle_instead_rotvel = True
    converter.frame_id = 'base_link'
    converter.get_logger = lambda: _Logger()
    converter.get_clock = lambda: _Clock()
    converter.ackermann_cmd_pub = _Publisher()
    command = Twist()
    command.linear.x = 1.0
    command.angular.z = requested

    converter.twist_callback(command)

    assert converter.ackermann_cmd_pub.message.drive.steering_angle == expected
