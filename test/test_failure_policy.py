"""Pure policy tests for solver-failure command handling.

The tracker is allocated with ``__new__`` so these tests exercise the policy without
constructing a ROS node. The methods under test read only the attributes populated here.
"""
import math

import pytest

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker


class _Tracker(BaseTrajectoryTracker):
    def _init_solver(self):
        return None


def _tracker(**overrides):
    tracker = _Tracker.__new__(_Tracker)
    tracker.solver_failure_mode = 'hold_last'
    tracker.solver_failure_hold_count = 1
    tracker.solver_failure_hold_time = 0.1
    tracker.solver_failure_zero_on_saturation = True
    tracker._consecutive_failures = 1
    tracker._last_successful_command_monotonic = 10.0

    tracker.acc_cmd = 0.5
    tracker.delta_cmd = 0.1
    tracker.velocity_cmd = 0.8
    tracker.MAX_DECEL = -3.0
    tracker.MAX_ACCEL = 3.0
    tracker.MIN_STEER_ANGLE = -math.radians(27.0)
    tracker.MAX_STEER_ANGLE = math.radians(27.0)
    tracker.MIN_SPEED = -1.5
    tracker.MAX_SPEED = 1.5

    for name, value in overrides.items():
        setattr(tracker, name, value)
    return tracker


def test_zero_mode_stops_on_first_failure():
    hold, reason = _tracker(solver_failure_mode='zero')._failure_hold_decision(now=10.01)

    assert not hold
    assert reason == 'solver_failure_mode=zero'


@pytest.mark.parametrize('failure_count,expected', [(1, True), (2, True), (3, False)])
def test_count_only_gate(failure_count, expected):
    tracker = _tracker(
        solver_failure_hold_count=2,
        solver_failure_hold_time=0.0,
        _consecutive_failures=failure_count,
    )

    hold, _ = tracker._failure_hold_decision(now=100.0)

    assert hold is expected


@pytest.mark.parametrize('now,expected', [(10.05, True), (10.1, True), (10.100001, False)])
def test_time_only_gate(now, expected):
    tracker = _tracker(
        solver_failure_hold_count=0,
        solver_failure_hold_time=0.1,
        _consecutive_failures=100,
    )

    hold, _ = tracker._failure_hold_decision(now=now)

    assert hold is expected


def test_count_and_time_gates_must_both_pass():
    count_expired = _tracker(_consecutive_failures=2)
    time_expired = _tracker(_consecutive_failures=1)

    assert not count_expired._failure_hold_decision(now=10.01)[0]
    assert not time_expired._failure_hold_decision(now=10.11)[0]


def test_hold_without_any_gate_fails_safe_to_zero():
    tracker = _tracker(solver_failure_hold_count=0, solver_failure_hold_time=0.0)

    hold, reason = tracker._failure_hold_decision(now=10.01)

    assert not hold
    assert reason == 'no failure-hold gate is enabled'


def test_time_gate_requires_a_previously_published_success():
    tracker = _tracker(_last_successful_command_monotonic=None)

    hold, reason = tracker._failure_hold_decision(now=10.01)

    assert not hold
    assert reason == 'no successful command has been published'


@pytest.mark.parametrize(
    'command',
    [
        {'acc_cmd': 3.0},
        {'acc_cmd': -3.0},
        {'delta_cmd': math.radians(27.0)},
        {'delta_cmd': -math.radians(27.0)},
        {'velocity_cmd': 1.5},
        {'velocity_cmd': -1.5},
    ],
)
def test_saturated_command_is_never_held_by_default(command):
    tracker = _tracker(**command)

    hold, reason = tracker._failure_hold_decision(now=10.01)

    assert not hold
    assert reason == 'last good command is saturated'


def test_saturation_gate_can_be_explicitly_disabled():
    tracker = _tracker(
        acc_cmd=3.0,
        solver_failure_zero_on_saturation=False,
    )

    hold, _ = tracker._failure_hold_decision(now=10.01)

    assert hold
