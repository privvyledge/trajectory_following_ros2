"""Simulator steering slew limit: measured from the applied steering, not the last command.

Exercises ``BaseSimulator._slew_steering`` through a stand-in carrying only the
attributes it reads, so the limit is testable without ROS.
"""
import pytest

from trajectory_following_ros2.simulator.base_simulator import BaseSimulator

RATE = 3.2   # rad/s
DT = 0.05    # s
STEP = RATE * DT


class _Stub:
    def __init__(self, rate=RATE, dt=DT):
        self.MAX_STEER_RATE = rate
        self.sample_time = dt
        self._delta_slewed = 0.0

    slew = BaseSimulator._slew_steering


def test_held_command_slews_one_step_per_call_without_a_jump():
    """Commands 0, 1, 1, ... must ramp; the old clamp jumped to 1.0 on the third."""
    s = _Stub()
    applied = [s.slew(c) for c in (0.0, 1.0, 1.0, 1.0)]
    assert applied[0] == 0.0
    assert applied[1:] == pytest.approx([STEP, 2 * STEP, 3 * STEP])


def test_several_sim_steps_per_command_keep_converging():
    """With the simulator stepping 4x per command, the steering never stalls."""
    s = _Stub()
    applied = [s.slew(0.5) for _ in range(4)]
    assert applied == pytest.approx([STEP, 2 * STEP, 3 * STEP, 0.5])
    assert s.slew(0.5) == pytest.approx(0.5)


def test_reverse_direction_is_limited_too():
    s = _Stub()
    s._delta_slewed = 0.3
    assert s.slew(-0.3) == pytest.approx(0.3 - STEP)
    assert s.slew(-0.3) == pytest.approx(0.3 - 2 * STEP)


def test_command_within_one_step_is_reached_exactly():
    s = _Stub()
    assert s.slew(0.1) == 0.1
    assert s.slew(0.1) == 0.1
