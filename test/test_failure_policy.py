"""Pure policy tests for solver-failure command handling.

The tracker is allocated with ``__new__`` so these tests exercise the policy without
constructing a ROS node. The methods under test read only the attributes populated here.
"""
import csv
import io
import math

import numpy as np
import pytest

from trajectory_following_ros2.backends.base_solver import SolverResult
from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker
from trajectory_following_ros2.utils.Trajectory import Trajectory


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
    tracker._brake_speed_bound = float('inf')

    tracker.acc_cmd = 0.5
    tracker.delta_cmd = 0.1
    tracker.velocity_cmd = 0.8
    tracker.MAX_DECEL = -3.0
    tracker.ENVELOPE_DECEL = 0.0  # 0 = follow |max_decel|
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


def _goal_tracker(**overrides):
    trajectory = Trajectory(goal_tolerance=0.2, stop_speed=0.1)
    values = {
        'trajectory': trajectory,
        'distance_tolerance': 0.2,
        'cumulative_distance': 1.0,
        'final_goal': np.array([0.0, 0.0, 0.0]),
        'path': np.zeros((10, 2)),
        'current_idx': 9,
    }
    values.update(overrides)
    tracker = _tracker(**values)
    return tracker


def test_goal_completion_requires_near_goal_and_stopped():
    tracker = _goal_tracker()

    assert tracker._goal_completion_ready(0.1, 0.0, 0.05)
    assert not tracker._goal_completion_ready(0.1, 0.0, 0.5)
    assert not tracker._goal_completion_ready(0.3, 0.0, 0.0)


def test_goal_completion_requires_progress_grace():
    tracker = _goal_tracker(cumulative_distance=0.5)

    assert not tracker._goal_completion_ready(0.0, 0.0, 0.0)


def _logging_tracker():
    """A tracker whose stats CSV is captured in memory."""
    buffer = io.StringIO()
    tracker = _tracker()
    tracker._solver_log_fh = buffer
    tracker._solver_log_writer = csv.writer(buffer)
    tracker._last_solver_log_ms = 0.0
    tracker.current_idx = 42
    tracker._keepout_side_hints = {}
    return tracker, buffer


def _row(buffer):
    return next(csv.reader(io.StringIO(buffer.getvalue())))


SAFETY_STOP = 15      # column indices into the stats row
SAFETY_REASON = 16
APPLIED = slice(21, 24)
PROPOSED = slice(6, 9)


def test_stats_row_separates_the_proposed_command_from_the_applied_one():
    """A safety stop publishes zero while the solver proposed a live command; the row
    must show both, or command analysis of an intervention run is meaningless."""
    tracker, buffer = _logging_tracker()
    result = SolverResult(accel_cmd=1.0, steering_cmd=0.2, velocity_cmd=1.5,
                          is_optimal=True, status='0')

    tracker._log_solver_stats(
        result, selected=[], safety={'obstacle_id': 7, 'physical_clearance': 0.09,
                                     'closing_speed': 1.4, 'stopping_room': 0.5},
        safety_reason='braking_envelope', applied=(0.0, 0.0, 0.0))

    row = _row(buffer)
    assert [float(v) for v in row[PROPOSED]] == [1.0, 0.2, 1.5]
    assert [float(v) for v in row[APPLIED]] == [0.0, 0.0, 0.0]
    assert row[SAFETY_STOP] == '1'
    assert row[SAFETY_REASON] == 'braking_envelope'
    assert float(row[17]) == 7        # safety_obstacle_id
    assert float(row[18]) == pytest.approx(0.09)   # physical_clearance
    assert float(row[19]) == pytest.approx(1.4)    # closing_speed
    assert float(row[20]) == pytest.approx(0.5)    # stopping_room


def test_stats_row_marks_an_uneventful_tick_as_not_stopped():
    tracker, buffer = _logging_tracker()
    result = SolverResult(accel_cmd=1.0, steering_cmd=0.2, velocity_cmd=1.5,
                          is_optimal=True, status='0')

    tracker._log_solver_stats(result, selected=[], safety={'obstacle_id': -1},
                              applied=(1.0, 0.2, 1.5))

    row = _row(buffer)
    assert row[SAFETY_STOP] == '0'
    assert row[SAFETY_REASON] == ''
    assert [float(v) for v in row[APPLIED]] == [1.0, 0.2, 1.5]


def test_stats_header_and_row_have_matching_widths(tmp_path):
    """The header is written at startup and the rows during the run; a column added to
    one and not the other silently misaligns every downstream analysis."""
    path = tmp_path / 'stats.csv'
    tracker = _tracker()
    tracker._solver = object()
    tracker.current_idx = 0
    tracker._keepout_side_hints = {}
    tracker._last_solver_log_ms = 0.0
    tracker.get_parameter = lambda name: type('P', (), {'value': str(path)})()
    tracker.get_logger = lambda: type('L', (), {'info': lambda *a, **k: None,
                                                'warn': lambda *a, **k: None})()

    tracker._setup_solver_log()
    tracker._log_solver_stats(SolverResult(), selected=[], safety={},
                              safety_reason='', applied=(0.0, 0.0, 0.0))
    tracker._solver_log_fh.close()

    header, row = list(csv.reader(path.open()))[:2]
    assert len(header) == len(row)
    for name in ('safety_stop', 'safety_reason', 'physical_clearance',
                 'closing_speed', 'stopping_room',
                 'applied_accel', 'applied_steering', 'applied_speed',
                 'avoidance_stop'):
        assert name in header
    assert header.index('safety_stop') == SAFETY_STOP
    assert header.index('applied_accel') == APPLIED.start
    # New columns are appended, never inserted: the analysis scripts index this
    # file positionally as well as by name. Assert the tail order explicitly so a
    # column added in the middle fails here rather than silently shifting them.
    assert header[-2:] == ['forward_escape_active', 'breakaway_floor']


def test_safety_brake_command_sheds_speed_and_keeps_solver_steering():
    tracker = _tracker(sample_time=0.05)
    result = SolverResult(steering_cmd=0.2, is_optimal=True)

    acc, steer, speed = tracker._safety_brake_command(result, speed=0.8)

    assert acc == pytest.approx(-3.0)
    assert steer == pytest.approx(0.2)
    assert speed == pytest.approx(0.8 - 3.0 * 0.05)


def test_safety_brake_command_sheds_speed_at_the_envelope_decel():
    """The action must brake at the rate the envelope reserved room for.

    Sizing ``stopping_room`` at 6 m/s² while the brake command sheds speed at
    3 m/s² fires later *and* stops slower — strictly worse than either value used
    consistently, and it would silently undo the envelope_decel change.
    """
    tracker = _tracker(sample_time=0.05)
    tracker.ENVELOPE_DECEL = 6.0
    result = SolverResult(steering_cmd=0.2, is_optimal=True)

    acc, _, speed = tracker._safety_brake_command(result, speed=0.8)

    assert acc == pytest.approx(-6.0)
    assert speed == pytest.approx(0.8 - 6.0 * 0.05)


def test_safety_brake_command_falls_back_to_last_steering_on_bad_iterate():
    tracker = _tracker(sample_time=0.05, delta_cmd=0.1)
    result = SolverResult(steering_cmd=5.0, is_optimal=False)

    acc, steer, speed = tracker._safety_brake_command(result, speed=0.1)

    assert steer == pytest.approx(0.1)
    assert speed == pytest.approx(0.0)


def test_safety_brake_command_brakes_a_reversing_vehicle_toward_zero():
    tracker = _tracker(sample_time=0.05)
    result = SolverResult(steering_cmd=0.0, is_optimal=True)

    acc, steer, speed = tracker._safety_brake_command(result, speed=-0.8)

    assert acc == pytest.approx(3.0)
    assert speed == pytest.approx(-0.65)


def test_safety_brake_command_is_monotone_through_actuation_lag():
    """Measured speed keeps rising through the plant's command filter right after
    the envelope fires; the brake must not track it back upward."""
    tracker = _tracker(sample_time=0.05)
    result = SolverResult(steering_cmd=0.0, is_optimal=True)

    _, _, first = tracker._safety_brake_command(result, speed=1.0)
    _, _, second = tracker._safety_brake_command(result, speed=1.3)  # lag rise

    assert first == pytest.approx(0.85)
    assert second == pytest.approx(0.70), 'bound must ratchet down, not follow speed'


def test_goal_completion_requires_route_progress():
    """A self-near route can satisfy proximity + stop mid-course after an
    obstacle shove; completion must also require the index to reach the tail."""
    tracker = _goal_tracker(path=np.zeros((100, 2)), current_idx=47)

    assert not tracker._goal_completion_ready(0.1, 0.0, 0.05)

    tracker.current_idx = 95
    assert tracker._goal_completion_ready(0.1, 0.0, 0.05)
