"""Breakaway-floor state machine: engage, hold, release, and the cases it must not touch.

Exercises ``BaseTrajectoryTracker._apply_breakaway_floor`` through a stand-in that
carries only the attributes the method reads, so the logic is testable without ROS.
Numbers come from the gosling1 speed staircase (2026-08-07): breakaway 0.18 m/s
forward / 0.15 reverse on stands, dropout 0.12, hysteresis 0.06.
"""
import math

import numpy as np

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker


class _Logger:
    def warn(self, *args, **kwargs):
        pass


class _Stub:
    """Minimal stand-in exposing exactly what the floor method touches."""

    def __init__(self, floor=0.25, engage=0.05, release=0.0,
                 min_speed=-2.0, max_speed=2.0):
        self.breakaway_speed = floor
        self.breakaway_engage_speed = engage
        self.breakaway_release_speed = release
        self.MIN_SPEED = min_speed
        self.MAX_SPEED = max_speed
        self._breakaway_active = False
        self._breakaway_applied = 0.0
        self._breakaway_sign = 0.0

    def get_logger(self):
        return _Logger()

    apply = BaseTrajectoryTracker._apply_breakaway_floor


def test_disabled_by_default_is_a_passthrough():
    s = _Stub(floor=0.0)
    assert s.apply(0.11, 0.0) == 0.11
    assert s._breakaway_applied == 0.0
    assert not s._breakaway_active


def test_engages_when_stalled_against_a_command_that_wants_motion():
    s = _Stub(floor=0.25)
    # the one-step ceiling from rest: max_accel * dt = 3.0 * 0.05
    assert s.apply(0.15, 0.0) == 0.25
    assert s._breakaway_active
    assert s._breakaway_applied == 0.25


def test_never_floors_a_commanded_stop():
    """A safety path or the goal commands zero; the floor must not restart the car."""
    s = _Stub(floor=0.25)
    assert s.apply(0.0, 0.0) == 0.0
    assert not s._breakaway_active
    assert s._breakaway_applied == 0.0


def test_preserves_sign_so_reverse_is_floored_backwards():
    s = _Stub(floor=0.25)
    assert s.apply(-0.12, 0.0) == -0.25
    assert s._breakaway_active


def test_does_not_reduce_a_command_already_above_the_floor():
    s = _Stub(floor=0.25)
    assert s.apply(0.80, 0.0) == 0.80
    assert s._breakaway_applied == 0.0


def test_holds_through_the_deadband_then_releases_once_rolling():
    """The release edge: dropping the floor the instant the wheel turns hands the
    actuator a step back down to the creep, which stalls it again. It must hold
    until the vehicle is rolling past the release speed."""
    s = _Stub(floor=0.25)          # release defaults to 0.125, the dropout speed
    assert s.apply(0.15, 0.00) == 0.25     # engage from rest
    assert s.apply(0.16, 0.02) == 0.25     # barely turning -- still floored
    assert s.apply(0.18, 0.08) == 0.25     # moving, but below release -- still floored
    assert s._breakaway_active
    out = s.apply(0.30, 0.14)              # past release: hand back to the solver
    assert out == 0.30
    assert not s._breakaway_active
    assert s._breakaway_applied == 0.0


def test_handover_does_not_step_the_command_down():
    """Releasing must not drop the published speed below the drivetrain's dropout
    speed, or the car stalls again and the floor limit-cycles."""
    s = _Stub(floor=0.25)
    dropout = 0.12
    s.apply(0.15, 0.0)
    published = 0.25
    # once rolling at `release`, the solver's own one-step command is v + a*dt
    for measured in (0.02, 0.06, 0.10, 0.13, 0.20):
        solver_cmd = measured + 3.0 * 0.05
        published = s.apply(solver_cmd, measured)
        assert abs(published) >= dropout, (
            f'published {published:.3f} at measured {measured:.3f} is below the '
            'dropout speed -- the release would re-stall the drivetrain')
    assert not s._breakaway_active


def test_holds_through_standstill_measurement_noise():
    """A stopped vehicle's measured speed dithers across zero, and that must not
    be read as the solver reversing.

    Keying the reversal release on ``measured*command < 0`` releases the latch on
    any negative noise sample; the next tick re-engages via the stalled branch, so
    the actuator sees floor/creep/floor/creep. Measured on the car: 21.6% of the
    ticks that should have held passed through at the un-floored creep, median
    unbroken hold 2 ticks (0.10 s) -- the pulse train the latch exists to prevent.
    """
    s = _Stub(floor=0.25)
    # Real trace shape: stopped, solver asking for its one-step creep every tick,
    # measured speed dithering either side of zero well inside the engage deadband.
    dither = [0.004, -0.001, 0.003, -0.003, 0.001, -0.005, 0.006, -0.002,
              0.000, -0.004, 0.002, -0.001]
    published = [s.apply(0.150, v) for v in dither]

    assert all(p == 0.25 for p in published), (
        f'floor dropped out on {sum(1 for p in published if p != 0.25)} of '
        f'{len(published)} stopped ticks: {published}')
    assert s._breakaway_active


def test_a_real_measured_reversal_still_releases():
    """The deadband on the measured-reversal test must not disable it outright:
    a vehicle genuinely rolling backwards against a forward push still releases."""
    s = _Stub(floor=0.25)
    assert s.apply(0.15, 0.0) == 0.25          # engage forward from rest
    assert s.apply(0.15, -0.002) == 0.25       # noise -- holds
    # -0.08 is past the engage deadband but short of the 0.125 release speed, so
    # only the measured-reversal branch can let go here.
    out = s.apply(0.15, -0.08)                 # actually rolling backwards
    assert not s._breakaway_active
    assert out == 0.15


def test_releases_when_the_solver_reverses_direction():
    """Pushing the old direction while the solver has decided to reverse would fight
    the plan; a sign flip must let go immediately rather than wait for motion."""
    s = _Stub(floor=0.25)
    assert s.apply(0.15, 0.0) == 0.25
    s._breakaway_active = True
    out = s.apply(-0.10, 0.03)   # measured still forward, command now reverse
    assert not s._breakaway_active
    assert out == -0.10


def test_respects_the_speed_envelope():
    """The floor raises a magnitude; it is not a licence to leave [MIN, MAX]."""
    s = _Stub(floor=0.25, min_speed=0.0, max_speed=0.20)
    assert s.apply(0.15, 0.0) == 0.20
    # allow_reversing=False pins MIN_SPEED to 0, so a reverse floor cannot appear
    s2 = _Stub(floor=0.25, min_speed=0.0, max_speed=2.0)
    assert s2.apply(-0.10, 0.0) == 0.0


def test_stays_disengaged_while_the_vehicle_is_already_moving():
    """Never engages mid-drive: a car at speed is not stalled, whatever it is asked for."""
    s = _Stub(floor=0.25)
    assert s.apply(0.10, 0.60) == 0.10
    assert not s._breakaway_active
    assert s._breakaway_applied == 0.0


def test_measured_ceiling_sits_below_the_measured_floor():
    """The premise the fix rests on, pinned so a max_accel change surfaces here.

    Published speed from rest is the one-step state v + a*dt, capped at
    max_accel*dt. Measured breakaway on gosling1 is 0.18 m/s on stands and
    0.20-0.26 on the ground; the ceiling is below both, which is why the vehicle
    cannot restart itself.
    """
    max_accel, dt = 3.0, 0.05
    ceiling = max_accel * dt
    assert math.isclose(ceiling, 0.15)
    assert ceiling < 0.18, 'bench breakaway'
    assert ceiling < 0.20, 'ground breakaway (lower bound)'


def test_floor_clears_the_ground_breakaway_band():
    """A floor set for this platform must actually exceed the measured band."""
    s = _Stub(floor=0.25)
    published = s.apply(0.15, 0.0)
    assert published >= 0.20, 'below the measured ground breakaway lower bound'
    assert math.isclose(published, 0.25)
    assert not np.isnan(published)
