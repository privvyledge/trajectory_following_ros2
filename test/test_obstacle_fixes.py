"""Pure-Python regression tests for shared obstacle-avoidance safety policies."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from trajectory_following_ros2.base_tracker import (
    AvoidanceStopLatch,
    BaseTrajectoryTracker,
    ProgressWatchdog,
    obstacle_shape_discs,
)
from trajectory_following_ros2.utils import trajectory_utils


class _Tracker(BaseTrajectoryTracker):
    def _init_solver(self):
        return None


def _new_tracker():
    tracker = _Tracker.__new__(_Tracker)
    # Ingestion-gate defaults: disabled, and no ego fix yet. Both are what a tracker
    # looks like before the first odometry, so the callback ingests everything.
    tracker.obstacle_ingest_radius = 0.0
    tracker._ingest_gate_xy = None
    return tracker


@pytest.mark.parametrize('dimensions', [
    (4.18, 1.99, 1.39),
    (6.27, 2.39, 2.10),
    (8.47, 2.89, 3.83),
    (0.375, 0.375, 1.86),
])
def test_box_radius_uses_only_planar_dimensions(dimensions):
    states, degenerate = obstacle_shape_discs(
        'BOX', dimensions, (0.0, 0.0, 0.0), min_radius=0.0)

    assert not degenerate
    assert len(states) == 1
    assert states[0][2] == pytest.approx(math.hypot(*dimensions[:2]) / 2.0 / 1.3)


def test_elongated_box_decomposes_along_its_yawed_axis():
    states, _ = obstacle_shape_discs(
        'BOX', (6.27, 2.39, 2.10), (10.0, 20.0, 0.0), yaw=math.pi / 2,
        decompose_boxes=True, max_discs=3, min_radius=0.3)

    half_length = (6.27 / 3.0) / 2.0
    extent = 6.27 / 2.0 - half_length
    assert len(states) == 3
    np.testing.assert_allclose([state[0] for state in states], 10.0, atol=1e-12)
    np.testing.assert_allclose(
        [state[1] for state in states], [20.0 - extent, 20.0, 20.0 + extent])
    assert [state[2] for state in states] == pytest.approx(
        [math.hypot(half_length, 2.39 / 2.0)] * 3)
    # CARLA truck keep-out with the scoped ego radius and reduced comfort margin.
    assert 1.5 + states[0][2] + 0.4 == pytest.approx(3.4875, abs=1e-3)


def test_box_at_aspect_threshold_remains_one_disc():
    states, _ = obstacle_shape_discs(
        'BOX', (3.0, 2.0, 1.0), (0.0, 0.0, 0.0),
        decompose_boxes=True, max_discs=3)

    assert len(states) == 1
    assert states[0][2] == pytest.approx(math.hypot(3.0, 2.0) / 2.0 / 1.3)


def test_zero_dimension_obstacle_uses_radius_floor_and_warns():
    warnings = []

    class Logger:
        def warn(self, message, **kwargs):
            warnings.append((message, kwargs))

    shape = SimpleNamespace(BOX=1, SPHERE=2, CYLINDER=3, type=1,
                            dimensions=[0.0, 0.0, 0.0])
    obj = SimpleNamespace(
        id=17,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)),
        shape=shape,
        twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0)))
    tracker = _new_tracker()
    tracker.decompose_obstacle_boxes = False
    tracker.obstacle_max_discs = 3
    tracker.min_obstacle_radius = 0.3
    tracker.get_logger = lambda: Logger()

    tracker._obstacle_callback(SimpleNamespace(objects=[obj]))

    assert tracker.obstacles[0]['state'] == [1.0, 2.0, 0.3]
    assert len(warnings) == 1
    assert 'Obstacle 17' in warnings[0][0]
    assert warnings[0][1]['throttle_duration_sec'] == 10.0


def test_decomposed_discs_keep_parent_id_and_velocity():
    shape = SimpleNamespace(BOX=1, SPHERE=2, CYLINDER=3, type=1,
                            dimensions=[6.27, 2.39, 2.10])
    obj = SimpleNamespace(
        id=23,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)),
        shape=shape,
        twist=SimpleNamespace(linear=SimpleNamespace(x=2.0, y=-0.5)))
    tracker = _new_tracker()
    tracker.decompose_obstacle_boxes = True
    tracker.obstacle_max_discs = 3
    tracker.min_obstacle_radius = 0.3
    tracker.get_logger = lambda: SimpleNamespace(warn=lambda *a, **k: None)

    tracker._obstacle_callback(SimpleNamespace(objects=[obj]))

    assert len(tracker.obstacles) == 3
    assert {obstacle['id'] for obstacle in tracker.obstacles} == {23}
    assert all(obstacle['velocity'] == [2.0, -0.5] for obstacle in tracker.obstacles)

    projection_tracker = _projection_tracker()
    centres = np.array([obstacle['state'][:2] for obstacle in tracker.obstacles])
    keepouts = projection_tracker._keepout_radii(tracker.obstacles)
    _, projection_radii, groups = projection_tracker._merge_overlapping_keepouts(
        centres, keepouts, obstacle_ids=[23, 23, 23])
    assert len(groups) == 1
    assert projection_radii[0] == pytest.approx(keepouts[0])


def test_progress_watchdog_pullaway_trip_release_and_intentional_stop():
    watchdog = ProgressWatchdog(timeout=5.0)
    assert not watchdog.update(0.0, 14, (0.0, 0.0), 0.0, 1.0, 3.0, 3.0)
    assert not watchdog.update(4.9, 14, (0.49, 0.0), 0.19, 1.0, 3.0, 3.0)
    assert not watchdog.update(5.0, 15, (0.51, 0.0), 0.2, 1.0, 3.0, 3.0)

    watchdog.reset()
    assert not watchdog.update(10.0, 20, (1.0, 1.0), 0.15, 1.0, 3.0, 3.0)
    assert watchdog.update(15.0, 20, (1.0, 1.0), 0.15, 1.0, 3.0, 3.0)
    assert not watchdog.update(15.1, 20, (1.6, 1.0), 0.15, 1.0, 3.0, 3.0)
    assert not watchdog.update(30.0, 20, (1.6, 1.0), 0.0, 0.0, -1.0, 3.0)
    assert not watchdog.update(40.0, 20, (1.6, 1.0), 0.0, -0.5, 1.0, 3.0)


def test_avoidance_stop_latch_hysteresis():
    latch = AvoidanceStopLatch()
    key = frozenset({9})

    assert latch.update({key: 2.1}, 2.0) == {key}
    assert latch.update({key: 1.7}, 2.0) == {key}
    assert latch.update({key: 2.1}, 2.0) == {key}
    assert latch.update({key: 1.59}, 2.0) == set()
    assert latch.update({}, 2.0) == set()


def _projection_tracker(max_offset=0.0, engagement=0.0):
    tracker = _new_tracker()
    tracker._gp = {'ego_radius': 1.5, 'safe_distance': 0.4,
                   'ego_disc_offsets': [0.0]}.__getitem__
    tracker.MAX_DECEL = -3.0
    tracker.sample_time = 0.05
    tracker.horizon = 25
    tracker.max_avoidance_offset = max_offset
    tracker.keepout_engagement_distance = engagement
    tracker._keepout_side_hints = {}
    tracker._avoidance_stop_latch = AvoidanceStopLatch()
    tracker._avoidance_stop_active = False
    tracker.x = tracker.y = tracker.yaw = 0.0
    tracker.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)
    return tracker


def _straight_reference():
    x = np.linspace(0.0, 10.0, 26)
    return np.vstack([x, np.zeros_like(x), np.full_like(x, 4.0), np.zeros_like(x)])


def test_large_detour_returns_raw_geometry_with_stopping_speed_ramp():
    tracker = _projection_tracker(max_offset=2.0)
    obstacle = {'id': 5, 'state': [5.0, 0.0, 1.5], 'velocity': [0.0, 0.0]}
    raw = _straight_reference()

    stopped = tracker._project_reference_out_of_keepouts(
        raw, [obstacle], ego_pose=(0.0, 0.0, 0.0))

    np.testing.assert_array_equal(stopped[[0, 1, 3], :], raw[[0, 1, 3], :])
    assert tracker._avoidance_stop_active
    assert np.all(np.diff(stopped[2, :]) <= 1e-12)
    first_intrusion = tracker._first_keepout_intrusion(raw, [obstacle])
    assert np.all(stopped[2, first_intrusion:] == 0.0)
    assert stopped[2, max(first_intrusion - 1, 0)] == 0.0


def test_disabled_detour_bound_is_the_existing_projection_exactly():
    tracker = _projection_tracker(max_offset=0.0)
    obstacle = {'id': 5, 'state': [5.0, 0.0, 1.5], 'velocity': [0.0, 0.0]}
    raw = _straight_reference()
    keepout = tracker._keepout_radii([obstacle])

    expected, _, _ = trajectory_utils.project_reference_out_of_keepouts(
        raw.copy(), [[5.0, 0.0]], keepout, side_hints=[0], disc_offsets=[0.0])
    actual = tracker._project_reference_out_of_keepouts(
        raw, [obstacle], ego_pose=(0.0, 0.0, 0.0))

    np.testing.assert_array_equal(actual, expected)
    assert not tracker._avoidance_stop_active


def test_engagement_ramp_removes_the_single_tick_projection_step():
    obstacle = {'id': 5, 'state': [5.0, 0.0, 1.5], 'velocity': [0.0, 0.0]}
    raw = _straight_reference()
    no_ramp = _projection_tracker()
    full = no_ramp._project_reference_out_of_keepouts(
        raw, [obstacle], ego_pose=(0.0, 0.0, 0.0))
    full_displacement = float(np.max(np.hypot(
        full[0, :] - raw[0, :], full[1, :] - raw[1, :])))

    ramped = _projection_tracker(engagement=25.0)
    offsets = []
    for vehicle_distance in np.linspace(25.0, 0.0, 51):
        projected = ramped._project_reference_out_of_keepouts(
            raw, [obstacle], ego_pose=(5.0 - vehicle_distance, 0.0, 0.0))
        offsets.append(float(np.max(np.hypot(
            projected[0, :] - raw[0, :], projected[1, :] - raw[1, :]))))

    assert max(np.diff(offsets)) < full_displacement / 5.0
    assert offsets[-1] == pytest.approx(full_displacement)


def _ingest_obj(obj_id, x, y):
    return SimpleNamespace(
        id=obj_id,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)),
        shape=SimpleNamespace(BOX=1, SPHERE=2, CYLINDER=3, type=1,
                              dimensions=[1.0, 1.0, 1.0]),
        twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0)))


def _ingest_tracker(gate, ego_xy):
    tracker = _new_tracker()
    tracker.decompose_obstacle_boxes = False
    tracker.obstacle_max_discs = 3
    tracker.min_obstacle_radius = 0.3
    tracker.get_logger = lambda: SimpleNamespace(warn=lambda *a, **k: None)
    tracker.obstacle_ingest_radius = gate
    tracker._ingest_gate_xy = ego_xy
    return tracker


def test_ingest_gate_drops_only_objects_beyond_the_radius():
    tracker = _ingest_tracker(gate=50.0, ego_xy=(0.0, 0.0))

    tracker._obstacle_callback(SimpleNamespace(objects=[
        _ingest_obj(1, 10.0, 0.0),    # well inside
        _ingest_obj(2, 0.0, 49.9),    # just inside
        _ingest_obj(3, 50.1, 0.0),    # just outside
        _ingest_obj(4, 400.0, 400.0),  # map-scale bystander
    ]))

    assert sorted(o['id'] for o in tracker.obstacles) == [1, 2]


def test_ingest_gate_disabled_keeps_every_object():
    tracker = _ingest_tracker(gate=0.0, ego_xy=(0.0, 0.0))

    tracker._obstacle_callback(SimpleNamespace(objects=[
        _ingest_obj(1, 10.0, 0.0),
        _ingest_obj(2, 400.0, 400.0),
    ]))

    assert sorted(o['id'] for o in tracker.obstacles) == [1, 2]


def test_ingest_gate_inactive_until_first_odometry():
    # No ego fix yet: dropping by distance would be guesswork, so nothing is dropped.
    tracker = _ingest_tracker(gate=50.0, ego_xy=None)

    tracker._obstacle_callback(SimpleNamespace(objects=[
        _ingest_obj(1, 400.0, 400.0),
    ]))

    assert [o['id'] for o in tracker.obstacles] == [1]


def test_ingest_yaw_matches_euler_from_quaternion():
    # The fast atan2 yaw must agree with the tf_transformations result it replaced.
    import tf_transformations

    for yaw in (0.0, 0.7, -1.9, 3.0):
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        expected = tf_transformations.euler_from_quaternion([0.0, 0.0, qz, qw])[2]
        got = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        assert got == pytest.approx(expected)


def test_required_detour_offset_is_recorded_for_the_stats_row():
    """``avoidance_stop`` says the bound was crossed but not by how much.

    "Needs 2.1 m of a 2.0 m bound" and "needs 9 m of a 2.0 m bound" call for opposite
    responses — raise the bound versus accept that the corridor is blocked — so the
    magnitude has to reach the CSV, not just the boolean.
    """
    tracker = _projection_tracker(max_offset=2.0)
    obstacle = {'id': 5, 'state': [5.0, 0.0, 1.5], 'velocity': [0.0, 0.0]}

    tracker._project_reference_out_of_keepouts(
        _straight_reference(), [obstacle], ego_pose=(0.0, 0.0, 0.0))

    assert tracker._avoidance_stop_active
    # The keep-out (ego 1.5 + obstacle 1.5 + safe 0.4) is centred on the reference,
    # so clearing it demands more than the 2.0 m bound — which is why it stopped.
    assert tracker._avoidance_required_offset > 2.0


def test_required_detour_offset_resets_when_nothing_is_selected():
    """A stale offset from an earlier tick would read as a live measurement."""
    tracker = _projection_tracker(max_offset=2.0)
    tracker._project_reference_out_of_keepouts(
        _straight_reference(), [{'id': 5, 'state': [5.0, 0.0, 1.5], 'velocity': [0.0, 0.0]}],
        ego_pose=(0.0, 0.0, 0.0))
    assert not np.isnan(tracker._avoidance_required_offset)

    tracker._project_reference_out_of_keepouts(
        _straight_reference(), [], ego_pose=(0.0, 0.0, 0.0))
    assert np.isnan(tracker._avoidance_required_offset)
