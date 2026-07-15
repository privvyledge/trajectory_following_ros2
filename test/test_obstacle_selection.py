"""Obstacle relevance ranking and horizon packing in the trajectory-tracker base.

The rollout harnesses mirror the control loop rather than running it, so they never
execute these methods; these tests drive them directly. The tracker is instantiated via
``__new__`` to bypass ``rclpy.Node.__init__`` — every method under test reads only plain
attributes, so no ROS context is needed.
"""
import numpy as np
import pytest

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker

HORIZON = 25
SAMPLE_TIME = 0.05
EGO_RADIUS = 0.15
SAFE_DISTANCE = 0.15
OBSTACLE_RADIUS = 0.3
# Keep-out = 0.15 + 0.3 + 0.15 = 0.6 m


class _RecordingSolver:
    """Stands in for a backend adapter that constrains obstacles."""

    def __init__(self):
        self.obstacle_states = None

    def update_obstacles(self, obstacle_states):
        self.obstacle_states = obstacle_states


def _make_tracker(obstacles, num_obstacles=1, predict_motion=True, solver=None):
    class _Tracker(BaseTrajectoryTracker):
        def _init_solver(self):
            return None

    tracker = _Tracker.__new__(_Tracker)
    tracker.obstacles = obstacles
    tracker._num_obstacles = num_obstacles
    tracker._predict_obstacle_motion = predict_motion
    tracker._solver = _RecordingSolver() if solver is None else solver
    tracker.horizon = HORIZON
    tracker.sample_time = SAMPLE_TIME
    tracker.n_obstacle_states = 3
    tracker._keepout_side_hints = {}
    tracker._gp = {'ego_radius': EGO_RADIUS, 'safe_distance': SAFE_DISTANCE}.__getitem__
    return tracker


def _obstacle(obs_id, x, y, radius=OBSTACLE_RADIUS, velocity=(0.0, 0.0)):
    return {'id': obs_id, 'state': [x, y, radius], 'velocity': list(velocity)}


def _straight_xref(start_x=0.0, spacing=0.075):
    """Reference running +x from `start_x`, one horizon's worth (~1.9 m at v=1.5)."""
    xs = start_x + spacing * np.arange(HORIZON + 1)
    return np.vstack([xs, np.zeros(HORIZON + 1),
                      np.full(HORIZON + 1, 1.5), np.zeros(HORIZON + 1)])


def test_obstacle_ahead_beats_obstacle_behind():
    """The reported bug: ranking by distance-to-ego let a near obstacle behind the
    vehicle take the only slot from the one actually on the path ahead.

    The behind obstacle is strictly nearer the vehicle (1.0 m vs 1.5 m), so a
    distance-to-ego rank picks it outright rather than merely tying.
    """
    behind = _obstacle(1, -1.0, 0.0)
    ahead = _obstacle(2, 1.5, 0.0)
    tracker = _make_tracker([behind, ahead], num_obstacles=1)

    selected = tracker._select_obstacles(_straight_xref(), ego_xy=(0.0, 0.0))

    assert [o['id'] for o in selected] == [2], 'obstacle ahead on the path must win the slot'


def test_nearer_of_two_on_path_obstacles_wins():
    """Both sit on the reference, so both are ~0 from it; a plain min-distance rank
    ties and can pick the farther one. The earliest-intrusion tiebreak orders them.

    Both centres land exactly on reference samples (0.075 m spacing), so the naive
    minimum really is 0.0 for each and the tie is decided by list order — with the
    farther obstacle listed first, a naive rank returns it.
    """
    near = _obstacle(1, 0.6, 0.0)
    far = _obstacle(2, 1.2, 0.0)
    tracker = _make_tracker([far, near], num_obstacles=1)  # far listed first

    selected = tracker._select_obstacles(_straight_xref(), ego_xy=(0.0, 0.0))

    assert [o['id'] for o in selected] == [1], 'the sooner-constraining obstacle must win'


def test_obstacle_on_the_vehicle_is_not_evicted_when_off_reference():
    """During an avoidance swerve the vehicle is off-reference by construction, so an
    obstacle on top of it is far from xref. The ego term in the point set keeps it."""
    on_ego = _obstacle(1, 0.2, 2.0)          # on the vehicle, 2 m off the reference
    on_path_far = _obstacle(2, 1.8, 0.0)     # on the reference, further along
    tracker = _make_tracker([on_path_far, on_ego], num_obstacles=1)

    selected = tracker._select_obstacles(_straight_xref(), ego_xy=(0.2, 2.0))

    assert [o['id'] for o in selected] == [1], 'obstacle on the vehicle must not be evicted'


def test_obstacle_far_from_everything_ranks_last():
    """Non-intruders sort behind intruders, and among themselves by distance."""
    far = _obstacle(1, 50.0, 50.0)
    on_path = _obstacle(2, 1.0, 0.0)
    tracker = _make_tracker([far, on_path], num_obstacles=2)

    selected = tracker._select_obstacles(_straight_xref(), ego_xy=(0.0, 0.0))

    assert [o['id'] for o in selected] == [2, 1]


def test_selection_is_empty_when_backend_has_no_obstacle_constraints():
    """do-mpc exposes no update_obstacles; bending its reference would fake avoidance
    that no constraint enforces."""
    tracker = _make_tracker([_obstacle(1, 1.0, 0.0)], solver=object())

    assert tracker._select_obstacles(_straight_xref(), ego_xy=(0.0, 0.0)) == []


def _legacy_pack(obstacles, num_obstacles, horizon):
    """The per-stage loop as it stood in both controller nodes before consolidation."""
    states = np.ones((3 * num_obstacles, horizon + 1)) * 1000.0
    states[2::3, :] = 1.0
    for k in range(horizon + 1):
        for j in range(num_obstacles):
            idx = 3 * j
            if len(obstacles) > j:
                states[idx:idx + 3, k] = obstacles[j]['state']
            else:
                states[idx:idx + 3, k] = [1000.0, 1000.0, 1.0]
    return states


@pytest.mark.parametrize('n_selected,num_obstacles', [(1, 1), (2, 2), (1, 3), (0, 2)])
def test_vectorized_packing_matches_the_legacy_loop(n_selected, num_obstacles):
    """Static obstacles must pack exactly as the replaced loop did, including the
    far-away padding of unused slots."""
    selected = [_obstacle(i, float(i), float(-i)) for i in range(n_selected)]
    tracker = _make_tracker(selected, num_obstacles=num_obstacles)

    tracker._pack_obstacle_states(selected)

    np.testing.assert_array_equal(
        tracker._solver.obstacle_states,
        _legacy_pack(selected, num_obstacles, HORIZON))


def test_zero_twist_packs_identically_whether_prediction_is_on_or_off():
    """Why predict_obstacle_motion defaults to True: with no reported velocity it is
    exactly the previous static fill."""
    selected = [_obstacle(1, 2.0, 1.0, velocity=(0.0, 0.0))]
    on = _make_tracker(selected, predict_motion=True)
    off = _make_tracker(selected, predict_motion=False)

    on._pack_obstacle_states(selected)
    off._pack_obstacle_states(selected)

    np.testing.assert_array_equal(on._solver.obstacle_states, off._solver.obstacle_states)


def test_moving_obstacle_is_propagated_over_the_horizon():
    selected = [_obstacle(1, 2.0, 1.0, velocity=(1.0, -0.5))]
    tracker = _make_tracker(selected, predict_motion=True)

    tracker._pack_obstacle_states(selected)
    states = tracker._solver.obstacle_states

    last = HORIZON * SAMPLE_TIME
    assert states[0, 0] == pytest.approx(2.0)
    assert states[1, 0] == pytest.approx(1.0)
    assert states[0, HORIZON] == pytest.approx(2.0 + 1.0 * last)
    assert states[1, HORIZON] == pytest.approx(1.0 - 0.5 * last)
    # radius must not be swept along with the centre
    assert np.all(states[2, :] == OBSTACLE_RADIUS)


def test_side_hints_follow_the_obstacle_not_its_rank():
    """Hints are keyed by id, so a selection reorder cannot hand one obstacle's
    committed go-around side to another."""
    first = _obstacle(7, 0.9, 0.0)
    second = _obstacle(9, 1.5, 0.0)
    tracker = _make_tracker([first, second], num_obstacles=2)
    xref = _straight_xref()

    tracker._project_reference_out_of_keepouts(xref, [first, second])
    hints_by_id = dict(tracker._keepout_side_hints)
    assert set(hints_by_id) == {7, 9}

    # Same obstacles, reversed rank: each id must retain its own hint.
    tracker._project_reference_out_of_keepouts(xref, [second, first])
    assert set(tracker._keepout_side_hints) == {7, 9}


def test_dropped_obstacles_do_not_accumulate_in_the_hint_store():
    tracker = _make_tracker([], num_obstacles=1)
    xref = _straight_xref()

    tracker._project_reference_out_of_keepouts(xref, [_obstacle(1, 0.9, 0.0)])
    assert set(tracker._keepout_side_hints) == {1}

    tracker._project_reference_out_of_keepouts(xref, [_obstacle(2, 0.9, 0.0)])
    assert set(tracker._keepout_side_hints) == {2}, 'stale ids must not linger'


def test_projection_leaves_the_callers_reference_untouched():
    """The raw reference still feeds the debug topic, so only a copy may be bent."""
    obstacle = _obstacle(1, 0.9, 0.0)
    tracker = _make_tracker([obstacle])
    xref = _straight_xref()
    original = xref.copy()

    projected = tracker._project_reference_out_of_keepouts(xref, [obstacle])

    np.testing.assert_array_equal(xref, original)
    assert not np.array_equal(projected, original), 'an on-path obstacle must bend the copy'
