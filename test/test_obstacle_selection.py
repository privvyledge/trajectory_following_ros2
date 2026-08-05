"""Obstacle relevance ranking and horizon packing in the trajectory-tracker base.

The rollout harnesses mirror the control loop rather than running it, so they never
execute these methods; these tests drive them directly. The tracker is instantiated via
``__new__`` to bypass ``rclpy.Node.__init__`` — every method under test reads only plain
attributes, so no ROS context is needed.
"""
import numpy as np
import pytest

from trajectory_following_ros2.base_tracker import AvoidanceStopLatch, BaseTrajectoryTracker

HORIZON = 25
SAMPLE_TIME = 0.05
EGO_RADIUS = 0.15
SAFE_DISTANCE = 0.15
OBSTACLE_RADIUS = 0.3
# Keep-out = 0.15 + 0.3 + 0.15 = 0.6 m
# Single collision disc on the rear-axle reference point: these tests are about the
# ranking, so the disc geometry is kept at the identity case. See test_ego_discs.py.
DISC_OFFSETS = [0.0]


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
    tracker.prediction_time = HORIZON * SAMPLE_TIME
    tracker.MAX_DECEL = -3.0
    tracker.ENVELOPE_DECEL = 0.0  # 0 = follow |max_decel|
    tracker.n_obstacle_states = 3
    tracker._keepout_side_hints = {}
    tracker._avoidance_stop_latch = AvoidanceStopLatch()
    tracker._avoidance_stop_active = False
    tracker.max_avoidance_offset = 0.0
    tracker.keepout_engagement_distance = 0.0
    tracker.x = tracker.y = tracker.yaw = 0.0

    class _NullLogger:
        def info(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass
    tracker.get_logger = _NullLogger
    tracker._gp = {'ego_radius': EGO_RADIUS, 'safe_distance': SAFE_DISTANCE,
                   'ego_disc_offsets': DISC_OFFSETS}.__getitem__
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

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

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

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

    assert [o['id'] for o in selected] == [1], 'the sooner-constraining obstacle must win'


def test_obstacle_on_the_vehicle_is_not_evicted_when_off_reference():
    """During an avoidance swerve the vehicle is off-reference by construction, so an
    obstacle on top of it is far from xref. The ego term in the point set keeps it."""
    on_ego = _obstacle(1, 0.2, 2.0)          # on the vehicle, 2 m off the reference
    on_path_far = _obstacle(2, 1.8, 0.0)     # on the reference, further along
    tracker = _make_tracker([on_path_far, on_ego], num_obstacles=1)

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.2, 2.0, 0.0))

    assert [o['id'] for o in selected] == [1], 'obstacle on the vehicle must not be evicted'


def test_obstacle_far_from_everything_ranks_last():
    """Non-intruders sort behind intruders, and among themselves by distance."""
    far = _obstacle(1, 50.0, 50.0)
    on_path = _obstacle(2, 1.0, 0.0)
    tracker = _make_tracker([far, on_path], num_obstacles=2)

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

    assert [o['id'] for o in selected] == [2, 1]


def test_selection_is_empty_when_backend_has_no_obstacle_constraints():
    """do-mpc exposes no update_obstacles; bending its reference would fake avoidance
    that no constraint enforces."""
    tracker = _make_tracker([_obstacle(1, 1.0, 0.0)], solver=object())

    assert tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0)) == []


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


def test_obstacle_braking_envelope_triggers_for_fast_closing_vehicle():
    obstacle = _obstacle(1, 1.0, 0.0)
    tracker = _make_tracker([obstacle])

    diag = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5,
        tick_interval_ms=100.0)

    assert diag['stop']
    assert diag['obstacle_id'] == 1
    # Physical clearance excludes safe_distance: 1.0 - 0.15 - 0.3 = 0.55 m.
    assert diag['physical_clearance'] == pytest.approx(0.55)
    assert diag['closing_speed'] == pytest.approx(1.5)
    lag = BaseTrajectoryTracker._ENVELOPE_ACTUATION_LAG_S
    assert diag['stopping_room'] == pytest.approx(1.5 * (0.1 + lag) + 1.5**2 / 6.0)


def test_obstacle_braking_envelope_allows_stationary_or_departing_vehicle():
    obstacle = _obstacle(1, 1.0, 0.0)
    tracker = _make_tracker([obstacle])

    assert not tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=0.0,
        tick_interval_ms=100.0)['stop']
    assert not tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, np.pi), speed=1.5,
        tick_interval_ms=100.0)['stop']


def test_obstacle_braking_envelope_is_quiet_far_from_the_obstacle():
    """Diagnostics are reported on every tick, so a non-firing tick must still say
    how much margin it had — that is what makes intervention chatter auditable."""
    obstacle = _obstacle(1, 20.0, 0.0)
    tracker = _make_tracker([obstacle])

    diag = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)

    assert not diag['stop']
    assert diag['obstacle_id'] == 1
    assert diag['physical_clearance'] == pytest.approx(19.55)
    assert diag['margin'] > 0.0


def test_obstacle_braking_envelope_uses_relative_closing_speed():
    """An obstacle fleeing at the vehicle's own speed is not being closed on, so the
    same geometry that fires against a static obstacle must not fire against it."""
    static = _obstacle(1, 1.0, 0.0)
    fleeing = _obstacle(1, 1.0, 0.0, velocity=(1.5, 0.0))
    tracker = _make_tracker([static])

    assert tracker._obstacle_safety_check(
        [static], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)['stop']
    departing = tracker._obstacle_safety_check(
        [fleeing], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)
    assert not departing['stop']
    assert departing['closing_speed'] == pytest.approx(0.0)


def test_delayed_tick_widens_the_braking_envelope():
    """The reaction term uses the *observed* tick interval, so a cadence stall must
    trip the envelope at a distance a healthy tick would clear."""
    obstacle = _obstacle(1, 1.6, 0.0)
    tracker = _make_tracker([obstacle])

    healthy = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)
    stalled = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=400.0)

    assert not healthy['stop']
    assert stalled['stop']
    assert stalled['stopping_room'] > healthy['stopping_room']


def test_envelope_decel_shrinks_the_envelope_without_touching_max_decel():
    """``envelope_decel`` overrides the deceleration the stopping-room term assumes.

    The braking distance is ``closing²/(2·decel)``, so a harder emergency decel
    shrinks the envelope quadratically — that is the whole point of separating it
    from ``max_decel``, which is a comfort value shaping the reference speed ramp
    and the stop-before-keep-out profile and must stay put.
    """
    # 1.8 m is between the two envelopes: room is 1.275 m at 3 m/s², 1.088 m at 6.
    obstacle = _obstacle(1, 1.8, 0.0)
    tracker = _make_tracker([obstacle])

    comfort = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=400.0)
    tracker.ENVELOPE_DECEL = 6.0
    emergency = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=400.0)

    assert comfort['stop'] and not emergency['stop']
    assert emergency['stopping_room'] < comfort['stopping_room']
    assert tracker.MAX_DECEL == -3.0  # untouched


def test_envelope_decel_zero_follows_max_decel():
    """0.0 is the documented 'inherit |max_decel|' sentinel, not a 0 m/s² stop."""
    obstacle = _obstacle(1, 1.6, 0.0)
    tracker = _make_tracker([obstacle])
    tracker.ENVELOPE_DECEL = 0.0
    inherited = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=400.0)
    tracker.ENVELOPE_DECEL = 3.0
    explicit = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=400.0)

    assert inherited['stopping_room'] == pytest.approx(explicit['stopping_room'])


def test_braking_diagnostics_report_the_obstacle_that_actually_fired():
    """An obstacle being driven away from can sit deeper inside its keep-out than the
    one being closed on, so a plain min-margin report would attribute the stop to the
    wrong obstacle — and to a *negative* closing speed."""
    passing = _obstacle(1, -0.2, 0.0)   # just behind, receding: inside keep-out, not closing
    ahead = _obstacle(2, 1.0, 0.0)      # closing fast enough to trip the envelope
    tracker = _make_tracker([passing, ahead], num_obstacles=2)

    diag = tracker._obstacle_safety_check(
        [passing, ahead], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)

    assert diag['stop']
    assert diag['obstacle_id'] == 2
    assert diag['closing_speed'] > 0.0


def test_braking_diagnostics_report_the_most_critical_obstacle():
    near = _obstacle(1, 1.0, 0.0)
    far = _obstacle(2, 8.0, 0.0)
    tracker = _make_tracker([near, far], num_obstacles=2)

    diag = tracker._obstacle_safety_check(
        [far, near], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)

    assert diag['stop']
    assert diag['obstacle_id'] == 1, 'the smallest-margin obstacle must be reported'


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


def test_stopped_vehicle_still_reports_clearance_diagnostics():
    """The tick right after a safety stop measures ~0 speed. A low-speed early-out
    left those rows blank (nan) and let the solver's creep command through
    unaudited — the stop/creep alternation that ratcheted the vehicle into
    contact. Stopped must mean "cannot fire", never "not measured"."""
    obstacle = _obstacle(1, 0.5, 0.0)
    tracker = _make_tracker([obstacle])

    diag = tracker._obstacle_safety_check(
        [obstacle], ego_pose=(0.0, 0.0, 0.0), speed=0.0, tick_interval_ms=50.0)

    assert not diag['stop']
    assert diag['obstacle_id'] == 1
    assert diag['physical_clearance'] == pytest.approx(0.5 - 0.15 - 0.3)


def test_latched_hold_admits_escapes_but_not_deep_closing_commands():
    """While a safety hold is latched the vehicle must still be allowed to leave:
    the solver's reverse escape is the recovery path, while a forward creep
    toward an almost-touching obstacle stays suppressed."""
    from trajectory_following_ros2.backends.base_solver import SolverResult
    obstacle = _obstacle(1, 0.5, 0.0)   # physical clearance 0.05 m: no budget left
    tracker = _make_tracker([obstacle])
    creep = SolverResult(velocity_cmd=0.15, is_optimal=True)
    reverse = SolverResult(velocity_cmd=-0.15, is_optimal=True)

    assert not tracker._hold_admissible_command(creep, [obstacle], (0.0, 0.0, 0.0))
    assert tracker._hold_admissible_command(reverse, [obstacle], (0.0, 0.0, 0.0))

    # Same forward command with the obstacle behind opens clearance.
    behind = _obstacle(1, -0.5, 0.0)
    assert tracker._hold_admissible_command(creep, [behind], (0.0, 0.0, 0.0))

    # Pinched between two keep-outs, reversing away from the one ahead closes
    # on the one behind: neither direction may pass.
    assert not tracker._hold_admissible_command(
        reverse, [obstacle, behind], (0.0, 0.0, 0.0))


def test_latched_hold_admits_a_slow_creep_with_stopping_budget_in_hand():
    """A vehicle parked just inside the comfort band (well clear of physical
    contact) must be allowed to move slowly, or the hold is a standstill
    deadlock: the comfort margin is spendable during recovery, and the
    admissible speed shrinks to zero before the physical floor is reached."""
    from trajectory_following_ros2.backends.base_solver import SolverResult
    obstacle = _obstacle(1, 0.63, 0.0)  # keep-out clearance 0.03, physical 0.18
    tracker = _make_tracker([obstacle])

    creep = SolverResult(velocity_cmd=0.15, is_optimal=True)
    fast = SolverResult(velocity_cmd=0.8, is_optimal=True)

    assert tracker._hold_admissible_command(creep, [obstacle], (0.0, 0.0, 0.0))
    assert not tracker._hold_admissible_command(fast, [obstacle], (0.0, 0.0, 0.0))


def test_overlapping_keepouts_merge_into_one_projection_circle():
    """Two keep-outs whose corridor is narrower than the vehicle must project as
    one enclosing circle — a per-obstacle projection threads the reference
    through the impassable gap and the solver accelerates into the pinch."""
    a = _obstacle(1, 2.0, 0.6)
    b = _obstacle(2, 2.0, -0.6)   # centres 1.2 m apart, keep-outs 0.6 m each
    tracker = _make_tracker([a, b], num_obstacles=2)
    centres = np.array([[2.0, 0.6], [2.0, -0.6]])

    proj_c, proj_r, groups = tracker._merge_overlapping_keepouts(
        centres, tracker._keepout_radii([a, b]))

    assert len(groups) == 1 and sorted(groups[0]) == [0, 1]
    np.testing.assert_allclose(proj_c[0], [2.0, 0.0])
    assert proj_r[0] == pytest.approx(1.2)


def test_separated_keepouts_stay_individual_projection_circles():
    a = _obstacle(1, 2.0, 2.0)
    b = _obstacle(2, 2.0, -2.0)
    tracker = _make_tracker([a, b], num_obstacles=2)
    centres = np.array([[2.0, 2.0], [2.0, -2.0]])

    proj_c, proj_r, groups = tracker._merge_overlapping_keepouts(
        centres, tracker._keepout_radii([a, b]))

    assert len(groups) == 2
    np.testing.assert_allclose(proj_r, [0.6, 0.6])


def test_merged_projection_routes_the_reference_around_the_pair():
    """The projected reference must clear BOTH keep-outs, not thread between."""
    a = _obstacle(1, 1.5, 0.55)
    b = _obstacle(2, 1.5, -0.55)   # reference on y=0 runs through the 4 cm gap
    tracker = _make_tracker([a, b], num_obstacles=2)
    xref = _straight_xref()

    projected = tracker._project_reference_out_of_keepouts(xref, [a, b])

    keepout = 0.6
    for cx, cy in ((1.5, 0.55), (1.5, -0.55)):
        d = np.hypot(projected[0, :] - cx, projected[1, :] - cy)
        assert d.min() >= keepout - 0.05, 'reference must not thread the pinch'


def test_selection_survives_a_feed_whose_object_count_changes_mid_rank():
    """A perception feed whose object count changes between reads must not crash.

    ``_obstacle_callback`` rebinds ``self.obstacles`` to a fresh list from another
    executor thread. Ranking that re-read the attribute could therefore build its
    per-obstacle arrays from one list and index them with another list's length —
    ``IndexError``, which propagates out of the control timer and kills the controller
    while the last command stays latched on the actuator. A fixed-count publisher can
    never expose this; a live feed whose count varies as actors appear and disappear
    does. The fix is a single read, so the tick works from a stale-but-consistent
    snapshot.

    The stub returns the short list for the first three reads and the long list after,
    which is exactly the read order the pre-fix code used (guard, centres, radii, then
    ``len``) — so this test fails with ``IndexError`` against that version.
    """
    short = [_obstacle(1, 1.0, 0.0)]
    grown = [_obstacle(1, 1.0, 0.0), _obstacle(2, 2.0, 0.0), _obstacle(3, 3.0, 0.0)]
    tracker = _make_tracker(short, num_obstacles=2)
    reads = []

    class _RebindingFeed(type(tracker)):
        @property
        def obstacles(self):
            reads.append(len(reads))
            return short if len(reads) <= 3 else grown

        @obstacles.setter
        def obstacles(self, value):   # the callback's rebind; ignored by the stub
            pass

    tracker.__class__ = _RebindingFeed

    selected = tracker._select_obstacles(_straight_xref(), (0.0, 0.0, 0.0))

    assert len(reads) == 1, 'ranking must read the cached detections exactly once'
    assert [o['id'] for o in selected] == [1]


# ---------------------------------------------------------------------------
# Relevance cutoff — a slot is only worth spending on a detection this solve can
# reach. Filling every slot with the merely-nearest detection is harmless on a
# handful of objects and destructive on a live map feed, where it hands the
# projection merge a set of bystanders to fuse into a phantom keep-out.
# ---------------------------------------------------------------------------

def test_unreachable_static_bystander_does_not_take_a_slot():
    """Clear of the window by more than the detour bound, with no speed to close it."""
    on_path = _obstacle(1, 1.0, 0.0)
    bystander = _obstacle(2, 1.0, 6.0)      # 6 m off, keep-out 0.6 -> 5.4 m clear
    tracker = _make_tracker([bystander, on_path], num_obstacles=2)
    tracker.max_avoidance_offset = 2.0

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

    assert [o['id'] for o in selected] == [1], 'bystander beyond the detour bound must not take a slot'


def test_bystander_within_the_detour_bound_keeps_its_slot():
    """Inside the swerve envelope it can still bite once the vehicle leaves the line."""
    on_path = _obstacle(1, 1.0, 0.0)
    near = _obstacle(2, 1.0, 2.0)          # 2 m off, keep-out 0.6 -> 1.4 m clear < 2.0
    tracker = _make_tracker([near, on_path], num_obstacles=2)
    tracker.max_avoidance_offset = 2.0

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

    assert sorted(o['id'] for o in selected) == [1, 2]


def test_fast_crossing_bystander_keeps_its_slot():
    """Reach grows with the obstacle's own travel over the horizon, so a crossing
    actor is not dismissed on its current position the way a lamp post is."""
    on_path = _obstacle(1, 1.0, 0.0)
    crossing = _obstacle(2, 1.0, 6.0, velocity=(0.0, -8.0))   # 8 m/s * 1.25 s = 10 m
    tracker = _make_tracker([crossing, on_path], num_obstacles=2)
    tracker.max_avoidance_offset = 2.0

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

    assert sorted(o['id'] for o in selected) == [1, 2]


def test_unbounded_detour_keeps_the_fill_all_behaviour():
    """With no bound on the swerve there is no bound to prove irrelevance from."""
    on_path = _obstacle(1, 1.0, 0.0)
    bystander = _obstacle(2, 1.0, 50.0)
    tracker = _make_tracker([bystander, on_path], num_obstacles=2)
    tracker.max_avoidance_offset = 0.0

    selected = tracker._select_obstacles(_straight_xref(), ego_pose=(0.0, 0.0, 0.0))

    assert sorted(o['id'] for o in selected) == [1, 2]


# ---------------------------------------------------------------------------
# Merge gating — proximity alone over-merges a chain running alongside the route.
# ---------------------------------------------------------------------------

def test_same_side_chain_is_not_merged_when_the_reference_threads_no_gap():
    """Roadside furniture in a line beside the route: consecutive members are close
    enough to trip the distance gate, but the reference passes none of the gaps
    between them, and the enclosing circle would reach across the carriageway."""
    a = _obstacle(1, 1.0, 0.9)
    b = _obstacle(2, 1.6, 0.9)     # 0.6 m apart, both 0.9 m to one side of y=0
    tracker = _make_tracker([a, b], num_obstacles=2)
    centres = np.array([[1.0, 0.9], [1.6, 0.9]])
    keepouts = tracker._keepout_radii([a, b])
    sides = tracker._keepout_sides(_straight_xref(), centres)

    _, _, gated = tracker._merge_overlapping_keepouts(
        centres, keepouts, obstacle_ids=[1, 2], sides=sides)
    _, _, ungated = tracker._merge_overlapping_keepouts(centres, keepouts)

    np.testing.assert_array_equal(sides, [1.0, 1.0])
    assert len(ungated) == 1, 'distance alone fuses the chain (the behaviour being gated)'
    assert len(gated) == 2, 'same side of the reference — no gap is threaded, so no merge'


def test_genuine_pinch_still_merges_when_the_reference_runs_through_it():
    a = _obstacle(1, 2.0, 0.6)
    b = _obstacle(2, 2.0, -0.6)     # straddling y=0: the reference runs between them
    tracker = _make_tracker([a, b], num_obstacles=2)
    centres = np.array([[2.0, 0.6], [2.0, -0.6]])
    sides = tracker._keepout_sides(_straight_xref(), centres)

    _, proj_r, groups = tracker._merge_overlapping_keepouts(
        centres, tracker._keepout_radii([a, b]), obstacle_ids=[1, 2], sides=sides)

    np.testing.assert_array_equal(sides, [1.0, -1.0])
    assert len(groups) == 1 and sorted(groups[0]) == [0, 1]
    assert proj_r[0] == pytest.approx(1.2)


def test_sub_discs_of_one_obstacle_merge_regardless_of_side():
    """A decomposed box is one physical body: its chain must project as one circle
    even though every sub-disc lies on the same side of the reference."""
    a = _obstacle(7, 1.0, 0.9)
    b = _obstacle(7, 1.6, 0.9)
    tracker = _make_tracker([a, b], num_obstacles=2)
    centres = np.array([[1.0, 0.9], [1.6, 0.9]])
    sides = tracker._keepout_sides(_straight_xref(), centres)

    _, _, groups = tracker._merge_overlapping_keepouts(
        centres, tracker._keepout_radii([a, b]), obstacle_ids=[7, 7], sides=sides)

    assert len(groups) == 1, 'sub-discs of one parent are exempt from the side test'


def test_braking_diagnostics_carry_the_reported_obstacle_position():
    """The stop-vs-swerve question cannot be answered from a clearance scalar.

    A blocked corridor (stopping is correct) and a passable gap the planner declined
    produce identical clearance traces; only the obstacle's position relative to the
    route separates them, and a live CARLA run is the wrong place to discover the
    column is missing. The centre must be the *reported* obstacle's, so it is written
    on the same branch that wins the report rank rather than from the last loop
    iteration.
    """
    receding = _obstacle(1, -0.2, 0.0)
    ahead = _obstacle(2, 1.0, 0.35)
    tracker = _make_tracker([receding, ahead], num_obstacles=2)

    diag = tracker._obstacle_safety_check(
        [receding, ahead], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)

    assert diag['obstacle_id'] == 2
    assert diag['obstacle_x'] == pytest.approx(1.0)
    assert diag['obstacle_y'] == pytest.approx(0.35)
    assert diag['obstacle_keepout'] == pytest.approx(
        EGO_RADIUS + OBSTACLE_RADIUS + SAFE_DISTANCE)


def test_braking_diagnostics_position_is_nan_with_nothing_selected():
    """No selection means no reported pair; a stale position would read as a real one."""
    tracker = _make_tracker([])
    diag = tracker._obstacle_safety_check(
        [], ego_pose=(0.0, 0.0, 0.0), speed=1.5, tick_interval_ms=50.0)
    assert np.isnan(diag['obstacle_x']) and np.isnan(diag['obstacle_y'])


# ---------------------------------------------------------------------------
# The CARLA terminal-stall scene, at CARLA scale.
#
# Reconstructed from a logged run in which the vehicle stopped short of a
# passable corridor and never restarted: the reference projection demanded a
# 5.9 m lateral detour against a 2.0 m bound, so every tick took the
# stop-before-the-keep-out branch. Everything below is that scene's geometry
# driven through the real merge/projection chain, with no ROS and no data files
# — the poses come from the run's logged geometry block and the disc radii from
# `obstacle_shape_discs` on the reported dimensions.
#
# The scene: a truck stopped essentially on the route centreline, a line of kerb
# furniture on the near side, and a bollard on the far side of the carriageway.
# The measured physical corridor between the truck's near edge and the bollard is
# 4.82 m, and the vehicle needs 2.65 m of it (1.85 m wide + 2 x safe_distance) —
# so the corridor is passable with ~2.2 m to spare, and the geometrically correct
# lateral offset is ~3.9 m. What the pipeline actually demands is ~6.95 m.
#
# The over-demand is NOT the truck's own projection: the truck alone asks for
# 3.81 m, which is the right answer. It is the merge fusing the truck with the
# kerb furniture *behind the vehicle's own side of the road* into one 7.44 m
# enclosing circle. That distinction decides where a fix belongs, so both halves
# are pinned below.
# ---------------------------------------------------------------------------

CARLA_EGO_RADIUS = 1.096
CARLA_SAFE_DISTANCE = 0.4
CARLA_DISC_OFFSETS = [-0.4125, 0.7625, 1.9375, 3.1125]
CARLA_ENGAGEMENT_DISTANCE = 25.0
CARLA_DETOUR_BOUND = 2.0            # the run's max_avoidance_offset

# Route centreline x in the stall zone; the route runs due south (yaw -pi/2).
ROUTE_X = -2.007
ROUTE_Y0 = -184.400
ROUTE_SPEED = 8.339
# Vehicle pose while stalled, from the run's logged geometry block.
STALL_EGO = (1.234420, -181.426559, -0.595800)

TRUCK_ID = 210
TRUCK_DISC_RADIUS = 2.020267        # 8.468 x 2.891 m box, decomposed into 3 discs
TRUCK_DISCS = [(-1.964147, -182.701504),
               (-1.936186, -185.524046),     # the parent centre
               (-1.908225, -188.346588)]
# Kerb furniture on the vehicle's own side, well clear of the driving line.
KERB_NEAR_ID, KERB_NEAR = 3263154216, (-4.440, -184.860)
KERB_FAR_ID, KERB_FAR = 1133245796, (-4.590, -191.060)
KERB_NEAR_RADIUS = 0.3              # 0.168 m post, lifted by min_obstacle_radius
KERB_FAR_RADIUS = 0.326357          # 0.6 x 0.6 m box, above the floor
# Bollard on the far side of the carriageway — the corridor's other wall.
BOLLARD_ID, BOLLARD = 2474949318, (4.410, -184.860)
BOLLARD_RADIUS = 0.3

# Corridor arithmetic, measured from the reported body dimensions.
CORRIDOR_GAP_M = 4.817              # truck near edge to bollard far edge
CORRIDOR_NEEDED_M = 2.650           # 1.85 m body + 2 x safe_distance
CORRECT_OFFSET_M = 3.925            # centre the vehicle in that corridor
TRUCK_ONLY_OFFSET_M = 3.8055        # what the truck's own projection asks for
FUSED_OFFSET_M = 6.954              # what the merged circle asks for


def _make_carla_tracker(obstacles, num_obstacles=10,
                        min_obstacle_radius=0.3,
                        max_avoidance_offset=CARLA_DETOUR_BOUND):
    """`_make_tracker` at CARLA vehicle scale, with the run's obstacle settings."""
    del min_obstacle_radius     # applied when building the fixture, not here
    tracker = _make_tracker(obstacles, num_obstacles=num_obstacles)
    tracker.max_avoidance_offset = max_avoidance_offset
    tracker.keepout_engagement_distance = CARLA_ENGAGEMENT_DISTANCE
    tracker.x, tracker.y, tracker.yaw = STALL_EGO
    tracker._gp = {'ego_radius': CARLA_EGO_RADIUS,
                   'safe_distance': CARLA_SAFE_DISTANCE,
                   'ego_disc_offsets': CARLA_DISC_OFFSETS}.__getitem__
    return tracker


def _stall_scene(kerb_near_radius=KERB_NEAR_RADIUS, bollard_radius=BOLLARD_RADIUS):
    """The stall scene's detections, in the order the ingestion produced them."""
    return ([_obstacle(BOLLARD_ID, *BOLLARD, radius=bollard_radius)]
            + [_obstacle(TRUCK_ID, x, y, radius=TRUCK_DISC_RADIUS)
               for x, y in TRUCK_DISCS]
            + [_obstacle(KERB_NEAR_ID, *KERB_NEAR, radius=kerb_near_radius),
               _obstacle(KERB_FAR_ID, *KERB_FAR, radius=KERB_FAR_RADIUS)])


def _stall_xref():
    """The reference window as the controller had it: due south, one horizon long."""
    ys = ROUTE_Y0 - ROUTE_SPEED * SAMPLE_TIME * np.arange(HORIZON + 1)
    return np.vstack([np.full(HORIZON + 1, ROUTE_X), ys,
                      np.full(HORIZON + 1, ROUTE_SPEED),
                      np.full(HORIZON + 1, -np.pi / 2)])


def _required_offsets(tracker, xref, selected):
    """Run the merge + projection chain and return (groups, circles, offsets)."""
    centres = np.array([o['state'][:2] for o in selected], dtype=float)
    radii = tracker._engaged_keepout_radii(xref, selected, STALL_EGO[:2])
    sides = tracker._keepout_sides(xref, centres)
    proj_c, proj_r, groups = tracker._merge_overlapping_keepouts(
        centres, radii, obstacle_ids=[o['id'] for o in selected], sides=sides)
    required = tracker._projection_group_offsets(
        xref, proj_c, proj_r, groups, selected)
    return groups, (proj_c, proj_r), required


def test_stall_scene_fuses_the_truck_with_the_kerb_line():
    """Characterization: what the pipeline does today, so a fix has to move it.

    Five keep-outs collapse to one 7.44 m enclosing circle straddling the whole
    carriageway, and the detour it implies (~6.95 m) is more than three times the
    2.0 m bound — so `_project_reference_out_of_keepouts` takes the stop-short
    branch on every tick and the vehicle never restarts. Guards the fixture
    against silent geometry drift, and proves the spec test below fails for the
    right reason rather than because the scene stopped being the scene.
    """
    scene = _stall_scene()
    tracker = _make_carla_tracker(scene)
    xref = _stall_xref()

    selected = tracker._select_obstacles(xref, ego_pose=STALL_EGO)
    groups, (proj_c, proj_r), required = _required_offsets(tracker, xref, selected)

    assert len(selected) == 6
    fused = max(groups, key=len)
    assert sorted(selected[i]['id'] for i in fused) == sorted(
        [TRUCK_ID, TRUCK_ID, TRUCK_ID, KERB_NEAR_ID, KERB_FAR_ID])
    assert float(proj_r[groups.index(fused)]) == pytest.approx(7.4436, abs=1e-3)
    assert max(required.values()) == pytest.approx(FUSED_OFFSET_M, abs=0.01)
    assert max(required.values()) > 3.0 * CARLA_DETOUR_BOUND


def test_stall_scene_truck_alone_asks_for_the_geometrically_correct_detour():
    """The over-demand is the merge, not the truck — which localizes the defect.

    Projected on its own, the truck's keep-out asks for 3.81 m, within 0.12 m of
    the 3.93 m that centring the vehicle in the measured corridor requires. So the
    circumscribing-circle inflation on a single body is tolerable here; what is not
    is fusing that body with furniture on the far kerb. A fix aimed at obstacle
    radius inflation would therefore not have moved this run.
    """
    truck = [o for o in _stall_scene() if o['id'] == TRUCK_ID]
    tracker = _make_carla_tracker(truck, num_obstacles=3)
    xref = _stall_xref()

    _, (_, proj_r), required = _required_offsets(tracker, xref, truck)

    assert len(proj_r) == 1, 'the sub-discs of one body are one projection circle'
    assert max(required.values()) == pytest.approx(TRUCK_ONLY_OFFSET_M, abs=0.01)
    assert abs(max(required.values()) - CORRECT_OFFSET_M) < 0.25


@pytest.mark.xfail(strict=True, reason='F3: the merge fuses the truck with the '
                                       'near kerb and hides a passable corridor')
def test_stall_scene_leaves_the_measured_corridor_passable():
    """Spec: the demanded detour must reflect the corridor that physically exists.

    Between the truck's near edge and the bollard there is 4.82 m of road and the
    vehicle occupies 2.65 m of it, so a ~3.9 m lateral offset drives through with
    ~2.2 m to spare. Anything approaching 7 m is not a statement about this scene's
    geometry — it is the enclosing circle of a group that never blocked a gap the
    reference uses. Flips loudly when the merge decision moves to lateral intervals
    per station instead of circle-to-circle distance.
    """
    scene = _stall_scene()
    tracker = _make_carla_tracker(scene)
    xref = _stall_xref()

    selected = tracker._select_obstacles(xref, ego_pose=STALL_EGO)
    _, _, required = _required_offsets(tracker, xref, selected)

    assert CORRIDOR_GAP_M - CORRIDOR_NEEDED_M > 2.0, 'the corridor is passable'
    assert max(required.values()) < CORRECT_OFFSET_M + 0.5


def test_the_far_side_bollard_is_held_out_of_the_group_by_the_side_gate():
    """The far wall of the corridor is *not* what fuses — the near kerb is.

    The truck's centre sub-disc and the bollard are 6.381 m apart against a 6.408 m
    distance gate, i.e. inside it by 27 mm; on distance alone they would fuse across
    the carriageway. They do not, because the truck sits 0.074 m off the centreline
    — just outside `_ON_REFERENCE_TOLERANCE_M` (0.05 m) — so it reports side +1 like
    the bollard and the same-side gate blocks the pair. Two consequences worth
    pinning: the bollard projects to a zero detour, and the whole outcome hangs on a
    24 mm margin in a tolerance constant that was chosen for something else.
    """
    scene = _stall_scene()
    tracker = _make_carla_tracker(scene)
    xref = _stall_xref()
    centres = np.array([o['state'][:2] for o in scene], dtype=float)
    keepouts = tracker._keepout_radii(scene)
    sides = tracker._keepout_sides(xref, centres)
    bollard, truck_mid = 0, 2

    gap = float(np.hypot(*(centres[bollard] - centres[truck_mid])))
    gate = keepouts[bollard] + keepouts[truck_mid] + CARLA_EGO_RADIUS

    assert gap < gate, 'the distance gate alone would fuse them'
    assert gate - gap == pytest.approx(0.027, abs=0.005)
    assert sides[bollard] == sides[truck_mid] == 1.0, 'same side: the gate blocks it'

    selected = tracker._select_obstacles(xref, ego_pose=STALL_EGO)
    groups, _, required = _required_offsets(tracker, xref, selected)
    assert [BOLLARD_ID] in [[selected[i]['id'] for i in g] for g in groups]
    assert required[frozenset([BOLLARD_ID])] == pytest.approx(0.0)


def test_lowering_the_radius_floor_does_not_dissolve_the_fusion():
    """The later run lowered `min_obstacle_radius` 0.3 -> 0.1 and did not fix F3.

    The floor only reaches the 0.168 m post, shrinking its keep-out by 0.2 m; the
    truck and the 0.6 m kerb box are both above the floor and unchanged, so the
    group and its detour survive intact. Documents that the fusion is driven by the
    truck's own inflated keep-out plus the merge allowance, not by the floor — the
    parameter that looks like the obvious lever is not one.
    """
    xref = _stall_xref()
    floored = _stall_scene()
    lowered = _stall_scene(kerb_near_radius=0.1, bollard_radius=0.1)

    out = {}
    for label, scene, bound in (('run_g', floored, 2.0), ('run_h', lowered, 3.5)):
        tracker = _make_carla_tracker(scene, max_avoidance_offset=bound)
        selected = tracker._select_obstacles(xref, ego_pose=STALL_EGO)
        groups, _, required = _required_offsets(tracker, xref, selected)
        out[label] = (max(groups, key=len), max(required.values()), bound)

    for label, (fused, offset, bound) in out.items():
        assert len(fused) == 5, f'{label}: the group survives the floor change'
        assert offset == pytest.approx(FUSED_OFFSET_M, abs=0.01)
        assert offset > bound, f'{label}: still over the bound, so still stop-short'


def test_truck_sub_disc_chain_drags_the_kerb_line_in_transitively():
    """One link is enough: union-find merging has no notion of group extent.

    Only the truck's rear and centre sub-discs are close enough to reach the kerb
    posts, but the sub-discs are same-parent-merged first, so a single sub-disc
    to post link pulls the entire 8.5 m truck chain and both posts into one
    cluster. The resulting enclosing circle spans a region no member ever blocked
    — which is how a group of five keep-outs, none wider than 3.6 m, becomes a
    7.44 m circle.
    """
    scene = _stall_scene()
    tracker = _make_carla_tracker(scene)
    xref = _stall_xref()
    centres = np.array([o['state'][:2] for o in scene], dtype=float)
    keepouts = tracker._keepout_radii(scene)
    sides = tracker._keepout_sides(xref, centres)
    front_disc, near_post = 1, 4

    reach = float(np.hypot(*(centres[front_disc] - centres[near_post])))
    gate = keepouts[front_disc] + keepouts[near_post] + CARLA_EGO_RADIUS
    _, proj_r, groups = tracker._merge_overlapping_keepouts(
        centres, keepouts, obstacle_ids=[o['id'] for o in scene], sides=sides)
    fused = max(groups, key=len)

    assert reach < gate, 'this pair does link directly'
    assert front_disc in fused and near_post in fused
    assert float(proj_r[groups.index(fused)]) > 2.0 * max(keepouts)


@pytest.mark.parametrize('body_gap', [0.62, 0.68, 0.74])
def test_merge_gate_demands_more_body_gap_than_the_vehicle_needs(body_gap):
    """Characterization of the merge allowance: it over-demands by one ego radius.

    Two obstacles straddling the reference fuse while their body-to-body gap is
    under ``2 * (ego_radius + safe_distance) + ego_radius`` — each keep-out already
    contains one ego radius and one safe distance, and the corridor term then adds a
    third ego radius on top. What the vehicle actually needs to drive between them
    is ``2 * ego_radius + 2 * safe_distance`` (its width plus a clearance either
    side). The parameters here bracket exactly that band: every gap listed is wide
    enough to drive through and still merges.

    Becomes the spec test, with the assertion inverted, once the decision is made on
    lateral intervals rather than circle distance.
    """
    needed = 2.0 * EGO_RADIUS + 2.0 * SAFE_DISTANCE
    demanded = 2.0 * (EGO_RADIUS + SAFE_DISTANCE) + EGO_RADIUS
    assert needed < body_gap < demanded, 'fixture must sit inside the over-demand band'

    half = body_gap / 2.0 + OBSTACLE_RADIUS
    a = _obstacle(1, 2.0, half)
    b = _obstacle(2, 2.0, -half)
    tracker = _make_tracker([a, b], num_obstacles=2)
    centres = np.array([[2.0, half], [2.0, -half]])
    sides = tracker._keepout_sides(_straight_xref(), centres)

    _, _, groups = tracker._merge_overlapping_keepouts(
        centres, tracker._keepout_radii([a, b]), obstacle_ids=[1, 2], sides=sides)

    np.testing.assert_array_equal(sides, [1.0, -1.0])
    assert len(groups) == 1, (
        f'{body_gap:.2f} m of body gap is driveable ({needed:.2f} m needed) '
        f'but the gate merges anything under {demanded:.2f} m')
