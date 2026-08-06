"""Reference generation across a stop-and-reverse cusp.

A route recorded by driving forward and then backing up stores the reverse as
*more* arc length, not less. The reference generator therefore has to keep
walking arc length forward while flipping only the sign of the reference speed.
Getting that backwards is silent: every solve stays optimal, no watchdog can see
it, and the vehicle simply stops short of the goal, because the reference
positions and the reference speed are asking for opposite directions of travel.
"""

import numpy as np

from trajectory_following_ros2.utils import trajectory_utils


COLUMNS = {'x': 0, 'y': 1, 'speed': 2, 'yaw': 3, 'omega': 4,
           'curvature': 5, 'cum_dist': 6, 'dt': 7, 'total_time_elapsed': 8}

FORWARD_SPEED = 0.8
REVERSE_SPEED = -0.3
CUSP_X = 2.0
SPACING = 0.05


def straight_out_and_back():
    """Drive +x to CUSP_X at FORWARD_SPEED, then back up 1 m at REVERSE_SPEED.

    Heading is 0 throughout -- a reversing vehicle does not turn around, which is
    why the recorded yaw cannot be used to detect the cusp; only the sign of the
    recorded speed marks it.
    """
    forward = np.arange(0.0, CUSP_X, SPACING)
    backward = np.arange(CUSP_X, CUSP_X - 1.0, -SPACING)
    xs = np.concatenate([forward, backward])
    zeros = np.zeros_like(xs)
    speed = np.concatenate([np.full(len(forward), FORWARD_SPEED),
                            np.full(len(backward), REVERSE_SPEED)])
    cum_dist = trajectory_utils.cumulative_distance_along_path(
        np.stack([xs, zeros], axis=1))
    return np.stack([xs, zeros, speed, zeros, zeros, zeros, cum_dist,
                     zeros, zeros], axis=1), len(forward)


def reference_at(trajectory, index, v_target, horizon=8):
    x = trajectory[index, COLUMNS['x']]
    return trajectory_utils.generate_reference_trajectory_by_interpolation(
        trajectory, init_pose=np.array([[x, 0.0, 0.0, 0.0]]),
        closest_index=index, waypoint_keys_to_columns=COLUMNS,
        horizon=horizon, v_target=float(v_target), dt=0.05,
        use_speed_profile=True, a_lat_max=3.0, v_min=0.3, v_max=1.0)


def assert_consistent(reference):
    """Every step of x_ref must move the way its own vel_ref says it should.

    Heading is 0, so a positive reference speed has to increase x and a negative
    one has to decrease it. This is the invariant the vehicle actually stalls on.
    """
    x_ref, vel_ref = reference['x_ref'], reference['vel_ref']
    for step, (dx, v) in enumerate(zip(np.diff(x_ref), vel_ref[:-1])):
        assert dx * v > 0.0, (
            f'step {step}: reference moves dx={dx:+.4f} while commanding '
            f'v={v:+.3f} -- position and speed disagree about direction')


def test_forward_anchor_advances_along_the_path():
    trajectory, cusp = straight_out_and_back()
    reference = reference_at(trajectory, cusp - 10, FORWARD_SPEED)
    assert np.all(reference['vel_ref'] > 0.0)
    assert np.all(np.diff(reference['x_ref']) > 0.0)
    assert_consistent(reference)


def test_reverse_anchor_continues_into_the_reverse_tail():
    """The regression: this used to retreat onto the forward leg."""
    trajectory, cusp = straight_out_and_back()
    reference = reference_at(trajectory, cusp + 5, REVERSE_SPEED)
    assert np.all(reference['vel_ref'] < 0.0)
    # x must DECREASE -- the vehicle is backing up along -x.
    assert np.all(np.diff(reference['x_ref']) < 0.0)
    assert_consistent(reference)


def test_horizon_spanning_the_cusp_flips_only_the_speed_sign():
    """Anchored just before the cusp, one horizon covers both directions."""
    trajectory, cusp = straight_out_and_back()
    reference = reference_at(trajectory, cusp - 2, FORWARD_SPEED)
    vel_ref = reference['vel_ref']
    assert vel_ref[0] > 0.0, 'the horizon must start out forward'
    assert vel_ref[-1] < 0.0, 'the horizon must reach the reverse tail'
    # The turnaround in the reference positions must sit at the recorded cusp,
    # not wherever the speed sign happens to change.
    assert reference['x_ref'].max() <= CUSP_X + SPACING
    assert_consistent(reference)


def test_arc_length_never_walks_backwards():
    """cum_dist_ref is monotonically increasing on both sides of the cusp.

    Arc length indexes position along the recorded route, so it advances with
    progress through the manoeuvre regardless of the direction of travel.
    """
    trajectory, cusp = straight_out_and_back()
    for index in (cusp - 10, cusp - 2, cusp + 5):
        v_target = trajectory[index, COLUMNS['speed']]
        cum_ref = reference_at(trajectory, index, v_target)['cum_dist_ref']
        assert np.all(np.diff(cum_ref) > 0.0), f'anchor {index} walked s backwards'


def test_forward_only_route_is_unchanged_by_the_signed_profile():
    """A route with no reverse content must produce exactly the legacy reference."""
    trajectory, cusp = straight_out_and_back()
    forward_only = trajectory[:cusp].copy()
    reference = reference_at(forward_only, 10, FORWARD_SPEED)
    assert np.allclose(reference['vel_ref'], FORWARD_SPEED)
    steps = np.diff(reference['x_ref'])
    assert np.allclose(steps, FORWARD_SPEED * 0.05, atol=1e-9)
