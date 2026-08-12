"""The goal-completion radius may be decoupled from the reference anchor distance.

`distance_tolerance` is both the reference anchor distance (`GOAL_DIS` /
`min_search_radius`) and the goal-completion radius, so the vehicle systematically
runs out of reference and stops about that far short of the final waypoint --
completion is then knife-edge against the very same number. `goal_tolerance` exists
to separate the two.

The load-bearing test here is the *default*: omitting the argument must reproduce
the historical `GOAL_DIS` behaviour exactly, so no existing route run changes.
"""
import numpy as np

from trajectory_following_ros2.utils.Trajectory import Trajectory


GOAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _traj(goal_tolerance=0.2):
    return Trajectory(goal_tolerance=goal_tolerance)


def test_default_uses_goal_dis_unchanged():
    """No goal_tolerance argument => the historical GOAL_DIS contract, exactly."""
    t = _traj(goal_tolerance=0.2)
    assert t.is_goal_reached(0.19, 0.0, 0.0, GOAL)
    assert not t.is_goal_reached(0.21, 0.0, 0.0, GOAL)
    # Explicitly passing None is the same as omitting it.
    assert t.is_goal_reached(0.19, 0.0, 0.0, GOAL, goal_tolerance=None)
    assert not t.is_goal_reached(0.21, 0.0, 0.0, GOAL, goal_tolerance=None)


def test_larger_goal_tolerance_latches_where_goal_dis_would_not():
    """The real case: the vehicle stops just outside the anchor distance."""
    t = _traj(goal_tolerance=0.2)
    # 0.22 m is the measured terminal distance of a run that held zero forever.
    assert not t.is_goal_reached(0.22, 0.0, 0.0, GOAL)
    assert t.is_goal_reached(0.22, 0.0, 0.0, GOAL, goal_tolerance=0.4)


def test_goal_tolerance_does_not_move_the_anchor():
    """Completion radius only -- GOAL_DIS still drives the reference lookahead."""
    t = _traj(goal_tolerance=0.2)
    t.is_goal_reached(0.22, 0.0, 0.0, GOAL, goal_tolerance=5.0)
    assert t.GOAL_DIS == 0.2


def test_stop_speed_contract_still_applies():
    """A loose radius must not let a moving vehicle report the goal."""
    t = _traj(goal_tolerance=0.2)
    fast = t.STOP_SPEED + 1.0
    assert not t.is_goal_reached(0.05, 0.0, fast, GOAL, goal_tolerance=2.0)
    assert t.is_goal_reached(0.05, 0.0, 0.0, GOAL, goal_tolerance=2.0)
