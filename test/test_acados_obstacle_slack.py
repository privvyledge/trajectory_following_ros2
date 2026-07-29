"""Obstacle slack penalties remain correctly shaped for every keep-out row."""
import numpy as np
import pytest

from trajectory_following_ros2.acados.acados_settings import obstacle_slack_penalties
from trajectory_following_ros2.coupled_kinematic_acados import (
    is_unsafe_obstacle_iterate, minimum_obstacle_clearance)


def test_quadratic_penalty_applies_only_to_lower_keepout_slack():
    penalties = obstacle_slack_penalties(4, linear_weight=1000.0,
                                         quadratic_weight=10000.0)

    np.testing.assert_array_equal(penalties['zl'], np.full(4, 1000.0))
    np.testing.assert_array_equal(penalties['Zl'], np.full(4, 10000.0))
    np.testing.assert_array_equal(penalties['zu'], np.zeros(4))
    np.testing.assert_array_equal(penalties['Zu'], np.zeros(4))


def test_minimum_obstacle_clearance_checks_every_disc_and_stage():
    x_sequence = np.zeros((4, 3))
    x_sequence[0] = [0.0, 0.5, 1.0]
    obstacle_states = np.array([
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.3, 0.3, 0.3],
    ])

    clearance = minimum_obstacle_clearance(
        x_sequence, obstacle_states, num_obstacles=1,
        ego_radius=0.2, safe_distance=0.1,
        ego_disc_offsets=np.array([0.0, 0.3]))

    assert clearance == pytest.approx(0.1)


def test_minimum_obstacle_clearance_is_unbounded_without_obstacles():
    assert minimum_obstacle_clearance(
        np.zeros((4, 2)), None, num_obstacles=0,
        ego_radius=0.2, safe_distance=0.1,
        ego_disc_offsets=np.array([0.0])) == float('inf')


def test_overlap_free_iterate_is_safe():
    assert not is_unsafe_obstacle_iterate(0.05, 0.05)


def test_iterate_predicting_new_overlap_is_unsafe():
    assert is_unsafe_obstacle_iterate(-0.01, 0.05)


def test_recovery_iterate_from_existing_overlap_is_accepted():
    """An already-overlapping stage 0 makes every plan 'predict overlap'; the
    plan that merely stays at the current depth (e.g. the reverse escape) must
    be accepted or the vehicle deadlocks in contact."""
    assert not is_unsafe_obstacle_iterate(-0.007, -0.007)


def test_iterate_pressing_deeper_into_overlap_is_still_rejected():
    assert is_unsafe_obstacle_iterate(-0.02, -0.007)
