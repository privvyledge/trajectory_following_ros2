import numpy as np

from trajectory_following_ros2.utils.Trajectory import Trajectory
from trajectory_following_ros2.utils import trajectory_utils


def test_reference_fast_path_skips_auxiliary_geometry(monkeypatch):
    trajectory = Trajectory()
    trajectory.trajectory = np.zeros((4, len(trajectory.trajectory_keys)))
    trajectory.state[0, trajectory.state_key_to_column['x']] = 1.0

    monkeypatch.setattr(
        trajectory, 'calc_nearest_index',
        lambda **_kwargs: (np.array([2]), np.array([0.0])))

    expected = {'x_ref': np.array([2.0])}
    monkeypatch.setattr(
        trajectory_utils, 'generate_reference_trajectory_by_interpolation',
        lambda *_args, **_kwargs: expected)

    def unexpected(*_args, **_kwargs):
        raise AssertionError('auxiliary waypoint geometry should be skipped')

    monkeypatch.setattr(trajectory, 'get_next_waypoint', unexpected)
    monkeypatch.setattr(trajectory_utils, 'get_target_point', unexpected)

    result = trajectory.calc_ref_trajectory(
        state=None, trajectory=None, current_index=None,
        dt=0.05, prediction_horizon=20,
        compute_auxiliary_waypoints=False)

    assert result == (2, expected, None, None, None)
