"""Anchor-ratchet guard tests for the arc-length reference-index projection.

Covers ``trajectory_utils.project_index_and_lookahead``'s ``max_advance`` gate
and the along-track progress budget kept by ``Trajectory.calc_nearest_index``.

Scenario under test: a path that revisits the same physical neighbourhood on a
later pass (a loop driving the same corner more than once). A vehicle held
laterally off its local segment — an obstacle-avoidance standoff — makes a
later-pass point the nearest candidate inside the sliding projection window;
without the gate, the monotonic anchor ratchets forward tick after tick with no
real progress, cascades to the path tail, and reports a false end-of-path.

Fixture discipline: every "must not capture" fixture has a discrimination twin
proving it DOES capture when the gate is disabled, so a fixture that cannot
detect the ratchet never counts as coverage.
"""
import numpy as np
import pytest

from trajectory_following_ros2.utils import trajectory_utils
from trajectory_following_ros2.utils.Trajectory import Trajectory


def two_pass_path(step=0.02, length=2.0, separation=0.5):
    """Out-and-back path: pass1 along y=0, a U-turn, pass2 back along y=separation.

    Returns (waypoints (N,2), cum_dist (N,), pass2_start_index).
    Pass separation is small, so a vehicle between the passes sits nearer to
    pass2 than to its own pass1 segment — the capture geometry.
    """
    xs1 = np.arange(0.0, length + 1e-9, step)
    pass1 = np.stack([xs1, np.zeros_like(xs1)], axis=1)
    r = separation / 2.0
    angles = np.linspace(-np.pi / 2, np.pi / 2, 15)[1:-1]
    turn = np.stack([length + r * np.cos(angles), r + r * np.sin(angles)], axis=1)
    xs2 = np.arange(length - step, -1e-9, -step)
    pass2 = np.stack([xs2, np.full_like(xs2, separation)], axis=1)
    wp = np.concatenate([pass1, turn, pass2], axis=0)
    cum = trajectory_utils.cumulative_distance_along_path(wp)
    return wp, cum, len(pass1) + len(turn)


def project(wp, cum, pos, floor, **kw):
    return trajectory_utils.project_index_and_lookahead(
        wp, cum, np.array([pos], dtype=float), floor_index=floor, **kw)


class TestHelperGate:
    """project_index_and_lookahead with an explicit max_advance."""

    def setup_method(self):
        self.wp, self.cum, self.pass2_start = two_pass_path()
        # anchor mid-pass1 at x=1.0; vehicle displaced toward pass2
        self.floor = int(np.argmin(np.abs(self.wp[:self.pass2_start - 20, 0] - 1.0)))
        self.pos = (1.0, 0.35)  # 0.35 m from pass1, 0.15 m from pass2

    def test_ungated_projection_captures_later_pass(self):
        """Discrimination baseline: without the gate the fixture MUST leap."""
        _, _, proj, status = project(self.wp, self.cum, self.pos, self.floor,
                                     projection_window=5.0)
        assert proj >= self.pass2_start, (
            'fixture cannot detect the ratchet: ungated projection stayed local')

    def test_zero_budget_pins_the_anchor(self):
        _, _, proj, _ = project(self.wp, self.cum, self.pos, self.floor,
                                projection_window=5.0, max_advance=0.0)
        assert proj == self.floor

    def test_small_budget_stays_on_local_pass(self):
        _, _, proj, _ = project(self.wp, self.cum, self.pos, self.floor,
                                projection_window=5.0, max_advance=0.05)
        assert self.floor <= proj < self.pass2_start
        assert self.cum[proj] - self.cum[self.floor] <= 0.05 + 1e-9

    def test_escape_hatch_reacquires_after_far_jump(self):
        """Gate active + vehicle far beyond the window => ungated re-projection."""
        line = np.stack([np.arange(0.0, 30.0, 0.05), np.zeros(600)], axis=1)
        cum = trajectory_utils.cumulative_distance_along_path(line)
        floor = 40  # x = 2.0
        _, _, proj, status = project(line, cum, (9.0, 0.0), floor,
                                     projection_window=5.0, max_advance=0.0)
        assert status == 'ok'
        assert cum[proj] - cum[floor] > 4.0  # advanced across the window

    def test_escape_hatch_does_not_fire_in_capture_geometry(self):
        """A standoff vehicle is NEAR its local segment, so the escape must not
        re-open the ungated window there."""
        _, _, proj, _ = project(self.wp, self.cum, self.pos, self.floor,
                                projection_window=5.0, max_advance=0.0)
        assert proj == self.floor

    def test_lost_still_reported_when_nowhere_near_path(self):
        idx, _, proj, status = project(self.wp, self.cum, (100.0, 100.0), self.floor,
                                       projection_window=5.0, max_advance=0.0,
                                       max_search_radius=30.0)
        assert status == 'lost' and len(idx) == 0


class TestTrajectoryBudget:
    """End-to-end budget accounting through Trajectory.calc_nearest_index."""

    def make_traj(self):
        t = Trajectory(goal_tolerance=0.1)
        t.projection_window = 5.0
        return t

    def tick(self, t, wp, pos):
        t.calc_nearest_index(waypoints=wp, state=np.array([pos], dtype=float),
                             min_search_radius=0.2, max_search_radius=30.0)
        return t.previous_index

    def walk_to_standoff(self, t, wp):
        """Drive along pass1 to x=1.0, then displace laterally to the standoff."""
        for x in np.arange(0.0, 1.0 + 1e-9, 0.02):
            self.tick(t, wp, (x, 0.0))
        return self.tick(t, wp, (1.0, 0.35))

    def test_stationary_standoff_does_not_ratchet(self):
        wp, cum, pass2_start = two_pass_path()
        t = self.make_traj()
        anchor = self.walk_to_standoff(t, wp)
        assert anchor < pass2_start
        for _ in range(300):
            anchor = self.tick(t, wp, (1.0, 0.35))
        assert anchor < pass2_start, 'anchor captured the later pass while parked'
        assert cum[anchor] - 1.0 < 0.5, 'anchor crept despite zero progress'

    def test_stationary_standoff_ratchets_when_gate_disabled(self):
        """Discrimination twin: same fixture MUST capture without the gate."""
        wp, _, pass2_start = two_pass_path()
        t = self.make_traj()
        self.walk_to_standoff(t, wp)
        for _ in range(5):
            t._previous_projection_position = None  # gate off: ungated every tick
            anchor = self.tick(t, wp, (1.0, 0.35))
        assert anchor >= pass2_start, (
            'fixture cannot detect the ratchet: ungated ticks stayed local')

    def test_thrashing_standoff_does_not_ratchet(self):
        """Oscillation must net to ~zero budget (signed along-track accrual)."""
        wp, cum, pass2_start = two_pass_path()
        t = self.make_traj()
        self.walk_to_standoff(t, wp)
        for k in range(300):
            x = 1.0 + (0.05 if k % 2 == 0 else -0.05)
            anchor = self.tick(t, wp, (x, 0.35))
        assert anchor < pass2_start
        # bounded creep: at most the banked budget cap (~0.5 m) plus slack,
        # nowhere near the ~1.76 m of arc to the later pass
        assert cum[anchor] - 1.0 < 1.0

    def test_offline_forward_progress_still_advances(self):
        """bug-090 non-regression: lateral offset must not freeze the index."""
        wp, cum, pass2_start = two_pass_path()
        t = self.make_traj()
        self.tick(t, wp, (0.0, 0.0))
        for x in np.arange(0.02, 1.5 + 1e-9, 0.02):
            anchor = self.tick(t, wp, (x, 0.35))  # held 0.35 m off the line
        assert anchor < pass2_start
        assert abs(cum[anchor] - 1.5) < 0.3, 'anchor did not track off-line progress'

    def test_far_jump_reacquires_and_catches_up(self):
        line = np.stack([np.arange(0.0, 30.0, 0.05), np.zeros(600)], axis=1)
        cum = trajectory_utils.cumulative_distance_along_path(line)
        t = self.make_traj()
        for x in np.arange(0.0, 2.0 + 1e-9, 0.05):
            self.tick(t, line, (x, 0.0))
        anchor = self.tick(t, line, (9.0, 0.0))  # 7 m jump, beyond the window
        assert cum[anchor] > 6.0, 'escape hatch did not re-acquire'
        x = 9.0
        for _ in range(200):
            x += 0.05
            anchor = self.tick(t, line, (x, 0.0))
        assert abs(cum[anchor] - x) < 0.5, 'anchor never caught back up after the jump'

    def test_end_of_path_still_fires_at_the_tail(self):
        line = np.stack([np.arange(0.0, 5.0, 0.05), np.zeros(100)], axis=1)
        t = self.make_traj()
        status = None
        for x in np.arange(0.0, 4.95 + 1e-9, 0.05):
            self.tick(t, line, (x, 0.0))
            status = t.projection_status
        assert status == 'end_of_path'

    def test_reset_progress_clears_budget_state(self):
        wp, _, _ = two_pass_path()
        t = self.make_traj()
        self.walk_to_standoff(t, wp)
        t.reset_progress()
        assert t._anchor_advance_budget == 0.0
        assert t._previous_projection_position is None
        assert self.tick(t, wp, (0.0, 0.0)) < 5  # ungated re-acquisition at start


class TestLocalPathTangent:
    def test_forward_tangent(self):
        wp = np.stack([np.arange(0.0, 1.0, 0.1), np.zeros(10)], axis=1)
        tangent = trajectory_utils.local_path_tangent(wp, 3)
        assert np.allclose(tangent, [1.0, 0.0])

    def test_duplicate_pile_falls_back_to_backward_scan(self):
        wp = np.concatenate([
            np.stack([np.arange(0.0, 1.0, 0.1), np.zeros(10)], axis=1),
            np.tile([[1.0, 0.0]], (30, 1))])
        tangent = trajectory_utils.local_path_tangent(wp, 15)
        assert np.allclose(tangent, [1.0, 0.0])

    def test_all_duplicates_returns_none(self):
        wp = np.tile([[1.0, 2.0]], (50, 1))
        assert trajectory_utils.local_path_tangent(wp, 25) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
