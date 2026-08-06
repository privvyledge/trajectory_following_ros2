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


def noisy_start_path(pile=200, jitter=0.001, creep=0.2, step=0.02, length=5.0, seed=0):
    """A recorded route that begins barely-moving: a creeping, jittering start.

    Models how a real recording opens — the logger is armed before the vehicle is
    really underway, so the first stretch of route inches forward (~1 mm per point)
    while sensor noise of the same order scatters each sample. The per-point drift
    and the jitter being comparable is the whole point: it makes the offset between
    neighbouring points noise rather than heading, while the stretch still lies
    along the route the vehicle drives, so its points stay the nearest candidates
    as the vehicle sets off. Returns (waypoints (N,2), cum_dist (N,), route_start).
    """
    rng = np.random.default_rng(seed)
    drift = np.stack([np.linspace(0.0, creep, pile), np.zeros(pile)], axis=1)
    parked = drift + rng.normal(0.0, jitter, size=(pile, 2))
    xs = np.arange(creep + step, length + 1e-9, step)
    route = np.stack([xs, np.zeros_like(xs)], axis=1)
    wp = np.concatenate([parked, route], axis=0)
    return wp, trajectory_utils.cumulative_distance_along_path(wp), pile


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

    def test_noisy_parked_start_pulls_away(self):
        """A route that opens parked must not strand the anchor inside the pile.

        No local direction exists in the pile, so no along-track progress can be
        measured there; gating on it anyway pins the anchor and the vehicle drives
        past a reference that never advances.
        """
        wp, cum, route_start = noisy_start_path()
        t = self.make_traj()
        anchor = self.tick(t, wp, (0.0, 0.0))
        for x in np.arange(0.04, 1.0 + 1e-9, 0.04):
            anchor = self.tick(t, wp, (x, 0.0))
        assert anchor >= route_start, 'anchor stranded in the parked start pile'
        assert abs(cum[anchor] - cum[route_start] - 1.0) < 0.3, (
            'anchor did not track progress along the route')

    def test_gating_resumes_after_the_parked_start(self):
        """Crossing the pile ungated must not hand gating a drained budget.

        Booking the pile crossing as consumption would leave the budget deeply
        negative, re-freezing the anchor once the direction becomes well defined.
        """
        wp, _, route_start = noisy_start_path()
        t = self.make_traj()
        for x in np.arange(0.0, 1.0 + 1e-9, 0.04):
            self.tick(t, wp, (x, 0.0))
        assert t._anchor_advance_budget >= 0.0, 'budget arrived drained past the pile'
        anchor_before = t.previous_index
        for _ in range(50):  # now parked on a well-defined stretch: must not ratchet
            anchor = self.tick(t, wp, (1.0, 0.0))
        assert anchor - anchor_before < 20, 'gate did not resume after the pile'

    def test_offline_forward_progress_still_advances(self):
        """Non-regression: lateral offset must not freeze the index."""
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


class TestWideGapIsCrossable:
    """A waypoint gap wider than the advance budget must not freeze the anchor.

    The budget is metres and what it gates is an index step, so before the
    forced-successor invariant a single wide gap made the window degenerate to
    {floor_index} and the reference index stuck there for the rest of the run --
    silently, with status 'ok'. Seen live on a recorded figure-8 carrying one
    0.555 m dropout among 0.045 m spacing: three closed-loop runs with different
    steering limits all stopped at exactly that waypoint.
    """

    @staticmethod
    def gapped_path(gap=0.555, spacing=0.045, n=40):
        xs = np.arange(n, dtype=float) * spacing
        xs[n // 2:] += gap - spacing          # one wide segment, mid-path
        return np.stack([xs, np.zeros(n)], axis=1)

    def test_anchor_crosses_a_gap_wider_than_the_budget(self):
        wp = self.gapped_path()
        cum = trajectory_utils.cumulative_distance_along_path(wp)
        floor = len(wp) // 2 - 1
        assert cum[floor + 1] - cum[floor] > 0.5, 'fixture gap must exceed the budget'
        # Vehicle sitting on the far side of the gap, budget at its +0.5 m ceiling.
        _, _, proj, status = trajectory_utils.project_index_and_lookahead(
            wp, cum, wp[floor + 1:floor + 2], floor_index=floor,
            lookahead_distance=0.2, projection_window=5.0, max_advance=0.5)
        assert status == 'ok'
        assert proj == floor + 1, 'anchor must be able to step across the gap'

    def test_spent_budget_still_admits_only_one_waypoint(self):
        """The invariant must not become a hole in the anchor-ratchet guard.

        With the budget fully spent the window may offer the successor and
        nothing beyond it, so a vehicle that is not making progress cannot
        ratchet forward through a later pass of the path.
        """
        wp = np.stack([np.arange(0.0, 4.0, 0.045), np.zeros(89)], axis=1)
        cum = trajectory_utils.cumulative_distance_along_path(wp)
        # Vehicle far down the path, but no budget: it may advance one step at most.
        _, _, proj, _ = trajectory_utils.project_index_and_lookahead(
            wp, cum, wp[60:61], floor_index=10, lookahead_distance=0.1,
            projection_window=5.0, max_advance=0.0)
        assert proj == 11, f'expected a single-waypoint step, advanced to {proj}'

    def test_a_dense_path_is_unaffected_by_the_invariant(self):
        """Discrimination twin: with no gap the budget alone still decides."""
        wp = np.stack([np.arange(0.0, 4.0, 0.045), np.zeros(89)], axis=1)
        cum = trajectory_utils.cumulative_distance_along_path(wp)
        _, _, proj, _ = trajectory_utils.project_index_and_lookahead(
            wp, cum, wp[60:61], floor_index=10, lookahead_distance=0.1,
            projection_window=5.0, max_advance=0.5)
        # 0.5 m of budget at 0.045 m spacing is ~11 waypoints, not the whole way.
        assert 11 < proj < 30, f'budget should bound the advance, got {proj}'


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

    def test_noise_pile_returns_none(self):
        """A recorded pile jitters rather than repeating: still no direction."""
        wp, _, _ = noisy_start_path()
        assert trajectory_utils.local_path_tangent(wp, 100) is None

    def test_noise_pile_reports_a_false_direction_without_a_baseline(self):
        """Discrimination twin: mm jitter clears a numerical-zero threshold easily.

        Proves the pile fixture detects the failure — with the baseline removed the
        tangent is confident and arbitrary, rather than the route's +x direction.
        """
        wp, _, _ = noisy_start_path()
        tangent = trajectory_utils.local_path_tangent(wp, 100, min_baseline=1e-9)
        assert tangent is not None
        assert abs(float(tangent @ np.array([1.0, 0.0]))) < 0.9, (
            'fixture cannot detect the failure: jitter happened to align with the route')


def reverse_cusp_trajectory(step=0.05, out_length=2.0, back_to=0.2, offset=0.02,
                            fwd_speed=0.9, rev_speed=-0.25):
    """A route driven out, then backed up along (nearly) the same line.

    Models the tail of the recorded gosling1 routes: the driver stops and reverses,
    so the return branch retraces the outbound one a couple of centimetres away and
    runs the opposite direction, and the only thing distinguishing the two is the
    sign of the recorded speed -- the yaw is continuous across the cusp, because a
    reversing vehicle does not turn around.

    Returns a full (N, 9) trajectory array in ``Trajectory``'s column order, so the
    signed speed column is populated the way the live path has it.
    """
    xs_out = np.arange(0.0, out_length + 1e-9, step)
    out = np.stack([xs_out, np.zeros_like(xs_out)], axis=1)
    xs_back = np.arange(out_length - step, back_to - 1e-9, -step)
    back = np.stack([xs_back, np.full_like(xs_back, offset)], axis=1)
    wp = np.concatenate([out, back], axis=0)

    traj = np.zeros((len(wp), 9), dtype=float)
    traj[:, 0], traj[:, 1] = wp[:, 0], wp[:, 1]
    traj[:len(out), 2] = fwd_speed
    traj[len(out):, 2] = rev_speed
    traj[:, 6] = trajectory_utils.cumulative_distance_along_path(wp)
    return traj, len(out)


class TestReverseCuspEscape:
    """The counter-branch escape: crossing a reverse cusp without stranding the anchor.

    At a cusp the anchor sits in a local distance MINIMUM on the outbound branch --
    every index between it and the return branch is farther from the vehicle, so the
    nearest-in-window projection will not walk across, and the arc that must be
    crossed costs about twice the displacement that funds it. Left alone the anchor
    freezes for the rest of the run, silently: status stays 'ok', every solve
    optimal, no watchdog trips, and the vehicle parks partway down its own tail.
    """

    def make_traj(self, traj):
        t = Trajectory(goal_tolerance=0.3)
        t.trajectory = traj
        t.projection_window = 5.0
        t.arclength_index_advance = True
        return t

    def tick(self, t, pos):
        t.state[0, 0], t.state[0, 1] = pos
        indices, _ = t.calc_nearest_index(min_search_radius=0.3, max_search_radius=30.0)
        return t.previous_index, (int(indices[0]) if len(indices) else None)

    def drive(self, t, positions):
        anchor = target = None
        for pos in positions:
            anchor, target = self.tick(t, pos)
        return anchor, target

    def out_and_back_positions(self, turnaround=1.6, back_to=0.8, offset=0.01, step=0.0125):
        """Drive out, stop SHORT of the route's own cusp, then reverse back beside it.

        Turning around short of the recorded cusp is what builds the barrier, and it
        is what the real runs do (the measured leg turned ~0.4 m early). It leaves the
        anchor on the outbound branch with the whole apex between it and the return
        branch: every index in between is farther from the retreating vehicle than the
        anchor already is, so the projection cannot walk across, and the arc it would
        have to jump is roughly twice the displacement that funds it.
        """
        out = [(x, 0.0) for x in np.arange(0.0, turnaround + 1e-9, step)]
        back = [(x, offset) for x in np.arange(turnaround, back_to - 1e-9, -step)]
        return out + back

    def test_anchor_crosses_the_cusp(self):
        traj, cusp = reverse_cusp_trajectory()
        t = self.make_traj(traj)
        anchor, target = self.drive(t, self.out_and_back_positions())
        assert anchor > cusp, (
            'anchor stranded on the outbound branch at %d (cusp %d)' % (anchor, cusp))
        assert target is not None and target > cusp, (
            'reference target %s never reached the reverse branch' % (target,))
        assert t.counter_branch_escapes == 1, (
            'expected exactly one escape at the one cusp, got %d'
            % (t.counter_branch_escapes,))

    def test_anchor_strands_without_the_escape(self):
        """Discrimination twin: the fixture MUST strand when the escape cannot fire.

        Raising the movement threshold out of reach disables only the escape, leaving
        the repaired budget accrual in place -- so this pins the escape specifically,
        not the accrual fix that precedes it.
        """
        traj, cusp = reverse_cusp_trajectory()
        t = self.make_traj(traj)
        t.COUNTER_BRANCH_MIN_STEP = 1e9
        anchor, _ = self.drive(t, self.out_and_back_positions())
        assert anchor <= cusp, (
            'fixture cannot detect the failure: anchor crossed at %d without the escape'
            % (anchor,))

    def test_parked_short_of_the_cusp_does_not_escape(self):
        """A stationary vehicle must never escape, cusp in the route or not."""
        traj, cusp = reverse_cusp_trajectory()
        t = self.make_traj(traj)
        self.drive(t, [(x, 0.0) for x in np.arange(0.0, 1.5 + 1e-9, 0.0125)])
        parked = t.previous_index
        assert parked <= cusp
        for _ in range(300):
            anchor, _ = self.tick(t, (1.5, 0.0))
        assert anchor <= cusp, 'parked vehicle escaped across the cusp'
        assert anchor - parked <= 1, 'parked vehicle crept along the route'
        assert t.counter_branch_escapes == 0

    def test_forward_only_uturn_does_not_escape(self):
        """The speed-sign gate: same doubling-back geometry, no recorded reversal.

        A tight U-turn driven entirely forwards produces counter-running branches
        centimetres apart -- the tangent test alone cannot tell it from a cusp. Only
        the recorded speed can, and here it never changes sign, so an oscillating
        vehicle between the branches must stay held by the ratchet guard.
        """
        traj, cusp = reverse_cusp_trajectory()
        traj[:, 2] = np.abs(traj[:, 2])  # same path, all of it driven forwards
        t = self.make_traj(traj)
        self.drive(t, [(x, 0.0) for x in np.arange(0.0, 1.5 + 1e-9, 0.0125)])
        for k in range(300):
            anchor, _ = self.tick(t, (1.5 + (0.05 if k % 2 else -0.05), 0.01))
        assert anchor <= cusp, (
            'oscillation escaped across a forward-only U-turn at %d' % (anchor,))
        assert t.counter_branch_escapes == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
