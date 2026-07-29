"""Ego collision-disc geometry: covering the body without inflating one radius.

A single keep-out disc on the rear-axle reference point does not cover a body that
extends ahead of the rear axle: the constraint reports positive clearance while the
front corner is inside the obstacle. These tests pin the multi-disc geometry that
replaces it — the helper that resolves the offsets, the pose->disc-centre mapping, the
heading-dependent rear-axle stand-off, and the reference projection that has to agree
with the constraint or the tracked target is infeasible where it matters most.
"""
import math

import numpy as np
import pytest

from trajectory_following_ros2.utils import trajectory_utils as tu


# f1tenth footprint: 0.58 x 0.31 m, rear axle 0.19 m behind the body centre, so the
# body spans [-0.10, +0.48] m along x from the reference point. Two circles covering
# the half-rectangles: radius hypot(0.29, 0.31) / 2, centres at +0.045 and +0.335.
F1TENTH_OFFSETS = (0.045, 0.335)
F1TENTH_DISC_R = math.hypot(0.29, 0.31) / 2.0


class TestResolveOffsets:

    @pytest.mark.parametrize('value', [None, [], np.array([]), [float('nan')]])
    def test_degenerate_values_fall_back_to_the_single_reference_point_disc(self, value):
        assert tu.resolve_ego_disc_offsets(value).tolist() == [0.0]

    def test_offsets_are_returned_as_given(self):
        got = tu.resolve_ego_disc_offsets(F1TENTH_OFFSETS)
        assert got.tolist() == pytest.approx(list(F1TENTH_OFFSETS))


class TestDiscCentres:

    def test_single_zero_offset_reproduces_the_pose(self):
        centres = tu.ego_disc_centres(2.0, -3.0, 1.1, [0.0])
        assert centres.shape == (1, 1, 2)
        assert centres[0, 0].tolist() == pytest.approx([2.0, -3.0])

    def test_discs_lie_ahead_along_the_heading(self):
        centres = tu.ego_disc_centres(0.0, 0.0, math.pi / 2, [0.0, 0.5])
        assert centres[0, 0].tolist() == pytest.approx([0.0, 0.0])
        assert centres[0, 1].tolist() == pytest.approx([0.0, 0.5], abs=1e-12)

    def test_vectorizes_over_poses(self):
        x = np.array([0.0, 1.0, 2.0])
        centres = tu.ego_disc_centres(x, np.zeros(3), np.zeros(3), [0.0, 0.25])
        assert centres.shape == (3, 2, 2)
        assert centres[:, 1, 0].tolist() == pytest.approx([0.25, 1.25, 2.25])

    def test_front_disc_covers_the_footprint_corner_the_single_disc_missed(self):
        """The regression this whole change exists for.

        Vehicle at the keep-out boundary pointed straight at a 0.3 m obstacle, with
        the shipped single-disc numbers (ego_radius 0.15 + safe_distance 0.15).
        """
        keepout = 0.15 + 0.3 + 0.15
        obstacle = np.array([keepout, 0.0])

        # Single disc: the reference point is exactly on the boundary, "clear"...
        single = tu.ego_disc_centres(0.0, 0.0, 0.0, [0.0])[0]
        assert np.hypot(*(single[0] - obstacle)) - keepout == pytest.approx(0.0)
        # ...while the front corner of the body sits inside the obstacle itself.
        corner = np.array([0.48, 0.155])
        assert np.hypot(*(corner - obstacle)) < 0.3

        # Two discs: the front disc registers the intrusion the single disc missed.
        pair = tu.ego_disc_centres(0.0, 0.0, 0.0, F1TENTH_OFFSETS)[0]
        front_gap = np.hypot(*(pair[1] - obstacle)) - (F1TENTH_DISC_R + 0.3 + 0.15)
        assert front_gap < 0.0


class TestRearAxleStandOff:

    def test_single_disc_is_the_plain_keepout_radius(self):
        for alpha in (0.0, 1.0, math.pi):
            got = tu.rear_axle_keepout_radius(0.6, [0.0], alpha)
            assert got.tolist() == pytest.approx([0.6])

    def test_pointing_at_the_obstacle_needs_the_full_offset(self):
        # alpha = pi: heading opposes the outward radial, i.e. nose toward the centre.
        got = tu.rear_axle_keepout_radius(0.6, [0.0, 0.3], math.pi)
        assert got.tolist() == pytest.approx([0.9])

    def test_tangential_passage_needs_less_than_the_keepout(self):
        # A lone offset disc, so the reference-point disc does not set the maximum.
        got = tu.rear_axle_keepout_radius(0.6, [0.3], math.pi / 2)
        assert got.tolist() == pytest.approx([math.sqrt(0.6 ** 2 - 0.3 ** 2)])
        assert got[0] < 0.6, 'a tangential disc swings alongside, not into, the circle'

    def test_the_binding_disc_is_the_worst_one(self):
        # Adding the reference-point disc back re-raises the stand-off to its own.
        got = tu.rear_axle_keepout_radius(0.6, [0.0, 0.3], math.pi / 2)
        assert got.tolist() == pytest.approx([0.6])

    def test_pointing_away_lets_the_rear_axle_sit_closer(self):
        got = tu.rear_axle_keepout_radius(0.6, [0.0, 0.3], 0.0)
        assert got.tolist() == pytest.approx([0.6])  # the rear disc still binds

    def test_stand_off_actually_clears_every_disc(self):
        """Property check: place the vehicle at the returned stand-off and verify."""
        keepout, offsets = 0.62, F1TENTH_OFFSETS
        for alpha in np.linspace(-math.pi, math.pi, 37):
            rho = float(tu.rear_axle_keepout_radius(keepout, offsets, alpha)[0])
            # Obstacle at the origin, rear axle out along +x, heading at alpha to it.
            centres = tu.ego_disc_centres(rho, 0.0, alpha, offsets)[0]
            gaps = np.hypot(centres[:, 0], centres[:, 1]) - keepout
            assert gaps.min() > -1e-9, f'alpha={alpha}: disc inside by {gaps.min()}'
            assert gaps.min() < 1e-6, f'alpha={alpha}: over-conservative by {gaps.min()}'

    def test_unclearable_disc_falls_back_to_the_conservative_radius(self):
        # |d*sin(alpha)| > R: the disc sweeps through the circle at every stand-off.
        got = tu.rear_axle_keepout_radius(0.2, [0.0, 0.5], math.pi / 2)
        assert got.tolist() == pytest.approx([0.7])


class TestProjectionAgreesWithTheConstraint:

    @staticmethod
    def _straight_reference(n=25, spacing=0.1):
        """Reference driving +x straight through an obstacle at the origin."""
        xref = np.zeros((4, n))
        xref[0, :] = np.linspace(-1.2, -1.2 + spacing * (n - 1), n)
        xref[2, :] = 1.0
        return xref

    def test_single_disc_behaviour_is_unchanged(self):
        keepout = 0.6
        a, _, _ = tu.project_reference_out_of_keepouts(
            self._straight_reference(), [[0.0, 0.0]], [keepout])
        b, _, _ = tu.project_reference_out_of_keepouts(
            self._straight_reference(), [[0.0, 0.0]], [keepout], disc_offsets=[0.0])
        assert np.allclose(a, b)

    def test_projected_reference_is_feasible_for_every_disc(self):
        """The point of threading the offsets through: a target the constraint allows.

        With the discs offset, a reference parked on the single-disc circle is inside
        the multi-disc keep-out where the vehicle points at the obstacle — exactly the
        park-at-the-bubble-edge deadlock the projection exists to prevent.
        """
        keepout, margin = 0.6, 0.05
        xref, n_projected, _ = tu.project_reference_out_of_keepouts(
            self._straight_reference(), [[0.0, 0.0]], [keepout],
            margin=margin, disc_offsets=F1TENTH_OFFSETS)
        assert n_projected > 0

        centres = tu.ego_disc_centres(xref[0, :], xref[1, :], xref[3, :], F1TENTH_OFFSETS)
        gaps = np.hypot(centres[..., 0], centres[..., 1]) - (keepout + margin)
        # Millimetre tolerance: the recomputed yaw is the chord direction between
        # successive arc points, not the tangent, so an offset disc cuts the circle by
        # a discretization residual that shrinks with the reference spacing. This is
        # the scale of that artifact, not of the geometry.
        assert gaps.min() > -5e-3, f'projected reference still {-gaps.min():.4f} m inside'

    def test_single_disc_projection_would_have_been_infeasible(self):
        """Discrimination twin: without the offsets the same case fails the check."""
        keepout, margin = 0.6, 0.05
        xref, _, _ = tu.project_reference_out_of_keepouts(
            self._straight_reference(), [[0.0, 0.0]], [keepout], margin=margin)
        centres = tu.ego_disc_centres(xref[0, :], xref[1, :], xref[3, :], F1TENTH_OFFSETS)
        gaps = np.hypot(centres[..., 0], centres[..., 1]) - (keepout + margin)
        assert gaps.min() < -1e-3

    def test_detour_is_tighter_than_one_circumscribing_disc(self):
        """The reason for discs over a bigger radius: less lateral deviation."""
        keepout = 0.6
        disc, _, _ = tu.project_reference_out_of_keepouts(
            self._straight_reference(), [[0.0, 0.0]], [keepout],
            disc_offsets=F1TENTH_OFFSETS)
        # Circumscribing equivalent: one disc grown by the largest offset.
        circum, _, _ = tu.project_reference_out_of_keepouts(
            self._straight_reference(), [[0.0, 0.0]], [keepout + max(F1TENTH_OFFSETS)])
        assert np.abs(disc[1, :]).max() < np.abs(circum[1, :]).max()
