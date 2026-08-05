"""Acados adapter failure-capture and reset tests using a deterministic fake solver."""
import numpy as np

from trajectory_following_ros2.coupled_kinematic_acados import AcadosSolverAdapter


class _FakeController:
    def __init__(self, horizon=2, status=4):
        self.horizon = horizon
        self.status = status
        self.constraints = {}
        self.states = {}
        self.inputs = {}
        self.params = {}
        self.reset_calls = 0

    def constraints_set(self, stage, field, value):
        self.constraints[(stage, field)] = np.asarray(value, dtype=float).copy()

    def cost_set(self, stage, field, value):
        pass

    def set(self, stage, field, value):  # noqa: A003 - mirrors acados solver API
        value = np.asarray(value, dtype=float).copy()
        if field == 'x':
            self.states[stage] = value
        elif field == 'u':
            self.inputs[stage] = value
        elif field == 'p':
            self.params[stage] = value

    def solve(self):
        return self.status

    def get(self, stage, field):
        if field == 'x':
            return self.states.get(stage, np.array([stage, 0.0, 0.5, 0.0]))
        if field == 'u':
            return self.inputs.get(stage, np.array([0.2, 0.1]))
        raise KeyError(field)

    def get_stats(self, name):
        return {
            'time_tot': 0.001,
            'residuals': np.array([1.0, 2.0, 3.0, 4.0]),
            'qp_stat': np.array([3]),
            'sqp_iter': 1,
        }[name]

    def reset(self):
        self.reset_calls += 1


def _adapter(controller, failure_dump_file='', rate_max=None):
    return AcadosSolverAdapter(
        controller,
        horizon=controller.horizon,
        wheelbase=0.256,
        stage_cost_type='EXTERNAL',
        terminal_cost_type='EXTERNAL',
        dt=0.05,
        u_min=np.array([-3.0, -0.5]),
        u_max=np.array([3.0, 0.5]),
        rate_max=rate_max,
        failure_dump_file=failure_dump_file,
    )


def test_stage_zero_rate_box_is_recorded_for_failure_replay():
    controller = _FakeController()
    adapter = _adapter(controller, rate_max=np.array([np.inf, 1.0]))

    adapter._apply_input_rate_bound(np.array([10.0, 0.4]))

    np.testing.assert_allclose(adapter._stage0_lbu, [-3.0, 0.35])
    np.testing.assert_allclose(adapter._stage0_ubu, [3.0, 0.45])
    np.testing.assert_array_equal(
        controller.constraints[(0, 'lbu')], adapter._stage0_lbu)
    np.testing.assert_array_equal(
        controller.constraints[(0, 'ubu')], adapter._stage0_ubu)


def test_hard_failure_dump_is_one_shot_and_precedes_clean_reseed(tmp_path):
    controller = _FakeController()
    dump_path = tmp_path / 'first_failure.npz'
    adapter = _adapter(
        controller,
        failure_dump_file=str(dump_path),
        rate_max=np.array([np.inf, 1.0]),
    )
    x0 = np.array([1.0, 2.0, 0.5, 0.1])
    xref = np.array([
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [0.5, 0.4, 0.3],
        [0.1, 0.2, 0.3],
    ])
    u_prev = np.array([0.2, 0.4])

    result = adapter.solve(x0, xref, u_prev)

    assert not result.is_optimal
    assert controller.reset_calls == 1
    assert dump_path.exists()
    with np.load(dump_path) as dump:
        np.testing.assert_array_equal(dump['x0'], x0)
        np.testing.assert_array_equal(dump['xref'], xref)
        np.testing.assert_array_equal(dump['u_prev'], u_prev)
        np.testing.assert_allclose(dump['stage0_lbu'], [-3.0, 0.35])
        np.testing.assert_allclose(dump['stage0_ubu'], [3.0, 0.45])
        np.testing.assert_array_equal(dump['qp_stat'], [3])
        assert dump['status'][0] == '4'

    # Recovery still performs the clean reset/reseed after the dump.
    np.testing.assert_array_equal(controller.states[0], x0)
    np.testing.assert_array_equal(controller.states[1], x0)
    np.testing.assert_array_equal(controller.states[2], x0)
    np.testing.assert_array_equal(controller.inputs[0], np.zeros(2))
    np.testing.assert_array_equal(controller.inputs[1], np.zeros(2))

    # A later failure must not overwrite the first-failure fixture.
    adapter.solve(x0 + 10.0, xref + 10.0, u_prev)
    with np.load(dump_path) as dump:
        np.testing.assert_array_equal(dump['x0'], x0)


def test_recovery_seed_is_dynamically_consistent_not_the_reference():
    """The post-failure re-seed must not snap the horizon onto the reference.

    The reference is anchored `distance_tolerance` metres ahead of the vehicle and
    is pushed aside by the keep-out projection, so seeding stages 1..N from it puts
    a metre-scale jump between stage 0 and stage 1 -- a step the dynamics cannot
    take in one `dt`. The first QP then fails on that seed and the adapter installs
    the same seed again, latching the failure permanently (the vehicle freezes with
    metres of clearance). Seeding every stage at the current state keeps the
    dynamics residual at ~0 and lets the next solve recover.
    """
    controller = _FakeController(horizon=3)
    adapter = _adapter(controller)

    x0 = np.array([1.0, 2.0, 6.9, 0.1])
    # A reference anchored ~2.5 m ahead and offset laterally, as the live one is.
    xref = np.array([
        [3.4, 3.8, 4.2, 4.6],
        [3.1, 3.5, 3.9, 4.3],
        [7.7, 7.9, 8.2, 8.2],
        [0.3, 0.3, 0.3, 0.3],
    ])

    adapter._recover_from_failure(x0, xref)

    for stage in range(controller.horizon + 1):
        np.testing.assert_array_equal(controller.states[stage], x0)
    # The seed must be dynamically benign: no stage-to-stage position jump.
    steps = [
        np.linalg.norm(controller.states[s + 1][:2] - controller.states[s][:2])
        for s in range(controller.horizon)
    ]
    assert max(steps) == 0.0
    for stage in range(controller.horizon):
        np.testing.assert_array_equal(controller.inputs[stage], np.zeros(2))
