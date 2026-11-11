"""CasADi adapter publishes input rates in SI units and reads the right layout.

Both CasADi formulations return ``u_rate`` as per-step increments
(``u_k - u_{k-1}``). The Ackermann ``jerk`` and ``steering_angle_velocity``
fields are rates, so the adapter must divide by the step. The Opti formulation
returns ``(N, nu)`` while the function-based one returns ``(nu, N)``.
"""
import numpy as np
import pytest

from trajectory_following_ros2.coupled_kinematic_casadi import CasAdiSolverAdapter

N = 5
TS = 0.05
D_ACC = [0.10, 0.20, 0.30, 0.40, 0.50]     # accel increments per stage
D_DELTA = [0.01, 0.02, 0.03, 0.04, 0.05]   # steering increments per stage


class _FakeController:
    def __init__(self, opti):
        self.Ts = TS
        self.horizon = N
        self._opti = opti

    def update(self, **kwargs):
        pass

    def solve(self):
        z = np.zeros((4, N + 1))
        u = np.zeros((2, N))
        rate = np.array([D_ACC, D_DELTA])
        if self._opti:
            z, u, rate = z.T, u.T, rate.T
        return {
            'z_mpc': z, 'u_mpc': u, 'u_control': [0.0, 0.0], 'u_rate': rate,
            'solver_status': True,
            'solver_stats': {'return_status': 'Solve_Succeeded'},
        }


@pytest.mark.parametrize('opti', [False, True])
def test_rates_are_stage0_increment_over_ts(opti):
    adapter = CasAdiSolverAdapter(_FakeController(opti), use_opti=opti,
                                  solver='ipopt')
    result = adapter.solve(np.zeros(4), np.zeros((4, N + 1)), np.zeros(2))
    assert result.jerk_cmd == pytest.approx(D_ACC[0] / TS)
    assert result.steering_rate_cmd == pytest.approx(D_DELTA[0] / TS)


def test_missing_u_rate_leaves_rates_unset():
    ctrl = _FakeController(False)
    base = ctrl.solve

    def solve_without_rate():
        sol = base()
        del sol['u_rate']
        return sol

    ctrl.solve = solve_without_rate
    adapter = CasAdiSolverAdapter(ctrl, use_opti=False, solver='ipopt')
    result = adapter.solve(np.zeros(4), np.zeros((4, N + 1)), np.zeros(2))
    assert result.jerk_cmd is None
    assert result.steering_rate_cmd is None
