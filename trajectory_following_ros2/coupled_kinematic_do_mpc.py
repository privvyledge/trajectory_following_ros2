import sys
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker, _make_executor
from trajectory_following_ros2.backends.base_solver import BaseSolver, SolverResult
from trajectory_following_ros2.do_mpc.mpc_setup import initialize_mpc_problem


class DoMpcSolverAdapter(BaseSolver):
    """Wraps the do-mpc Controller to conform to BaseSolver."""

    def __init__(self, controller):
        self._controller = controller

    def initialize(self, x0: np.ndarray) -> None:
        self._controller.mpc.x0 = x0.reshape(-1, 1)
        self._controller.mpc.set_initial_guess()

    def set_weights(self, Q: np.ndarray, R: np.ndarray,
                    Rd: np.ndarray, Qf: np.ndarray) -> None:
        self._controller.Q_diag_value = np.diag(Q)
        self._controller.R_diag_value = np.diag(R)
        self._controller.Rd_diag_value = np.diag(Rd)
        self._controller.Qf_diag_value = np.diag(Qf)

    def solve(self, x0: np.ndarray, xref: np.ndarray,
              u_prev: np.ndarray) -> SolverResult:
        # xref: (4, N+1) → reference_states expects (N+1, 4)
        self._controller.reference_states[:, :] = xref.T

        u = self._controller.get_control(x0.reshape(-1, 1))  # (2, 1)

        # retrieve predicted trajectories
        pred_x = self._controller.mpc.data.prediction(('_x', 'pos_x'), t_ind=-1)[0]  # (N+1, 1)
        pred_y = self._controller.mpc.data.prediction(('_x', 'pos_y'), t_ind=-1)[0]
        pred_vel = self._controller.mpc.data.prediction(('_x', 'vel'), t_ind=-1)[0]
        pred_psi = self._controller.mpc.data.prediction(('_x', 'psi'), t_ind=-1)[0]
        pred_acc = self._controller.mpc.data.prediction(('_u', 'acc'), t_ind=-1)[0]    # (N, 1)
        pred_delta = self._controller.mpc.data.prediction(('_u', 'delta'), t_ind=-1)[0]

        x_seq = np.hstack([pred_x, pred_y, pred_vel, pred_psi]).T       # (4, N+1)
        u_seq = np.hstack([pred_acc, pred_delta]).T                       # (2, N)

        N = self._controller.mpc.n_horizon
        k_next = min(1, N - 1)
        vel_next = float(pred_vel[k_next, 0])

        acc_cmd = float(u[0, 0])
        delta_cmd = float(u[1, 0])

        stats = self._controller.mpc.solver_stats
        is_optimal = bool(stats.get('success', False))
        solve_time = float(stats.get('t_wall_total', 0.0))

        return SolverResult(
            accel_cmd=acc_cmd,
            steering_cmd=delta_cmd,
            velocity_cmd=vel_next,
            u_sequence=u_seq,
            x_sequence=x_seq,
            u_prev=np.array([acc_cmd, delta_cmd]),
            is_optimal=is_optimal,
            solve_time=solve_time,
            status='optimal' if is_optimal else 'infeasible',
        )

    @property
    def nx(self) -> int:
        return 4

    @property
    def nu(self) -> int:
        return 2


class KinematicCoupledDoMpc(BaseTrajectoryTracker):

    def __init__(self):
        super().__init__('kinematic_coupled_do_mpc_controller')
        self.get_logger().info('kinematic_coupled_do_mpc_controller started.')

    def _declare_backend_parameters(self):
        self.declare_parameter('ode_type', 'continuous_kinematic_coupled')
        self.declare_parameter('max_iter', 300)
        self.declare_parameter('termination_condition', 1e-6)
        self.declare_parameter('generate_mpc_model', False)

    def _init_solver(self) -> Optional[BaseSolver]:
        ode_type = self.get_parameter('ode_type').value
        max_iter = int(self.get_parameter('max_iter').value)
        tol = self.get_parameter('termination_condition').value
        compile_model = self.get_parameter('generate_mpc_model').value
        model_type = 'continuous' if 'continuous' in ode_type else 'discrete'

        _, controller, _ = initialize_mpc_problem(
            reference_path=None,
            horizon=self.horizon,
            sample_time=self.sample_time,
            Q=self.Q, R=self.R, Qf=self.Qf, Rd=self.Rd,
            wheelbase=self.WHEELBASE,
            delta_min=np.degrees(self.MIN_STEER_ANGLE),
            delta_max=np.degrees(self.MAX_STEER_ANGLE),
            vel_min=self.MIN_SPEED,
            vel_max=self.MAX_SPEED,
            ay_max=4.0,
            acc_min=self.MAX_DECEL,
            acc_max=self.MAX_ACCEL,
            max_iterations=max_iter,
            tolerance=tol,
            suppress_ipopt_output=True,
            model_type=model_type,
            warmstart=True,
            quad_prog_mode=True,
            compile_model=compile_model,
            jit_compilation=False,
        )

        self.get_logger().info(f'do_mpc MPC built (model_type={model_type}).')
        return DoMpcSolverAdapter(controller)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = KinematicCoupledDoMpc()
        executor = _make_executor(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
