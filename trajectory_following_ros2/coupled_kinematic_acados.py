import math
import os
import sys
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from ament_index_python.packages import get_package_share_directory

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker
from trajectory_following_ros2.backends.base_solver import BaseSolver, SolverResult
from trajectory_following_ros2.acados.acados_settings import acados_settings


class AcadosSolverAdapter(BaseSolver):
    """Wraps the acados OCP solver to conform to BaseSolver."""

    def __init__(self, controller, horizon: int, wheelbase: float,
                 stage_cost_type: str = 'NONLINEAR_LS',
                 terminal_cost_type: str = 'NONLINEAR_LS'):
        self._controller = controller
        self._horizon = horizon
        self._wheelbase = wheelbase
        self._stage_cost_type = stage_cost_type
        self._terminal_cost_type = terminal_cost_type

    def initialize(self, x0: np.ndarray) -> None:
        for i in range(self._horizon + 1):
            self._controller.set(i, 'x', x0)
        for i in range(self._horizon):
            self._controller.set(i, 'u', np.zeros(2))
        self._controller.constraints_set(0, 'lbx', x0)
        self._controller.constraints_set(0, 'ubx', x0)
        self._controller.solve()  # initial guess

    def solve(self, x0: np.ndarray, xref: np.ndarray,
              u_prev: np.ndarray) -> SolverResult:
        # x0: (4,); xref: (4, N+1); u_prev: (2,)
        self._controller.constraints_set(0, 'lbx', x0)
        self._controller.constraints_set(0, 'ubx', x0)

        for j in range(self._horizon):
            yref = xref[:, j]
            if self._stage_cost_type in ('LINEAR_LS', 'NONLINEAR_LS'):
                self._controller.cost_set(j, 'yref',
                                          np.hstack([yref, np.zeros(2)]))
            self._controller.set(j, 'p', np.array([
                self._wheelbase, *yref, *np.zeros(2), *x0, *u_prev]))

        yref_N = xref[:, self._horizon]
        if self._terminal_cost_type in ('LINEAR_LS', 'NONLINEAR_LS'):
            self._controller.set(self._horizon, 'yref', yref_N)
        self._controller.set(self._horizon, 'p', np.array([
            self._wheelbase, *yref_N, *np.zeros(2), *x0, *u_prev]))

        t0 = time.process_time()
        status = self._controller.solve()
        solve_time_cpu = time.process_time() - t0

        u = self._controller.get(0, 'u')
        x_seq = np.array(
            [self._controller.get(i, 'x') for i in range(self._horizon + 1)]).T  # (4, N+1)
        u_seq = np.array(
            [self._controller.get(i, 'u') for i in range(self._horizon)]).T       # (2, N)

        try:
            solve_time = float(self._controller.get_stats('time_tot') or solve_time_cpu)
        except Exception:
            solve_time = solve_time_cpu

        return SolverResult(
            accel_cmd=float(u[0]),
            steering_cmd=float(u[1]),
            velocity_cmd=float(x_seq[2, 1]),
            u_sequence=u_seq,
            x_sequence=x_seq,
            u_prev=u.copy(),
            is_optimal=(status == 0),
            solve_time=solve_time,
            status=str(status),
        )

    @property
    def nx(self) -> int:
        return 4

    @property
    def nu(self) -> int:
        return 2


class KinematicCoupledAcados(BaseTrajectoryTracker):

    def __init__(self):
        super().__init__('kinematic_coupled_acados_controller')
        self.get_logger().info('kinematic_coupled_acados_controller started.')

    def _declare_backend_parameters(self):
        self.declare_parameter('ode_type', 'continuous_kinematic_coupled')
        self.declare_parameter('stage_cost_type', 'NONLINEAR_LS')
        self.declare_parameter('terminal_cost_type', 'NONLINEAR_LS')
        self.declare_parameter('max_iter', 15)
        self.declare_parameter('termination_condition', 1e-6)
        self.declare_parameter('scale_cost', False)
        self.declare_parameter('generate_mpc_model', True)
        self.declare_parameter('build_with_cython', True)
        self.declare_parameter('model_directory',
                               os.path.join(
                                   get_package_share_directory('trajectory_following_ros2'),
                                   'data', 'model'))

    def _init_solver(self) -> Optional[BaseSolver]:
        stage_cost_type = self.get_parameter('stage_cost_type').value
        terminal_cost_type = self.get_parameter('terminal_cost_type').value
        max_iter = int(self.get_parameter('max_iter').value)
        tol = self.get_parameter('termination_condition').value
        scale_cost = self.get_parameter('scale_cost').value
        generate = self.get_parameter('generate_mpc_model').value
        with_cython = self.get_parameter('build_with_cython').value
        model_dir = self.get_parameter('model_directory').value

        build_path = os.path.join(model_dir, 'c_generated_code')
        config_path = os.path.join(model_dir, 'kinematic_bicycle_acados_ocp.json')
        os.makedirs(build_path, exist_ok=True)

        cwd = os.getcwd()
        os.chdir(build_path)

        _, _, controller, _ = acados_settings(
            Tf=self.prediction_time,
            N=self.horizon,
            x0=self.zk,
            scale_cost=scale_cost,
            Q=self.Q, R=self.R, Qe=self.Qf, Rd=self.Rd,
            wheelbase=self.WHEELBASE,
            cost_module=stage_cost_type,
            cost_module_e=terminal_cost_type,
            generate=True,
            build=generate,
            with_cython=with_cython,
            num_iterations=max_iter,
            tolerance=tol,
            mpc_config_file=config_path,
            code_export_directory=build_path,
        )

        os.chdir(cwd)

        self._stage_cost_type = stage_cost_type
        self._terminal_cost_type = terminal_cost_type

        self.get_logger().info('acados MPC built.')
        return AcadosSolverAdapter(
            controller,
            horizon=self.horizon,
            wheelbase=self.WHEELBASE,
            stage_cost_type=stage_cost_type,
            terminal_cost_type=terminal_cost_type,
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = KinematicCoupledAcados()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
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
