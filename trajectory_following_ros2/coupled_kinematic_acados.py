import os
import sys
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker, _make_executor
from trajectory_following_ros2.backends.base_solver import BaseSolver, SolverResult
from trajectory_following_ros2.acados.acados_settings import acados_settings


class AcadosSolverAdapter(BaseSolver):
    """Wraps the acados OCP solver to conform to BaseSolver."""

    def __init__(self, controller, horizon: int, wheelbase: float,
                 stage_cost_type: str = 'NONLINEAR_LS',
                 terminal_cost_type: str = 'NONLINEAR_LS',
                 num_obstacles: int = 0, ego_radius: float = 1.0,
                 has_weight_params: bool = False,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 Qe: Optional[np.ndarray] = None,
                 Rd: Optional[np.ndarray] = None):
        self._controller = controller
        self._horizon = horizon
        self._wheelbase = wheelbase
        self._stage_cost_type = stage_cost_type
        self._terminal_cost_type = terminal_cost_type
        self._num_obstacles = num_obstacles
        self._ego_radius = ego_radius
        self._obstacle_states: Optional[np.ndarray] = None  # (3*n_obs, N+1)
        self._has_weight_params = has_weight_params
        if has_weight_params:
            self._Q_diag = np.diag(Q) if Q is not None else np.ones(4)
            self._R_diag = np.diag(R) if R is not None else np.ones(2)
            self._Qe_diag = np.diag(Qe) if Qe is not None else np.ones(4)
            self._Rd_diag = np.diag(Rd) if Rd is not None else np.ones(2)

    def update_obstacles(self, obstacle_states: np.ndarray) -> None:
        self._obstacle_states = obstacle_states

    def set_weights(self, Q: np.ndarray, R: np.ndarray,
                    Rd: np.ndarray, Qf: np.ndarray) -> None:
        if self._has_weight_params:
            self._Q_diag = np.diag(Q)
            self._R_diag = np.diag(R)
            self._Rd_diag = np.diag(Rd)
            self._Qe_diag = np.diag(Qf)

    def _obs_params(self, k: int) -> list:
        """Return the obstacle portion of the p vector for stage k."""
        if self._obstacle_states is not None:
            return [*self._obstacle_states[:, k], self._ego_radius]
        return [*([1000.0, 1000.0, 1.0] * self._num_obstacles), self._ego_radius]

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

        weight_params = (
            [*self._Q_diag, *self._R_diag, *self._Qe_diag, *self._Rd_diag]
            if self._has_weight_params else []
        )
        for j in range(self._horizon):
            yref = xref[:, j]
            if self._stage_cost_type in ('LINEAR_LS', 'NONLINEAR_LS'):
                self._controller.cost_set(j, 'yref',
                                          np.hstack([yref, np.zeros(2)]))
            p = [self._wheelbase, *yref, *np.zeros(2), *x0, *u_prev]
            if self._num_obstacles > 0:
                p = [*p, *self._obs_params(j)]
            if self._has_weight_params:
                p = [*p, *weight_params]
            self._controller.set(j, 'p', np.array(p))

        yref_N = xref[:, self._horizon]
        if self._terminal_cost_type in ('LINEAR_LS', 'NONLINEAR_LS'):
            self._controller.set(self._horizon, 'yref', yref_N)
        p_N = [self._wheelbase, *yref_N, *np.zeros(2), *x0, *u_prev]
        if self._num_obstacles > 0:
            p_N = [*p_N, *self._obs_params(self._horizon)]
        if self._has_weight_params:
            p_N = [*p_N, *weight_params]
        self._controller.set(self._horizon, 'p', np.array(p_N))

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
        self.declare_parameter('qp_solver', 'PARTIAL_CONDENSING_HPIPM')
        self.declare_parameter('nlp_solver_type', 'SQP_RTI')
        self.declare_parameter('model_directory',
                               os.path.join(
                                   get_package_share_directory('trajectory_following_ros2'),
                                   'data', 'model'))
        self.declare_parameter('obstacle_slack_weight', 100.0)

    def _init_solver(self) -> Optional[BaseSolver]:
        stage_cost_type = self.get_parameter('stage_cost_type').value
        terminal_cost_type = self.get_parameter('terminal_cost_type').value
        max_iter = int(self.get_parameter('max_iter').value)
        tol = self.get_parameter('termination_condition').value
        scale_cost = self.get_parameter('scale_cost').value
        generate = self.get_parameter('generate_mpc_model').value
        with_cython = self.get_parameter('build_with_cython').value
        qp_solver = self.get_parameter('qp_solver').value
        nlp_solver_type = self.get_parameter('nlp_solver_type').value
        model_dir = self.get_parameter('model_directory').value
        num_obstacles = self.get_parameter('num_obstacles').value
        collision_method = self.get_parameter('obstacle_collision_avoidance_method').value
        ego_radius = self.get_parameter('ego_radius').value
        obstacle_slack_weight = self.get_parameter('obstacle_slack_weight').value

        if ego_radius <= 0.0:
            ego_radius = 2.731977273419954 / 1.3  # Carla Model 3 default

        if collision_method == 'cbf':
            self.get_logger().warn(
                "obstacle_collision_avoidance_method='cbf' is not supported for acados "
                "(CBF links two stages; requires augmented state). Falling back to 'euclidean'.")

        if num_obstacles > 0:
            self.get_logger().warn(
                f"num_obstacles={num_obstacles}: acados parameter vector includes obstacle "
                "states. If num_obstacles changed since last build, set generate_mpc_model=True.")

        build_path = os.path.join(model_dir, 'c_generated_code')
        config_path = os.path.join(model_dir, 'kinematic_bicycle_acados_ocp.json')
        os.makedirs(build_path, exist_ok=True)

        cwd = os.getcwd()
        os.chdir(build_path)

        _, _, controller, _, has_weight_params = acados_settings(
            Tf=self.prediction_time,
            N=self.horizon,
            x0=self.zk,
            scale_cost=scale_cost,
            Q=self.Q, R=self.R, Qe=self.Qf, Rd=self.Rd,
            wheelbase=self.WHEELBASE,
            vel_min=self.MIN_SPEED, vel_max=self.MAX_SPEED,
            acc_min=self.MAX_DECEL, acc_max=self.MAX_ACCEL,
            delta_min=self.MIN_STEER_ANGLE, delta_max=self.MAX_STEER_ANGLE,
            cost_module=stage_cost_type,
            cost_module_e=terminal_cost_type,
            qp_solver=qp_solver,
            nlp_solver_type=nlp_solver_type,
            generate=True,
            build=generate,
            with_cython=with_cython,
            num_iterations=max_iter,
            tolerance=tol,
            mpc_config_file=config_path,
            code_export_directory=build_path,
            num_obstacles=num_obstacles,
            ego_radius=ego_radius,
            obstacle_slack_weight=obstacle_slack_weight,
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
            num_obstacles=num_obstacles,
            ego_radius=ego_radius,
            has_weight_params=has_weight_params,
            Q=self.Q, R=self.R, Qe=self.Qf, Rd=self.Rd,
        )

    def _control_timer_callback(self):
        """Populate obstacle states before each solve, then delegate to base."""
        if self._solver is not None and self._num_obstacles > 0:
            if self.obstacle_states is None:
                self.obstacle_states = np.ones(
                    (self.n_obstacle_states * self._num_obstacles,
                     self.horizon + 1)) * 1000.0
                self.obstacle_states[2::3, :] = 1.0  # radii

            for k in range(self.horizon + 1):  # N+1 to cover terminal stage (con_h_expr_e)
                for j in range(self._num_obstacles):
                    idx = 3 * j
                    if len(self.obstacles) > j:
                        self.obstacle_states[idx:idx + 3, k] = self.obstacles[j]['state']
                    else:
                        self.obstacle_states[idx:idx + 3, k] = [1000.0, 1000.0, 1.0]

            self._solver.update_obstacles(self.obstacle_states)  # type: ignore[attr-defined]

        super()._control_timer_callback()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = KinematicCoupledAcados()
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
