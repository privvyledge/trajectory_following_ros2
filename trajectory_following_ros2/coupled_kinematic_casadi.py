import math
import sys
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker
from trajectory_following_ros2.backends.base_solver import BaseSolver, SolverResult

try:
    OBSTACLES_AVAILABLE = True
    from derived_object_msgs.msg import ObjectArray
except ImportError:
    OBSTACLES_AVAILABLE = False

from trajectory_following_ros2.casadi.kinematic_mpc_casadi_opti import KinematicMPCCasadiOpti
from trajectory_following_ros2.casadi.kinematic_mpc_casadi import KinematicMPCCasadi
from trajectory_following_ros2.casadi.discrete_kinematic_mpc_casadi import DiscreteKinematicMPCCasadi
import tf_transformations


class CasAdiSolverAdapter(BaseSolver):
    """Wraps KinematicMPCCasadi / Opti / Discrete to conform to BaseSolver."""

    def __init__(self, controller, use_opti: bool = False,
                 num_obstacles: int = 0, n_obstacle_states: int = 3):
        self._controller = controller
        self._use_opti = use_opti
        self._warmstart = {}
        self._num_obstacles = num_obstacles
        self._obstacle_states: Optional[np.ndarray] = None
        self._n_obs_states = n_obstacle_states

    def initialize(self, x0: np.ndarray) -> None:
        pass  # CasADi is ready after construction

    def update_obstacles(self, obstacle_states: np.ndarray):
        self._obstacle_states = obstacle_states

    def solve(self, x0: np.ndarray, xref: np.ndarray,
              u_prev: np.ndarray) -> SolverResult:
        # xref: (4, N+1); u_prev: (2,)
        state = x0.tolist()
        ref_traj = [xref[0, :], xref[1, :], xref[2, :], xref[3, :]]
        prev_input = [float(u_prev[0]), float(u_prev[1])]

        if self._use_opti:
            # opti expects ref_traj without terminal point on some formulations;
            # pass full N+1 and let the controller slice as needed
            self._controller.update(
                state=state,
                ref_traj=[r[:-1] for r in ref_traj],  # N points for opti
                previous_input=prev_input,
                warmstart_variables=self._warmstart)
        else:
            self._controller.update(
                state=state,
                ref_traj=ref_traj,
                previous_input=prev_input,
                warmstart_variables=self._warmstart)

        if self._num_obstacles > 0 and self._obstacle_states is not None:
            self._controller.update_obstacles_state(self._obstacle_states)

        sol = self._controller.solve()

        # Persist warmstart variables
        self._warmstart['z_ws'] = sol['z_mpc']
        self._warmstart['u_ws'] = sol['u_mpc']
        self._warmstart['lam_x'] = sol.get('lam_x')
        self._warmstart['lam_g'] = sol.get('lam_g')
        self._warmstart['lam_p'] = sol.get('lam_p')
        self._warmstart['sl_ws'] = sol.get('sl_mpc')
        self._warmstart['sl_obs_ws'] = sol.get('sl_obs_ws')

        z_mpc = sol['z_mpc']  # opti: (N+1, nx); non-opti: (nx, N+1)
        u_mpc = sol['u_mpc']  # opti: (N, nu); non-opti: (nu, N)

        if self._use_opti:
            # rows = timesteps, cols = states
            x_sequence = z_mpc.T          # (nx, N+1)
            u_sequence = u_mpc.T          # (nu, N)
            vel_next = float(z_mpc[1, 2]) # row=k+1, col=vel
        else:
            x_sequence = z_mpc             # (nx, N+1)
            u_sequence = u_mpc             # (nu, N)
            vel_next = float(z_mpc[2, 1]) # row=vel, col=k+1

        acc_cmd = float(sol['u_control'][0])
        delta_cmd = float(sol['u_control'][1])

        u_rate = sol.get('u_rate')
        jerk = float(u_rate[0, 0]) if u_rate is not None else None
        delta_rate = float(u_rate[1, 0]) if u_rate is not None else None

        return SolverResult(
            accel_cmd=acc_cmd,
            steering_cmd=delta_cmd,
            velocity_cmd=vel_next,
            jerk_cmd=jerk,
            steering_rate_cmd=delta_rate,
            u_sequence=u_sequence,
            x_sequence=x_sequence,
            u_prev=np.array([acc_cmd, delta_cmd]),
            is_optimal=bool(sol['solver_status']),
            solve_time=float(sol.get('solve_time', 0.0)),
            status=str(sol.get('solver_stats', {}).get('return_status', '')),
        )

    @property
    def nx(self) -> int:
        return 4

    @property
    def nu(self) -> int:
        return 2


class KinematicCoupledCasadi(BaseTrajectoryTracker):

    def __init__(self):
        super().__init__('kinematic_coupled_casadi_controller')
        self.get_logger().info('kinematic_coupled_casadi_controller started.')

    def _declare_backend_parameters(self):
        self.declare_parameter('ode_type', 'discrete_kinematic_coupled')
        self.declare_parameter('use_opti', False)
        self.declare_parameter('max_iter', 15)
        self.declare_parameter('termination_condition', 1e-6)
        self.declare_parameter('normalize_yaw_error', True)
        self.declare_parameter('solver_type', 'quad')
        self.declare_parameter('solver', 'qrqp')
        self.declare_parameter('slack_weights_input_rate', [1.0, 1.0])
        self.declare_parameter('slack_scale_input_rate', [1.0, 1.0])
        self.declare_parameter('slack_upper_bound_input_rate', [1e9, 1e9])
        self.declare_parameter('slack_objective_is_quadratic', False)
        self.declare_parameter('ego_radius', -1.0)
        self.declare_parameter('obstacle_collision_avoidance_method', 'euclidean')
        self.declare_parameter('obstacle_topic', 'fake_obstacles/object_array')
        self.declare_parameter('num_obstacles', 0)

    def _init_solver(self) -> Optional[BaseSolver]:
        ode_type = self.get_parameter('ode_type').value
        use_opti = self.get_parameter('use_opti').value
        normalize_yaw_error = self.get_parameter('normalize_yaw_error').value
        solver_type = self.get_parameter('solver_type').value
        solver = self.get_parameter('solver').value
        slack_weights = list(
            self.get_parameter('slack_weights_input_rate').get_parameter_value().double_array_value)
        slack_scale = list(
            self.get_parameter('slack_scale_input_rate').get_parameter_value().double_array_value)
        slack_ub = list(
            self.get_parameter('slack_upper_bound_input_rate').get_parameter_value().double_array_value)
        slack_quad = self.get_parameter('slack_objective_is_quadratic').value
        num_obstacles = self.get_parameter('num_obstacles').value
        collision_method = self.get_parameter('obstacle_collision_avoidance_method').value

        ego_radius = self.get_parameter('ego_radius').value
        if ego_radius <= 0.0:
            ego_radius = 2.731977273419954 / 1.3  # Carla Model 3 default

        model_type = 'continuous' if 'continuous' in ode_type else 'discrete'

        common_kwargs = dict(
            horizon=self.horizon,
            sample_time=self.sample_time,
            wheelbase=self.WHEELBASE,
            nx=self.NX, nu=self.NU,
            x0=self.zk, u0=self.uk,
            Q=self.Q.diagonal(), R=self.R.diagonal(),
            Qf=self.Qf.diagonal(), Rd=self.Rd.diagonal(),
            vel_bound=(self.MIN_SPEED, self.MAX_SPEED),
            delta_bound=(np.degrees(self.MIN_STEER_ANGLE), np.degrees(self.MAX_STEER_ANGLE)),
            acc_bound=(self.MAX_DECEL, self.MAX_ACCEL),
            jerk_bound=(self.MIN_JERK, self.MAX_JERK),
            delta_rate_bound=(-np.degrees(self.MAX_STEER_RATE),
                              np.degrees(self.MAX_STEER_RATE)),
            solver_type=solver_type, solver=solver,
            suppress_ipopt_output=True,
            normalize_yaw_error=normalize_yaw_error,
            slack_weights_u_rate=slack_weights,
            slack_scale_u_rate=slack_scale,
            slack_upper_bound_u_rate=slack_ub,
            slack_objective_is_quadratic=slack_quad,
        )

        if use_opti:
            controller = KinematicMPCCasadiOpti(**common_kwargs)
        elif model_type == 'continuous':
            controller = KinematicMPCCasadi(symbol_type='MX', warmstart=True, **common_kwargs)
        else:
            controller = DiscreteKinematicMPCCasadi(
                symbol_type='MX', warmstart=True,
                num_obstacles=num_obstacles,
                collision_avoidance_scheme=collision_method,
                ego_radius=ego_radius,
                **common_kwargs)

        self.get_logger().info(f'CasADi MPC built (ode={ode_type}, opti={use_opti}).')

        # obstacle subscription (casadi only)
        self._num_obstacles = num_obstacles
        self.obstacles = []
        self.n_obstacle_states = 3
        self.obstacle_states: Optional[np.ndarray] = None
        self.obstacle_collision_avoidance_method = collision_method
        if num_obstacles > 0 and OBSTACLES_AVAILABLE:
            obstacle_topic = self.get_parameter('obstacle_topic').value
            self.obstacle_sub = self.create_subscription(
                ObjectArray, obstacle_topic, self._obstacle_callback, 1,
                callback_group=self.subscription_group)
            self.get_logger().info(
                f'Obstacle avoidance enabled: {num_obstacles} obstacles on {obstacle_topic}')
        elif num_obstacles > 0:
            self.get_logger().warn(
                'num_obstacles > 0 but derived_object_msgs not available.')

        return CasAdiSolverAdapter(
            controller, use_opti=use_opti,
            num_obstacles=num_obstacles,
            n_obstacle_states=3)

    def _control_timer_callback(self):
        """Update obstacle states before each solve, then delegate to base."""
        if self._num_obstacles > 0 and self._solver is not None:
            if self.obstacle_states is None:
                self.obstacle_states = np.ones(
                    (self.n_obstacle_states * self._num_obstacles,
                     self.horizon + 1)) * 1000.0
                self.obstacle_states[2::3, :] = 1.0  # radii

            for k in range(self.horizon):
                for j in range(self._num_obstacles):
                    idx = 3 * j
                    if len(self.obstacles) > j:
                        self.obstacle_states[idx:idx + 3, k] = self.obstacles[j]['state']
                    else:
                        self.obstacle_states[idx:idx + 3, k] = [1000.0, 1000.0, 1.0]

            self._solver.update_obstacles(self.obstacle_states)  # type: ignore[attr-defined]

        super()._control_timer_callback()

    def _obstacle_callback(self, data: 'ObjectArray'):
        obstacles = []
        for obj in data.objects:
            pos = [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z]
            shape = {
                obj.shape.BOX: 'BOX',
                obj.shape.SPHERE: 'SPHERE',
                obj.shape.CYLINDER: 'CYLINDER',
            }.get(obj.shape.type, 'BOX')

            if shape == 'SPHERE' and obj.shape.dimensions:
                radius = obj.shape.dimensions[0]
            elif shape == 'CYLINDER' and len(obj.shape.dimensions) >= 2:
                radius = obj.shape.dimensions[1]
            elif shape == 'BOX' and len(obj.shape.dimensions) >= 3:
                l, w, h = obj.shape.dimensions[:3]
                radius = (math.sqrt(l**2 + w**2 + h**2) / 2) / 1.3
            else:
                continue

            dist = np.linalg.norm(np.array(pos[:2]) - np.array([self.x, self.y]))
            obstacles.append({'state': [pos[0], pos[1], radius], 'distance': dist})

        self.obstacles = sorted(obstacles, key=lambda o: o['distance'])


def main(args=None):
    rclpy.init(args=args)
    try:
        node = KinematicCoupledCasadi()
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
