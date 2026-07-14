import os
import sys
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker, _make_executor
from trajectory_following_ros2.backends.base_solver import BaseSolver, SolverResult

from trajectory_following_ros2.casadi.kinematic_mpc_casadi_opti import KinematicMPCCasadiOpti
from trajectory_following_ros2.casadi.kinematic_mpc_casadi import KinematicMPCCasadi


# Supported (solver_type, solver) combinations for the merged KinematicMPCCasadi.
# The continuous and discrete ode_type entry points share the same solver paths
# after the merge; the prediction model differs (continuous == nonlinear+euler)
# but the NLP/QP solver routing in setup_solver() is identical.
#   ('nlp',  'ipopt')   -- interior-point NLP solver.
#   ('quad', 'qrqp')    -- sqpmethod with QRQP inner QP (default).
#   ('quad', 'osqp')    -- sqpmethod with OSQP inner QP.
#   ('quad', 'qpoases') -- sqpmethod with qpOASES inner QP.
#   ('quad', 'ipopt')   -- sqpmethod with IPOPT used as the inner QP solver (via nlpsol).
# All 'quad' (SQP/QP) paths require Rd > 0 (a non-zero input-rate penalty) and the
# input-rate slack so the QP stays feasible/convex. The old direct casadi.qpsol
# path for the continuous formulation (broken, qrqp silently fell back to osqp) was
# removed by the merge.
_VALID_SOLVER_COMBOS = {
    ('nlp', 'ipopt'),
    ('quad', 'qrqp'),
    ('quad', 'osqp'),
    ('quad', 'qpoases'),
    ('quad', 'ipopt'),
    ('qp', 'qrqp'),
    ('qp', 'osqp'),
    ('qp', 'qpoases'),
}


class CasAdiSolverAdapter(BaseSolver):
    """Wraps KinematicMPCCasadi / Opti / Discrete to conform to BaseSolver."""

    def __init__(self, controller, use_opti: bool = False,
                 num_obstacles: int = 0, n_obstacle_states: int = 3,
                 solver: str = 'ipopt', vel_bound: Optional[tuple] = None):
        self._controller = controller
        self._use_opti = use_opti
        self._warmstart = {}
        self._num_obstacles = num_obstacles
        self._obstacle_states: Optional[np.ndarray] = None
        self._n_obs_states = n_obstacle_states
        self._solver_name = solver
        if vel_bound is not None:
            self._v_min, self._v_max = vel_bound
        else:
            self._v_min, self._v_max = -np.inf, np.inf

    def initialize(self, x0: np.ndarray) -> None:
        pass  # CasADi is ready after construction

    def set_weights(self, Q: np.ndarray, R: np.ndarray,
                    Rd: np.ndarray, Qf: np.ndarray) -> None:
        if not self._use_opti:
            self._controller.set_weights(Q, R, Rd, Qf)

    def update_obstacles(self, obstacle_states: np.ndarray):
        self._obstacle_states = obstacle_states

    def solve(self, x0: np.ndarray, xref: np.ndarray,
              u_prev: np.ndarray) -> SolverResult:
        # xref: (4, N+1); u_prev: (2,)
        x0 = np.array(x0, dtype=float).copy()
        if hasattr(self, '_v_min') and hasattr(self, '_v_max'):
            x0[2] = np.clip(x0[2], self._v_min, self._v_max)

        if not self._use_opti and not self._warmstart:
            n = getattr(self._controller, 'horizon', xref.shape[1] - 1)
            self._warmstart = {
                'z_ws': np.asarray(xref, dtype=float).copy(),
                'u_ws': np.tile(np.asarray(u_prev, dtype=float).reshape(2, 1), (1, n)),
                'sl_ws': None, 'sl_obs_ws': None,
                'lam_x': None, 'lam_g': None, 'lam_p': None,
            }

        if not self._use_opti and self._warmstart and 'z_ws' in self._warmstart and self._warmstart['z_ws'] is not None:
            psi_ws = self._warmstart['z_ws'][3, 0]
            shift = 2.0 * np.pi * np.round((x0[3] - psi_ws) / (2.0 * np.pi))
            if shift != 0.0:
                self._warmstart['z_ws'][3, :] += shift

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

        # Persist warmstart variables only when the solution is usable.
        # Degenerate statuses leave the NLP at a bad point — clear the dict so
        # the next call does a cold start rather than inheriting a poisoned guess.
        _DEGENERATE = {
            'Search_Direction_Becomes_Too_Small',
            'Infeasible_Problem_Detected',
            'Restoration_Failed',
            'Error_In_Step_Computation',
        }
        return_status = str(sol.get('solver_stats', {}).get('return_status', ''))
        if return_status not in _DEGENERATE:
            # Receding-horizon shift: the warm-start for step k+1 should begin
            # at the state the solver predicted for k+1, not the current k.
            # Shift left by one column and hold the terminal state/input.
            z_ws = sol['z_mpc']   # (nx, N+1)
            u_ws = sol['u_mpc']   # (nu, N)
            self._warmstart['z_ws'] = np.concatenate(
                [z_ws[:, 1:], z_ws[:, -1:]], axis=1)
            self._warmstart['u_ws'] = np.concatenate(
                [u_ws[:, 1:], u_ws[:, -1:]], axis=1)
            # Dual variables cut IPOPT from ~30 cold iterations to 5-15.
            # For sqpmethod they hurt convergence so we skip them there.
            _use_duals = self._solver_name == 'ipopt'
            self._warmstart['lam_x'] = sol.get('lam_x') if _use_duals else None
            self._warmstart['lam_g'] = sol.get('lam_g') if _use_duals else None
            self._warmstart['lam_p'] = sol.get('lam_p') if _use_duals else None
            self._warmstart['sl_ws'] = sol.get('sl_mpc')
            self._warmstart['sl_obs_ws'] = sol.get('sl_obs_ws')
        else:
            self._warmstart.clear()

        z_mpc = sol['z_mpc']  # opti: (N+1, nx); non-opti: (nx, N+1)
        u_mpc = sol['u_mpc']  # opti: (N, nu); non-opti: (nu, N)

        if self._use_opti:
            # rows = timesteps, cols = states
            x_sequence = z_mpc.T          # (nx, N+1)
            u_sequence = u_mpc.T          # (nu, N)
            vel_next = float(z_mpc[1, 2])  # row=k+1, col=vel
        else:
            x_sequence = z_mpc             # (nx, N+1)
            u_sequence = u_mpc             # (nu, N)
            vel_next = float(z_mpc[2, 1])  # row=vel, col=k+1

        acc_cmd = float(sol['u_control'][0])
        delta_cmd = float(sol['u_control'][1])

        u_rate = sol.get('u_rate')
        jerk = float(u_rate[0, 0]) if u_rate is not None else None
        delta_rate = float(u_rate[1, 0]) if u_rate is not None else None

        # A budget-limited solve still returns a usable, warm-started, near-converged
        # iterate — accept it (real-time-iteration style) rather than discarding it and
        # holding a stale command. At the +/-pi heading zones IPOPT can't prove optimality
        # within the cpu-time cap, but its iterate keeps cte ~0.2 (on track); treating that
        # as a hard failure tripped the 5-consecutive zero-command safety and stopped the
        # vehicle. Only genuine failures (infeasible/restoration, in _DEGENERATE) stay
        # non-optimal so the safety net still protects against real divergence.
        _USABLE_INCOMPLETE = ('Maximum_CpuTime_Exceeded', 'Maximum_Iterations_Exceeded')
        _solver_ok = bool(sol['solver_status']) or return_status in _USABLE_INCOMPLETE

        return SolverResult(
            accel_cmd=acc_cmd,
            steering_cmd=delta_cmd,
            velocity_cmd=vel_next,
            jerk_cmd=jerk,
            steering_rate_cmd=delta_rate,
            u_sequence=u_sequence,
            x_sequence=x_sequence,
            u_prev=np.array([acc_cmd, delta_cmd]),
            is_optimal=_solver_ok,
            solve_time=float(sol.get('solve_time', 0.0)),
            status=str(return_status),
            error=sol.get('error'),
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
        # Discrete-formulation model form and discretization scheme (only used
        # when ode_type selects a discrete formulation):
        #   discrete_model_type: 'nonlinear' (full ODE, default) | 'ltv'
        #                        (forward-Jacobian/Taylor linearization).
        #   discrete_integration_method: 'rk4' (default) | 'euler'. Only these
        #     explicit schemes are valid for the MPC model -- the black-box
        #     casadi.integrator schemes ('cvodes'/'rk'/'collocation') cannot be
        #     SX-expanded inside the NLP. Ignored when discrete_model_type='ltv'
        #     (the LTV Jacobian form is forward-Euler by construction).
        self.declare_parameter('discrete_model_type', 'nonlinear')
        self.declare_parameter('discrete_integration_method', 'rk4')
        # Directory for CasADi JIT codegen artifacts (jit_tmp.c + tmp_*.o/.so) so they
        # do not litter the working directory. Unified name with the acados node (which
        # uses it for its generated C-code + compiled model). Empty string -> dump in the
        # current working directory (legacy).
        self.declare_parameter(
            'code_gen_directory',
            os.path.join(get_package_share_directory('trajectory_following_ros2'),
                         'data', 'casadi_codegen'))
        # Iteration budget — meaning depends on solver/solver_type.
        # Node default (15) is appropriate for qrqp/sqpmethod. Raise to >=100 for IPOPT.
        self.declare_parameter('max_iter', 15)
        self.declare_parameter('termination_condition', 1e-6)
        self.declare_parameter('normalize_yaw_error', True)
        self.declare_parameter('solver_type', 'quad')
        self.declare_parameter('solver', 'qrqp')
        self.declare_parameter('sqp_convexify_strategy', 'regularize')
        self.declare_parameter('sqp_hessian_approximation', 'exact')
        self.declare_parameter('qp_inner_max_iter', 0)
        self.declare_parameter('suppress_solver_output', True)
        self.declare_parameter('slack_weights_input_rate', [1.0, 1.0])
        self.declare_parameter('slack_scale_input_rate', [1.0, 1.0])
        self.declare_parameter('slack_upper_bound_input_rate', [float('inf'), float('inf')])
        self.declare_parameter('slack_objective_is_quadratic', False)
        # Per-obstacle slack (soft collision constraint). Empty lists ([]) mean the
        # solver falls back to a hard constraint. These are sized to num_obstacles at
        # read time (a scalar is broadcast); the slacked form is what lets the
        # linearized LTV/QP obstacle constraint stay feasible when the operating point
        # sits inside the inflated obstacle.
        self.declare_parameter('slack_weights_obstacle_avoidance', [1000.0])
        self.declare_parameter('slack_upper_bound_obstacle_avoidance', [1000.0])

    def _init_solver(self) -> Optional[BaseSolver]:
        ode_type = self.get_parameter('ode_type').value
        use_opti = self.get_parameter('use_opti').value
        discrete_model_type = self.get_parameter('discrete_model_type').value
        discrete_integration_method = self.get_parameter('discrete_integration_method').value
        code_gen_directory = self.get_parameter('code_gen_directory').value or None
        max_iter = int(self.get_parameter('max_iter').value)
        normalize_yaw_error = self.get_parameter('normalize_yaw_error').value
        solver_type = self.get_parameter('solver_type').value
        solver = self.get_parameter('solver').value
        sqp_convexify_strategy = self.get_parameter('sqp_convexify_strategy').value
        sqp_hessian_approximation = self.get_parameter('sqp_hessian_approximation').value
        qp_inner_max_iter = int(self.get_parameter('qp_inner_max_iter').value)
        suppress_solver_output = self.get_parameter('suppress_solver_output').value

        if solver_type == 'qp' and normalize_yaw_error:
            self.get_logger().warn(
                "solver_type='qp' requires normalize_yaw_error=False because atan2 wrapping breaks "
                "the quadratic cost requirement of casadi.qpsol. Overriding normalize_yaw_error to False.")
            normalize_yaw_error = False

        _is_ipopt = (solver_type == 'nlp' and solver == 'ipopt') or \
                    (solver_type == 'quad' and solver == 'ipopt')
        _is_osqp = (solver == 'osqp')
        if _is_ipopt and max_iter < 50:
            self.get_logger().warn(
                f'max_iter={max_iter} is very low for IPOPT (solver_type={solver_type}, '
                f'solver={solver}). IPOPT typically needs 50-300 iterations without warm-start. '
                f'Consider max_iter >= 100.')
        elif _is_osqp and max_iter < 100:
            self.get_logger().warn(
                f'max_iter={max_iter} is very low for OSQP (first-order ADMM). '
                f'OSQP typically needs 100-4000 steps. Consider max_iter >= 500.')
        slack_weights = list(
            self.get_parameter('slack_weights_input_rate').get_parameter_value().double_array_value)
        slack_scale = list(
            self.get_parameter('slack_scale_input_rate').get_parameter_value().double_array_value)
        slack_ub = list(
            self.get_parameter('slack_upper_bound_input_rate').get_parameter_value().double_array_value)
        slack_quad = self.get_parameter('slack_objective_is_quadratic').value
        num_obstacles = self.get_parameter('num_obstacles').value
        collision_method = self.get_parameter('obstacle_collision_avoidance_method').value

        def _size_obstacle_slack(param_name):
            vals = list(self.get_parameter(param_name).get_parameter_value().double_array_value)
            if not vals or num_obstacles <= 0:
                return None
            if len(vals) == 1:
                return vals * num_obstacles
            return vals
        slack_weights_obs = _size_obstacle_slack('slack_weights_obstacle_avoidance')
        slack_ub_obs = _size_obstacle_slack('slack_upper_bound_obstacle_avoidance')

        ego_radius = self.get_parameter('ego_radius').value
        if ego_radius <= 0.0:
            ego_radius = 2.731977273419954 / 1.3  # Carla Model 3 default
        safe_distance = self.get_parameter('safe_distance').value

        model_type = 'continuous' if 'continuous' in ode_type else 'discrete'

        # Validate the (solver_type, solver) combination against the matrix above.
        # Opti uses its own solver setup, so it is exempt from this check.
        if not use_opti and (solver_type, solver) not in _VALID_SOLVER_COMBOS:
            matrix = '\n'.join(f'    {st} + {sv}' for st, sv in sorted(_VALID_SOLVER_COMBOS))
            raise ValueError(
                f"Unsupported solver combination (solver_type='{solver_type}', "
                f"solver='{solver}') for ode_type='{ode_type}'. Supported combinations:\n"
                f"{matrix}\nAll 'quad' paths require Rd > 0.")
        if solver_type == 'qp':
            if use_opti or discrete_model_type != 'ltv' or 'discrete' not in ode_type:
                raise ValueError(
                    f"solver_type='qp' requires discrete_model_type='ltv', a discrete ode_type, "
                    f"and use_opti=False. Got ode_type='{ode_type}', "
                    f"discrete_model_type='{discrete_model_type}', "
                    f"use_opti={use_opti}.")
        if discrete_model_type == 'ltv' and num_obstacles > 0 and solver_type != 'qp':
            raise ValueError(
                f"discrete_model_type='ltv' does not support obstacle avoidance for solver_type='{solver_type}'. "
                f"Set discrete_model_type='nonlinear' or num_obstacles=0.")
        if solver_type == 'qp' and num_obstacles > 0 and collision_method == 'cbf':
            raise ValueError(
                "solver_type='qp' does not support CBF obstacle avoidance. "
                "CBF stays nlp-only. Use collision_avoidance_method='euclidean'.")
        # 'quad' / 'qp' (SQP/QP) paths need a non-zero input-rate penalty Rd to stay feasible.
        if not use_opti and solver_type in ('quad', 'qp') and not np.any(self.Rd.diagonal() > 0):
            self.get_logger().warn(
                f"solver_type='{solver_type}' (solver='{solver}') needs Rd > 0 for the input-rate "
                f"slack to keep the QP feasible/convex, but Rd is all zeros. Expect "
                f"infeasible/suboptimal solves; set a non-zero Rd.")

        common_kwargs = dict(
            horizon=self.horizon,
            sample_time=self.sample_time,
            wheelbase=self.WHEELBASE,
            nx=self.NX, nu=self.NU,
            x0=self.zk, u0=self.uk,
            Q=self.Q.diagonal(), R=self.R.diagonal(),
            Qf=self.Qf.diagonal(), Rd=self.Rd.diagonal(),
            vel_bound=(self.MIN_SPEED, self.MAX_SPEED),
            delta_bound=(self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE),
            acc_bound=(self.MAX_DECEL, self.MAX_ACCEL),
            jerk_bound=(self.MIN_JERK, self.MAX_JERK),
            delta_rate_bound=(-self.MAX_STEER_RATE, self.MAX_STEER_RATE),
            solver_type=solver_type, solver=solver,
            suppress_solver_output=suppress_solver_output,
            sqp_convexify_strategy=sqp_convexify_strategy,
            sqp_hessian_approximation=sqp_hessian_approximation,
            qp_inner_max_iter=qp_inner_max_iter,
            max_iter=max_iter,
            normalize_yaw_error=normalize_yaw_error,
            slack_weights_u_rate=slack_weights,
            slack_scale_u_rate=slack_scale,
            slack_upper_bound_u_rate=slack_ub,
            slack_objective_is_quadratic=slack_quad,
        )

        self.get_logger().info(
            f'Building CasADi NLP (ode={ode_type}, solver_type={solver_type}, '
            f'solver={solver}); generated C may JIT-compile on first solve '
            '(a few seconds, then cached)...')

        if use_opti:
            controller = KinematicMPCCasadiOpti(**common_kwargs)
        elif model_type == 'continuous':
            # The continuous formulation is mathematically identical to the merged
            # class with nonlinear dynamics + forward Euler. No obstacle avoidance on
            # the continuous entry point (matches legacy behaviour).
            controller = KinematicMPCCasadi(
                symbol_type='MX', warmstart=True,
                discrete_model_type='nonlinear',
                discrete_integration_method='euler',
                code_gen_directory=code_gen_directory,
                num_obstacles=0,
                **common_kwargs)
        else:
            controller = KinematicMPCCasadi(
                symbol_type='MX', warmstart=True,
                discrete_model_type=discrete_model_type,
                discrete_integration_method=discrete_integration_method,
                code_gen_directory=code_gen_directory,
                num_obstacles=num_obstacles,
                collision_avoidance_scheme=collision_method,
                ego_radius=ego_radius, safe_distance=safe_distance,
                slack_weights_obstacle_avoidance=slack_weights_obs,
                slack_upper_bound_obstacle_avoidance=slack_ub_obs,
                **common_kwargs)

        if model_type == 'discrete' and not use_opti:
            self.get_logger().info(
                f'CasADi MPC built (ode={ode_type}, opti={use_opti}, '
                f'discrete_model={discrete_model_type}, '
                f'discretization={discrete_integration_method}).')
        else:
            self.get_logger().info(f'CasADi MPC built (ode={ode_type}, opti={use_opti}).')

        return CasAdiSolverAdapter(
            controller, use_opti=use_opti,
            num_obstacles=num_obstacles,
            n_obstacle_states=3,
            solver=solver,
            vel_bound=(self.MIN_SPEED, self.MAX_SPEED))

    def _control_timer_callback(self):
        """Update obstacle states before each solve, then delegate to base."""
        if self._solver is not None and self._num_obstacles > 0:
            if self.obstacle_states is None:
                self.obstacle_states = np.ones(
                    (self.n_obstacle_states * self._num_obstacles,
                     self.horizon + 1)) * 1000.0
                self.obstacle_states[2::3, :] = 1.0  # radii

            for k in range(self.horizon + 1):  # N+1 to fill terminal column used by Euclidean k=N
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
        node = KinematicCoupledCasadi()
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
