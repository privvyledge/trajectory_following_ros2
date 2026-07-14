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
from trajectory_following_ros2.utils.trajectory_utils import project_reference_out_of_keepouts


class AcadosSolverAdapter(BaseSolver):
    """Wraps the acados OCP solver to conform to BaseSolver."""

    def __init__(self, controller, horizon: int, wheelbase: float,
                 stage_cost_type: str = 'NONLINEAR_LS',
                 terminal_cost_type: str = 'NONLINEAR_LS',
                 num_obstacles: int = 0, ego_radius: float = 1.0,
                 safe_distance: float = 0.5,
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
        self._safe_distance = safe_distance
        self._obstacle_states: Optional[np.ndarray] = None  # (3*n_obs, N+1)
        self._keepout_side_hints: Optional[list] = None  # go-around side memory
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

    def _recover_from_failure(self, x0: np.ndarray, xref: np.ndarray) -> None:
        """Reset the solver after a failed solve and re-seed a clean iterate.

        Zeroes the primal/dual iterate and the QP-solver memory, then seeds the
        state trajectory from the current state + (keep-out-projected) reference
        and the inputs with zeros, so the next solve is a well-posed cold start
        instead of a re-linearization at a degenerate (possibly NaN) iterate.
        """
        try:
            self._controller.reset()
        except Exception:  # older interface without reset(): reseeding still helps
            pass
        self._controller.set(0, 'x', np.asarray(x0, dtype=float))
        for i in range(1, self._horizon + 1):
            self._controller.set(i, 'x', np.asarray(xref[:, i], dtype=float))
        for i in range(self._horizon):
            self._controller.set(i, 'u', np.zeros(2))

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

        # Keep the tracked target feasible: reference points inside an obstacle
        # keep-out are swept onto its boundary (go-around arc). Without this, a
        # reference threading the keep-out makes "park at the bubble edge" the
        # optimum; at v=0 steering has no yaw authority and the vehicle never
        # detours. Copy first — the caller's xref feeds the debug/reference
        # publishers and must stay the raw reference.
        if self._num_obstacles > 0 and self._obstacle_states is not None:
            obs0 = np.asarray(self._obstacle_states[:, 0], dtype=float).reshape(
                self._num_obstacles, 3)
            keepouts = self._ego_radius + obs0[:, 2] + self._safe_distance
            xref, _, self._keepout_side_hints = project_reference_out_of_keepouts(
                xref.copy(), obs0[:, :2], keepouts,
                side_hints=self._keepout_side_hints)

        weight_params = (
            [*self._Q_diag, *self._R_diag, *self._Qe_diag, *self._Rd_diag]
            if self._has_weight_params else []
        )

        # Heading-error wrapping at +/-pi crossings differs by cost type:
        #   EXTERNAL / NONLINEAR_LS -> wrapped in-solver via atan2 in the cost
        #     expression (using the psi_ref carried in the zref parameter), so the
        #     psi entry of yref must be ZERO here to avoid double-subtraction.
        #   LINEAR_LS -> cannot wrap in-solver (linear residual), so we pre-unwrap
        #     the psi reference trajectory and align it to the current heading;
        #     then psi - psi_ref already lies in [-pi, pi].
        # The p-vector (zref) always carries the true psi_ref for the in-solver wrap.
        psi_ref_lin = None
        if 'LINEAR_LS' in (self._stage_cost_type, self._terminal_cost_type):
            psi_ref_lin = np.unwrap(xref[3, :])
            psi_ref_lin = psi_ref_lin + np.round(
                (x0[3] - psi_ref_lin[0]) / (2.0 * np.pi)) * 2.0 * np.pi

        for j in range(self._horizon):
            yref = xref[:, j]
            if self._stage_cost_type == 'NONLINEAR_LS':
                self._controller.cost_set(
                    j, 'yref', np.array([yref[0], yref[1], yref[2], 0.0, 0.0, 0.0]))
            elif self._stage_cost_type == 'LINEAR_LS':
                self._controller.cost_set(
                    j, 'yref',
                    np.array([yref[0], yref[1], yref[2], psi_ref_lin[j], 0.0, 0.0]))
            p = [self._wheelbase, *yref, *np.zeros(2), *x0, *u_prev]
            if self._num_obstacles > 0:
                p = [*p, *self._obs_params(j)]
            if self._has_weight_params:
                p = [*p, *weight_params]
            self._controller.set(j, 'p', np.array(p))

        yref_N = xref[:, self._horizon]
        if self._terminal_cost_type == 'NONLINEAR_LS':
            self._controller.set(
                self._horizon, 'yref',
                np.array([yref_N[0], yref_N[1], yref_N[2], 0.0]))
        elif self._terminal_cost_type == 'LINEAR_LS':
            self._controller.set(
                self._horizon, 'yref',
                np.array([yref_N[0], yref_N[1], yref_N[2], psi_ref_lin[self._horizon]]))
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

        # Real-time-iteration-style acceptance: status 2 (max SQP iterations) is a
        # budget-limited return, not a blow-up — near a nonconvex obstacle keep-out
        # the SQP can cycle on the stationarity residual while every inner QP
        # succeeds, and applying the near-converged iterate keeps the vehicle
        # progressing (the CasADi adapter applies the same policy to
        # Maximum_Iterations_Exceeded). Only a non-finite iterate is rejected;
        # genuinely degenerate statuses (QP failure, NaN detection) still count as
        # failures and feed the consecutive-failure zero-command safety fallback.
        finite = bool(np.isfinite(u).all() and np.isfinite(x_seq).all())
        is_optimal = finite and status in (0, 2)

        if not is_optimal:
            # A failed solve can leave a degenerate iterate in the solver memory
            # (worst case NaN after an inner-QP failure). acados warm-starts every
            # solve from that memory, so without a reset each later solve
            # re-linearizes at the poisoned iterate and fails at the first QP —
            # the failure becomes permanent even when a cold-started solve would
            # succeed. Reset the solver (incl. QP-solver memory) and re-seed from
            # the current state/reference so the next tick starts well-posed;
            # the CasADi adapter recovers the same way (reference cold start
            # after a degenerate solve).
            self._recover_from_failure(x0, xref)

        if not finite:
            # Never let a non-finite iterate reach the published command, the
            # u_prev echo, or the predicted-path debug topic.
            u = np.zeros(2)
            u_seq = np.zeros((2, self._horizon))
            x_seq = np.tile(np.asarray(x0, dtype=float).reshape(4, 1),
                            (1, self._horizon + 1))

        return SolverResult(
            accel_cmd=float(u[0]),
            steering_cmd=float(u[1]),
            velocity_cmd=float(x_seq[2, 1]),
            u_sequence=u_seq,
            x_sequence=x_seq,
            u_prev=u.copy(),
            is_optimal=is_optimal,
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
        # EXTERNAL is the default: it is the only cost module that applies the Rd
        # input-rate penalty (LINEAR_LS/NONLINEAR_LS silently drop it, so tuned
        # steering-rate damping vanishes and steering jitters) and the only one that
        # enables obstacle/CBF constraints. Its EXACT-Hessian cost (~5 ms/solve) is
        # negligible against the control budget. Override to *_LS only for a pure
        # tracking cost with no rate penalty.
        self.declare_parameter('stage_cost_type', 'EXTERNAL')
        self.declare_parameter('terminal_cost_type', 'EXTERNAL')
        self.declare_parameter('max_iter', 15)
        self.declare_parameter('termination_condition', 1e-6)
        self.declare_parameter('scale_cost', False)
        self.declare_parameter('generate_mpc_model', True)
        self.declare_parameter('build_with_cython', True)
        self.declare_parameter('qp_solver', 'PARTIAL_CONDENSING_HPIPM')
        # Full SQP by default (not single-iteration SQP_RTI): one RTI step is too
        # weak at a sharp corner under odometry noise + actuator lag and silently
        # stalls (index freezes, cross-track error diverges, no failure flag). Full
        # SQP tracks robustly. Use SQP_RTI only for a clean sim or a hard timing budget.
        self.declare_parameter('nlp_solver_type', 'SQP')
        # 'ERK' (default, continuous ODE) or 'DISCRETE' (RK4 one-step map via
        # model.disc_dyn_expr). Consumed at model-build time; a restart is
        # required to change it (not hot-reloadable — the base parameter callback
        # rejects unrecognized params).
        self.declare_parameter('integrator_type', 'ERK')
        # Unified name across backends (CasADi controller uses the same param). Holds
        # the generated acados C-code + compiled model here.
        self.declare_parameter('code_gen_directory',
                               os.path.join(
                                   get_package_share_directory('trajectory_following_ros2'),
                                   'data', 'model'))
        self.declare_parameter('obstacle_slack_weight', 100.0)
        # In-solver input-rate (slew) limiting. acados otherwise only penalizes the
        # input rate via the Rd cost and never bounds it, so the solver can plan a
        # slew faster than the actuator and get post-clipped (tracking mismatch). When
        # enabled, a soft stage-0 constraint bounds |u0 - u_prev| to MAX_JERK*dt and
        # MAX_STEER_RATE*dt — parity with the CasADi backend. Restart-only (changes the
        # OCP structure; needs generate_mpc_model=True to re-codegen).
        self.declare_parameter('enforce_input_rate_constraint', True)
        self.declare_parameter('input_rate_slack_weight', 1000.0)

    def _init_solver(self) -> Optional[BaseSolver]:
        stage_cost_type = self.get_parameter('stage_cost_type').value
        terminal_cost_type = self.get_parameter('terminal_cost_type').value

        # The Rd input-rate penalty and obstacle/CBF constraints live ONLY in the
        # EXTERNAL cost branch. LINEAR_LS/NONLINEAR_LS silently drop Rd (steering then
        # jitters) and cannot carry obstacle constraints — warn loudly so a non-EXTERNAL
        # choice is a deliberate decision, not a silent regression.
        if stage_cost_type != 'EXTERNAL':
            self.get_logger().warn(
                f"stage_cost_type='{stage_cost_type}' (not EXTERNAL): the Rd input-rate "
                "penalty is dropped (steering may jitter) and obstacle/CBF constraints are "
                "unavailable. Set stage_cost_type:=EXTERNAL to enable them.")
        if terminal_cost_type != 'EXTERNAL':
            self.get_logger().warn(
                f"terminal_cost_type='{terminal_cost_type}' (not EXTERNAL): no terminal Rd "
                "rate penalty. Set terminal_cost_type:=EXTERNAL for parity with the stage cost.")

        max_iter = int(self.get_parameter('max_iter').value)
        tol = self.get_parameter('termination_condition').value
        scale_cost = self.get_parameter('scale_cost').value
        generate = self.get_parameter('generate_mpc_model').value
        with_cython = self.get_parameter('build_with_cython').value
        qp_solver = self.get_parameter('qp_solver').value
        nlp_solver_type = self.get_parameter('nlp_solver_type').value
        integrator_type = str(self.get_parameter('integrator_type').value).upper()
        if integrator_type not in ('ERK', 'DISCRETE'):
            self.get_logger().warn(
                f"integrator_type='{integrator_type}' not supported by this node "
                "(only 'ERK'/'DISCRETE'); falling back to 'ERK'.")
            integrator_type = 'ERK'
        model_dir = self.get_parameter('code_gen_directory').value
        num_obstacles = self.get_parameter('num_obstacles').value
        collision_method = self.get_parameter('obstacle_collision_avoidance_method').value
        ego_radius = self.get_parameter('ego_radius').value
        safe_distance = self.get_parameter('safe_distance').value
        obstacle_slack_weight = self.get_parameter('obstacle_slack_weight').value
        enforce_input_rate = bool(self.get_parameter('enforce_input_rate_constraint').value)
        input_rate_slack_weight = float(self.get_parameter('input_rate_slack_weight').value)

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

        if generate:
            self.get_logger().info(
                'Generating acados C-code and compiling the OCP solver'
                + (' (+ Cython wrapper)' if with_cython else '')
                + ' — first run can take ~30s-2min and the node will appear idle; '
                'set generate_mpc_model=False to reuse the built model...')
        else:
            self.get_logger().info('Loading pre-built acados OCP solver...')

        self.get_logger().info(f'acados integrator_type: {integrator_type}')
        if enforce_input_rate:
            self.get_logger().info(
                'acados input-rate constraint: ON (soft stage-0 bound |u0 - u_prev| <= '
                f'[jerk {self.MAX_JERK:.3g} m/s^3, steer_rate {self.MAX_STEER_RATE:.3g} rad/s] * dt, '
                f'slack weight {input_rate_slack_weight:.3g}).')
        else:
            self.get_logger().info(
                'acados input-rate constraint: OFF (input rate penalized only via Rd cost).')

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
            integrator_type=integrator_type,
            generate=True,
            build=generate,
            with_cython=with_cython,
            num_iterations=max_iter,
            tolerance=tol,
            mpc_config_file=config_path,
            code_export_directory=build_path,
            num_obstacles=num_obstacles,
            ego_radius=ego_radius,
            safe_distance=safe_distance,
            obstacle_slack_weight=obstacle_slack_weight,
            steer_rate_max=(self.MAX_STEER_RATE if enforce_input_rate else None),
            jerk_max=(self.MAX_JERK if enforce_input_rate else None),
            input_rate_slack_weight=input_rate_slack_weight,
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
            safe_distance=safe_distance,
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
