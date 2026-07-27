import ctypes
import hashlib
import logging
import os
import sys
import tempfile
import time
from typing import Optional

try:
    # For flushing the C runtime's stdio buffers: the generated acados C code
    # printf()s, and when stdout is a file/pipe those messages sit in the C buffer —
    # they must be flushed while the fds still point at /dev/null or the "silenced"
    # output simply appears after the redirect window closes.
    _libc = ctypes.CDLL(None)
except OSError:  # non-POSIX fallback: silencing degrades gracefully
    _libc = None

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
                 Rd: Optional[np.ndarray] = None,
                 dt: float = 0.05,
                 u_min: Optional[np.ndarray] = None,
                 u_max: Optional[np.ndarray] = None,
                 rate_max: Optional[np.ndarray] = None,
                 failure_dump_file: str = '',
                 solver_config_file: str = '',
                 suppress_solver_output: bool = True,
                 logger=None):
        self._controller = controller
        self._ros_logger = logger
        self._horizon = horizon
        self._wheelbase = wheelbase
        self._stage_cost_type = stage_cost_type
        self._terminal_cost_type = terminal_cost_type
        self._num_obstacles = num_obstacles
        self._ego_radius = ego_radius
        self._obstacle_states: Optional[np.ndarray] = None  # (3*n_obs, N+1)
        self._has_weight_params = has_weight_params
        # Input-rate (slew) bound applied to stage 0 each tick; None disables it.
        # rate_max is [max_jerk (m/s^3), max_steer_rate (rad/s)]; see
        # _apply_input_rate_bound.
        self._dt = float(dt)
        self._u_min = (np.asarray(u_min, dtype=float) if u_min is not None
                       else np.array([-np.inf, -np.inf]))
        self._u_max = (np.asarray(u_max, dtype=float) if u_max is not None
                       else np.array([np.inf, np.inf]))
        self._rate_max = np.asarray(rate_max, dtype=float) if rate_max is not None else None
        self._failure_dump_file = os.path.expanduser(str(failure_dump_file).strip())
        if self._failure_dump_file and not self._failure_dump_file.endswith('.npz'):
            self._failure_dump_file += '.npz'
        self._failure_dump_attempted = False
        self._failure_dump_written = False
        self._solver_config_file = os.path.abspath(
            os.path.expanduser(str(solver_config_file).strip())
        ) if solver_config_file else ''
        self._stage0_lbu = self._u_min.copy()
        self._stage0_ubu = self._u_max.copy()
        # Slow-solve diagnostics: any solve above 80% of the tick budget gets its SQP
        # statistics logged (throttled), failure or not — an optimal 1 s solve breaks
        # real time just as surely as a failed one.
        self._slow_solve_threshold = 0.8 * self._dt
        self._slow_solve_count = 0
        self._slow_solve_worst = 0.0
        self._slow_solve_last_log = 0.0
        # The generated acados C code printf()s directly (e.g. 'ocp_nlp_sqp: maximum
        # iterations reached' on every budget-limited solve), bypassing Python logging
        # entirely — at an on-path obstacle that is 90+ identical lines. When
        # suppressed, fds 1/2 are pointed at /dev/null for the duration of the C solve
        # call only; the throttled status-2 summary in solve() keeps the condition
        # diagnosable.
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY) if suppress_solver_output else None
        self._budget_limited_count = 0
        self._budget_limited_last_log = 0.0
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

    def _apply_input_rate_bound(self, u_prev: np.ndarray) -> None:
        """Bound the applied command's slew: ``|u0 - u_prev| <= rate_max * dt``.

        Implemented as a per-tick tightening of the stage-0 input box rather than as a
        generated constraint, because the OCP already box-bounds both inputs at every
        stage (``idxbu``) and acados lets those bounds be overridden per stage at
        runtime — the same mechanism the initial state uses just above. Nothing is
        generated, so this needs no re-codegen and no particular acados version.

        Stage 0 only: ``u_prev`` is the single command applied on the previous tick, so
        a rate bound is physically meaningful only for ``u0``, the command that reaches
        the actuator this tick. Interior-stage rate stays shaped by the ``Rd`` cost (a
        true per-stage rate bound would need the inputs augmented into the state).

        The bound is hard rather than slacked, and cannot make the QP infeasible:
        ``u_prev`` is clipped into the input box first, so the intersection of
        ``[u_prev - r, u_prev + r]`` with the box always contains ``u_prev`` and is
        therefore non-empty.
        """
        self._stage0_lbu = self._u_min.copy()
        self._stage0_ubu = self._u_max.copy()
        if self._rate_max is None:
            return
        u_prev = np.clip(np.asarray(u_prev, dtype=float).flatten(),
                         self._u_min, self._u_max)
        step = self._rate_max * self._dt
        self._stage0_lbu = np.maximum(self._u_min, u_prev - step)
        self._stage0_ubu = np.minimum(self._u_max, u_prev + step)
        self._controller.constraints_set(
            0, 'lbu', self._stage0_lbu)
        self._controller.constraints_set(
            0, 'ubu', self._stage0_ubu)

    def _solve_raw(self):
        """Run the C solve, silencing the solver's own fd-level prints when configured.

        The redirect covers only the C call, but fds 1/2 are process-wide, so another
        thread's output landing inside that ~ms window is swallowed too — an accepted
        cost at 20 Hz. Python-side logging in this adapter happens outside the window.
        """
        if self._devnull_fd is None:
            return self._controller.solve()
        sys.stdout.flush()
        sys.stderr.flush()
        if _libc is not None:
            _libc.fflush(None)
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(self._devnull_fd, 1)
        os.dup2(self._devnull_fd, 2)
        try:
            return self._controller.solve()
        finally:
            # Flush the solver's buffered printf output into /dev/null BEFORE
            # restoring the fds — C stdio is block-buffered on files/pipes, so
            # without this the flood re-emerges on the restored fd later.
            if _libc is not None:
                _libc.fflush(None)
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)

    def _log(self, level: str, msg: str, *args) -> None:
        """Emit adapter diagnostics, preferring the ROS node logger.

        The module-level Python logger has no handler configured under ROS, so only
        WARNING and above reach the root ``lastResort`` handler — an ``info()`` call
        here is silently dropped. Route through the node logger when one was passed.
        """
        text = (msg % args) if args else msg
        if self._ros_logger is not None:
            getattr(self._ros_logger, 'warn' if level == 'warning' else level)(text)
        else:
            getattr(logging.getLogger(__name__), level)(text)

    def _solver_stat(self, name: str) -> np.ndarray:
        """Read one acados statistic without letting diagnostics affect control."""
        try:
            value = self._controller.get_stats(name)
            return np.asarray(value) if value is not None else np.array([])
        except Exception:
            return np.array([])

    def _dump_failure(self, x0: np.ndarray, xref: np.ndarray, u_prev: np.ndarray,
                      status, finite: bool, u: np.ndarray, x_seq: np.ndarray,
                      seed_x: np.ndarray, seed_u: np.ndarray) -> None:
        """Write the first hard-failure inputs and solver statistics for offline replay."""
        if not self._failure_dump_file or self._failure_dump_attempted:
            return
        self._failure_dump_attempted = True

        payload = {
            'x0': np.asarray(x0, dtype=float),
            'xref': np.asarray(xref, dtype=float),
            'u_prev': np.asarray(u_prev, dtype=float),
            'stage0_lbu': self._stage0_lbu,
            'stage0_ubu': self._stage0_ubu,
            'u_min': self._u_min,
            'u_max': self._u_max,
            'rate_max': (self._rate_max if self._rate_max is not None else np.array([])),
            'obstacle_states': (
                self._obstacle_states if self._obstacle_states is not None else np.array([])),
            'returned_u': np.asarray(u, dtype=float),
            'returned_x': np.asarray(x_seq, dtype=float),
            'seed_x': np.asarray(seed_x, dtype=float),
            'seed_u': np.asarray(seed_u, dtype=float),
            'status': np.array([str(status)]),
            'finite': np.array([int(finite)]),
            'horizon': np.array([self._horizon]),
            'wheelbase': np.array([self._wheelbase]),
            'ego_radius': np.array([self._ego_radius]),
            'dt': np.array([self._dt]),
            'stage_cost_type': np.array([self._stage_cost_type]),
            'terminal_cost_type': np.array([self._terminal_cost_type]),
            'solver_config_file': np.array([self._solver_config_file]),
            'num_obstacles': np.array([self._num_obstacles]),
            'has_weight_params': np.array([int(self._has_weight_params)]),
            'residuals': self._solver_stat('residuals'),
            'qp_stat': self._solver_stat('qp_stat'),
            'sqp_iter': self._solver_stat('sqp_iter'),
            'time_tot': self._solver_stat('time_tot'),
        }
        if self._has_weight_params:
            payload.update({
                'Q_diag': self._Q_diag,
                'R_diag': self._R_diag,
                'Qe_diag': self._Qe_diag,
                'Rd_diag': self._Rd_diag,
            })

        try:
            parent = os.path.dirname(self._failure_dump_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez_compressed(self._failure_dump_file, **payload)
            self._failure_dump_written = True
            self._log(
                'warning', 'Saved first acados hard-failure replay dump to %s',
                self._failure_dump_file)
        except (OSError, TypeError, ValueError) as exc:
            self._log(
                'warning', 'Could not save acados failure replay dump to %s: %s',
                self._failure_dump_file, exc)

    def _log_slow_solve(self, status, solve_time: float,
                        solve_time_cpu: float) -> None:
        """Log SQP statistics for solves that blow the tick budget (throttled).

        Aggregates between logs (at most one line per second) so a batch of slow
        solves at an obstacle is reported, not turned into its own flood.
        """
        self._slow_solve_count += 1
        self._slow_solve_worst = max(self._slow_solve_worst, solve_time)
        now = time.monotonic()
        if now - self._slow_solve_last_log < 1.0:
            return
        self._slow_solve_last_log = now

        def _ms(name):
            arr = self._solver_stat(name)
            try:
                return float(np.asarray(arr).flatten()[0]) * 1e3
            except (IndexError, TypeError, ValueError):
                return float('nan')

        sqp_iter = self._solver_stat('sqp_iter')
        residuals = self._solver_stat('residuals')
        # cpu << wall on a slow solve means the process was descheduled (OS/CPU
        # contention), not that the solver burned that much compute.
        self._log(
            'warning', 'Slow acados solve: %.1f ms wall / %.1f ms cpu (budget %.0f ms), '
            'status=%s, sqp_iter=%s, time_qp=%.1f ms, time_lin=%.1f ms, '
            'time_reg=%.1f ms, acados time_tot=%.1f ms, residuals=%s; '
            '%d slow solve(s) since last report, worst %.1f ms',
            solve_time * 1e3, solve_time_cpu * 1e3,
            self._slow_solve_threshold * 1e3, status,
            np.asarray(sqp_iter).flatten(), _ms('time_qp'), _ms('time_lin'),
            _ms('time_reg'), _ms('time_tot'),
            np.array2string(np.asarray(residuals, dtype=float),
                                             precision=2),
            self._slow_solve_count, self._slow_solve_worst * 1e3)
        self._slow_solve_count = 0
        self._slow_solve_worst = 0.0

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
        self._solve_raw()  # initial guess

    def solve(self, x0: np.ndarray, xref: np.ndarray,
              u_prev: np.ndarray) -> SolverResult:
        # x0: (4,); xref: (4, N+1); u_prev: (2,)
        self._controller.constraints_set(0, 'lbx', x0)
        self._controller.constraints_set(0, 'ubx', x0)
        self._apply_input_rate_bound(u_prev)

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

        seed_x = np.array([])
        seed_u = np.array([])
        if self._failure_dump_file and not self._failure_dump_attempted:
            try:
                seed_x = np.array([
                    self._controller.get(i, 'x')
                    for i in range(self._horizon + 1)]).T
                seed_u = np.array([
                    self._controller.get(i, 'u')
                    for i in range(self._horizon)]).T
            except Exception:
                # The primary x0/xref/u_prev snapshot is still useful on older wrappers
                # that cannot expose the current primal warm-start iterate.
                seed_x = np.array([])
                seed_u = np.array([])

        t0 = time.process_time()
        t0_wall = time.monotonic()
        status = self._solve_raw()
        solve_time_cpu = time.process_time() - t0
        solve_time = time.monotonic() - t0_wall

        if status == 2:
            # Replaces the C-level per-solve 'maximum iterations reached' print with a
            # throttled aggregate, so budget-limited batches stay visible without the
            # 90-line flood.
            self._budget_limited_count += 1
            now = time.monotonic()
            if now - self._budget_limited_last_log >= 5.0:
                self._budget_limited_last_log = now
                self._log(
                    'info',
                    '%d budget-limited acados solve(s) since last report (status=2: '
                    'SQP hit max_iter; near-converged iterate applied, RTI-style)',
                    self._budget_limited_count)
                self._budget_limited_count = 0

        u = self._controller.get(0, 'u')
        x_seq = np.array(
            [self._controller.get(i, 'x') for i in range(self._horizon + 1)]).T  # (4, N+1)
        u_seq = np.array(
            [self._controller.get(i, 'u') for i in range(self._horizon)]).T       # (2, N)

        # solve_time is measured with time.monotonic() around the C call, not from
        # acados' own 'time_tot': that statistic is taken from a settable clock, so a
        # system-clock step (e.g. WSL2 time resync) makes it report absurd — even
        # negative — durations. time_tot is still logged in the slow-solve line.
        if solve_time > self._slow_solve_threshold:
            self._log_slow_solve(status, solve_time, solve_time_cpu)

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
            self._dump_failure(
                x0, xref, u_prev, status, finite, u, x_seq, seed_x, seed_u)
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
            # u_prev echo, or the predicted-path debug topic. velocity_cmd is zeroed
            # explicitly rather than read off the sanitized x_seq: tiling x0 would
            # make it the *current* speed, i.e. "hold speed, steer straight" — the
            # worst command to publish while the solver is in a degenerate state, and
            # it would be applied for up to 4 more ticks before the consecutive-failure
            # fallback zeroes commands.
            u = np.zeros(2)
            u_seq = np.zeros((2, self._horizon))
            x_seq = np.tile(np.asarray(x0, dtype=float).reshape(4, 1),
                            (1, self._horizon + 1))

        return SolverResult(
            accel_cmd=float(u[0]),
            steering_cmd=float(u[1]),
            velocity_cmd=float(x_seq[2, 1]) if finite else 0.0,
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
        # input rate via the Rd cost and never bounds it, so the solver can plan a slew
        # faster than the actuator and get post-clipped (tracking mismatch). When
        # enabled, the stage-0 input box is tightened each tick to
        # |u0 - u_prev| <= [MAX_JERK, MAX_STEER_RATE] * dt — parity with the CasADi
        # backend's slacked rate bound. Applied at runtime via constraints_set, so no
        # re-codegen is needed to change it (restart-only only because the adapter reads
        # it once at construction).
        self.declare_parameter('enforce_input_rate_constraint', True)
        # Optional one-shot .npz dump of the first hard failure, written before reset.
        self.declare_parameter('acados_failure_dump_file', '')
        # Silence the generated C code's per-solve printf flood (e.g. 'ocp_nlp_sqp:
        # maximum iterations reached'); a throttled Python-side aggregate replaces it.
        # Same name/default as the CasADi node's parameter.
        self.declare_parameter('suppress_solver_output', True)

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
        # acados' generated Cython Makefile does not quote paths, so a space anywhere in
        # the code-generation directory breaks the build (`cython: error: unknown option
        # -`) AND the runtime load (the generated .json and compiled .so carry absolute
        # paths). Copying artifacts back into a spaced directory would only move the
        # failure to load time, so redirect to a space-free mirror under the system temp
        # dir and use it for both codegen and load. The mirror is deterministic (hashed
        # from the original path) so the built model is reused across launches.
        abs_model_dir = os.path.abspath(model_dir) if model_dir else os.getcwd()
        if ' ' in abs_model_dir:
            digest = hashlib.sha1(abs_model_dir.encode('utf-8')).hexdigest()[:12]
            redirected = os.path.join(
                tempfile.gettempdir(), f'trajectory_acados_codegen_{digest}')
            self.get_logger().warn(
                f"code_gen_directory '{abs_model_dir}' contains a space, which breaks "
                f"acados Cython code generation; redirecting to '{redirected}'. Set a "
                "space-free code_gen_directory to silence this.")
            model_dir = redirected
        num_obstacles = self.get_parameter('num_obstacles').value
        collision_method = self.get_parameter('obstacle_collision_avoidance_method').value
        safe_distance = self.get_parameter('safe_distance').value
        obstacle_slack_weight = self.get_parameter('obstacle_slack_weight').value
        enforce_input_rate = bool(self.get_parameter('enforce_input_rate_constraint').value)
        failure_dump_file = self.get_parameter('acados_failure_dump_file').value

        ego_radius = self.effective_ego_radius()

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
                'acados input-rate constraint: ON (stage-0 steering box tightened each tick '
                f'to |delta0 - delta_prev| <= {self.MAX_STEER_RATE:.3g} rad/s * '
                f'{self.sample_time:.3g} s = '
                f'{np.degrees(self.MAX_STEER_RATE * self.sample_time):.2f} deg/tick; '
                'acceleration rate is left to the Rd cost).')
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
            dt=self.sample_time,
            u_min=np.array([self.MAX_DECEL, self.MIN_STEER_ANGLE]),
            u_max=np.array([self.MAX_ACCEL, self.MAX_STEER_ANGLE]),
            # Steering rate only; the acceleration rate is left unbounded (inf) and
            # stays shaped by the Rd cost. max_steer_rate is a real servo slew limit,
            # but max_jerk is a comfort/Bryson-weight parameter, not an actuator limit —
            # and acceleration is not directly actuated here anyway (the published
            # command is speed + steering angle). Enforcing max_jerk as a hard bound at
            # its 1.5 m/s^3 default lets the acceleration move only 0.075 m/s^2 per
            # 0.05 s tick, so crossing the +/-3 m/s^2 range takes 2 s; measured on the
            # on-path-obstacle hairpin rollout that starves the accel input badly enough
            # to trip the consecutive-failure fallback (7 failures, 0.548 m max CTE vs
            # 0 failures, 0.086 m with steering-rate only).
            rate_max=(np.array([np.inf, self.MAX_STEER_RATE])
                      if enforce_input_rate else None),
            failure_dump_file=failure_dump_file,
            solver_config_file=config_path,
            suppress_solver_output=bool(
                self.get_parameter('suppress_solver_output').value),
            logger=self.get_logger(),
        )


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
