import csv
import logging
import math
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import interpolate
from scipy.spatial import distance

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.clock import JumpThreshold, TimeJump
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult, ParameterType

from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf_transformations

from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import AccelWithCovarianceStamped, PoseStamped

from trajectory_following_ros2.utils.Trajectory import Trajectory
import trajectory_following_ros2.utils.trajectory_utils as trajectory_utils
from trajectory_following_ros2.backends.base_solver import BaseSolver, SolverResult
from trajectory_following_ros2.utils.mpc_weight_utils import bryson_weights

try:
    OBSTACLES_AVAILABLE = True
    from derived_object_msgs.msg import ObjectArray
except ImportError:
    OBSTACLES_AVAILABLE = False

_STALE_ODOM_THRESHOLD_S = 0.5  # seconds before odom is considered stale
# Seconds without an ObjectArray before the obstacle feed is reported as silent. Well
# above any plausible publish period (a 1 Hz feed is already slow for avoidance) so a
# normal feed never trips it.
_SILENT_OBSTACLE_FEED_S = 2.0


class BaseTrajectoryTracker(Node, ABC):
    """
    Common ROS 2 boilerplate for all trajectory tracking controllers.

    Subclasses must implement:
      _init_solver()               → Optional[BaseSolver]
      _declare_backend_parameters() (optional override, default no-op)

    For non-MPC controllers (e.g. Pure Pursuit) that do not use BaseSolver,
    override _control_timer_callback() completely and return None from _init_solver().

    Override ``_odom_topic_default`` as a class attribute to change the default
    odometry topic (MPC nodes: ``odometry/local``; PurePursuit: ``odometry/filtered``).
    """

    _odom_topic_default: str = 'odometry/local'

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._declare_common_parameters()
        self._declare_backend_parameters()
        self._read_parameters()
        self._init_state()
        self._setup_tf()
        self._setup_pub_sub()
        self._solver: Optional[BaseSolver] = self._init_solver()
        self._setup_solver_log()

    # ------------------------------------------------------------------
    # Abstract / override points
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_solver(self) -> Optional[BaseSolver]:
        """Create and return the backend solver adapter. Return None for non-MPC nodes."""

    def _declare_backend_parameters(self) -> None:
        """Declare backend-specific ROS parameters. Override in subclasses."""

    def _setup_solver_log(self) -> None:
        """Open the per-solve stats CSV if ``solver_log_file`` is set (else disabled).

        Backend-agnostic: writes one row per solve from the returned ``SolverResult``,
        so every MPC backend is covered without touching the adapters. Failures here
        never abort startup — a bad path just disables logging with a warning.
        """
        self._solver_log_fh = None
        self._solver_log_writer = None
        path = str(self.get_parameter('solver_log_file').value).strip()
        if not path or self._solver is None:
            return
        try:
            path = os.path.expanduser(path)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._solver_log_fh = open(path, 'w', newline='')
            self._solver_log_writer = csv.writer(self._solver_log_fh)
            self._solver_log_writer.writerow([
                'wall_time', 'ref_idx', 'solve_time_ms', 'status', 'is_optimal',
                'consecutive_failures', 'accel_cmd', 'steering_cmd', 'velocity_cmd', 'error',
                'n_selected', 'sel_id', 'sel_side', 'sel_min_clearance',
                'sel_ego_clearance',
                'safety_stop', 'safety_reason', 'safety_obstacle_id',
                'physical_clearance', 'closing_speed', 'stopping_room',
                'applied_accel', 'applied_steering', 'applied_speed',
                'tick_interval_ms', 'reference_ms', 'obstacle_ms', 'solver_wall_ms',
                'pre_log_ms', 'previous_log_write_ms',
            ])
            self._solver_log_fh.flush()
            self.get_logger().info(f'Solver stats logging to {path}')
        except OSError as exc:
            self.get_logger().warn(f'Could not open solver_log_file ({path}): {exc}; disabling.')
            self._solver_log_fh = None
            self._solver_log_writer = None

    def _obstacle_diag(self, result: SolverResult, selected: list, ego_pose=None):
        """Per-tick obstacle diagnostics for the stats CSV.

        Returns ``(n_selected, sel_id, sel_side, sel_min_clearance,
        sel_ego_clearance)`` for the top-ranked selected obstacle: its id, its
        committed go-around side (``self._keepout_side_hints`` after this tick's
        projection), and two clearances to that obstacle's keep-out boundary, both
        negative when inside the keep-out:

        - ``sel_min_clearance`` — minimum over the *planned* trajectory
          (``result.x_sequence``). The constraint is slacked, so a plan dipping
          into the margin is expected and does not mean the vehicle went there.
        - ``sel_ego_clearance`` — the *executed* clearance: where the vehicle
          measurably was this tick. This is the one to judge encroachment by.

        Both measure the worst of the ego collision discs (``ego_disc_offsets``)
        against ``ego_radius + obstacle_radius + safe_distance`` — the same discs
        the OCP constrains. With the default single disc on the rear-axle reference
        point that is the reference point alone, and a body extending past the rear
        axle can then clip an obstacle at positive clearance; configure the discs to
        cover the footprint and the reported number covers the body too.

        A weak detour shows clearance dipping negative with the side steady; a
        side-hint flip shows ``sel_side`` changing sign tick to tick.
        """
        n_sel = len(selected) if selected else 0
        if n_sel == 0:
            return 0, -1, 0, float('nan'), float('nan')
        obs = selected[0]
        sel_id = obs.get('id', -1)
        sel_side = self._keepout_side_hints.get(sel_id, 0)
        min_clear = float('nan')
        ego_clear = float('nan')
        try:
            centre = np.asarray(obs['state'][:2], dtype=float)
            keepout = float(self._keepout_radii([obs])[0])
            offsets = self.effective_ego_disc_offsets()

            def _clearance(x, y, yaw):
                discs = trajectory_utils.ego_disc_centres(x, y, yaw, offsets)
                d = np.hypot(discs[..., 0] - centre[0], discs[..., 1] - centre[1])
                return float(np.min(d) - keepout)

            xseq = getattr(result, 'x_sequence', None)
            if xseq is not None:
                min_clear = _clearance(xseq[0, :], xseq[1, :], xseq[3, :])
            if ego_pose is not None:
                ego_clear = _clearance(ego_pose[0], ego_pose[1], ego_pose[2])
        except (ValueError, IndexError, TypeError):
            pass
        return n_sel, sel_id, sel_side, min_clear, ego_clear

    def _obstacle_safety_check(self, selected: list, ego_pose,
                               speed: float, tick_interval_ms: float) -> dict:
        """State-based obstacle braking envelope, with per-tick diagnostics.

        Fires when the closing speed toward a selected obstacle leaves less
        clearance than the room needed to stop: one reaction tick at the current
        closing speed (the *observed* tick interval, so a cadence stall widens the
        envelope) plus the braking distance at ``max_decel``.

        Returns a dict — ``stop`` (bool), ``obstacle_id``, ``physical_clearance``,
        ``closing_speed``, ``stopping_room``, ``margin`` — describing the most
        critical obstacle/ego-disc pair, i.e. the smallest
        ``clearance - stopping_room`` margin, **whether or not the envelope fired**.
        So every logged row shows how close that tick came, not just the ones that
        braked. ``physical_clearance`` is body-to-body
        (``distance - ego_radius - obstacle_radius``): it excludes ``safe_distance``,
        so negative means actual overlap, while the envelope itself is judged on the
        full keep-out clearance.
        """
        diag = {'stop': False, 'obstacle_id': -1, 'physical_clearance': float('nan'),
                'closing_speed': float('nan'), 'stopping_room': float('nan'),
                'margin': float('nan')}
        if not selected:
            return diag
        # No low-speed early-out: a stopped vehicle cannot fire (closing ≈ 0 fails
        # the closing > 0.05 gate) but its clearance/margin must still be reported —
        # skipping here blinded the tick right after a safety stop, and the solver's
        # creep command was applied unchecked on exactly those ticks.
        x, y, yaw = ego_pose
        offsets = self.effective_ego_disc_offsets()
        # ego_disc_centres returns (n_points, n_discs, 2); this is a single pose, so
        # flatten to one row per disc — reducing over the wrong axis here silently
        # turns the distance into per-coordinate magnitudes.
        discs = trajectory_utils.ego_disc_centres(x, y, yaw, offsets).reshape(-1, 2)
        ego_velocity = np.array([speed * np.cos(yaw), speed * np.sin(yaw)])
        decel = max(abs(float(self.MAX_DECEL)), 1e-3)
        # Reaction window: one observed tick plus the actuation lag — the plant keeps
        # executing the *previous* command while a brake command works through the
        # lag, so measured speed can still be rising for that long after a fire.
        reaction = min(max(self.sample_time, tick_interval_ms * 1e-3)
                       + self._ENVELOPE_ACTUATION_LAG_S, 0.7)
        # If the last applied command was accelerating, assume it keeps doing so
        # through the reaction window: an envelope that ignores this fires only
        # after a full-throttle launch toward the keep-out has already made the
        # stop physically impossible.
        accel_headroom = max(0.0, float(getattr(self, 'acc_cmd', 0.0)))
        margin_gate = float(self._gp('safe_distance'))
        # Report rank: an obstacle that actually fired outranks any that did not, and
        # within each group the smallest margin wins. Without the first key a stop can
        # be reported alongside a *different* obstacle's (non-closing) numbers.
        best_rank = (2, float('inf'))

        for obs in selected:
            centre = np.asarray(obs['state'][:2], dtype=float)
            obstacle_velocity = np.asarray(obs.get('velocity', [0.0, 0.0]), dtype=float)
            keepout = float(self._keepout_radii([obs])[0])
            toward = centre[None, :] - discs
            distance = np.linalg.norm(toward, axis=1)
            unit_toward = toward / np.maximum(distance[:, None], 1e-9)
            relative_velocity = ego_velocity - obstacle_velocity
            closing = unit_toward @ relative_velocity
            clearance = distance - keepout
            reacted = np.maximum(closing, 0.0) + accel_headroom * reaction
            stopping_room = (closing * reaction
                             + 0.5 * accel_headroom * reaction**2
                             + reacted**2 / (2.0 * decel))
            margin = clearance - stopping_room
            fires = (closing > 0.05) & (margin <= 0.0)
            candidates = np.flatnonzero(fires) if fires.any() else np.arange(margin.size)
            worst = int(candidates[np.argmin(margin[candidates])])
            if fires.any():
                diag['stop'] = True
            rank = (0 if fires.any() else 1, float(margin[worst]))
            if rank < best_rank:
                best_rank = rank
                diag.update({
                    'obstacle_id': obs.get('id', -1),
                    'physical_clearance': float(clearance[worst] + margin_gate),
                    'closing_speed': float(closing[worst]),
                    'stopping_room': float(stopping_room[worst]),
                    'margin': float(margin[worst]),
                })
        return diag

    #: Keep-out margin (m) the envelope must recover before a latched safety hold
    #: releases. At standstill stopping_room ≈ 0, so this is effectively "clearance
    #: has grown this far past the keep-out boundary again".
    _SAFETY_HOLD_RELEASE_MARGIN = 0.05

    #: Actuation lag (s) budgeted into the envelope's reaction window: the plant
    #: keeps executing the previous command while a brake command works through
    #: the command filter, so measured speed can still rise this long after a fire.
    _ENVELOPE_ACTUATION_LAG_S = 0.2

    #: Body-to-body clearance (m) a held vehicle must keep in hand: while a safety
    #: hold is latched, a closing proposal is admitted only if it could stop with
    #: at least this much physical clearance remaining.
    _HOLD_PHYSICAL_FLOOR = 0.05

    def _hold_admissible_command(self, result: SolverResult,
                                 selected: list, ego_pose) -> bool:
        """Whether a proposal may be applied while a safety hold is latched.

        A held vehicle must still be allowed to *leave* — the solver's
        reverse/step-aside escape is exactly the recovery path. A proposal is
        judged against every selected obstacle still inside the release margin
        (not just the one that fired: between two obstacles, backing away from
        the one ahead can close on the one behind) and passes when either

        - it does not close on that keep-out, or
        - it closes slowly enough to stop with ``_HOLD_PHYSICAL_FLOOR`` of
          body-to-body clearance still in hand. The comfort margin
          (``safe_distance``) is spendable during recovery, exactly as the OCP's
          slack treats it — without this, a vehicle parked just inside the
          comfort band admits no command at all and the hold becomes a
          standstill deadlock (observed as a 1400-tick latched stop).
        """
        x, y, yaw = ego_pose
        offsets = self.effective_ego_disc_offsets()
        discs = trajectory_utils.ego_disc_centres(x, y, yaw, offsets).reshape(-1, 2)
        command_velocity = (float(result.velocity_cmd)
                            * np.array([np.cos(yaw), np.sin(yaw)]))
        decel = max(abs(float(self.MAX_DECEL)), 1e-3)
        reaction = self.sample_time + self._ENVELOPE_ACTUATION_LAG_S
        margin_gate = float(self._gp('safe_distance'))
        keepouts = self._keepout_radii(selected)
        for obs, keepout in zip(selected, keepouts):
            centre = np.asarray(obs['state'][:2], dtype=float)
            toward = centre[None, :] - discs
            dist = np.linalg.norm(toward, axis=1)
            clearance = float(dist.min()) - float(keepout)
            if clearance >= self._SAFETY_HOLD_RELEASE_MARGIN:
                continue
            unit = toward / np.maximum(dist[:, None], 1e-9)
            closing = float(np.max(unit @ command_velocity))
            if closing <= 0.02:
                continue
            stopping_room = closing * reaction + closing**2 / (2.0 * decel)
            if stopping_room > clearance + margin_gate - self._HOLD_PHYSICAL_FLOOR:
                return False
        return True

    def _safety_brake_command(self, result: SolverResult, speed: float):
        """Bounded-decel stop action: ``(accel, steering, speed)`` for a safety stop.

        Sheds one tick of speed at ``max_decel`` — the same deceleration the
        envelope's stopping-room formula assumes — while keeping the solver's
        steering, so braking mid-avoidance-arc does not straighten the wheel the
        way a hard zero did (that straightened lurch is what ratcheted the vehicle
        into contact). Falls back to the last applied steering when the iterate is
        not trusted.

        The commanded speed is monotone over a braking sequence: measured speed can
        keep *rising* through the actuation lag right after the envelope fires, and
        ``measured - step`` alone would then command increasing speeds while
        nominally braking. The bound resets whenever a tick publishes normally.
        """
        decel = max(abs(float(self.MAX_DECEL)), 1e-3)
        step = decel * self.sample_time
        self._brake_speed_bound = max(
            0.0, min(abs(speed), self._brake_speed_bound) - step)
        speed_cmd = math.copysign(self._brake_speed_bound, speed)
        accel_cmd = -math.copysign(decel, speed) if abs(speed) > 1e-3 else 0.0
        steering = float(result.steering_cmd) if result.is_optimal else self.delta_cmd
        steering = float(np.clip(steering, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE))
        return accel_cmd, steering, speed_cmd

    def _log_solver_stats(self, result: SolverResult, selected: list = None,
                          timing: Optional[dict] = None, ego_pose=None,
                          safety: Optional[dict] = None, safety_reason: str = '',
                          applied=None) -> None:
        """Append one CSV row for this solve (no-op when logging is disabled).

        ``accel_cmd``/``steering_cmd``/``velocity_cmd`` are what the solver
        *proposed*; ``applied_*`` are what the controller actually published this
        tick (zeros on a safety stop, the held command on a bridged failure), so
        the two are never conflated. ``safety_reason`` names the intervention that
        zeroed the tick (empty when the proposed command was applied).
        """
        if self._solver_log_writer is None:
            return
        try:
            timing = timing or {}
            safety = safety or {}
            applied = applied if applied is not None else (
                float('nan'), float('nan'), float('nan'))
            n_sel, sel_id, sel_side, sel_clear, ego_clear = self._obstacle_diag(
                result, selected or [], ego_pose)
            previous_log_ms = self._last_solver_log_ms
            log_started = time.monotonic()
            self._solver_log_writer.writerow([
                f'{time.time():.6f}',
                getattr(self, 'current_idx', -1),
                f'{result.solve_time * 1e3:.4f}',
                result.status,
                int(result.is_optimal),
                self._consecutive_failures,
                f'{result.accel_cmd:.6f}',
                f'{result.steering_cmd:.6f}',
                f'{result.velocity_cmd:.6f}',
                result.error or '',
                n_sel,
                sel_id,
                sel_side,
                f'{sel_clear:.6f}',
                f'{ego_clear:.6f}',
                int(bool(safety_reason)),
                safety_reason,
                safety.get('obstacle_id', -1),
                f"{safety.get('physical_clearance', float('nan')):.6f}",
                f"{safety.get('closing_speed', float('nan')):.6f}",
                f"{safety.get('stopping_room', float('nan')):.6f}",
                f'{applied[0]:.6f}',
                f'{applied[1]:.6f}',
                f'{applied[2]:.6f}',
                f"{timing.get('tick_interval_ms', float('nan')):.4f}",
                f"{timing.get('reference_ms', float('nan')):.4f}",
                f"{timing.get('obstacle_ms', float('nan')):.4f}",
                f"{timing.get('solver_wall_ms', float('nan')):.4f}",
                f"{timing.get('pre_log_ms', float('nan')):.4f}",
                f'{previous_log_ms:.4f}',
            ])
            self._solver_log_fh.flush()
            self._last_solver_log_ms = (time.monotonic() - log_started) * 1e3
        except (OSError, ValueError):
            pass  # never let logging disturb the control loop

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _declare_common_parameters(self):
        self.declare_parameter('robot_frame', 'base_link',
                               ParameterDescriptor(description='Local/body frame of the vehicle.'))
        self.declare_parameter('global_frame', 'odom',
                               ParameterDescriptor(description='Global/world frame (odom or map).'))
        self.declare_parameter('control_rate', 20.0)
        # Debug/viz publish rate. Runs on its own ReentrantCallbackGroup thread,
        # independent of the MPC control timer, so raising it does not slow the
        # control loop (only adds viz CPU). 10 Hz keeps the predicted path/goal
        # point within ~100 ms of the live state; raise toward control_rate (20)
        # for tighter viz, lower on resource-constrained hardware (Jetson).
        self.declare_parameter('debug_frequency', 10.0)
        # Per-solve solver-stats CSV log. Empty string (default) = disabled. When set
        # to a path, one row per solve tick is appended (backend-agnostic: works for
        # every MPC backend since it logs the returned SolverResult). Restart-only.
        self.declare_parameter('solver_log_file', '')
        self.declare_parameter(
            'solver_failure_mode', 'hold_last',
            ParameterDescriptor(description=(
                "Solver failure action: 'zero' publishes a zero command immediately; "
                "'hold_last' may bridge a failure subject to the count/time/saturation gates.")))
        self.declare_parameter(
            'solver_failure_hold_count', 1,
            ParameterDescriptor(description=(
                'Maximum consecutive failures allowed to hold the last command. '
                '0 disables the count gate.')))
        self.declare_parameter(
            'solver_failure_hold_time', 0.1,
            ParameterDescriptor(description=(
                'Maximum wall time in seconds since the last successful command publication '
                'during which hold_last may bridge a failure. 0 disables the time gate.')))
        self.declare_parameter(
            'solver_failure_zero_on_saturation', False,
            ParameterDescriptor(description=(
                'When true, never hold a command at an accel, steering, or speed limit. '
                'Default False: a legitimately saturated last-good command (max steer '
                'mid-corner, max-speed cruise) is safer to hold through a transient '
                'failure than to zero.')))
        self.declare_parameter('distance_tolerance', 0.2)
        self.declare_parameter('speed_tolerance', 0.5)
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('min_steer', -27.0)   # degrees
        self.declare_parameter('max_steer', 27.0)    # degrees
        self.declare_parameter('min_jerk', -1.5)     # m/s³
        self.declare_parameter('max_jerk', 1.5)
        self.declare_parameter('max_steer_rate', 60 / 0.17)   # deg/s
        self.declare_parameter('max_speed', 1.5)     # m/s
        self.declare_parameter('min_speed', -1.5)
        self.declare_parameter('max_accel', 3.0)     # m/s²
        self.declare_parameter('max_decel', -3.0)
        self.declare_parameter('saturate_input', True)
        self.declare_parameter('allow_reversing', True)
        self.declare_parameter('n_states', 4)
        self.declare_parameter('n_inputs', 2)
        self.declare_parameter('horizon', 25)
        self.declare_parameter('prediction_time', 0.0)
        self.declare_parameter('R', [0.1, 0.1],
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('Rd', [10.0, 100.0],
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('Q', [1.0, 1.0, 10.0, 10.0],
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('Qf', [0.04, 0.04, 0.1, 1.0],
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('path_topic', 'trajectory/path')
        self.declare_parameter(
            'path_qos', 'transient_local',
            ParameterDescriptor(description=(
                "QoS durability for the path subscription. "
                "'transient_local' (default): latching — waypoint_loader style. "
                "'volatile': live replanning — Nav2 /plan style.")))
        self.declare_parameter('speed_topic', 'trajectory/speed')
        self.declare_parameter('odom_topic', self._odom_topic_default)
        self.declare_parameter('acceleration_topic', 'accel/local')
        self.declare_parameter('ackermann_cmd_topic', 'drive')
        self.declare_parameter('publish_twist_topic', True)
        self.declare_parameter('twist_topic', 'cmd_vel')
        self.declare_parameter('desired_speed', 0.0)
        # Reference speed policy (see _apply_speed_policy / generate_reference_trajectory_by_interpolation)
        self.declare_parameter('use_speed_profile', True)    # track the recorded speed profile over the horizon
        self.declare_parameter('max_lateral_accel', 3.0)     # m/s²; curvature speed cap v ≤ sqrt(a_lat/|κ|). 0 disables
        self.declare_parameter('min_reference_speed', 0.3)   # m/s; forward creep floor to lift a noisy near-zero start
        # Reference-index advance mode. True = along-track (arc-length) projection: the
        # target index advances with longitudinal progress even when the vehicle is held
        # laterally off the line (obstacle-avoidance swerve), so it never freezes at a
        # standoff. False = legacy Euclidean distance-gate. See utils/Trajectory.py.
        self.declare_parameter('arclength_index_advance', True)
        # Forward arc-length span (m) of the arc-length projection window. Kept short
        # so the projection cannot leap to end-of-path points sitting physically near
        # the start on a closed loop; must be < loop length and > one tick's travel.
        self.declare_parameter('projection_window', 5.0)
        self.declare_parameter(
            'loop', 0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    'Trajectory repetition mode. '
                    '0 = stop at goal (default). '
                    '-1 = loop indefinitely. '
                    'N > 0 = run the trajectory N times total then stop.')))
        self.declare_parameter('n_ind_search', 10)
        self.declare_parameter('smooth_yaw', False)
        self.declare_parameter('debug', False)
        self.declare_parameter('num_obstacles', 0)
        self.declare_parameter('ego_radius', -1.0)
        # Longitudinal offsets (m, + forward from the rear-axle reference point) of the
        # ego collision discs; every disc carries ego_radius. The default [0.0] is the
        # single legacy disc on the reference point. Two or more discs cover a body that
        # extends past the rear axle without inflating one radius to the circumscribing
        # one — see trajectory_utils.resolve_ego_disc_offsets. Restart-only (the disc
        # count is baked into the OCP constraint rows at build time).
        self.declare_parameter('ego_disc_offsets', [0.0])
        # Extra keep-out buffer added to (ego_radius + obstacle_radius) when building
        # the obstacle constraint. Restart-only (baked into the OCP at build time).
        self.declare_parameter('safe_distance', 0.5)
        self.declare_parameter('obstacle_topic', 'fake_obstacles/object_array')
        self.declare_parameter('obstacle_collision_avoidance_method', 'euclidean')
        # Propagate each obstacle over the horizon at the constant velocity reported in
        # its twist, so stage k keeps out of where it will be rather than where it was.
        # A zero/absent twist reduces this exactly to a static fill; turn it off when a
        # perception feed reports velocities too noisy to extrapolate.
        self.declare_parameter('predict_obstacle_motion', True)
        # State-based braking envelope on/off (hot-reloadable). Off, the check still
        # runs and its diagnostics are logged every tick — it just never intervenes —
        # so an A/B run compares identical CSV columns.
        self.declare_parameter('obstacle_braking_envelope', True)
        self.declare_parameter('actuator_feedback_topic', '')
        self.declare_parameter('delay_compensation_enabled', False)
        self.declare_parameter('delay_compensation_method', 'forward_simulation')
        self.declare_parameter('estimated_delay', 0.0)

        self.declare_parameter(
            'executor_threads', 3,
            ParameterDescriptor(description=(
                'Number of threads for MultiThreadedExecutor. '
                'Minimum 3 (MPC timer + subscriptions + debug timer). '
                'Increase on multi-core hardware (e.g. 4-6 on Jetson Orin).')))

        # Bryson's rule weight parameters (active when use_bryson_weights: true)
        self.declare_parameter('use_bryson_weights', False)
        self.declare_parameter('max_error_x', 1.0)    # m
        self.declare_parameter('max_error_y', 0.3)    # m
        self.declare_parameter('max_error_v', 0.5)    # m/s
        self.declare_parameter('max_error_psi', 0.1)  # rad
        self.declare_parameter('bryson_max_accel', 3.0)   # m/s²
        self.declare_parameter('bryson_max_steer', 0.4)   # rads

    def _gp(self, name):
        return self.get_parameter(name).value

    def _read_parameters(self):
        self.robot_frame = self._gp('robot_frame')
        self.global_frame = self._gp('global_frame')
        self.control_rate = self._gp('control_rate')
        self.debug_frequency = self._gp('debug_frequency')
        self.solver_failure_mode = str(self._gp('solver_failure_mode')).strip().lower()
        if self.solver_failure_mode not in ('zero', 'hold_last'):
            self.get_logger().warn(
                f"Invalid solver_failure_mode='{self.solver_failure_mode}'; "
                "falling back to the fail-safe 'zero' mode.")
            self.solver_failure_mode = 'zero'
        self.solver_failure_hold_count = max(
            0, int(self._gp('solver_failure_hold_count')))
        self.solver_failure_hold_time = max(
            0.0, float(self._gp('solver_failure_hold_time')))
        self.solver_failure_zero_on_saturation = bool(
            self._gp('solver_failure_zero_on_saturation'))
        if (self.solver_failure_mode == 'hold_last'
                and self.solver_failure_hold_count == 0
                and self.solver_failure_hold_time == 0.0):
            self.get_logger().warn(
                'solver_failure_mode=hold_last has both hold gates disabled; '
                'failures will publish zero commands rather than hold without a bound.')
        self.distance_tolerance = self._gp('distance_tolerance')
        self.speed_tolerance = self._gp('speed_tolerance')
        self.WHEELBASE = self._gp('wheelbase')
        self.MAX_STEER_ANGLE = math.radians(self._gp('max_steer'))
        self.MIN_STEER_ANGLE = math.radians(self._gp('min_steer'))
        _max_steer_deg = self._gp('max_steer')
        if _max_steer_deg > 180.0:
            self.get_logger().warn(
                f'max_steer={_max_steer_deg} deg exceeds 180° — physically impossible; '
                f'clamping will apply. Check your config.')
        elif 0 < _max_steer_deg < 5.0:
            self.get_logger().warn(
                f'max_steer={_max_steer_deg} deg is suspiciously small. '
                f'This parameter expects degrees — did you accidentally enter radians? '
                f'({_max_steer_deg} rad ≈ {math.degrees(_max_steer_deg):.1f}°)')
        self.MIN_JERK = self._gp('min_jerk')
        self.MAX_JERK = self._gp('max_jerk')
        self.MAX_STEER_RATE = math.radians(self._gp('max_steer_rate'))
        self.MAX_SPEED = self._gp('max_speed')
        self.MIN_SPEED = self._gp('min_speed')
        self.MAX_ACCEL = self._gp('max_accel')
        self.MAX_DECEL = self._gp('max_decel')
        self.saturate_input = self._gp('saturate_input')
        self.allow_reversing = self._gp('allow_reversing')
        if not self.allow_reversing:
            self.MIN_SPEED = 0.0
        self.NX = self._gp('n_states')
        self.NU = self._gp('n_inputs')
        self.horizon = int(self._gp('horizon'))
        self.prediction_time = self._gp('prediction_time')
        self._recompute_weights()
        self.path_topic = self._gp('path_topic')
        self.path_qos = self._gp('path_qos')
        self.speed_topic = self._gp('speed_topic')
        self.odom_topic = self._gp('odom_topic')
        self.acceleration_topic = self._gp('acceleration_topic')
        self.ackermann_cmd_topic = self._gp('ackermann_cmd_topic')
        self.actuator_feedback_topic = self._gp('actuator_feedback_topic')
        self.delay_compensation_enabled = self._gp('delay_compensation_enabled')
        self.delay_compensation_method = self._gp('delay_compensation_method')
        self.estimated_delay = self._gp('estimated_delay')
        self.publish_twist_topic = self._gp('publish_twist_topic')
        self.twist_topic = self._gp('twist_topic')
        self.desired_speed = self._gp('desired_speed')
        self.use_speed_profile = self._gp('use_speed_profile')
        self.max_lateral_accel = self._gp('max_lateral_accel')
        self.arclength_index_advance = self._gp('arclength_index_advance')
        self.projection_window = self._gp('projection_window')
        self.min_reference_speed = self._gp('min_reference_speed')
        self.loop = int(self._gp('loop'))
        self.n_ind_search = self._gp('n_ind_search')
        self.smooth_yaw = self._gp('smooth_yaw')
        self.debug = self._gp('debug')
        self._num_obstacles = self._gp('num_obstacles')
        self._predict_obstacle_motion = self._gp('predict_obstacle_motion')
        self.obstacle_braking_envelope = bool(self._gp('obstacle_braking_envelope'))
        # Latched safety hold: armed on the first envelope fire, released only once
        # the keep-out margin recovers (see _control_timer_callback). Without the
        # latch the envelope zeroes single ticks while the solver's creep command is
        # applied on the alternating ticks, ratcheting the vehicle into contact.
        self._safety_hold = False
        self._brake_speed_bound = float('inf')
        self.dt = self.sample_time = 1.0 / self.control_rate
        if not self.prediction_time:
            self.prediction_time = self.sample_time * self.horizon

        np.set_printoptions(suppress=True)

    def _init_state(self):
        self.x = self.y = self.yaw = self.speed = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.omega = self.yaw_rate = self.acceleration = 0.0
        self.direction = 1.0
        self.rear_x = self.rear_y = 0.0
        self.cumulative_distance = 0.0

        self.acc_cmd = self.delta_cmd = self.velocity_cmd = 0.0
        self.jerk_cmd: Optional[float] = None
        self.delta_rate_cmd: Optional[float] = None

        self.zk = np.zeros((self.NX, 1))
        self.uk = np.zeros((self.NU, 1))
        self.u_prev = np.zeros((self.NU, 1))
        self.xref = np.zeros((self.horizon + 1, self.NX))
        self.mpc_predicted_states = np.zeros((self.horizon + 1, self.NX))
        self.mpc_predicted_inputs = np.zeros((self.horizon, self.NU))

        self.initial_pose_received = False
        self.initial_accel_received = False
        self.path_received = False
        self.desired_speed_received = False
        self.mpc_initialized = False
        self.trajectory_initialized = False
        self.stop_flag = False

        self.path = None
        self.des_yaw_list = None
        self.speeds = None
        self.current_idx = 0
        self.target_point = None
        self.final_idx = None
        self.final_goal = None
        self.final_goal_reached = False
        self._loop_count = 0
        self.state_frame_id = ''

        self.solution_time = 0.0
        self.solver_iteration_count = 0
        self.run_count = 0
        self.solution_status = False
        self.warmstart_variables = {}
        self._consecutive_failures = 0
        self._last_successful_command_monotonic: Optional[float] = None
        self._last_control_tick_monotonic: Optional[float] = None
        self._last_solver_log_ms = float('nan')

        self._last_odom_stamp = None

        self.obstacles: list = []
        self._obstacle_topic_name = ''
        self._last_obstacle_msg_time = None
        self.n_obstacle_states: int = 3
        self.obstacle_states: Optional[np.ndarray] = None
        # Per-obstacle go-around side memory for the keep-out reference projection.
        # Head-on, the shortest-way arc side is knife-edge and flips tick-to-tick
        # without this, and the vehicle chases an alternating left/right target.
        # Keyed by obstacle id rather than by position in the selected list: the
        # selection order shifts as the vehicle moves, and a positional list would
        # hand one obstacle's committed side to another.
        self._keepout_side_hints: dict = {}

        self.trajectory = Trajectory(
            search_index_number=10,
            goal_tolerance=self.distance_tolerance,
            stop_speed=self.speed_tolerance,
        )

        self.mutex = threading.Lock()
        self._u_prev_from_echo = False
        self._warned_twist_frame = False
        self._warned_pose_frame = False
        # These are latest-value holders, not history buffers: the control loop
        # (control_rate Hz) pushes one frame per tick and the debug publisher
        # (debug_frequency Hz) consumes the most recent one. With maxsize=1,
        # update_queue() always overwrites with the newest frame and the
        # consumer's get() returns it — so the viz never trails the live state.
        # A larger buffer made get() (FIFO front) return stale frames, which
        # showed up in RViz as the goal point / predicted path lagging the vehicle.
        _q = 1
        self.location_queue = queue.Queue(maxsize=_q)
        self.mpc_reference_states_queue = queue.Queue(maxsize=_q)
        self.mpc_predicted_states_queue = queue.Queue(maxsize=_q)
        self.goal_queue = queue.Queue(maxsize=_q)

        self.add_on_set_parameters_callback(self.parameter_change_callback)

    def _setup_tf(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Detect clock jumps (e.g. ROSBAG loop/seek or sim-time resets). A backward
        # jump invalidates the TF buffer (holds "future" transforms), the stale-odom
        # timestamp comparison, and the trajectory's notion of elapsed time. Trigger on
        # any backward jump or a large forward jump, and on a clock-type change.
        threshold = JumpThreshold(
            min_forward=Duration(seconds=10.0),
            min_backward=Duration(seconds=-0.001),
            on_clock_change=True)
        # Keep the handle alive — it is unregistered when garbage-collected.
        self._jump_handle = self.get_clock().create_jump_callback(
            threshold, pre_callback=None, post_callback=self._on_time_jump)

    def _on_time_jump(self, time_jump: TimeJump):
        """Recover from a clock discontinuity (ROSBAG loop/seek, sim-time reset).

        Clears the TF buffer (its cached transforms are now stamped in the wrong
        epoch) and resets timing-derived state so the stale-odom guard and warm
        starts do not act on pre-jump data. Sensor callbacks re-populate state on
        the next message after the jump.

        On a backward jump (bag replay loop): if looping is enabled, the trajectory
        index is also reset to the start because the bag's odometry is replaying from
        the beginning. This does not count against the loop limit — the bag reset is
        an external event, not a controller-driven lap completion.
        """
        delta_s = time_jump.delta.nanoseconds * 1e-9
        self.get_logger().warn(
            f'Time jump detected (delta={delta_s:.3f}s); clearing TF buffer and resetting timing state.')

        self.tf_buffer.clear()
        with self.mutex:
            self._last_odom_stamp = None
            self.u_prev[:, :] = 0.0
            self._u_prev_from_echo = False
        self._consecutive_failures = 0

        if delta_s < 0 and self.loop != 0 and self.trajectory_initialized:
            self._reset_lap_progress()
            self.get_logger().info(
                'Backward time jump with looping enabled — trajectory index reset to start '
                '(loop count unchanged).')

    def _setup_pub_sub(self):
        self.subscription_group = ReentrantCallbackGroup()
        self.mpc_group = MutuallyExclusiveCallbackGroup()
        self.debug_group = ReentrantCallbackGroup()

        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        if self.path_qos == 'volatile':
            path_sub_qos = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE)
            self.get_logger().info(
                f'Path subscription QoS: VOLATILE (Nav2 mode) on {self.path_topic}')
        else:
            path_sub_qos = latching_qos

        self.path_sub = self.create_subscription(
            Path, self.path_topic, self.path_callback,
            qos_profile=path_sub_qos)
        self.speed_sub = self.create_subscription(
            Float32MultiArray, self.speed_topic, self.desired_speed_callback,
            qos_profile=latching_qos)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback,
            1, callback_group=self.subscription_group)
        self.accel_sub = self.create_subscription(
            AccelWithCovarianceStamped, self.acceleration_topic, self.acceleration_callback,
            1, callback_group=self.subscription_group)
        self.drive_echo_sub = self.create_subscription(
            AckermannDriveStamped, self.ackermann_cmd_topic, self._drive_echo_callback,
            1, callback_group=self.subscription_group)
        if self.actuator_feedback_topic and self.actuator_feedback_topic != self.ackermann_cmd_topic:
            self.actuator_feedback_sub = self.create_subscription(
                AckermannDriveStamped, self.actuator_feedback_topic, self._drive_echo_callback,
                1, callback_group=self.subscription_group)
            self.get_logger().info(
                f'Actuator feedback u_prev source: {self.actuator_feedback_topic}')

        self.ackermann_cmd_pub = self.create_publisher(
            AckermannDriveStamped, self.ackermann_cmd_topic, 1)
        self.steer_pub = self.create_publisher(Float32, 'mpc/des_steer', 1)
        self.speed_pub = self.create_publisher(Float32, 'mpc/des_speed', 1)
        self.yaw_rate_pub = self.create_publisher(Float32, 'mpc/des_yaw_rate', 1)
        if self.publish_twist_topic:
            self.twist_cmd_pub = self.create_publisher(TwistStamped, self.twist_topic, 1)
        self.mpc_goal_pub = self.create_publisher(PointStamped, 'mpc/goal_point', 1)
        self.mpc_path_pub = self.create_publisher(Path, 'mpc/predicted_path', 1)
        self.mpc_reference_path_pub = self.create_publisher(Path, 'mpc/reference_path', 1)
        self.solve_time_pub = self.create_publisher(Float32, 'mpc/solve_time', 1)

        if self._num_obstacles > 0 and OBSTACLES_AVAILABLE:
            obstacle_topic = self._gp('obstacle_topic')
            self.obstacle_sub = self.create_subscription(
                ObjectArray, obstacle_topic, self._obstacle_callback, 1,
                callback_group=self.subscription_group)
            self.get_logger().info(
                f'Obstacle avoidance enabled: {self._num_obstacles} obstacles on {obstacle_topic}')
            self._obstacle_topic_name = obstacle_topic
            self._last_obstacle_msg_time = None
        elif self._num_obstacles > 0:
            self.get_logger().warn(
                'num_obstacles > 0 but derived_object_msgs not available.')

        self.mpc_timer = self.create_timer(
            self.sample_time, self._control_timer_callback,
            callback_group=self.mpc_group)
        if self.debug_frequency > 0:
            self.debug_timer = self.create_timer(
                1.0 / self.debug_frequency, self.publish_debug_topics,
                callback_group=self.debug_group)

    # ------------------------------------------------------------------
    # Subscriptions callbacks
    # ------------------------------------------------------------------

    async def path_callback(self, data: Path):
        path_frame_id = data.header.frame_id

        # Resolve TF if path is not already in the controller's global frame.
        tf_tx = tf_ty = tf_yaw = 0.0
        if path_frame_id and self.global_frame and path_frame_id != self.global_frame:
            try:
                # wait_for_transform_async blocks this coroutine (non-blocking to executor)
                # until the transform is available. This is critical when path arrives via
                # transient-local QoS before TF is up — lookup_transform_async raises
                # immediately on a missing transform and the callback would return early,
                # never to be called again by the latching publisher.
                self.get_logger().info(
                    f'path_callback: waiting for TF {path_frame_id} → {self.global_frame}')
                await self.tf_buffer.wait_for_transform_async(
                    self.global_frame, path_frame_id, Time())
                tf_stamped = self.tf_buffer.lookup_transform(
                    self.global_frame, path_frame_id, Time())
                t = tf_stamped.transform.translation
                r = tf_stamped.transform.rotation
                tf_tx, tf_ty = t.x, t.y
                _, _, tf_yaw = tf_transformations.euler_from_quaternion(
                    [r.x, r.y, r.z, r.w])
            except Exception as e:
                self.get_logger().error(
                    f'path_callback: cannot transform {path_frame_id} → '
                    f'{self.global_frame}: {e}')
                return
        elif path_frame_id and not self.global_frame:
            self.get_logger().warn(
                f'global_frame is empty; path frame "{path_frame_id}" will be used as-is '
                '(no TF transform applied). Set global_frame to enable frame correction.',
                throttle_duration_sec=10.0)

        # todo: switch to numpy and vectorize since data.poses is just a list of poses
        cos_tf = math.cos(tf_yaw)
        sin_tf = math.sin(tf_yaw)

        coordinate_list, yaw_list = [], []
        for pose_msg in data.poses:
            p = pose_msg.pose.position
            # todo: compare with tf2_geometry_msgs.do_transform_pose()
            coordinate_list.append([
                cos_tf * p.x - sin_tf * p.y + tf_tx,
                sin_tf * p.x + cos_tf * p.y + tf_ty,
            ])
            q = pose_msg.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            yaw_list.append(yaw + tf_yaw)

        new_path = np.array(coordinate_list)
        new_yaw = np.array(yaw_list)

        if new_path.shape[0] < 2:
            self.get_logger().warn(
                f'path_callback: received path with {new_path.shape[0]} pose(s) '
                '— need ≥ 2 for curvature; ignoring.')
            return

        # Reject malformed paths (NaN/inf coordinates or yaw) at ingestion rather than
        # letting them propagate: a non-finite path otherwise crashes downstream at the
        # curvature filter, the speed/dt timing, or the KD-tree build. Drop it with a
        # warning and retain the previously latched path.
        if not (np.all(np.isfinite(new_path)) and np.all(np.isfinite(new_yaw))):
            self.get_logger().warn(
                'path_callback: received a path with non-finite (NaN/inf) coordinates; '
                'ignoring and retaining the previous path.')
            return

        # Ignore identical re-delivery of the same path: transient-local/latched
        # redelivery, or a planner republishing an unchanged plan. Nothing to rebuild.
        if self.path_received and np.array_equal(new_path, self.path):
            return

        # A different path arriving after one is already active is a replan
        # (e.g. Nav2 publishing a fresh /plan). The trajectory's speed/dt columns,
        # final_goal speed, KD-tree and lap progress are all tied to the previous
        # path, so force a lazy re-init via _init_trajectory. Drop the
        # trajectory_initialized flag first so the control timer stops consuming the
        # old trajectory while we rewrite it, and re-anchor progress to the new start.
        is_replan = self.path_received
        if is_replan:
            self.trajectory_initialized = False
            # Clear the goal latch so a fresh plan resumes tracking even if the
            # previous one had already finished (final_goal_reached) or stacked up
            # solver failures.
            self.final_goal_reached = False
            self._consecutive_failures = 0
            self._reset_lap_progress()
            if not self.desired_speed_received:
                # Speed was synthesized from desired_speed for the previous length;
                # clear it so _init_trajectory re-synthesizes it for the new path.
                self.speeds = None

        self.path = new_path
        self.des_yaw_list = new_yaw
        self.path_received = True
        self.final_idx = self.path.shape[0] - 1
        self.final_goal = self.path[self.final_idx, :]
        self.target_point = self.path[self.current_idx, :]
        self.update_queue(self.goal_queue, self.target_point[0:2])

        self.trajectory.trajectory = np.zeros(
            (self.path.shape[0], len(self.trajectory.trajectory_keys)))
        tc = self.trajectory.trajectory_key_to_column
        self.trajectory.trajectory[:, tc['x']] = self.path[:, 0]
        self.trajectory.trajectory[:, tc['y']] = self.path[:, 1]
        self.trajectory.trajectory[:, tc['yaw']] = self.des_yaw_list

        cdists = trajectory_utils.cumulative_distance_along_path(
            self.trajectory.trajectory[:, [tc['x'], tc['y']]])
        curvs = trajectory_utils.calculate_curvature_all(
            cdists, self.trajectory.trajectory[:, tc['yaw']], smooth=True)
        self.trajectory.trajectory[:, tc['curvature']] = curvs
        self.trajectory.trajectory[:, tc['cum_dist']] = cdists

        # reset kdtree so it is rebuilt against the new path
        self.trajectory.waypoint_kdtree = None

    def desired_speed_callback(self, data: Float32MultiArray):
        self.speeds = np.array(list(data.data))
        self.desired_speed_received = True

    def odom_callback(self, data: Odometry):
        self.state_frame_id = data.header.frame_id
        self._last_odom_stamp = self.get_clock().now()

        pose = data.pose.pose
        twist = data.twist.twist
        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Transform pose to global_frame when the localizer publishes in a different frame
        # (e.g. odom frame while global_frame='map'). Uses the latest cached TF entry so
        # this is non-blocking at 50 Hz. Skips silently (with one logged warning) until TF
        # is available — the raw pose is used in the meantime.
        pose_frame = data.header.frame_id
        if pose_frame and self.global_frame and pose_frame != self.global_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.global_frame, pose_frame, Time())
                t = tf.transform.translation
                r = tf.transform.rotation
                _, _, tf_yaw = tf_transformations.euler_from_quaternion(
                    [r.x, r.y, r.z, r.w])
                c, s = math.cos(tf_yaw), math.sin(tf_yaw)
                x, y = c * x - s * y + t.x, s * x + c * y + t.y
                yaw += tf_yaw
            except Exception:
                if not self._warned_pose_frame:
                    self.get_logger().warn(
                        f'odom frame "{pose_frame}" != global_frame "{self.global_frame}"; '
                        'TF not yet available — using raw odom pose until transform appears.')
                    self._warned_pose_frame = True

        # Project twist to body frame when the localizer publishes it in the world frame.
        # ROS 2 convention: twist is in child_frame_id. If that differs from robot_frame,
        # rotate the planar components into the body frame using the current heading.
        child_frame = data.child_frame_id
        if child_frame and child_frame != self.robot_frame:
            if not self._warned_twist_frame:
                self.get_logger().warn(
                    f'odom child_frame_id "{child_frame}" != robot_frame "{self.robot_frame}"; '
                    'projecting twist.linear to body frame.')
                self._warned_twist_frame = True
            c, s = math.cos(yaw), math.sin(yaw)
            vx = twist.linear.x * c + twist.linear.y * s
            vy = -twist.linear.x * s + twist.linear.y * c
        else:
            vx = twist.linear.x
            vy = twist.linear.y
        vz = twist.linear.z
        omega = twist.angular.z
        try:
            direction = vx / abs(vx)
        except ZeroDivisionError:
            direction = 1.0
        speed = vx
        rear_x = x - (self.WHEELBASE / 2) * math.cos(yaw)
        rear_y = y - (self.WHEELBASE / 2) * math.sin(yaw)

        with self.mutex:
            self.x, self.y = x, y
            self.yaw = yaw
            self.vx, self.vy, self.vz = vx, vy, vz
            self.omega = self.yaw_rate = omega
            self.direction = direction
            self.speed = speed
            self.rear_x, self.rear_y = rear_x, rear_y

        self.update_queue(self.location_queue, [x, y])

        if not self.initial_pose_received:
            self.initial_pose_received = True
            self.get_logger().info(
                f'First odometry received (x={x:.2f}, y={y:.2f}, v={speed:.2f} m/s); '
                'solver initializes on the next control tick.')

    def acceleration_callback(self, data: AccelWithCovarianceStamped):
        self.initial_accel_received = True
        self.acceleration = data.accel.accel.linear.x

    def _drive_echo_callback(self, msg: AckermannDriveStamped):
        with self.mutex:
            self.u_prev[0, 0] = msg.drive.acceleration
            self.u_prev[1, 0] = msg.drive.steering_angle
            if not self._u_prev_from_echo:
                self.get_logger().info('u_prev warm-start: sourcing from drive echo.')
            self._u_prev_from_echo = True

    def _snapshot_state(self):
        """Return a consistent (x, y, speed, yaw, omega) snapshot under the state mutex."""
        with self.mutex:
            return self.x, self.y, self.speed, self.yaw, self.omega

    def _obstacle_callback(self, data: 'ObjectArray'):
        """Parse and cache every reported obstacle; ranking happens at the point of use.

        Deliberately does no ranking: relevance is measured against the reference
        horizon, which only exists inside the control loop, so ``_select_obstacles``
        scores these per tick. Keeping this callback free of ego state also keeps it
        free of the state mutex.
        """
        self._last_obstacle_msg_time = time.monotonic()
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

            obstacles.append({
                'id': int(obj.id),
                'state': [pos[0], pos[1], radius],
                'velocity': [obj.twist.linear.x, obj.twist.linear.y],
            })

        self.obstacles = obstacles

    def effective_ego_radius(self) -> float:
        """Ego collision radius, resolving the ``<= 0`` sentinel to the default.

        Shared by the solver adapters (which bake it into the OCP) and the keep-out
        reference projection, so the radius the constraint enforces and the radius the
        reference is swept out of can never disagree.
        """
        return trajectory_utils.resolve_ego_radius(self._gp('ego_radius'))

    def effective_ego_disc_offsets(self) -> np.ndarray:
        """Ego collision-disc offsets, resolved to at least the single reference-point disc.

        Shared, like :meth:`effective_ego_radius`, by the adapters that bake the discs
        into the OCP, the relevance ranking, the reference projection and the executed
        clearance diagnostics — one definition, so the enforced, the tracked and the
        reported geometry agree.
        """
        return trajectory_utils.resolve_ego_disc_offsets(self._gp('ego_disc_offsets'))

    def _solver_has_obstacle_constraints(self) -> bool:
        """Whether the active backend actually constrains obstacles.

        ``update_obstacles`` is the adapter-side marker for a backend that feeds
        obstacle states into its OCP (acados, CasADi). do-mpc has no obstacle
        constraints at all, so bending its reference around an obstacle would fake
        avoidance the solver cannot enforce — reference shaping is only sound when a
        constraint backs it.
        """
        return hasattr(self._solver, 'update_obstacles')

    def _keepout_radii(self, obstacles: list) -> np.ndarray:
        """Keep-out radius per obstacle: ``ego_radius + obstacle_radius + safe_distance``.

        The single definition shared by the relevance ranking, the reference projection
        and the OCP constraint, so the three cannot disagree on the geometry.
        """
        radii = np.array([o['state'][2] for o in obstacles], dtype=float)
        return self.effective_ego_radius() + radii + float(self._gp('safe_distance'))

    def _obstacles_are_active(self) -> bool:
        """Whether obstacle handling should run at all this tick."""
        return (self._num_obstacles > 0 and self._solver is not None
                and self._solver_has_obstacle_constraints())

    def _select_obstacles(self, xref: np.ndarray, ego_pose) -> list:
        """Rank detections by relevance to this solve; return at most ``num_obstacles``.

        Relevance is measured against the point set ``{ego} + xref[0..N]`` rather than
        against the vehicle alone, because that is the set the solve can actually act
        on: the OCP constrains stages 0..N and the keep-out projection bends those same
        reference points, so an obstacle far from all of them cannot influence the
        result. Ranking by distance-to-ego instead lets an obstacle behind the vehicle
        evict the one ahead — at ``num_obstacles: 1`` it takes the only slot. The window
        spans ``v * prediction_time``, so it scales with speed for free, and "behind"
        needs no heading test: the horizon only runs forward.

        Ego joins the point set because ``xref[:, 0]`` is anchored ``distance_tolerance``
        *ahead* of the vehicle, leaving the vehicle's own position uncovered — during an
        avoidance swerve the car is off-reference by construction, and an obstacle on top
        of it must not rank as irrelevant.

        Obstacles whose keep-out actually bites somewhere on the window sort first, by
        the earliest stage at which it bites; the rest follow by distance to the window.
        Ordering intruders by *when* rather than by how deeply matters when two obstacles
        both sit on the reference: the reference runs through both, so a plain minimum
        distance scores both ~0 and breaks the tie arbitrarily — which can pick the
        farther one and reinstate the very bug this ranking removes.
        """
        # Bind the cached list ONCE. ``_obstacle_callback`` runs on another executor
        # thread and rebinds ``self.obstacles`` to a fresh list; re-reading the
        # attribute below would let the length change mid-ranking, so the per-obstacle
        # arrays built here would be indexed by a different obstacle count (IndexError,
        # which kills the controller and leaves the last command latched). A stale-but-
        # consistent snapshot is the intended contract — that is why the callback needs
        # no mutex. Feeds with a fixed object count never expose this; a live perception
        # feed whose count varies as actors appear and disappear does.
        obstacles = self.obstacles
        if not self._obstacles_are_active() or not obstacles:
            return []

        centres = np.array([o['state'][:2] for o in obstacles], dtype=float)
        keepout = self._keepout_radii(obstacles)
        # One entry per (stage, disc): an obstacle the body reaches must rank as an
        # intruder even when the rear-axle point itself stays clear of the keep-out.
        # Discs stay grouped by stage so a column index still maps back to its stage.
        offsets = self.effective_ego_disc_offsets()
        n_discs = offsets.size
        poses = np.column_stack([np.r_[ego_pose[0], xref[0, :]],
                                 np.r_[ego_pose[1], xref[1, :]],
                                 np.r_[ego_pose[2], xref[3, :]]])  # (N+2, 3)
        points = trajectory_utils.ego_disc_centres(
            poses[:, 0], poses[:, 1], poses[:, 2], offsets).reshape(-1, 2)

        dist = distance.cdist(centres, points)      # (n_detected, (N+2) * n_discs)
        intrudes = dist < keepout[:, None]
        bites = intrudes.any(axis=1)
        # argmax on a boolean row gives the first True — the earliest constraining
        # stage. Ego is column 0, so an obstacle already on the vehicle sorts first.
        first_stage = intrudes.argmax(axis=1) // n_discs
        closest = dist.min(axis=1)

        keys = [(0, int(first_stage[i]), 0.0) if bites[i] else (1, 0, float(closest[i]))
                for i in range(len(obstacles))]
        order = sorted(range(len(obstacles)), key=keys.__getitem__)
        return [obstacles[i] for i in order[:self._num_obstacles]]

    def _warn_if_obstacle_feed_silent(self) -> None:
        """Warn (throttled) when obstacle avoidance is configured but nothing arrives.

        A silent feed is invisible in the control output: with no detections the run
        tracks cleanly, every clearance column logs as ``nan``, and an obstacle test
        that never saw an obstacle reads as a pass. Covers both "never arrived" (wrong
        topic, publisher not started) and "went quiet mid-run" (publisher died).
        """
        if not self._obstacles_are_active():
            return
        if self._last_obstacle_msg_time is None:
            self.get_logger().warn(
                f'num_obstacles={self._num_obstacles} but no obstacle message has '
                f'arrived on {self._obstacle_topic_name}; the vehicle is running with '
                'no keep-out. Check that the obstacle publisher is up and on this '
                'topic.', throttle_duration_sec=5.0)
            return
        silent_for = time.monotonic() - self._last_obstacle_msg_time
        if silent_for > _SILENT_OBSTACLE_FEED_S:
            self.get_logger().warn(
                f'No obstacle message on {self._obstacle_topic_name} for '
                f'{silent_for:.1f} s; still enforcing the last reported obstacles.',
                throttle_duration_sec=5.0)

    def _pack_obstacle_states(self, selected: list) -> None:
        """Fill the ``(3 * num_obstacles, N+1)`` obstacle block and hand it to the solver.

        Slots beyond the selected set are parked far away so their constraint row exists
        but never activates. With ``predict_obstacle_motion`` the keep-out tracks where
        each obstacle will be at stage k rather than where it was reported; a zero twist
        makes that identical to a static fill.
        """
        if not self._obstacles_are_active():
            return

        n_obs = self._num_obstacles
        n_stages = self.horizon + 1
        selected = selected[:n_obs]

        # Unused slots: far-away centre with a nominal radius, so the constraint is
        # structurally present but slack at every stage.
        packed = np.tile(np.array([1000.0, 1000.0, 1.0]), (n_obs, 1))
        velocity = np.zeros((n_obs, 2))
        if selected:
            packed[:len(selected)] = [o['state'] for o in selected]
            if self._predict_obstacle_motion:
                velocity[:len(selected)] = [o['velocity'] for o in selected]

        elapsed = np.arange(n_stages) * self.sample_time            # (n_stages,)
        states = np.empty((n_obs, self.n_obstacle_states, n_stages))
        states[:, 0, :] = packed[:, 0:1] + velocity[:, 0:1] * elapsed
        states[:, 1, :] = packed[:, 1:2] + velocity[:, 1:2] * elapsed
        states[:, 2, :] = packed[:, 2:3]                            # radius is constant
        self.obstacle_states = states.reshape(self.n_obstacle_states * n_obs, n_stages)

        self._solver.update_obstacles(self.obstacle_states)  # type: ignore[attr-defined]

    def _merge_overlapping_keepouts(self, centres: np.ndarray,
                                    keepouts: np.ndarray):
        """Group keep-outs whose mutual gap is too narrow to drive through.

        Two keep-out circles separated by less than an ego radius leave a corridor
        the vehicle cannot physically use, yet the per-obstacle reference
        projection would happily thread the reference through it — the solver then
        accelerates into a pinch no constraint set is feasible for. For the
        *projection only* (the OCP keeps the individual circles, so the real
        detour still hugs the true boundaries), such circles are replaced by one
        enclosing circle so the reference is routed around the group.

        Returns ``(proj_centres, proj_radii, groups)`` where ``groups`` lists the
        member indices of each projection circle.
        """
        n = len(centres)
        parent = list(range(n))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        min_corridor = float(self.effective_ego_radius())
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.hypot(*(centres[i] - centres[j])))
                if d < keepouts[i] + keepouts[j] + min_corridor:
                    parent[_find(i)] = _find(j)

        clusters = {}
        for i in range(n):
            clusters.setdefault(_find(i), []).append(i)

        proj_centres, proj_radii, groups = [], [], []
        for members in clusters.values():
            if len(members) == 1:
                i = members[0]
                proj_centres.append(centres[i])
                proj_radii.append(float(keepouts[i]))
            else:
                # Enclosing circle about the members' mean centre (exact for two
                # equal circles, conservative otherwise — a slightly generous
                # guide arc is fine, the OCP enforces the true boundaries).
                c = centres[members].mean(axis=0)
                r = max(float(np.hypot(*(centres[i] - c)) + keepouts[i])
                        for i in members)
                proj_centres.append(c)
                proj_radii.append(r)
            groups.append(members)
        return np.array(proj_centres), np.array(proj_radii), groups

    def _project_reference_out_of_keepouts(self, xref: np.ndarray,
                                           selected: list) -> np.ndarray:
        """Sweep reference points that fall inside an obstacle keep-out onto its boundary.

        A reference threading a keep-out hands the tracking cost a target the
        constraints forbid; the closest feasible answer is then to park at the bubble
        edge, where at v=0 steering has no yaw authority and the optimizer cannot see
        the detour. Projecting turns that target into a go-around arc.

        Backend-agnostic and applied once per tick here rather than inside a single
        adapter, so every backend with obstacle constraints benefits and the keep-out
        geometry is defined in exactly one place. Takes the same selected set the OCP
        was given, so the reference is only bent around obstacles a constraint backs.
        Returns the caller's array untouched when there is nothing to do; otherwise
        returns a projected copy, leaving ``self.xref`` (and so the reference debug
        topic) as the raw reference.
        """
        if not selected:
            return xref

        centres = np.array([o['state'][:2] for o in selected], dtype=float)
        proj_centres, proj_radii, groups = self._merge_overlapping_keepouts(
            centres, self._keepout_radii(selected))
        if len(groups) < len(selected):
            self.get_logger().info(
                f'{len(selected)} obstacle keep-outs merged into {len(groups)} '
                'projection circle(s): the corridor between them is narrower than '
                'the vehicle, so the reference is routed around the group.',
                throttle_duration_sec=5.0)
        # Carry each obstacle's committed go-around side across ticks by id: the
        # projection takes hints positionally, but the selection order shifts as the
        # vehicle moves, so a positional store would leak one obstacle's side to
        # another. A merged group negotiates one shared side, keyed by its smallest
        # member id and written back to every member so the commitment survives the
        # group splitting. Obstacles that drop out are forgotten, matching the
        # projection's own "negotiation over, next encounter re-decides" reset.
        hint_keys = [min(selected[i]['id'] for i in g) for g in groups]
        hints = [self._keepout_side_hints.get(k, 0) for k in hint_keys]
        xref, _, hints = trajectory_utils.project_reference_out_of_keepouts(
            xref.copy(), proj_centres, proj_radii,
            side_hints=hints, disc_offsets=self.effective_ego_disc_offsets())
        self._keepout_side_hints = {
            selected[i]['id']: h for g, h in zip(groups, hints) for i in g}
        return xref

    def _last_command_is_saturated(self, atol: float = 1e-6) -> bool:
        """Return whether the last applied command sits on any configured limit."""
        limits = (
            (self.acc_cmd, self.MAX_DECEL, self.MAX_ACCEL),
            (self.delta_cmd, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE),
            (self.velocity_cmd, self.MIN_SPEED, self.MAX_SPEED),
        )
        return any(
            math.isclose(value, lower, rel_tol=0.0, abs_tol=atol)
            or math.isclose(value, upper, rel_tol=0.0, abs_tol=atol)
            for value, lower, upper in limits
        )

    def _failure_hold_decision(self, now: Optional[float] = None) -> tuple[bool, str]:
        """Decide whether a failed solve may re-publish the last good command.

        Count and wall-time gates are independent: a zero value disables that gate;
        when both are enabled, both must pass. An unbounded hold (both gates disabled)
        is deliberately rejected. The return reason is suitable for a throttled log.
        """
        if self.solver_failure_mode == 'zero':
            return False, 'solver_failure_mode=zero'

        count_enabled = self.solver_failure_hold_count > 0
        time_enabled = self.solver_failure_hold_time > 0.0
        if not (count_enabled or time_enabled):
            return False, 'no failure-hold gate is enabled'

        if (self.solver_failure_zero_on_saturation
                and self._last_command_is_saturated()):
            return False, 'last good command is saturated'

        if (count_enabled
                and self._consecutive_failures > self.solver_failure_hold_count):
            return False, 'failure count exceeded the hold limit'

        if time_enabled:
            if self._last_successful_command_monotonic is None:
                return False, 'no successful command has been published'
            if now is None:
                now = time.monotonic()
            age = max(0.0, now - self._last_successful_command_monotonic)
            if age > self.solver_failure_hold_time:
                return False, 'last good command exceeded the hold-time limit'

        return True, 'within configured failure-hold gates'

    def _goal_completion_ready(self, x, y, vel) -> bool:
        """Whether the run may be reported as completed.

        Arc-length projection can reach the tail while the vehicle is still moving or
        while a spatially overlapping route tail remains untraversed. Completion must
        therefore use the ordinary goal distance and stop-speed contract, not a looser
        proximity-only gate.
        """
        past_grace = self.cumulative_distance >= 3.0 * self.distance_tolerance
        # Route-progress gate: a self-near path can put the final waypoint within
        # centimetres of a mid-course corner, so a vehicle shoved off-line there
        # (e.g. by an obstacle standoff) satisfies the proximity + stop contract
        # without having driven the route. Completion additionally requires the
        # tracked index to have actually reached the tail.
        last = len(self.path) - 1
        progressed = last <= 0 or self.current_idx >= 0.9 * last
        return past_grace and progressed and self.trajectory.is_goal_reached(
            x, y, vel, self.final_goal)

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _control_timer_callback(self):
        """MPC control loop — called at control_rate Hz.
        Non-MPC subclasses (e.g. Pure Pursuit) should override this entirely."""

        tick_started = time.monotonic()
        tick_interval_ms = float('nan')
        if self._last_control_tick_monotonic is not None:
            tick_interval_ms = (
                tick_started - self._last_control_tick_monotonic) * 1e3
        self._last_control_tick_monotonic = tick_started

        # 1. Stale-odometry guard
        if not self._odom_is_fresh():
            if self.initial_pose_received:
                self.get_logger().warn(
                    'Stale odometry, skipping solve', throttle_duration_sec=1.0)
                self._publish_zero_command()
            return

        # 2. Solver initialization — once, after first odom received
        if not self.mpc_initialized and self.initial_pose_received and self._solver is not None:
            _x, _y, _vel, _psi, _ = self._snapshot_state()
            x0 = np.array([_x, _y, _vel, _psi])
            self._solver.initialize(x0)
            self.mpc_initialized = True
            self.get_logger().info('MPC solver initialized.')

        # 3. Lazy trajectory initialization
        # Speed topic is optional: if desired_speed is set directly (e.g. Nav2 mode),
        # proceed without waiting for the speed subscription.
        _speeds_ready = self.desired_speed_received or abs(self.desired_speed) > 0.0
        if self.path_received and _speeds_ready and not self.trajectory_initialized:
            self._init_trajectory()

        if not (self.mpc_initialized and self.trajectory_initialized):
            return

        # 4. Final-goal latch: once the run is complete (and not looping), hold zero.
        if self.final_goal_reached:
            self._publish_zero_command()
            return

        # 5. Snapshot state (avoid race with odom callback)
        x, y, vel, psi, omega = self._snapshot_state()

        # 6. Update trajectory state
        self._update_trajectory_state(x, y, vel, psi, omega)

        # 7. Reference trajectory
        target_speed = self._get_target_speed()
        self.current_idx, ref_traj, _, _, _ = self.trajectory.calc_ref_trajectory(
            state=None, trajectory=None, current_index=None,
            dt=self.sample_time, prediction_horizon=self.horizon,
            lookahead_time=1.0, lookahead=30.0,
            num_points_to_interpolate=self.horizon, target_speed=target_speed,
            compute_auxiliary_waypoints=False)

        # Terminator (single path): the lap is finished if calc_ref_trajectory found no
        # waypoints ahead (ref_traj is None — the usual case on dense/closed paths,
        # where the stopped-at-goal test never fires) OR the vehicle has come to rest at
        # the final goal (open paths). The grace distance gates the stopped-at-goal test
        # so it cannot fire at the start, where start ≈ goal on a closed loop;
        # cumulative_distance resets to 0 on every lap, so it doubles as a re-anchor
        # cooldown after a reset.
        # A 'lost' projection also yields ref_traj is None, but means the opposite of
        # end-of-path: the vehicle is nowhere near the path (localization jump, large
        # disturbance) rather than done with it. Hold zero and keep trying to
        # re-acquire — latching the final goal here would report the run complete and
        # park. Only the arc-length path can tell them apart (status is None on the
        # legacy gate, which preserves its treat-empty-as-end-of-path behaviour).
        if self.trajectory.projection_status == 'lost':
            self.get_logger().warn(
                'Vehicle is farther than the search radius from every nearby waypoint '
                '(lost the path?); holding zero command until it re-acquires.',
                throttle_duration_sec=2.0)
            self._publish_zero_command()
            return

        end_of_path = ref_traj is None
        at_goal = self._goal_completion_ready(x, y, vel)
        if end_of_path and not at_goal:
            # Reaching the projection tail is necessary but not sufficient. Require
            # the same near-goal + stopped contract as is_goal_reached; otherwise an
            # overlapping/reverse tail can be skipped merely because its endpoint is
            # spatially close. Hold zero until the contract becomes true.
            dist_to_goal = float(np.hypot(x - self.final_goal[0], y - self.final_goal[1]))
            speed_error = abs(vel - self.final_goal[2])
            self.get_logger().warn(
                f'Reference reports end-of-path but the completion contract is not met '
                f'(goal distance {dist_to_goal:.2f} m, speed error {speed_error:.2f} m/s); '
                'not latching the goal — holding zero command.',
                throttle_duration_sec=2.0)
            self._publish_zero_command()
            return
        if end_of_path or at_goal:
            if self._advance_lap():
                # Another lap: return and let the KD-tree re-anchor on the next tick.
                # Solving now would hand the solver a large position error (vehicle at
                # end of path, reference at start) and risk extreme commands.
                return
            self._publish_zero_command()
            return

        # xref: (4, N+1); self.xref: (N+1, 4)
        xref = np.array([ref_traj['x_ref'], ref_traj['y_ref'],
                         ref_traj['vel_ref'], ref_traj['yaw_ref']])
        self.xref[:, :] = xref.T

        # 8. Update cumulative distance
        yaw_diff = trajectory_utils.normalize_angle(
            psi - self.xref[0, 3], minus_pi_to_pi=True, pi_is_negative=True, degrees=False)
        self.cumulative_distance += trajectory_utils.calculate_current_arc_length(
            vel, yaw_diff, self.sample_time)

        # 9. Build state vector
        x0 = np.array([x, y, vel, psi])
        self.zk[:, 0] = x0

        # 10. Solve
        with self.mutex:
            u_prev_snapshot = self.u_prev.copy()

        if self.delay_compensation_enabled and self.estimated_delay > 0.0:
            x0_raw = x0.copy()
            x0 = trajectory_utils.predict_state_rk4(
                x0_raw, u_prev_snapshot.flatten(), self.estimated_delay, self.WHEELBASE)
            self.get_logger().info(
                f'Delay compensation ({self.estimated_delay * 1e3:.0f} ms): x0 '
                f'{np.round(x0_raw, 4)} -> {np.round(x0, 4)}',
                throttle_duration_sec=5.0)

        # Obstacles are ranked against this tick's reference and handed to the OCP, then
        # the same selected set is swept out of the reference — one selection feeding
        # both, so the constraint and the target can never disagree about what is there.
        reference_ms = (time.monotonic() - tick_started) * 1e3
        obstacle_started = time.monotonic()
        self._warn_if_obstacle_feed_silent()
        selected = self._select_obstacles(xref, (x, y, psi))
        self._pack_obstacle_states(selected)
        projected_xref = self._project_reference_out_of_keepouts(xref, selected)
        obstacle_ms = (time.monotonic() - obstacle_started) * 1e3

        solve_started = time.monotonic()
        result: SolverResult = self._solver.solve(
            x0, projected_xref, u_prev_snapshot.flatten())
        solver_wall_ms = (time.monotonic() - solve_started) * 1e3

        # 11. Track consecutive failures. A failed solve never applies its returned
        #     iterate. The configured failure policy either zeroes immediately or
        #     permits the last good command to bridge bounded count/time windows.
        #     The counter is updated before the stats log so each row carries its own
        #     tick's count (not the previous tick's), and logged before the policy can
        #     return early so the row that trips zero-command fallback is preserved.
        self._consecutive_failures = (
            0 if result.is_optimal else self._consecutive_failures + 1)

        if not result.is_optimal:
            err_suffix = f', error={result.error}' if result.error else ''
            self.get_logger().warn(
                f'Solver suboptimal (status={result.status}, '
                f'consecutive={self._consecutive_failures}{err_suffix})',
                throttle_duration_sec=1.0)

        # Decide the safety action *before* the stats row is written, so the row
        # records what was actually published, not merely what the solver proposed.
        # A last-good-command bridge is unsafe when an obstacle is actively
        # constrained: the previous command can still be driving toward it. The
        # failure that preceded the reproduced collision occurred with 92 mm of
        # physical clearance, then one held 1.5 m/s command crossed the boundary.
        # Brake on every failed obstacle solve; keep the bounded hold policy for
        # ordinary tracking failures where no selected keep-out is involved.
        safety = self._obstacle_safety_check(
            selected, (x, y, psi), vel, tick_interval_ms)
        if not self.obstacle_braking_envelope:
            # Advisory mode (A/B): diagnostics still reach the CSV, but the envelope
            # never intervenes and any latched hold is dropped.
            safety['stop'] = False
            self._safety_hold = False
        else:
            # Latch: one envelope fire holds the vehicle until the keep-out margin
            # recovers past a hysteresis threshold. A per-tick stop alone alternated
            # with unchecked creep ticks (the stop zeroes speed, the next tick's
            # low measured speed produced no fire) and ratcheted into contact.
            # While latched, a proposal that does not close on the firing obstacle
            # is let through — that is the recovery path (reverse / step aside).
            if safety['stop']:
                self._safety_hold = True
            elif self._safety_hold:
                margin = safety.get('margin', float('nan'))
                if not selected or (np.isfinite(margin)
                                    and margin > self._SAFETY_HOLD_RELEASE_MARGIN):
                    self._safety_hold = False
                elif not (result.is_optimal and self._hold_admissible_command(
                        result, selected, (x, y, psi))):
                    safety['stop'] = True
        safety_reason = ''
        policy_reason = ''
        if safety['stop']:
            safety_reason = 'braking_envelope'
        elif not result.is_optimal:
            if result.requires_immediate_stop:
                safety_reason = 'unsafe_iterate'
            elif selected:
                safety_reason = 'obstacle_solve_failure'
            else:
                hold_last, policy_reason = self._failure_hold_decision()
                if not hold_last:
                    safety_reason = 'failure_policy'

        if not safety_reason:
            if result.is_optimal:
                if not self._u_prev_from_echo:
                    with self.mutex:
                        self.u_prev[:, 0] = result.u_prev

                # Unpack result (only the optimal solve updates the applied command).
                self.acc_cmd = result.accel_cmd
                self.delta_cmd = result.steering_cmd
                self.velocity_cmd = result.velocity_cmd
                self.jerk_cmd = result.jerk_cmd
                self.delta_rate_cmd = result.steering_rate_cmd

                self.uk[0, 0] = self.acc_cmd
                self.uk[1, 0] = self.delta_cmd

                try:
                    self.mpc_predicted_states[:, :] = result.x_sequence.T  # (N+1, nx)
                    self.mpc_predicted_inputs[:, :] = result.u_sequence.T  # (N, nu)
                except ValueError:
                    pass  # shape mismatch on first call if horizon changed
            else:
                self.get_logger().warn(
                    f'Holding last good command: {policy_reason}.',
                    throttle_duration_sec=1.0)

            # 12. Saturation (applied command is final once this has run)
            if self.saturate_input:
                self._input_saturation()

        # Stop action: the braking envelope sheds speed at max_decel while keeping
        # the solver's steering (geometrically coherent mid-avoidance-arc); every
        # other intervention hard-zeroes, because there the iterate itself is not
        # trusted so its steering must not be applied either.
        brake_cmd = (self._safety_brake_command(result, vel)
                     if safety_reason == 'braking_envelope' else None)
        if brake_cmd is None:
            self._brake_speed_bound = float('inf')
        if safety_reason:
            applied = brake_cmd if brake_cmd is not None else (0.0, 0.0, 0.0)
        else:
            applied = (self.acc_cmd, self.delta_cmd, self.velocity_cmd)
        self._log_solver_stats(result, selected, timing={
            'tick_interval_ms': tick_interval_ms,
            'reference_ms': reference_ms,
            'obstacle_ms': obstacle_ms,
            'solver_wall_ms': solver_wall_ms,
            'pre_log_ms': (time.monotonic() - tick_started) * 1e3,
        }, ego_pose=(x, y, psi), safety=safety, safety_reason=safety_reason,
            applied=applied)

        if safety_reason:
            action = 'Braking' if brake_cmd is not None else 'Zeroing commands'
            detail = ''
            if safety_reason == 'braking_envelope':
                detail = (f", margin={safety.get('margin', float('nan')):.3f} m"
                          f", phys={safety.get('physical_clearance', float('nan')):.3f} m"
                          f", hold={self._safety_hold}")
            self.get_logger().error(
                f'{action} ({safety_reason}'
                f'{": " + policy_reason if policy_reason else ""}{detail}).',
                throttle_duration_sec=1.0)
            if brake_cmd is not None:
                self.acc_cmd, self.delta_cmd, self.velocity_cmd = brake_cmd
                self.jerk_cmd = self.delta_rate_cmd = None
                self._publish_command()
            else:
                self._publish_zero_command()
            return

        self.solution_time = result.solve_time
        self.solution_status = result.is_optimal

        self._publish_command()
        if result.is_optimal:
            self._last_successful_command_monotonic = time.monotonic()

        if self.publish_twist_topic:
            lat_vel = 0.0
            if result.is_optimal and result.x_sequence.shape[1] > 1:
                lat_vel = float(result.x_sequence[1, 1])
            self._publish_twist(lateral_velocity=lat_vel)

        # 13. Update debug state
        self.target_point = self.xref[0, :].tolist()
        self.run_count += 1
        if self.run_count == 1:
            self.get_logger().info(
                f'First control solve complete (solve_time={result.solve_time * 1e3:.1f} ms, '
                f'optimal={result.is_optimal}); controller is live.')
        self.update_queue(self.goal_queue, self.target_point[0:2])
        self.update_queue(self.mpc_reference_states_queue, self.xref.copy())
        self.update_queue(self.mpc_predicted_states_queue, self.mpc_predicted_states.copy())

        # Cross-track error: true perpendicular distance from the rear axle to the
        # reference polyline, found by projecting onto the path segments in a small
        # index window around the tracked index. NOTE: current_idx is the look-ahead
        # target (~distance_tolerance ahead), so the distance to path[current_idx]
        # would carry a ~distance_tolerance baseline offset — projecting onto the
        # nearest segments removes that and reports the real lateral error.
        try:
            _lo = max(0, self.current_idx - 10)
            _hi = min(len(self.path) - 1, self.current_idx + 3)
            _a = self.path[_lo:_hi, 0:2]
            _b = self.path[_lo + 1:_hi + 1, 0:2]
            _ab = _b - _a
            _p = np.array([x, y])
            _t = np.clip(np.sum((_p - _a) * _ab, axis=1)
                         / np.maximum(np.sum(_ab * _ab, axis=1), 1e-12), 0.0, 1.0)
            _proj = _a + _t[:, None] * _ab
            _cte = float(np.min(np.hypot(_proj[:, 0] - x, _proj[:, 1] - y)))
        except (IndexError, AttributeError, TypeError, ValueError):
            _cte = float('nan')
        # Throttled to ~1 Hz: this runs every control tick (e.g. 20 Hz); throttling keeps
        # the log readable while still surfacing live tracking state. cte is the true
        # perpendicular cross-track error (see above), not a look-ahead proxy.
        self.get_logger().info(
            f'acc={self.acc_cmd:.3f} delta={np.degrees(self.delta_cmd):.1f}deg '
            f'vel_cmd={self.velocity_cmd:.2f} status={self.solution_status} '
            f't_solve={self.solution_time * 1e3:.1f}ms idx={self.current_idx} '
            f'cte={_cte:.3f} run={self.run_count}',
            throttle_duration_sec=1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _odom_is_fresh(self) -> bool:
        if self._last_odom_stamp is None:
            return False
        age = (self.get_clock().now() - self._last_odom_stamp).nanoseconds * 1e-9
        return age <= _STALE_ODOM_THRESHOLD_S

    def _apply_speed_policy(self):
        """Push the reference-speed policy parameters onto the Trajectory.

        Called once at trajectory init and again on every relevant runtime
        parameter change (hot-reload). max_reference_speed mirrors MAX_SPEED so
        the per-horizon reference is never commanded above the saturation bound.
        """
        self.trajectory.use_speed_profile = self.use_speed_profile
        self.trajectory.a_lat_max = self.max_lateral_accel
        self.trajectory.min_reference_speed = self.min_reference_speed
        self.trajectory.max_reference_speed = self.MAX_SPEED
        self.trajectory.arclength_index_advance = self.arclength_index_advance
        self.trajectory.projection_window = self.projection_window

    def _reset_lap_progress(self):
        """Reset trajectory progress to the start of the path.

        Shared by the lap-restart path (looping) and the bag-replay backward time
        jump in _on_time_jump. Resets the node-side progress counters and the
        Trajectory's traversal indices; the stored trajectory data is untouched.
        """
        self.current_idx = 0
        self.cumulative_distance = 0.0
        self.trajectory.reset_progress()

    def _advance_lap(self) -> bool:
        """Apply the loop policy on reaching the end of the trajectory.

        `loop`: 0 = stop at the end, -1 = loop forever, N > 0 = run N laps.
        Returns True when another lap should run (caller returns and lets the
        KD-tree re-anchor next tick), False when the run is complete and
        final_goal_reached has been latched (caller publishes a zero command).
        """
        if self.loop == 0:
            self.get_logger().info('Final goal reached.')
            self.final_goal_reached = True
            return False

        self._loop_count += 1
        if self.loop == -1 or self._loop_count < self.loop:
            self.get_logger().info(
                f'Lap {self._loop_count}'
                + (f'/{self.loop}' if self.loop > 0 else '')
                + ' complete, restarting trajectory.')
            self._reset_lap_progress()
            return True

        self.get_logger().info(f'All {self._loop_count} lap(s) complete.')
        self.final_goal_reached = True
        return False

    def _init_trajectory(self):
        tc = self.trajectory.trajectory_key_to_column
        if self.speeds is None or len(self.speeds) != len(self.path):
            # No speed topic received, or the speeds are stale from a previous path of a
            # different length (e.g. a replan on the latched /trajectory/path topic whose
            # matching speed message was not re-published) — synthesize a constant-speed
            # profile from desired_speed sized to the current path so the assignment below
            # cannot raise a broadcast-shape error.
            v = max(abs(self.desired_speed), 0.1)
            self.speeds = np.full(len(self.path), v)
            self.get_logger().info(
                f'No matching speed profile for the current path ({len(self.path)} pts); '
                f'using constant desired_speed={v:.2f} m/s for trajectory timing.')
        relative_times, relative_dts = trajectory_utils.calc_path_relative_time(
            self.path, self.speeds, min_dt=1.0)

        interp = interpolate.interp1d(
            range(len(self.path)), relative_times, kind='slinear', fill_value='extrapolate')
        relative_times = interp(range(len(self.path)))

        interp = interpolate.interp1d(
            range(len(self.path)), relative_dts, kind='slinear', fill_value='extrapolate')
        relative_dts = interp(range(len(self.path)))

        self.trajectory.trajectory[:, tc['dt']] = relative_dts
        self.trajectory.trajectory[:, tc['total_time_elapsed']] = relative_times
        self.trajectory.trajectory[:, tc['speed']] = self.speeds
        self._apply_speed_policy()
        self.trajectory_initialized = True

        # Trajectory.is_goal_reached reads goal[2] as the target speed at the endpoint.
        # self.path only has (x, y) columns, so append the final waypoint's speed here
        # once the speeds array is available.
        if self.final_idx is not None:
            self.final_goal = np.append(
                self.path[self.final_idx, :], float(self.speeds[self.final_idx]))

        if self.loop != 0:
            start = self.path[0, :2]
            end = self.path[-1, :2]
            gap = float(np.linalg.norm(end - start))
            if gap > 2.0 * self.distance_tolerance:
                self.get_logger().warn(
                    f'loop != 0 but path is not closed: first-to-last waypoint gap is {gap:.3f} m '
                    f'(distance_tolerance={self.distance_tolerance:.3f} m). '
                    'The KD-tree will jump to an incorrect index at lap transitions.')

        # Goal tolerance vs path scale: if distance_tolerance is large relative to the
        # whole path's spatial extent (bounding-box diagonal), the vehicle starts
        # "within goal tolerance" of the entire path. calc_ref_trajectory then finds no
        # waypoints ahead, the end_of_path terminator latches final_goal_reached, and the
        # controller reports "Final goal reached." on tick 1 without ever moving. This is
        # the classic platform/waypoint scale mismatch (e.g. CARLA's 5 m tolerance on the
        # F1/10 ~3 m path).
        xy = self.path[:, :2]
        extent = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
        if self.distance_tolerance >= extent:
            self.get_logger().warn(
                f'distance_tolerance ({self.distance_tolerance:.3f} m) is >= the path extent '
                f'({extent:.3f} m, bounding-box diagonal): the vehicle is within goal tolerance '
                'of the entire path from the start, so the controller will report "Final goal '
                'reached" on the first tick without moving. Check that the platform/waypoint '
                'scales match (e.g. do not run a CARLA-scale tolerance against an F1/10 path).')

    def _update_trajectory_state(self, x, y, vel, psi, omega):
        sc = self.trajectory.state_key_to_column
        tc = self.trajectory.trajectory_key_to_column
        self.trajectory.current_index = self.current_idx
        self.trajectory.state[0, sc['x']] = x
        self.trajectory.state[0, sc['y']] = y
        self.trajectory.state[0, sc['speed']] = vel
        self.trajectory.state[0, sc['yaw']] = psi
        self.trajectory.state[0, sc['omega']] = omega
        self.trajectory.state[0, sc['curvature']] = \
            trajectory_utils.calculate_curvature_single(
                self.trajectory.trajectory[:, [tc['x'], tc['y']]],
                goal_index=self.current_idx)
        self.trajectory.state[0, sc['cum_dist']] = self.cumulative_distance

    def _get_target_speed(self) -> Optional[float]:
        if self.desired_speed_received and self.desired_speed == 0.0 and self.speeds is not None:
            idx = min(self.current_idx, len(self.speeds) - 1)
            return float(self.speeds[idx])
        if abs(self.desired_speed) > 0.0:
            return float(self.desired_speed)
        return None

    # Tolerance for saturation warnings: suppress log noise when the solver
    # returns a value infinitesimally outside the constraint boundary due to
    # floating-point arithmetic (common when an NLP constraint is active).
    _SAT_WARN_TOL = 1e-4  # rad / (m/s²) — ~0.006°, well below any real violation. Todo: move to the top

    def _input_saturation(self):
        if self.acc_cmd < self.MAX_DECEL - self._SAT_WARN_TOL \
                or self.acc_cmd > self.MAX_ACCEL + self._SAT_WARN_TOL:
            self.get_logger().info(
                f'Accel cmd {self.acc_cmd:.3f} out of bounds, saturating.')
        self.acc_cmd = float(np.clip(self.acc_cmd, self.MAX_DECEL, self.MAX_ACCEL))

        if self.delta_cmd < self.MIN_STEER_ANGLE - self._SAT_WARN_TOL \
                or self.delta_cmd > self.MAX_STEER_ANGLE + self._SAT_WARN_TOL:
            self.get_logger().info(
                f'Steer cmd {np.degrees(self.delta_cmd):.1f}° out of bounds, saturating.')
        self.delta_cmd = float(np.clip(self.delta_cmd, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE))

        if self.velocity_cmd < self.MIN_SPEED - self._SAT_WARN_TOL \
                or self.velocity_cmd > self.MAX_SPEED + self._SAT_WARN_TOL:
            self.get_logger().info(
                f'Vel cmd {self.velocity_cmd:.2f} out of bounds, saturating.')
        self.velocity_cmd = float(np.clip(self.velocity_cmd, self.MIN_SPEED, self.MAX_SPEED))

        if self.jerk_cmd is not None:
            self.jerk_cmd = float(np.clip(self.jerk_cmd, self.MIN_JERK, self.MAX_JERK))
        if self.delta_rate_cmd is not None:
            self.delta_rate_cmd = float(
                np.clip(self.delta_rate_cmd, -self.MAX_STEER_RATE, self.MAX_STEER_RATE))

    def _publish_command(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_frame
        msg.drive.steering_angle = self.delta_cmd
        if self.delta_rate_cmd is not None:
            msg.drive.steering_angle_velocity = self.delta_rate_cmd
        msg.drive.speed = self.velocity_cmd
        msg.drive.acceleration = self.acc_cmd
        if self.jerk_cmd is not None:
            msg.drive.jerk = self.jerk_cmd
        self.ackermann_cmd_pub.publish(msg)

        self.steer_pub.publish(Float32(data=float(self.delta_cmd)))
        self.speed_pub.publish(Float32(data=float(self.velocity_cmd)))
        self.solve_time_pub.publish(Float32(data=float(self.solution_time)))

    def _publish_twist(self, lateral_velocity: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_frame
        msg.twist.linear.x = self.velocity_cmd
        msg.twist.linear.y = lateral_velocity
        msg.twist.angular.z = (
            (self.velocity_cmd / self.WHEELBASE) * math.tan(self.delta_cmd)
            if self.WHEELBASE > 0 else 0.0)
        self.twist_cmd_pub.publish(msg)

    def _publish_zero_command(self):
        self.acc_cmd = self.delta_cmd = self.velocity_cmd = 0.0
        self.jerk_cmd = self.delta_rate_cmd = None
        self._publish_command()

    def publish_debug_topics(self):
        if not self.mpc_initialized or self.run_count == 0:
            return

        goal_has_subs = self.mpc_goal_pub.get_subscription_count() > 0
        path_has_subs = self.mpc_path_pub.get_subscription_count() > 0
        ref_has_subs = self.mpc_reference_path_pub.get_subscription_count() > 0

        if not (goal_has_subs or path_has_subs or ref_has_subs):
            return

        timestamp = self.get_clock().now().to_msg()

        goal = self.goal_queue.get(block=False) if not self.goal_queue.empty() else None
        mpc_predicted_states = (
            self.mpc_predicted_states_queue.get(block=False)
            if not self.mpc_predicted_states_queue.empty() else None)
        mpc_reference_states = (
            self.mpc_reference_states_queue.get(block=False)
            if not self.mpc_reference_states_queue.empty() else None)

        if goal is not None and goal_has_subs:
            pt = PointStamped()
            pt.header.stamp = timestamp
            pt.header.frame_id = self.global_frame
            pt.point.x, pt.point.y = float(goal[0]), float(goal[1])
            self.mpc_goal_pub.publish(pt)

        # todo: vectorize the loops since they are just lists and use np.tolist()
        if mpc_predicted_states is not None and path_has_subs:
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = self.global_frame
            for k in range(mpc_predicted_states.shape[0]):
                pose = PoseStamped()
                pose.header.stamp = timestamp
                pose.header.frame_id = self.global_frame
                pose.pose.position.x = float(mpc_predicted_states[k, 0])
                pose.pose.position.y = float(mpc_predicted_states[k, 1])
                q = tf_transformations.quaternion_from_euler(0, 0, float(mpc_predicted_states[k, 3]))
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                path_msg.poses.append(pose)
            self.mpc_path_pub.publish(path_msg)

        if mpc_reference_states is not None and ref_has_subs:
            ref_msg = Path()
            ref_msg.header.stamp = timestamp
            ref_msg.header.frame_id = self.global_frame
            for k in range(mpc_reference_states.shape[0]):
                pose = PoseStamped()
                pose.header.stamp = timestamp
                pose.header.frame_id = self.global_frame
                pose.pose.position.x = float(mpc_reference_states[k, 0])
                pose.pose.position.y = float(mpc_reference_states[k, 1])
                q = tf_transformations.quaternion_from_euler(
                    0, 0, float(mpc_reference_states[k, 3]))
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                ref_msg.poses.append(pose)
            self.mpc_reference_path_pub.publish(ref_msg)

    def _recompute_weights(self, overrides=None):
        """Read Q/R/Rd/Qf from current ROS parameters (or Bryson's rule) and update self.

        ``overrides`` maps parameter names to proposed values supplied by an
        in-progress set-parameters callback. Inside such a callback the parameters
        have not been committed yet, so ``get_parameter`` still returns the
        pre-change values; the overrides take precedence so a runtime weight update
        applies the just-set values immediately instead of lagging by one update.
        """
        overrides = overrides or {}

        def _val(name):
            return overrides[name] if name in overrides else self._gp(name)

        if _val('use_bryson_weights'):
            self.Q, self.R, _Rd = bryson_weights(
                max_state_errors={
                    'x':   _val('max_error_x'),
                    'y':   _val('max_error_y'),
                    'v':   _val('max_error_v'),
                    'psi': _val('max_error_psi'),
                },
                max_inputs={
                    'a':     _val('bryson_max_accel'),
                    'delta': _val('bryson_max_steer'),
                },
                max_input_rates={
                    'jerk':       abs(self._gp('max_jerk')),
                    'steer_rate': self.MAX_STEER_RATE,
                },
            )
            self.Rd = _Rd
            self.Qf = self.Q.copy()
        else:
            self.R = np.diag(_val('R'))
            self.Rd = np.diag(_val('Rd'))
            self.Q = np.diag(_val('Q'))
            qf_vals = _val('Qf')
            self.Qf = np.diag(qf_vals) if qf_vals is not None and len(qf_vals) \
                else self.Q.copy()

    def parameter_change_callback(self, params):
        result = SetParametersResult(successful=True)
        # Weight params are validated + accumulated across the batch, then applied
        # once after the loop (an atomic set may change several at the same time).
        _weight_names = ('Q', 'R', 'Rd', 'Qf', 'use_bryson_weights',
                         'max_error_x', 'max_error_y', 'max_error_v', 'max_error_psi',
                         'bryson_max_accel', 'bryson_max_steer')
        _weight_expected_len = {'Q': self.NX, 'Qf': self.NX,
                                'R': self.NU, 'Rd': self.NU}
        weight_overrides = {}
        weights_changed = False
        failure_policy_changed = False
        for param in params:
            success = True
            if param.name == 'robot_frame':
                self.robot_frame = param.value
            elif param.name == 'global_frame':
                self.global_frame = param.value
            elif param.name == 'distance_tolerance':
                self.distance_tolerance = param.value
            elif param.name == 'speed_tolerance':
                self.speed_tolerance = param.value
            elif param.name == 'wheelbase':
                self.WHEELBASE = param.value
            elif param.name == 'max_speed':
                self.MAX_SPEED = param.value
                self._apply_speed_policy()
            elif param.name == 'use_speed_profile':
                self.use_speed_profile = param.value
                self._apply_speed_policy()
            elif param.name == 'max_lateral_accel':
                self.max_lateral_accel = param.value
                self._apply_speed_policy()
            elif param.name == 'min_reference_speed':
                self.min_reference_speed = param.value
                self._apply_speed_policy()
            elif param.name == 'arclength_index_advance':
                self.arclength_index_advance = param.value
                self._apply_speed_policy()
            elif param.name == 'projection_window':
                self.projection_window = param.value
                self._apply_speed_policy()
            elif param.name == 'min_speed':
                self.MIN_SPEED = param.value
            elif param.name == 'max_accel':
                self.MAX_ACCEL = param.value
            elif param.name == 'max_decel':
                self.MAX_DECEL = param.value
            elif param.name == 'obstacle_braking_envelope':
                self.obstacle_braking_envelope = bool(param.value)
                if not self.obstacle_braking_envelope:
                    self._safety_hold = False
            elif param.name == 'desired_speed':
                self.desired_speed = param.value
            elif param.name == 'loop':
                self.loop = int(param.value)
            elif param.name == 'solver_failure_mode':
                mode = str(param.value).strip().lower()
                if mode not in ('zero', 'hold_last'):
                    success = False
                    self.get_logger().error(
                        f"Rejected solver_failure_mode='{param.value}': "
                        "expected 'zero' or 'hold_last'.")
                else:
                    self.solver_failure_mode = mode
                    failure_policy_changed = True
            elif param.name == 'solver_failure_hold_count':
                if int(param.value) < 0:
                    success = False
                    self.get_logger().error(
                        'Rejected solver_failure_hold_count: expected an integer >= 0.')
                else:
                    self.solver_failure_hold_count = int(param.value)
                    failure_policy_changed = True
            elif param.name == 'solver_failure_hold_time':
                if float(param.value) < 0.0:
                    success = False
                    self.get_logger().error(
                        'Rejected solver_failure_hold_time: expected seconds >= 0.')
                else:
                    self.solver_failure_hold_time = float(param.value)
                    failure_policy_changed = True
            elif param.name == 'solver_failure_zero_on_saturation':
                self.solver_failure_zero_on_saturation = bool(param.value)
                failure_policy_changed = True
            elif param.name in _weight_names:
                exp = _weight_expected_len.get(param.name)
                if exp is not None and len(param.value) != exp:
                    success = False
                    self.get_logger().error(
                        f'Rejected {param.name}: expected {exp} elements, got '
                        f'{len(param.value)}; retaining previous weights.')
                else:
                    weight_overrides[param.name] = param.value
                    weights_changed = True
            else:
                success = False
            if not success:
                result.successful = False
            self.get_logger().info(
                f'Param change: {param.name}={param.value} success={success}')
        # A pre-set callback that returns unsuccessful vetoes the whole atomic set,
        # so only apply the recompute when every param in the batch was accepted.
        if weights_changed and result.successful:
            self._recompute_weights(overrides=weight_overrides)
            if self._solver is not None:
                self._solver.set_weights(self.Q, self.R, self.Rd, self.Qf)
        if (failure_policy_changed and result.successful
                and self.solver_failure_mode == 'hold_last'
                and self.solver_failure_hold_count == 0
                and self.solver_failure_hold_time == 0.0):
            self.get_logger().warn(
                'Both solver failure hold gates are disabled; failures will publish zero.')
        return result

    @staticmethod
    def update_queue(data_queue: queue.Queue, data, overwrite_if_full: bool = True):
        """Thread-safe queue update. Drops oldest item if full before inserting."""
        if data_queue.full() and overwrite_if_full:
            try:
                data_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            data_queue.put_nowait(data)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Executor factory — used by every controller node's main()
# ---------------------------------------------------------------------------

_EXECUTOR_THREADS_MIN = 3  # MPC timer + subscriptions + debug timer


def _make_executor(node: BaseTrajectoryTracker) -> MultiThreadedExecutor:
    """Create a ``MultiThreadedExecutor`` sized from *node*'s ``executor_threads`` parameter.

    Reads the parameter after node construction so the value already reflects
    any overrides from the launch file, YAML config, or ``--ros-args``.

    The minimum is ``_EXECUTOR_THREADS_MIN`` (3): one thread each for the MPC
    timer (``MutuallyExclusiveCallbackGroup``), sensor subscriptions
    (``ReentrantCallbackGroup``), and debug publishing
    (``ReentrantCallbackGroup``).  Values below the minimum are clamped with a
    logged warning.  The node is added to the executor before returning.
    """
    # ROS path: base_tracker re-emits solver errors via get_logger().warn() (see the
    # suboptimal branch in _control_timer_callback), so silence the pure-Python module
    # loggers in the CasADi backends to avoid duplicate stderr tracebacks. Standalone
    # pure-Python use never runs this, so those loggers stay active there.
    logging.getLogger('trajectory_following_ros2.casadi').setLevel(logging.CRITICAL + 1)

    requested = node.get_parameter('executor_threads').value
    if requested < _EXECUTOR_THREADS_MIN:
        node.get_logger().warn(
            f'executor_threads={requested} is below the minimum of '
            f'{_EXECUTOR_THREADS_MIN}; clamping. '
            'Needs one thread each for the MPC timer, subscriptions, and debug publishing.')
        requested = _EXECUTOR_THREADS_MIN
    executor = MultiThreadedExecutor(num_threads=requested)
    executor.add_node(node)
    return executor
