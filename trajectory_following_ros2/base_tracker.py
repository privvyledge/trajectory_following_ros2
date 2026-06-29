import logging
import math
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import interpolate

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

    # ------------------------------------------------------------------
    # Abstract / override points
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_solver(self) -> Optional[BaseSolver]:
        """Create and return the backend solver adapter. Return None for non-MPC nodes."""

    def _declare_backend_parameters(self) -> None:
        """Declare backend-specific ROS parameters. Override in subclasses."""

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
        self.declare_parameter('obstacle_topic', 'fake_obstacles/object_array')
        self.declare_parameter('obstacle_collision_avoidance_method', 'euclidean')
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
        self.min_reference_speed = self._gp('min_reference_speed')
        self.loop = int(self._gp('loop'))
        self.n_ind_search = self._gp('n_ind_search')
        self.smooth_yaw = self._gp('smooth_yaw')
        self.debug = self._gp('debug')
        self._num_obstacles = self._gp('num_obstacles')
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

        self._last_odom_stamp = None

        self.obstacles: list = []
        self.n_obstacle_states: int = 3
        self.obstacle_states: Optional[np.ndarray] = None

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

    def acceleration_callback(self, data: AccelWithCovarianceStamped):
        self.initial_accel_received = True
        self.acceleration = data.accel.accel.linear.x

    def _drive_echo_callback(self, msg: AckermannDriveStamped):
        with self.mutex:
            self.u_prev[0, 0] = msg.drive.acceleration
            self.u_prev[1, 0] = msg.drive.steering_angle
            self._u_prev_from_echo = True

    def _snapshot_state(self):
        """Return a consistent (x, y, speed, yaw, omega) snapshot under the state mutex."""
        with self.mutex:
            return self.x, self.y, self.speed, self.yaw, self.omega

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

            # todo: acquire the thread lock to prevent race conditions
            dist = np.linalg.norm(np.array(pos[:2]) - np.array([self.x, self.y]))
            obstacles.append({'state': [pos[0], pos[1], radius], 'distance': dist})

        self.obstacles = sorted(obstacles, key=lambda o: o['distance'])

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _control_timer_callback(self):
        """MPC control loop — called at control_rate Hz.
        Non-MPC subclasses (e.g. Pure Pursuit) should override this entirely."""

        # 1. Stale-odometry guard (Bug 3 fix)
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
            num_points_to_interpolate=self.horizon, target_speed=target_speed)

        # Terminator (single path): the lap is finished if calc_ref_trajectory found no
        # waypoints ahead (ref_traj is None — the usual case on dense/closed paths,
        # where the stopped-at-goal test never fires) OR the vehicle has come to rest at
        # the final goal (open paths). The grace distance gates the stopped-at-goal test
        # so it cannot fire at the start, where start ≈ goal on a closed loop;
        # cumulative_distance resets to 0 on every lap, so it doubles as a re-anchor
        # cooldown after a reset.
        end_of_path = ref_traj is None
        past_grace = self.cumulative_distance >= 3.0 * self.distance_tolerance
        at_goal = past_grace and self.trajectory.is_goal_reached(x, y, vel, self.final_goal)
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
            x0 = trajectory_utils.predict_state_rk4(
                x0, u_prev_snapshot.flatten(), self.estimated_delay, self.WHEELBASE)

        result: SolverResult = self._solver.solve(x0, xref, u_prev_snapshot.flatten())

        # 11. Track consecutive failures. On an isolated suboptimal solve, hold the
        #     last good command instead of applying the (possibly saturated/garbage)
        #     iterate; zero-command safety after N=5.
        if result.is_optimal:
            self._consecutive_failures = 0
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
            self._consecutive_failures += 1
            err_suffix = f', error={result.error}' if result.error else ''
            self.get_logger().warn(
                f'Solver suboptimal (status={result.status}, '
                f'consecutive={self._consecutive_failures}{err_suffix})',
                throttle_duration_sec=1.0)
            if self._consecutive_failures >= 5:
                self.get_logger().error(
                    f'{self._consecutive_failures} consecutive solver failures — zeroing commands.',
                    throttle_duration_sec=1.0)
                self._publish_zero_command()
                return
            # Hold last good command: fall through to re-publish self.{acc,delta,velocity}_cmd
            # unchanged. A single 395 ms IPOPT spike once emitted delta=27° + hard brake here,
            # poisoning the loop into a reverse; bridging the spike avoids that cascade.

        self.solution_time = result.solve_time
        self.solution_status = result.is_optimal

        # 12. Saturation + publish
        if self.saturate_input:
            self._input_saturation()

        self._publish_command()

        if self.publish_twist_topic:
            lat_vel = 0.0
            if result.is_optimal and result.x_sequence.shape[1] > 1:
                lat_vel = float(result.x_sequence[1, 1])
            self._publish_twist(lateral_velocity=lat_vel)

        # 13. Update debug state
        self.target_point = self.xref[0, :].tolist()
        self.run_count += 1
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
        if self.speeds is None:
            # No speed topic received — synthesize a constant-speed profile from desired_speed.
            v = max(abs(self.desired_speed), 0.1)
            self.speeds = np.full(len(self.path), v)
            self.get_logger().info(
                f'No speed topic received; using constant desired_speed={v:.2f} m/s '
                'for trajectory timing.')
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
        # F1/10 ~3 m path). See buglog bug-042.
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

    def _recompute_weights(self):
        """Read Q/R/Rd/Qf from current ROS parameters (or Bryson's rule) and update self."""
        if self._gp('use_bryson_weights'):
            self.Q, self.R, _Rd = bryson_weights(
                max_state_errors={
                    'x':   self._gp('max_error_x'),
                    'y':   self._gp('max_error_y'),
                    'v':   self._gp('max_error_v'),
                    'psi': self._gp('max_error_psi'),
                },
                max_inputs={
                    'a':     self._gp('bryson_max_accel'),
                    'delta': self._gp('bryson_max_steer'),
                },
                max_input_rates={
                    'jerk':       abs(self._gp('max_jerk')),
                    'steer_rate': self.MAX_STEER_RATE,
                },
            )
            self.Rd = _Rd
            self.Qf = self.Q.copy()
        else:
            self.R = np.diag(self.get_parameter('R').get_parameter_value().double_array_value)
            self.Rd = np.diag(self.get_parameter('Rd').get_parameter_value().double_array_value)
            self.Q = np.diag(self.get_parameter('Q').get_parameter_value().double_array_value)
            qf_vals = self.get_parameter('Qf').get_parameter_value().double_array_value
            self.Qf = np.diag(qf_vals) if qf_vals is not None else self.Q.copy()

    def parameter_change_callback(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
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
            elif param.name == 'min_speed':
                self.MIN_SPEED = param.value
            elif param.name == 'max_accel':
                self.MAX_ACCEL = param.value
            elif param.name == 'max_decel':
                self.MAX_DECEL = param.value
            elif param.name == 'desired_speed':
                self.desired_speed = param.value
            elif param.name == 'loop':
                self.loop = int(param.value)
            elif param.name in ('Q', 'R', 'Rd', 'Qf', 'use_bryson_weights',
                                'max_error_x', 'max_error_y', 'max_error_v', 'max_error_psi',
                                'bryson_max_accel', 'bryson_max_steer'):
                self._recompute_weights()
                if self._solver is not None:
                    self._solver.set_weights(self.Q, self.R, self.Rd, self.Qf)
            else:
                result.successful = False
            self.get_logger().info(
                f'Param change: {param.name}={param.value} success={result.successful}')
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
