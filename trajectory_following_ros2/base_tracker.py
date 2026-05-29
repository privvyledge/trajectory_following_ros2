import math
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import interpolate

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
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
        self.declare_parameter('debug_frequency', 4.0)
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
        self.declare_parameter('loop', False)
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

    def _read_parameters(self):
        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.control_rate = self.get_parameter('control_rate').value
        self.debug_frequency = self.get_parameter('debug_frequency').value
        self.distance_tolerance = self.get_parameter('distance_tolerance').value
        self.speed_tolerance = self.get_parameter('speed_tolerance').value
        self.WHEELBASE = self.get_parameter('wheelbase').value
        self.MAX_STEER_ANGLE = math.radians(self.get_parameter('max_steer').value)
        self.MIN_STEER_ANGLE = math.radians(self.get_parameter('min_steer').value)
        if abs(self.MAX_STEER_ANGLE) > math.pi:
            self.get_logger().warn(
                f'max_steer={self.get_parameter("max_steer").value} deg converts to '
                f'{self.MAX_STEER_ANGLE:.3f} rad — value > π rad; '
                'did you declare it in radians instead of degrees?')
        self.MIN_JERK = self.get_parameter('min_jerk').value
        self.MAX_JERK = self.get_parameter('max_jerk').value
        self.MAX_STEER_RATE = math.radians(self.get_parameter('max_steer_rate').value)
        self.MAX_SPEED = self.get_parameter('max_speed').value
        self.MIN_SPEED = self.get_parameter('min_speed').value
        self.MAX_ACCEL = self.get_parameter('max_accel').value
        self.MAX_DECEL = self.get_parameter('max_decel').value
        self.saturate_input = self.get_parameter('saturate_input').value
        self.allow_reversing = self.get_parameter('allow_reversing').value
        if not self.allow_reversing:
            self.MIN_SPEED = 0.0
        self.NX = self.get_parameter('n_states').value
        self.NU = self.get_parameter('n_inputs').value
        self.horizon = int(self.get_parameter('horizon').value)
        self.prediction_time = self.get_parameter('prediction_time').value
        self._recompute_weights()
        self.path_topic = self.get_parameter('path_topic').value
        self.path_qos = self.get_parameter('path_qos').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value
        self.actuator_feedback_topic = self.get_parameter('actuator_feedback_topic').value
        self.delay_compensation_enabled = self.get_parameter('delay_compensation_enabled').value
        self.delay_compensation_method = self.get_parameter('delay_compensation_method').value
        self.estimated_delay = self.get_parameter('estimated_delay').value
        self.publish_twist_topic = self.get_parameter('publish_twist_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.desired_speed = self.get_parameter('desired_speed').value
        self.loop = self.get_parameter('loop').value
        self.n_ind_search = self.get_parameter('n_ind_search').value
        self.smooth_yaw = self.get_parameter('smooth_yaw').value
        self.debug = self.get_parameter('debug').value
        self._num_obstacles = self.get_parameter('num_obstacles').value
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
        _q = 5
        self.location_queue = queue.Queue(maxsize=_q)
        self.mpc_reference_states_queue = queue.Queue(maxsize=_q)
        self.mpc_predicted_states_queue = queue.Queue(maxsize=_q)
        self.goal_queue = queue.Queue(maxsize=_q)

        self.add_on_set_parameters_callback(self.parameter_change_callback)

    def _setup_tf(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        if self._num_obstacles > 0 and OBSTACLES_AVAILABLE:
            obstacle_topic = self.get_parameter('obstacle_topic').value
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

        self.path = np.array(coordinate_list)
        self.des_yaw_list = np.array(yaw_list)
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

        # 4. Goal check
        if self.trajectory.check_goal(
                self.x, self.y, self.speed, self.final_goal,
                self.current_idx, self.path.shape[0]):
            self.get_logger().info('Final goal reached.')
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

        if ref_traj is None:
            self.get_logger().info('No more waypoints, final goal reached.')
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

        # 11. Track consecutive failures; zero-command safety after N=5.
        if result.is_optimal:
            self._consecutive_failures = 0
            if not self._u_prev_from_echo:
                with self.mutex:
                    self.u_prev[:, 0] = result.u_prev
        else:
            self._consecutive_failures += 1
            self.get_logger().warn(
                f'Solver suboptimal (status={result.status}, '
                f'consecutive={self._consecutive_failures})',
                throttle_duration_sec=1.0)
            if self._consecutive_failures >= 5:
                self.get_logger().error(
                    f'{self._consecutive_failures} consecutive solver failures — zeroing commands.',
                    throttle_duration_sec=1.0)
                self._publish_zero_command()
                return

        # Unpack result
        self.acc_cmd = result.accel_cmd
        self.delta_cmd = result.steering_cmd
        self.velocity_cmd = result.velocity_cmd
        self.jerk_cmd = result.jerk_cmd
        self.delta_rate_cmd = result.steering_rate_cmd
        self.solution_time = result.solve_time
        self.solution_status = result.is_optimal

        self.uk[0, 0] = self.acc_cmd
        self.uk[1, 0] = self.delta_cmd

        try:
            self.mpc_predicted_states[:, :] = result.x_sequence.T  # (N+1, nx)
            self.mpc_predicted_inputs[:, :] = result.u_sequence.T  # (N, nu)
        except ValueError:
            pass  # shape mismatch on first call if horizon changed

        # 12. Saturation + publish
        if self.saturate_input:
            self._input_saturation()

        self._publish_command()

        if self.publish_twist_topic:
            lat_vel = 0.0
            if result.x_sequence.shape[1] > 1:
                lat_vel = float(result.x_sequence[1, 1])
            self._publish_twist(lateral_velocity=lat_vel)

        # 13. Update debug state
        self.target_point = self.xref[0, :].tolist()
        self.run_count += 1
        self.update_queue(self.goal_queue, self.target_point[0:2])
        self.update_queue(self.mpc_reference_states_queue, self.xref.copy())
        self.update_queue(self.mpc_predicted_states_queue, self.mpc_predicted_states.copy())

        self.get_logger().info(
            f'acc={self.acc_cmd:.3f} delta={np.degrees(self.delta_cmd):.1f}deg '
            f'vel_cmd={self.velocity_cmd:.2f} status={self.solution_status} '
            f't_solve={self.solution_time * 1e3:.1f}ms run={self.run_count}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _odom_is_fresh(self) -> bool:
        if self._last_odom_stamp is None:
            return False
        age = (self.get_clock().now() - self._last_odom_stamp).nanoseconds * 1e-9
        return age <= _STALE_ODOM_THRESHOLD_S

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
        self.trajectory_initialized = True

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

    def _input_saturation(self):
        if not (self.MAX_DECEL <= self.acc_cmd <= self.MAX_ACCEL):
            self.get_logger().info(
                f'Accel cmd {self.acc_cmd:.3f} out of bounds, saturating.')
        self.acc_cmd = float(np.clip(self.acc_cmd, self.MAX_DECEL, self.MAX_ACCEL))

        if not (self.MIN_STEER_ANGLE <= self.delta_cmd <= self.MAX_STEER_ANGLE):
            self.get_logger().info(
                f'Steer cmd {np.degrees(self.delta_cmd):.1f}° out of bounds, saturating.')
        self.delta_cmd = float(np.clip(self.delta_cmd, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE))

        if not (self.MIN_SPEED <= self.velocity_cmd <= self.MAX_SPEED):
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
        if self.get_parameter('use_bryson_weights').value:
            self.Q, self.R, _Rd = bryson_weights(
                max_state_errors={
                    'x':   self.get_parameter('max_error_x').value,
                    'y':   self.get_parameter('max_error_y').value,
                    'v':   self.get_parameter('max_error_v').value,
                    'psi': self.get_parameter('max_error_psi').value,
                },
                max_inputs={
                    'a':     self.get_parameter('bryson_max_accel').value,
                    'delta': self.get_parameter('bryson_max_steer').value,
                },
                max_input_rates={
                    'jerk':       abs(self.get_parameter('max_jerk').value),
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
            elif param.name == 'min_speed':
                self.MIN_SPEED = param.value
            elif param.name == 'max_accel':
                self.MAX_ACCEL = param.value
            elif param.name == 'max_decel':
                self.MAX_DECEL = param.value
            elif param.name == 'desired_speed':
                self.desired_speed = param.value
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
