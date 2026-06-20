"""
ROS 2 visualizer node — forwards controller and simulator state to one or
more visualization backends (Rerun, matplotlib, or both).

Subscribes to the standard topics published by BaseTrajectoryTracker and
fans out spatial and time-series data to each active backend.

Optional subscriptions (activated when the corresponding parameter is
non-empty):
  actuator_feedback_topic — AckermannDriveStamped echoed by the hardware
  reference_cmd_topic     — AckermannDriveStamped from a driver-recorded dataset
"""
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped, AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
import tf_transformations


class VisualizerNode(Node):
    """Logs controller + simulator state to one or more visualization backends."""

    def __init__(self):
        super().__init__('trajectory_visualizer')

        self._declare_parameters()
        self._read_parameters()

        self._mutex = threading.Lock()
        self._ref_first_xy  = None
        self._ref_first_yaw = None

        self._backends: list = []
        self._init_backends()

        self._setup_subscriptions()
        self.get_logger().info('Trajectory visualizer started.')

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('odom_topic',              'odometry/local')
        self.declare_parameter('path_topic',              'trajectory/path')
        self.declare_parameter('predicted_path_topic',    'mpc/predicted_path')
        self.declare_parameter('reference_path_topic',    'mpc/reference_path')
        self.declare_parameter('goal_topic',              'mpc/goal_point')
        self.declare_parameter('des_yaw_rate_topic',      'mpc/des_yaw_rate')
        self.declare_parameter('ackermann_cmd_topic',     'drive')
        self.declare_parameter('accel_topic',             'accel/local')
        self.declare_parameter('solve_time_topic',        'mpc/solve_time')
        self.declare_parameter('actuator_feedback_topic', '')
        self.declare_parameter('reference_cmd_topic',     '')
        # Rerun connection
        self.declare_parameter('app_name',        'trajectory_following_ros2')
        self.declare_parameter('spawn_viewer',    True)
        self.declare_parameter('connect_addr',    '')
        self.declare_parameter('recording_path',  '')
        # Browser web viewer (WebGPU) — bypasses native Vulkan (broken on WSL2).
        self.declare_parameter('serve_web',        False)
        self.declare_parameter('web_port',         9090)
        self.declare_parameter('web_open_browser', True)
        # Backend selection
        self.declare_parameter('viz_backend',     'both')   # 'rerun'|'native'|'both'
        self.declare_parameter('plot_buffer_size', 300)
        # Native (matplotlib) video recording. Empty = off. .gif uses Pillow,
        # any other extension (.mp4/.mkv/...) uses ffmpeg.
        self.declare_parameter('native_video_path', '')
        self.declare_parameter('native_video_fps',  10)

    def _read_parameters(self):
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.odom_topic              = gp('odom_topic')
        self.path_topic              = gp('path_topic')
        self.predicted_path_topic    = gp('predicted_path_topic')
        self.reference_path_topic    = gp('reference_path_topic')
        self.goal_topic              = gp('goal_topic')
        self.des_yaw_rate_topic      = gp('des_yaw_rate_topic')
        self.ackermann_cmd_topic     = gp('ackermann_cmd_topic')
        self.accel_topic             = gp('accel_topic')
        self.solve_time_topic        = gp('solve_time_topic')
        self.actuator_feedback_topic = gp('actuator_feedback_topic')
        self.reference_cmd_topic     = gp('reference_cmd_topic')
        self.app_name                = gp('app_name')
        self.spawn_viewer            = gp('spawn_viewer')
        self.connect_addr            = gp('connect_addr')
        self.recording_path          = gp('recording_path')
        self.serve_web               = gp('serve_web')
        self.web_port                = gp('web_port')
        self.web_open_browser        = gp('web_open_browser')

    def _init_backends(self):
        backend_choice = self.get_parameter('viz_backend').value
        buf_size       = self.get_parameter('plot_buffer_size').value

        use_rerun  = backend_choice in ('rerun',  'both')
        use_native = backend_choice in ('native', 'both')

        if use_rerun:
            try:
                from trajectory_following_ros2.viz.rerun_backend import RerunBackend
                rb = RerunBackend(
                    app_name=self.app_name,
                    spawn_viewer=self.spawn_viewer,
                    connect_addr=self.connect_addr,
                    recording_path=self.recording_path,
                    stamp_fn=lambda: self.get_clock().now().to_msg(),
                    serve_web=self.serve_web,
                    web_port=self.web_port,
                    open_browser=self.web_open_browser,
                )
                self._backends.append(rb)
                self.get_logger().info('Rerun backend active.')
            except ImportError as e:
                self.get_logger().warn(f'Rerun backend skipped (import failed): {e}')
            except Exception as e:
                self.get_logger().error(f'Rerun backend failed to initialize: {e}')

        if use_native:
            try:
                from trajectory_following_ros2.viz.matplotlib_backend import MatplotlibBackend
                mb = MatplotlibBackend(
                    buffer_size=buf_size,
                    video_path=self.get_parameter('native_video_path').value,
                    video_fps=self.get_parameter('native_video_fps').value,
                )
                self._backends.append(mb)
                self.get_logger().info('Matplotlib backend active.')
            except ImportError as e:
                self.get_logger().warn(f'Matplotlib (native) backend skipped (import failed): {e}')
            except Exception as e:
                self.get_logger().error(f'Matplotlib backend failed to initialize: {e}')

        if not self._backends:
            self.get_logger().error(
                'No visualization backends could be initialized. '
                'Install rerun-sdk and/or matplotlib.')

    def _setup_subscriptions(self):
        latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        cs = self.create_subscription

        cs(Odometry,                   self.odom_topic,           self._odom_cb,       10)
        cs(Path,                       self.path_topic,           self._full_path_cb,  latch)
        cs(Path,                       self.predicted_path_topic, self._predicted_cb,  10)
        cs(Path,                       self.reference_path_topic, self._ref_path_cb,   10)
        cs(PointStamped,               self.goal_topic,           self._goal_cb,       10)
        cs(Float32,                    self.des_yaw_rate_topic,   self._yaw_rate_cb,   10)
        cs(AckermannDriveStamped,      self.ackermann_cmd_topic,  self._cmd_cb,        10)
        cs(AccelWithCovarianceStamped, self.accel_topic,          self._accel_cb,      10)
        cs(Float32,                    self.solve_time_topic,     self._solve_time_cb, 10)

        if self.actuator_feedback_topic:
            cs(AckermannDriveStamped, self.actuator_feedback_topic,
               self._actuator_feedback_cb, 10)
            self.get_logger().info(
                f'Actuator feedback visualization: {self.actuator_feedback_topic}')

        if self.reference_cmd_topic:
            cs(AckermannDriveStamped, self.reference_cmd_topic,
               self._reference_cmd_cb, 10)
            self.get_logger().info(
                f'Reference command visualization: {self.reference_cmd_topic}')

    # ------------------------------------------------------------------
    # Fan-out helper
    # ------------------------------------------------------------------

    def _log(self, method: str, *args, **kwargs) -> None:
        for b in self._backends:
            try:
                getattr(b, method)(*args, **kwargs)
            except Exception as e:
                self.get_logger().warn(
                    f'Backend {type(b).__name__}.{method} raised: {e}',
                    throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Callbacks — spatial
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        q     = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        speed = msg.twist.twist.linear.x

        self._log('log_vehicle_pose', x, y, yaw, speed, stamp=msg.header.stamp)

        with self._mutex:
            ref_xy  = self._ref_first_xy
            ref_yaw = self._ref_first_yaw

        if ref_xy is not None:
            ry = ref_yaw if ref_yaw is not None else yaw
            dx, dy = ref_xy[0] - x, ref_xy[1] - y
            cte = -math.sin(ry) * dx + math.cos(ry) * dy
            heading_err = math.degrees(math.atan2(math.sin(ry - yaw), math.cos(ry - yaw)))
            self._log('log_errors', cte, heading_err)

    def _full_path_cb(self, msg: Path):
        if not msg.poses:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._log('log_full_path', pts, stamp=msg.header.stamp)

    def _predicted_cb(self, msg: Path):
        if not msg.poses:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._log('log_predicted_path', pts, stamp=msg.header.stamp)

    def _ref_path_cb(self, msg: Path):
        if not msg.poses:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        q = msg.poses[0].pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        with self._mutex:
            self._ref_first_xy  = pts[0]
            self._ref_first_yaw = yaw
        self._log('log_ref_window', pts, stamp=msg.header.stamp)

    def _goal_cb(self, msg: PointStamped):
        self._log('log_goal', msg.point.x, msg.point.y, stamp=msg.header.stamp)

    # ------------------------------------------------------------------
    # Callbacks — commanded actions
    # ------------------------------------------------------------------

    def _cmd_cb(self, msg: AckermannDriveStamped):
        self._log('log_commands',
                  msg.drive.acceleration,
                  math.degrees(msg.drive.steering_angle),
                  msg.drive.speed,
                  stamp=msg.header.stamp)

    def _yaw_rate_cb(self, msg: Float32):
        self._log('log_yaw_rate', msg.data)

    # ------------------------------------------------------------------
    # Callbacks — state feedback
    # ------------------------------------------------------------------

    def _accel_cb(self, msg: AccelWithCovarianceStamped):
        self._log('log_accel_actual', msg.accel.accel.linear.x, stamp=msg.header.stamp)

    def _solve_time_cb(self, msg: Float32):
        self._log('log_solve_time', msg.data)

    # ------------------------------------------------------------------
    # Callbacks — optional
    # ------------------------------------------------------------------

    def _actuator_feedback_cb(self, msg: AckermannDriveStamped):
        self._log('log_actuator_feedback',
                  msg.drive.acceleration,
                  math.degrees(msg.drive.steering_angle),
                  msg.drive.speed,
                  stamp=msg.header.stamp)

    def _reference_cmd_cb(self, msg: AckermannDriveStamped):
        self._log('log_reference_cmd',
                  msg.drive.acceleration,
                  math.degrees(msg.drive.steering_angle),
                  msg.drive.speed,
                  stamp=msg.header.stamp)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self):
        for b in self._backends:
            b.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
