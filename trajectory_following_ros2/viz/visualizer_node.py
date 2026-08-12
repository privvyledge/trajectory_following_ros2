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
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped, AccelWithCovarianceStamped, PolygonStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.srv import GetParameters
import tf_transformations

from trajectory_following_ros2.utils.trajectory_utils import (
    resolve_ego_disc_offsets, resolve_ego_radius)

try:
    OBSTACLES_AVAILABLE = True
    from derived_object_msgs.msg import ObjectArray
except ImportError:
    OBSTACLES_AVAILABLE = False


class VisualizerNode(Node):
    """Logs controller + simulator state to one or more visualization backends."""

    def __init__(self):
        super().__init__('trajectory_visualizer')

        self._declare_parameters()
        self._read_parameters()

        self._mutex = threading.Lock()
        self._ref_first_xy = None
        self._last_ego_xy = None
        self._ref_first_yaw = None

        # Odometry arrives in the localizer's frame (odom); the route, predicted path,
        # reference window and goal all arrive in the controller's global_frame. Drawing
        # them in one world without transforming the pose offsets the vehicle by the
        # whole map->odom correction whenever those frames differ.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._warned_pose_frame = False

        self._backends: list = []
        self._init_backends()

        self._setup_subscriptions()
        self._setup_keepout_sync()
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
        self.declare_parameter('obstacle_topic',          'fake_obstacles/object_array')
        self.declare_parameter('footprint_topic',          '')
        # Frame every spatial entity is drawn in. The odometry pose is transformed into
        # it before logging, exactly as the controller's odom_callback does. Empty ('')
        # adopts the frame the route Path arrives in, which is the controller's
        # global_frame by construction — so this needs no launch wiring to be correct,
        # and set it explicitly only when running the visualizer with no route.
        self.declare_parameter('global_frame',             '')
        # The drawn keep-out must match the one the solver enforces
        # (ego_radius + obstacle_radius + safe_distance), so these are fetched from the
        # controller at startup and the viz_* values below are only a fallback for when
        # no controller is running. Empty ('') auto-discovers the controller among the
        # running nodes, which the launch files rename per backend.
        self.declare_parameter('controller_node_name',    '')
        # Draw only obstacles within this radius of the vehicle (0 = draw every
        # reported object). A live CARLA feed carries the whole map's static geometry —
        # 1235 objects on Town01 — and each one costs two patches that the backend
        # tears down and rebuilds per message, which wedges the matplotlib GUI thread
        # solid and leaves an apparently blank window. The controller only ever
        # constrains obstacles near the horizon, so drawing the rest shows nothing.
        self.declare_parameter('viz_obstacle_radius',     80.0)
        self.declare_parameter('viz_ego_radius',          0.15)
        self.declare_parameter('viz_safe_distance',        0.15)
        self.declare_parameter('viz_ego_disc_offsets',     [0.0])
        self.declare_parameter('vehicle_length',          0.58)
        self.declare_parameter('vehicle_width',           0.31)
        self.declare_parameter('vehicle_height',          0.12)
        self.declare_parameter('footprint_rear_axle_offset', 0.19)
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
        self.declare_parameter('rerun_spatial_frequency', 5.0)
        self.declare_parameter('plot_buffer_size', 300)
        # Native (matplotlib) video recording. Empty = off. .gif uses Pillow,
        # any other extension (.mp4/.mkv/...) uses ffmpeg.
        self.declare_parameter('native_video_path', '')
        self.declare_parameter('native_video_fps',  10)

    # ------------------------------------------------------------------
    # Keep-out sync
    # ------------------------------------------------------------------

    _KEEPOUT_SYNC_PERIOD_S = 2.0
    _KEEPOUT_SYNC_MAX_ATTEMPTS = 5

    def _setup_keepout_sync(self):
        """Adopt the controller's ego_radius/safe_distance so the drawn keep-out matches.

        The keep-out the solver enforces is ego_radius + obstacle_radius +
        safe_distance, all owned by the controller. Re-declaring them here would let the
        drawn circle silently disagree with the enforced one (they default to F1/10
        values, while a Carla controller resolves ego_radius to ~2.1 m), so the viz_*
        params are only a fallback for running the visualizer without a controller.

        Polled from a timer rather than resolved in __init__ because the controller may
        not have started yet, and a node cannot spin its own service call during
        construction.
        """
        self._keepout_attempts = 0
        self._keepout_failed_nodes = set()
        self._keepout_timer = self.create_timer(
            self._KEEPOUT_SYNC_PERIOD_S, self._try_sync_keepout)

    def _try_sync_keepout(self):
        self._keepout_attempts += 1
        for node_name in self._keepout_candidates():
            if node_name in self._keepout_failed_nodes:
                continue
            client = self.create_client(GetParameters, f'{node_name}/get_parameters')
            if not client.service_is_ready():
                self.destroy_client(client)
                continue
            future = client.call_async(
                GetParameters.Request(
                    names=['ego_radius', 'safe_distance', 'ego_disc_offsets']))
            future.add_done_callback(
                partial(self._on_keepout_response, node_name, client))
            self._keepout_timer.cancel()
            return

        if self._keepout_attempts >= self._KEEPOUT_SYNC_MAX_ATTEMPTS:
            self._keepout_timer.cancel()
            self.get_logger().warn(
                f'No controller reported the keep-out after '
                f'{self._keepout_attempts} attempts; drawing it from viz_ego_radius='
                f'{self.viz_ego_radius} + viz_safe_distance={self.viz_safe_distance}, '
                'which may not match what the solver enforces.')

    def _keepout_candidates(self):
        """Controller nodes to ask, most specific first.

        Falls back to discovery because the launch files rename the controller per
        backend (acados_mpc_node / casadi_mpc_node / do_mpc_node), so no single default
        name is right. Simulator nodes are excluded: 'kinematic_dompc_simulator'
        contains 'mpc' but owns no keep-out parameters.
        """
        configured = str(self.controller_node_name).strip()
        if configured:
            return [configured if configured.startswith('/') else f'/{configured}']
        return [f'{ns.rstrip("/")}/{name}'
                for name, ns in self.get_node_names_and_namespaces()
                if name != self.get_name() and 'simulator' not in name
                and ('mpc' in name or 'controller' in name)]

    def _on_keepout_response(self, node_name, client, future):
        try:
            values = future.result().values
            # An unset/undeclared name comes back as PARAMETER_NOT_SET (type 0).
            if len(values) < 2 or any(v.type == 0 for v in values[:2]):
                raise ValueError(f'{node_name} did not report ego_radius/safe_distance')
            ego_radius = resolve_ego_radius(values[0].double_value)
            safe_distance = float(values[1].double_value)
            # Disc offsets are optional: a controller predating them reports
            # PARAMETER_NOT_SET, which resolves to the single reference-point disc.
            offsets = resolve_ego_disc_offsets(
                list(values[2].double_array_value)
                if len(values) > 2 and values[2].type != 0 else None)
        except Exception as exc:
            # A node without the parameters will never gain them (declared at
            # construction), so skip it permanently and resume the timer to try the
            # remaining candidates; the final fallback warning fires only when every
            # candidate has been ruled out.
            self._keepout_failed_nodes.add(node_name)
            self.get_logger().info(
                f'Could not read the keep-out from {node_name} ({exc}); '
                'trying other candidates.')
            self._keepout_timer.reset()
            return
        finally:
            self.destroy_client(client)

        self.viz_ego_radius = ego_radius
        self.viz_safe_distance = safe_distance
        self.viz_ego_disc_offsets = offsets
        for backend in self._backends:
            backend.set_keepout(ego_radius, safe_distance, offsets)
        self.get_logger().info(
            f'Keep-out adopted from {node_name}: ego_radius={ego_radius:.3f} m + '
            f'safe_distance={safe_distance:.3f} m at disc offsets '
            f'[{", ".join(f"{o:.3f}" for o in offsets)}] m (drawn keep-out = that + '
            'each obstacle radius).')

    def _read_parameters(self):
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.odom_topic = gp('odom_topic')
        self.path_topic = gp('path_topic')
        self.predicted_path_topic = gp('predicted_path_topic')
        self.reference_path_topic = gp('reference_path_topic')
        self.goal_topic = gp('goal_topic')
        self.des_yaw_rate_topic = gp('des_yaw_rate_topic')
        self.ackermann_cmd_topic = gp('ackermann_cmd_topic')
        self.accel_topic = gp('accel_topic')
        self.solve_time_topic = gp('solve_time_topic')
        self.actuator_feedback_topic = gp('actuator_feedback_topic')
        self.reference_cmd_topic = gp('reference_cmd_topic')
        self.obstacle_topic = gp('obstacle_topic')
        self.footprint_topic = gp('footprint_topic')
        self.global_frame = str(gp('global_frame')).strip()
        self.controller_node_name = gp('controller_node_name')
        # Fallbacks only — _setup_keepout_sync replaces these with the controller's
        # live values once it is reachable.
        self.viz_obstacle_radius = float(gp('viz_obstacle_radius'))
        self.viz_ego_radius = gp('viz_ego_radius')
        self.viz_safe_distance = gp('viz_safe_distance')
        self.viz_ego_disc_offsets = resolve_ego_disc_offsets(gp('viz_ego_disc_offsets'))
        self.vehicle_length = gp('vehicle_length')
        self.vehicle_width = gp('vehicle_width')
        self.vehicle_height = gp('vehicle_height')
        self.footprint_rear_axle_offset = gp('footprint_rear_axle_offset')
        self.app_name = gp('app_name')
        self.spawn_viewer = gp('spawn_viewer')
        self.connect_addr = gp('connect_addr')
        self.recording_path = gp('recording_path')
        self.serve_web = gp('serve_web')
        self.web_port = gp('web_port')
        self.web_open_browser = gp('web_open_browser')
        self.rerun_spatial_frequency = gp('rerun_spatial_frequency')

    def _init_backends(self):
        backend_choice = self.get_parameter('viz_backend').value
        buf_size = self.get_parameter('plot_buffer_size').value

        use_rerun = backend_choice in ('rerun', 'both')
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
                    ego_radius=self.viz_ego_radius,
                    safe_distance=self.viz_safe_distance,
                    ego_disc_offsets=self.viz_ego_disc_offsets,
                    vehicle_length=self.vehicle_length,
                    vehicle_width=self.vehicle_width,
                    vehicle_height=self.vehicle_height,
                    footprint_rear_axle_offset=self.footprint_rear_axle_offset,
                    spatial_frequency=self.rerun_spatial_frequency,
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
                    ego_radius=self.viz_ego_radius,
                    safe_distance=self.viz_safe_distance,
                    ego_disc_offsets=self.viz_ego_disc_offsets,
                    vehicle_length=self.vehicle_length,
                    vehicle_width=self.vehicle_width,
                    vehicle_height=self.vehicle_height,
                    footprint_rear_axle_offset=self.footprint_rear_axle_offset,
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

        if self.obstacle_topic:
            if OBSTACLES_AVAILABLE:
                cs(ObjectArray, self.obstacle_topic, self._obstacle_cb, 10)
                self.get_logger().info(f'Obstacle visualization enabled on: {self.obstacle_topic}')
            else:
                self.get_logger().warn(
                    'Obstacle topic configured but derived_object_msgs is not available. '
                    'Obstacles will not be rendered.'
                )

        if self.footprint_topic:
            cs(PolygonStamped, self.footprint_topic, self._footprint_cb, 10)
            self.get_logger().info(f'Ego footprint visualization enabled on: {self.footprint_topic}')

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
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        speed = msg.twist.twist.linear.x
        x, y, yaw = self._to_global_frame(x, y, yaw, msg.header.frame_id)

        self._log('log_vehicle_pose', x, y, yaw, speed, stamp=msg.header.stamp)

        with self._mutex:
            self._last_ego_xy = (x, y)
            ref_xy = self._ref_first_xy
            ref_yaw = self._ref_first_yaw

        if ref_xy is not None:
            ry = ref_yaw if ref_yaw is not None else yaw
            dx, dy = ref_xy[0] - x, ref_xy[1] - y
            cte = -math.sin(ry) * dx + math.cos(ry) * dy
            heading_err = math.degrees(math.atan2(math.sin(ry - yaw), math.cos(ry - yaw)))
            self._log('log_errors', cte, heading_err)

    def _to_global_frame(self, x, y, yaw, pose_frame):
        """Transform a planar pose into the frame the route is drawn in.

        Mirrors BaseTrajectoryTracker.odom_callback: latest cached TF entry (so this
        stays non-blocking at 50 Hz), and on failure one warning followed by the raw
        pose. That fallback is silent by design after the first line, which is why a
        missing localizer shows up as a plausible-looking but wrong overlay.
        """
        with self._mutex:
            target = self.global_frame
        if not target or not pose_frame or pose_frame == target:
            return x, y, yaw
        try:
            tf = self.tf_buffer.lookup_transform(target, pose_frame, Time())
        except Exception:
            if not self._warned_pose_frame:
                self.get_logger().warn(
                    f'odom frame "{pose_frame}" != global_frame "{target}"; TF not yet '
                    'available — drawing the raw odom pose, which is offset from the '
                    'route by the whole correction until the transform appears.')
                self._warned_pose_frame = True
            return x, y, yaw
        t = tf.transform.translation
        r = tf.transform.rotation
        _, _, tf_yaw = tf_transformations.euler_from_quaternion([r.x, r.y, r.z, r.w])
        c, s = math.cos(tf_yaw), math.sin(tf_yaw)
        return c * x - s * y + t.x, s * x + c * y + t.y, yaw + tf_yaw

    def _full_path_cb(self, msg: Path):
        if not msg.poses:
            return
        # The route defines the drawing frame when global_frame was left empty.
        if not self.global_frame and msg.header.frame_id:
            with self._mutex:
                self.global_frame = msg.header.frame_id
            self.get_logger().info(
                f'Drawing in frame "{msg.header.frame_id}" (adopted from '
                f'{self.path_topic}).')
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
            self._ref_first_xy = pts[0]
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

    def _obstacle_cb(self, msg: 'ObjectArray'):
        obstacles = []
        with self._mutex:
            ego_xy = self._last_ego_xy
        for obj in msg.objects:
            pos = [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z]
            if (self.viz_obstacle_radius > 0.0 and ego_xy is not None
                    and math.hypot(pos[0] - ego_xy[0],
                                   pos[1] - ego_xy[1]) > self.viz_obstacle_radius):
                continue
            q = obj.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

            shape_type = {
                obj.shape.BOX: 'BOX',
                obj.shape.SPHERE: 'SPHERE',
                obj.shape.CYLINDER: 'CYLINDER',
            }.get(obj.shape.type, 'BOX')

            obstacles.append({
                'type': shape_type,
                'x': pos[0],
                'y': pos[1],
                'yaw': yaw,
                'dimensions': list(obj.shape.dimensions)
            })

        self._log('log_obstacles', obstacles, margin_offset=self.viz_ego_radius + self.viz_safe_distance, stamp=msg.header.stamp)

    def _footprint_cb(self, msg: PolygonStamped):
        pts = [(p.x, p.y) for p in msg.polygon.points]
        self._log('log_footprint_polygon', pts, stamp=msg.header.stamp)

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
