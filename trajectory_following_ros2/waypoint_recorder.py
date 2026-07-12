"""
Notes:
    * waypoints/poses should be recorded in map frame or transformed to map frame.
        Could record global poses, e.g. AMCL directly or convert odom pose to global frame
    * global planner/costmap works in map frame
    * local planner/costmap (the controller) works in odom frame and should transform global poses to odom frame
    * some controllers work in robot frame, so tranform to that frame (e.g collision checker)

    Todo:
        * catch exception if the directory does not exist and create parent directories if they don't exist
        * save control commands (speed, delta)
        * add the option to either save Odometry or subscribe to a path topic and save
        * wire up pose_callback and twist_callback as alternative waypoint sources

Msgs:
    Path: http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Path.html
    Marker: text and pose
"""
import sys
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.clock import Clock, ClockType

from tf2_ros import TransformException, LookupException
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf_transformations
import tf2_geometry_msgs

from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry, Path


class WaypointRecorderNode(Node):
    """Record odometry to a CSV file for later replay as a waypoint trajectory."""

    def __init__(self, ):
        """Constructor for WaypointRecorderNode"""
        super(WaypointRecorderNode, self).__init__('waypoint_recorder')

        # declare parameters
        self.declare_parameter('file_path', '')
        self.declare_parameter('waypoint_source', 'odometry')
        self.declare_parameter('save_interval', 0.1)  # seconds
        self.declare_parameter('target_frame_id', 'map',
                               ParameterDescriptor(description='Frame to transform poses into before saving.'))
        self.declare_parameter('odom_topic', 'odometry/local')
        self.declare_parameter('pose_topic', 'pose')
        self.declare_parameter('twist_topic', 'twist')
        self.declare_parameter('path_topic', 'waypoint_recorder/path')
        self.declare_parameter('marker_topic', 'waypoint_recorder/markers')
        self.declare_parameter('marker_max_speed', 10.0,
                               ParameterDescriptor(description='Speed (m/s) mapped to the fast end of the trail '
                                                              'arrow colour gradient (blue=slow → red=fast).'))
        self.declare_parameter('publish_path', True,
                               ParameterDescriptor(description='Publish the accumulating Path trail. Disable to '
                                                              'remove all path-viz overhead during recording.'))
        self.declare_parameter('publish_markers', True,
                               ParameterDescriptor(description='Publish the velocity-arrow trail + live position '
                                                              'sphere. Disable to remove all marker-viz overhead.'))
        self.declare_parameter('viz_publish_interval', 0.0,
                               ParameterDescriptor(description='Minimum seconds between visualization publishes. '
                                                              '0.0 (default) publishes every recorded row. A larger '
                                                              'value throttles the O(N) Path/marker publish without '
                                                              'dropping trail points (accumulation is unthrottled), '
                                                              'protecting the CSV write on long recordings.'))
        self.declare_parameter('save_if_transform_fails', False,
                               ParameterDescriptor(description='Save pose in source frame if TF to '
                                                               'target_frame_id is unavailable.'))
        self.declare_parameter('stale_odom_timeout', 0.5,
                               ParameterDescriptor(description='Skip write if no odom message has arrived within '
                                                               'this many seconds. Prevents recording stale data '
                                                               'on topic loss without suppressing intentional '
                                                               'stops (a stopped vehicle still publishes odom).'))
        self.declare_parameter('min_distance', 0.0,
                               ParameterDescriptor(description='Minimum distance (m) between consecutive recorded '
                                                               'waypoints. 0.0 (default) disables the check. '
                                                               'Note: a non-zero value suppresses dwell time at '
                                                               'intentional stops (stop signs, pickups).'))
        self.declare_parameter('log_duration', 4.0,
                               ParameterDescriptor(description='Logging at high frequencies can flood the console.'
                                                               'Set this parameter to a value <= 0.0 and the logs '
                                                               'only be displayed once. Positive values will be used '
                                                               'to throttle the logging duration.'))

        # get parameters
        self.file_path = str(self.get_parameter('file_path').value)
        self.waypoint_source = self.get_parameter('waypoint_source').value
        self.save_interval = self.get_parameter('save_interval').value
        self.target_frame_id = self.get_parameter('target_frame_id').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.marker_max_speed = self.get_parameter('marker_max_speed').value
        self._enable_path_pub = self.get_parameter('publish_path').value
        self._enable_marker_pub = self.get_parameter('publish_markers').value
        self._viz_publish_interval = self.get_parameter('viz_publish_interval').value
        self.save_if_transform_fails = self.get_parameter('save_if_transform_fails').value
        self.stale_odom_timeout = self.get_parameter('stale_odom_timeout').value
        self.min_distance = self.get_parameter('min_distance').value
        self.log_duration = self.get_parameter('log_duration').value
        self.log_kwarg = {'once': True} if self.log_duration <= 0.0 else {'throttle_duration_sec': self.log_duration}

        # vehicle state (updated by odom_callback after successful TF)
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.qx, self.qy, self.qz, self.qw = 0.0, 0., 0., 0.
        self.roll, self.pitch, self.yaw = 0., 0., 0.
        self.vx, self.vy, self.vz = 0., 0., 0.
        self.speed, self.omega = 0.0, 0.0
        self.global_frame = ''
        self.total_time_elapsed = 0.0

        # timing state
        self._start_time = None           # rclpy.time.Time of first successful odom+TF update
        self._last_odom_wall_time = None  # STEADY_TIME of latest odom callback arrival
        self._last_write_time = None      # node-clock time of last successful write
        self._last_recorded_x = None     # position at last write, for min_distance check
        self._last_recorded_y = None
        self._last_msg_time = None        # msg timestamp of last processed odom, for reversal detection
        self._row_count = 0               # total rows written to CSV
        self._is_stale = False            # True while odom is stale (edge-triggered logging)
        self._new_odom_since_last_write = False  # cleared after each write; prevents duplicate rows
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._trail_marker_id = 0         # monotonic id so recorded trail arrows persist (never overwritten)
        self._pending_arrows = []         # arrows accumulated since the last (throttled) marker publish
        self._last_viz_pub_time = None    # node-clock time of last visualization publish, for throttling

        # buffering=1: line-buffered — each line ending with '\n' is flushed to disk immediately
        header = 'frame_id, total_time_elapsed, dt, x, y, z, yaw, qx, qy, qz, qw, vx, vy, speed, omega\n'
        self.waypoint_file = open(self.file_path, 'w', encoding="utf-8", buffering=1)
        self.get_logger().info(f'Saving to {self.file_path}. ')
        self.waypoint_file.write(header)

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # visualization messages
        self.path_msg = Path()

        # publishers
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)

        # subscribers
        self.odom_sub = self.create_subscription(
                Odometry, self.odom_topic, self.odom_callback, 1)
        self.pose_sub = self.create_subscription(
                PoseWithCovarianceStamped, self.pose_topic, self.pose_callback, 1)
        self.twist_sub = self.create_subscription(
                TwistStamped, self.twist_topic, self.twist_callback, 1)

        # recording timer
        self.recording_timer = self.create_timer(self.save_interval, self.recording_callback)

        self.get_logger().info('waypoint_recorder started. ')

    def odom_callback(self, data):
        self.get_logger().info("Odom message received", **self.log_kwarg)
        # Stamp STEADY_TIME arrival before TF lookup — staleness check must use wall clock
        # because sim clock stops advancing when the bag pauses (get_clock() would never age out).
        self._last_odom_wall_time = self._steady_clock.now()

        msg_time = data.header.stamp
        frame_id = data.header.frame_id
        pose = data.pose.pose
        twist = data.twist.twist
        position = pose.position
        orientation = pose.orientation

        source_frame = frame_id
        target_frame = self.target_frame_id
        self.global_frame = source_frame
        # Only transform when a target frame is requested; otherwise record in the message's own frame.
        if target_frame:
            try:
                # timeout=0: non-blocking, uses the latest available transform
                trans = self.tf_buffer.lookup_transform(
                        target_frame, source_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0))
                self.global_frame = target_frame
                new_pose = tf2_geometry_msgs.do_transform_pose(pose, trans)
                position = new_pose.position
                orientation = new_pose.orientation
            except (TransformException, LookupException) as e:
                self.get_logger().info(f"Could not transform {source_frame} → {target_frame}: {e}")
                if not self.save_if_transform_fails:
                    return

        # Update state — only reached on successful transform or save_if_transform_fails=True
        msg_time_obj = rclpy.time.Time.from_msg(msg_time)

        # Time reversal detection: bag restarted or looped — timestamps jump backward.
        # Reset all session state so the new playback writes a clean, monotonic sequence.
        if self._last_msg_time is not None:
            dt_msg = (msg_time_obj - self._last_msg_time).nanoseconds / 1e9
            if dt_msg < -1.0:
                self.get_logger().warning(
                    f'Time reversal detected ({dt_msg:.1f}s jump) — bag restarted or looped. '
                    f'Resetting recording state.')
                self._start_time = None
                self._last_write_time = None
                self._last_recorded_x = None
                self._last_recorded_y = None
                self._is_stale = False
                self._new_odom_since_last_write = False
                # Clear stale TF data so the buffer accepts the rewound timestamps
                # from the restarted bag without TF_OLD_DATA warnings.
                self.tf_buffer.clear()
        self._last_msg_time = msg_time_obj

        if self._start_time is None:
            self._start_time = msg_time_obj
        self.total_time_elapsed = (msg_time_obj - self._start_time).nanoseconds / 1e9

        self.x, self.y, self.z = position.x, position.y, position.z
        self.qx, self.qy, self.qz, self.qw = orientation.x, orientation.y, orientation.z, orientation.w
        self.roll, self.pitch, self.yaw = tf_transformations.euler_from_quaternion(
                [self.qx, self.qy, self.qz, self.qw])

        self.vx = twist.linear.x
        self.vy = twist.linear.y
        self.vz = twist.linear.z
        self.omega = twist.angular.z
        self.speed = np.linalg.norm([self.vx, self.vy, self.vz])

        self._new_odom_since_last_write = True

    def pose_callback(self, data):
        pass

    def twist_callback(self, data):
        pass

    def recording_callback(self):
        if self._start_time is None:
            return  # no valid odom+TF received yet

        # Staleness check: uses STEADY_TIME so it fires even when sim clock stops advancing.
        # A genuinely stopped vehicle still publishes odom, so this does NOT suppress dwell time.
        now_wall = self._steady_clock.now()
        odom_age = (now_wall - self._last_odom_wall_time).nanoseconds / 1e9
        if odom_age > self.stale_odom_timeout:
            if not self._is_stale:
                self.get_logger().warning(
                    f'Odom stale ({odom_age:.2f}s > {self.stale_odom_timeout}s threshold), '
                    f'pausing recording.')
                self._is_stale = True
            return
        if self._is_stale:
            self.get_logger().info('Odom resumed, recording continues.')
            self._is_stale = False

        if not self._new_odom_since_last_write:
            return

        # Optional distance threshold (disabled by default: min_distance=0.0).
        # Note: this suppresses dwell time at intentional stops.
        if self.min_distance > 0.0 and self._last_recorded_x is not None:
            dist = np.hypot(self.x - self._last_recorded_x, self.y - self._last_recorded_y)
            if dist < self.min_distance:
                return

        now = self.get_clock().now()
        # dt = time since last write (≈ save_interval); 0.0 on the first row of each session
        dt = 0.0 if self._last_write_time is None else (now - self._last_write_time).nanoseconds / 1e9

        if self._last_write_time is None:
            self.get_logger().info(f'Recording started — saving to {self.file_path}')

        self.waypoint_file.write(
                f"{self.global_frame}, {self.total_time_elapsed:.6f}, {dt:.6f}, "
                f"{self.x}, {self.y}, {self.z}, {self.yaw}, "
                f"{self.qx}, {self.qy}, {self.qz}, {self.qw}, "
                f"{self.vx}, {self.vy}, "
                f"{self.speed}, {self.omega}\n")

        self._new_odom_since_last_write = False
        self._row_count += 1
        self._last_write_time = now
        self._last_recorded_x = self.x
        self._last_recorded_y = self.y

        # Visualization runs AFTER the CSV write so it can never delay or interrupt
        # recording. Accumulation below is O(1) per row and always runs (so no trail
        # points are dropped); only the O(N) publish is throttled by viz_publish_interval.
        if self._enable_path_pub:
            self._append_path_pose()
        if self._enable_marker_pub:
            self._pending_arrows.append(self._build_trail_arrow())
        if (self._enable_path_pub or self._enable_marker_pub) and self._viz_publish_due(now):
            self._last_viz_pub_time = now
            if self._enable_path_pub:
                self.publish_path()
            if self._enable_marker_pub:
                self._publish_markers()

        self.get_logger().info(
            f'Recording: {self._row_count} rows written ({self.total_time_elapsed:.1f}s elapsed)',
            **self.log_kwarg)

    def _viz_publish_due(self, now):
        """Return whether the throttle interval has elapsed since the last viz publish."""
        if self._viz_publish_interval <= 0.0 or self._last_viz_pub_time is None:
            return True
        return (now - self._last_viz_pub_time).nanoseconds / 1e9 >= self._viz_publish_interval

    def _append_path_pose(self):
        """Append the current pose to the accumulating Path (O(1), always runs)."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.global_frame
        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        pose_msg.pose.position.z = self.z
        pose_msg.pose.orientation.x = self.qx
        pose_msg.pose.orientation.y = self.qy
        pose_msg.pose.orientation.z = self.qz
        pose_msg.pose.orientation.w = self.qw
        self.path_msg.poses.append(pose_msg)

    def publish_path(self):
        """Republish the full accumulated Path (O(N); throttled by the caller)."""
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.header.frame_id = self.global_frame
        self.path_pub.publish(self.path_msg)

    def _publish_markers(self):
        """Publish the batch of pending trail arrows + the live position sphere.

        Arrows carry unique monotonic ids and an infinite lifetime, so RViz
        accumulates them into a persistent trail; the sphere (fixed id 0) is
        overwritten each publish to track the latest recorded position.
        """
        markers = self._pending_arrows + [self._build_current_sphere()]
        self.marker_pub.publish(MarkerArray(markers=markers))
        self._pending_arrows = []

    def _build_current_sphere(self):
        """Build the live current-position sphere marker (green, id 0, overwritten)."""
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'waypoint_recorder/current'
        marker.id = 0  # always overwrite the same marker — live current-position indicator

        marker.type = marker.SPHERE
        marker.action = marker.ADD

        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0

        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = self.z
        marker.pose.orientation.x = self.qx
        marker.pose.orientation.y = self.qy
        marker.pose.orientation.z = self.qz
        marker.pose.orientation.w = self.qw

        marker.text = f"{self.speed:.2f}"
        marker.lifetime = Duration(seconds=0).to_msg()  # 0 means forever
        return marker

    def _speed_to_color(self, speed):
        """Map speed to an RGB gradient: blue (slow) → green → red (fast)."""
        if self.marker_max_speed <= 0.0:
            return 0.0, 1.0, 0.0  # gradient disabled → constant green
        t = min(max(abs(speed) / self.marker_max_speed, 0.0), 1.0)
        if t < 0.5:  # blue → green
            s = t / 0.5
            return 0.0, s, 1.0 - s
        s = (t - 0.5) / 0.5  # green → red
        return s, 1.0 - s, 0.0

    def _build_trail_arrow(self):
        """Build a persistent velocity arrow at the just-recorded waypoint.

        One arrow per recorded row, oriented along the vehicle heading, length
        scaled by speed and colour-coded by speed. Unique monotonic ids + an
        infinite lifetime mean the arrows accumulate into a trail rather than
        overwriting each other (unlike the live current-position sphere).
        """
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'waypoint_recorder/trail'
        marker.id = self._trail_marker_id
        self._trail_marker_id += 1

        marker.type = marker.ARROW
        marker.action = marker.ADD

        # Arrow points along the vehicle's +x (heading ≈ velocity direction for a car).
        # Shaft length encodes speed; a floor keeps stopped waypoints visible.
        marker.scale.x = max(0.5, abs(self.speed) * 0.3)  # shaft length
        marker.scale.y = 0.2                              # shaft diameter
        marker.scale.z = 0.3                              # head diameter

        r, g, b = self._speed_to_color(self.speed)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, 1.0

        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = self.z
        marker.pose.orientation.x = self.qx
        marker.pose.orientation.y = self.qy
        marker.pose.orientation.z = self.qz
        marker.pose.orientation.w = self.qw

        marker.lifetime = Duration(seconds=0).to_msg()  # 0 means forever
        return marker


def main(args=None):
    rclpy.init(args=args)
    try:
        wr_node = WaypointRecorderNode()
        try:
            rclpy.spin(wr_node)
        finally:
            wr_node.waypoint_file.close()
            wr_node.destroy_node()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
