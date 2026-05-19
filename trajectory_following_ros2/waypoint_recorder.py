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
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

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
        self.declare_parameter('target_frame_id', 'odom',
                               ParameterDescriptor(description='Frame to transform poses into before saving.'))
        self.declare_parameter('odom_topic', 'odometry/local')
        self.declare_parameter('pose_topic', 'pose')
        self.declare_parameter('twist_topic', 'twist')
        self.declare_parameter('path_topic', 'waypoint_recorder/path')
        self.declare_parameter('marker_topic', 'waypoint_recorder/markers')
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
        self.save_if_transform_fails = self.get_parameter('save_if_transform_fails').value
        self.stale_odom_timeout = self.get_parameter('stale_odom_timeout').value
        self.min_distance = self.get_parameter('min_distance').value

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
        self._last_odom_wall_time = None  # wall-clock time of latest odom callback arrival
        self._last_write_time = None      # wall-clock time of last successful write
        self._last_recorded_x = None     # position at last write, for min_distance check
        self._last_recorded_y = None

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
        self.marker_array_msg = MarkerArray()

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
        # Stamp wall-clock arrival before TF lookup so staleness check works even on TF failure
        self._last_odom_wall_time = self.get_clock().now()

        msg_time = data.header.stamp
        frame_id = data.header.frame_id
        pose = data.pose.pose
        twist = data.twist.twist
        position = pose.position
        orientation = pose.orientation

        source_frame = frame_id
        target_frame = self.target_frame_id
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
            if self.save_if_transform_fails:
                self.global_frame = source_frame
            else:
                self.global_frame = source_frame
                return

        # Update state — only reached on successful transform or save_if_transform_fails=True
        msg_time_obj = rclpy.time.Time.from_msg(msg_time)
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

        self.publish_marker(position, orientation, self.speed)

    def pose_callback(self, data):
        pass

    def twist_callback(self, data):
        pass

    def recording_callback(self):
        if self._start_time is None:
            return  # no valid odom+TF received yet

        now = self.get_clock().now()

        # Staleness check: skip if odom topic has gone silent (topic loss, node crash, etc.).
        # A genuinely stopped vehicle still publishes odom, so this does NOT suppress dwell time.
        odom_age = (now - self._last_odom_wall_time).nanoseconds / 1e9
        if odom_age > self.stale_odom_timeout:
            self.get_logger().warning(
                    f'Odom stale ({odom_age:.2f}s > {self.stale_odom_timeout}s), skipping write.')
            return

        # Optional distance threshold (disabled by default: min_distance=0.0).
        # Note: this suppresses dwell time at intentional stops.
        if self.min_distance > 0.0 and self._last_recorded_x is not None:
            dist = np.hypot(self.x - self._last_recorded_x, self.y - self._last_recorded_y)
            if dist < self.min_distance:
                return

        # dt = time since last write (≈ save_interval); 0.0 on the first row
        dt = 0.0 if self._last_write_time is None else (now - self._last_write_time).nanoseconds / 1e9

        self.waypoint_file.write(
                f"{self.global_frame}, {self.total_time_elapsed:.6f}, {dt:.6f}, "
                f"{self.x}, {self.y}, {self.z}, {self.yaw}, "
                f"{self.qx}, {self.qy}, {self.qz}, {self.qw}, "
                f"{self.vx}, {self.vy}, "
                f"{self.speed}, {self.omega}\n")

        self._last_write_time = now
        self._last_recorded_x = self.x
        self._last_recorded_y = self.y
        self.publish_path()

    def publish_path(self):
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

        self.path_msg.header.stamp = pose_msg.header.stamp
        self.path_msg.header.frame_id = self.global_frame
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

    def publish_marker(self, position, orientation, speed, scale=None, rgb=None):
        if rgb is None:
            rgb = [0., 1., 0.]
        if scale is None:
            scale = [1., 1., 1.]

        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'waypoint_recorder'
        marker.id = 0  # always overwrite the same marker — only current position shown

        marker.type = marker.SPHERE  # SPHERE, ARROW, TEXT_VIEW_FACING
        marker.action = marker.ADD

        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0

        marker.pose.position = position
        marker.pose.orientation = orientation

        marker.text = f"{speed:.2f}"
        marker.lifetime = Duration(seconds=0).to_msg()  # 0 means forever

        # Replace rather than append: only the current position marker is kept
        self.marker_array_msg.markers = [marker]
        self.marker_pub.publish(self.marker_array_msg)


def main(args=None):
    rclpy.init(args=args)
    wr_node = WaypointRecorderNode()
    rclpy.spin(wr_node)
    wr_node.waypoint_file.close()
    wr_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()