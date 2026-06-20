"""
Load path from waypoints and set QoS profile to transient.

Todo:
    Switch to the new trajectory class and add other columns
    Make the marker type configurable and choose what to display
    add a flag to replace zero speeds with a default value
    Add flag to publish initialpose as the first data row
    Pass data into ROS arrays at once instead of appending
    Publish as a list of poses for NavigateThroughPoses action servers
    Limit the number of nodes/states published
    Switch to custom message or actions to specify Path + speeds, i.e trajectory
    Smooth/interpolate (in a separate replanner node)
    Move to a separate package
"""
import sys

import pandas as pd
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf_transformations
import tf2_geometry_msgs

from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

import trajectory_following_ros2.utils.filters as filters
import trajectory_following_ros2.utils.trajectory_utils as trajectory_utils


class WaypointLoaderNode(Node):
    """Load waypoints from a CSV file and publish as Path, speed, and MarkerArray."""

    def __init__(self, ):
        """Constructor for WaypointLoaderNode"""
        super(WaypointLoaderNode, self).__init__('waypoint_loader')

        # declare parameters
        self.declare_parameter('file_path', '')
        self.declare_parameter('path_topic', 'waypoint_loader/path')
        self.declare_parameter('speed_topic', 'waypoint_loader/speed')
        self.declare_parameter('marker_topic', 'waypoint_loader/markers')
        self.declare_parameter('target_frame_id', 'map',
                               ParameterDescriptor(description='The static frame to publish waypoints in.'))
        self.declare_parameter('publish_if_transform_fails', False,
                               ParameterDescriptor(description='Should a poses be published in the default frame '
                                                               'if there is no valid '
                                                               'transformation to the target frame.'))
        self.declare_parameter('remove_duplicates', True)
        self.declare_parameter('smooth_path', True)
        self.declare_parameter('smooth_speed', True)
        self.declare_parameter('smooth_yaw', True)
        # Vehicle-specific starting points for the three smoothing parameters.
        # start_vel is the filter's initial condition at waypoint 0 — use 0.0 when the
        # recorded trajectory begins from rest; match the entry speed for highway segments.
        # accel_limit and time_constant are comfort/capability constraints, not hard vehicle limits.
        #
        #   Platform                       start_vel    accel_limit   time_constant
        #   F1/10 — indoor/campus          3.0 m/s      2.0 m/s²      0.15 s
        #   F1/10 — outdoor, aggressive    5.0 m/s      3.0 m/s²      0.10 s
        #   Chrysler Pacifica — urban      4.0 m/s      1.5 m/s²      0.50 s
        #   Chrysler Pacifica — highway   10.0 m/s      1.0 m/s²      0.60 s
        #   CARLA Tesla Model 3 — urban    5.0 m/s      2.5 m/s²      0.30 s
        #   CARLA Tesla Model 3 — highway 15.0 m/s      2.0 m/s²      0.40 s
        self.declare_parameter('smooth_speed_start_vel', 0.0,
                               ParameterDescriptor(description='Initial velocity seed (m/s) for the speed '
                                                               'smoothing profile. Use 0.0 for trajectories '
                                                               'starting from rest.'))
        self.declare_parameter('smooth_speed_accel_limit', 3.0,
                               ParameterDescriptor(description='Acceleration limit (m/s^2) for speed smoothing. '
                                                               'Comfort range: ~2.0 (full-size AV) to ~3.0 '
                                                               '(F1/10 / CARLA sim).'))
        self.declare_parameter('smooth_speed_time_constant', 0.3,
                               ParameterDescriptor(description='Velocity filter time constant (s). Larger = '
                                                               'smoother but slower speed transitions. Typical: '
                                                               '0.1-0.2 (F1/10), 0.3-0.4 (CARLA), '
                                                               '0.5-0.6 (Pacifica).'))
        self.declare_parameter('reverse_speed_threshold', 0.05,
                               ParameterDescriptor(description='Speed magnitude (m/s) below which vx is treated '
                                                               'as forward when restoring sign. Deadbands the '
                                                               'sign-flip dithering of a noisy near-stationary '
                                                               'start; a waypoint counts as reverse only when '
                                                               'vx < -threshold.'))

        # get parameters
        self.file_path = str(self.get_parameter('file_path').value)
        self.path_topic = self.get_parameter('path_topic').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.target_frame_id = self.get_parameter('target_frame_id').value
        self.publish_if_transform_fails = self.get_parameter('publish_if_transform_fails').value

        self.remove_duplicates = self.get_parameter('remove_duplicates').value
        self.smooth_path = self.get_parameter('smooth_path').value
        self.smooth_speed = self.get_parameter('smooth_speed').value
        self.smooth_yaw = self.get_parameter('smooth_yaw').value
        self.smooth_speed_start_vel = self.get_parameter('smooth_speed_start_vel').value
        self.smooth_speed_accel_limit = self.get_parameter('smooth_speed_accel_limit').value
        self.smooth_speed_time_constant = self.get_parameter('smooth_speed_time_constant').value
        self.reverse_speed_threshold = self.get_parameter('reverse_speed_threshold').value

        # Setup transformations to transform pose from one frame to another.
        # Created before parsing so transform_waypoints() can use it on the loaded data.
        # node=self shares the node's clock — critical when use_sim_time=True.
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Load waypoints (csv_data: frame_id, total_time_elapsed, dt, x, y, z, yaw, qx, qy, qz, qw,
        #                           vx, vy, speed, omega)
        # Load immediately to fail fast on invalid CSV paths
        self.csv_data = self.get_waypoints_from_csv(self.file_path)
        # Normalize frame_id values (CSV is read with skipinitialspace, but guard against
        # trailing whitespace) so frame comparisons are reliable.
        self.csv_data["frame_id"] = self.csv_data["frame_id"].astype(str).str.strip()
        # Cache once — frame_id column doesn't change until transform_waypoints() runs (after which the timer is cancelled).
        self._source_frames = set(self.csv_data["frame_id"])
        # Cache zero-time sentinel and initial warn timestamp; both need the clock to be ready.
        self._time_zero = rclpy.time.Time(clock_type=self.get_clock().clock_type)
        self._last_warn_time = self.get_clock().now()

        # Setup empty messages
        self.path_msg = Path()
        self.marker_array_msg = MarkerArray()
        self.speed_msg = Float32MultiArray()

        # Setup publishers (transient-local = latching: late subscribers still receive the message)
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos_profile=latching_qos)
        self.speed_pub = self.create_publisher(Float32MultiArray, self.speed_topic, qos_profile=latching_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, qos_profile=latching_qos)

        # Create a timer to attempt transforms and publishing once TF is ready
        self.init_timer = self.create_timer(0.5, self.init_callback)

        self.get_logger().info('waypoint_loader started. Waiting for transforms...')

    def get_waypoints_from_csv(self, path_to_csv):
        return pd.read_csv(path_to_csv, skipinitialspace=True, encoding='utf-8-sig')

    def waypoint_parser(self):
        """Build Path, speed, and MarkerArray messages from loaded waypoints. Runs once."""
        self._parse_stamp = self.get_clock().now().to_msg()
        for row in range(len(self.xs)):
            self.parse_trajectory(index=row)
            self.parse_marker(index=row, display_type='orientation')  # orientation, speed, position
        # Set Path header once; target_frame_id if transforms ran, else first row's frame.
        self.path_msg.header.stamp = self._parse_stamp
        self.path_msg.header.frame_id = self.target_frame_id if self.target_frame_id else self.frame_ids[0]

    def init_callback(self):
        """Timer callback to check for transforms, process waypoints, and publish once."""
        now = self.get_clock().now()
        missing_frame = None

        if self.target_frame_id:
            for source_frame in self._source_frames:
                if source_frame == self.target_frame_id:
                    continue
                if not self.tf_buffer.can_transform(self.target_frame_id, source_frame, self._time_zero):
                    missing_frame = source_frame
                    break

            if missing_frame is not None:
                if (now - self._last_warn_time).nanoseconds > 5_000_000_000:
                    self._last_warn_time = now
                    self.get_logger().warning(
                        f'Waiting for transform "{missing_frame}" → "{self.target_frame_id}" in TF buffer...')
                if not self.publish_if_transform_fails:
                    return
                self.get_logger().warning(
                    f'Transform "{missing_frame}" → "{self.target_frame_id}" unavailable; '
                    f'publishing in source frame (publish_if_transform_fails=True).')

        # Transform coordinates
        try:
            self.transform_waypoints()
        except Exception as e:
            self.get_logger().error(f'Failed during waypoint transform: {e}')
            self.init_timer.cancel()
            return

        if self.remove_duplicates:
            self.csv_data = self.csv_data.drop_duplicates(subset=["x", "y"], keep='first')

        # optional smoothing
        if self.smooth_path:
            self.csv_data[["x", "y"]] = filters.smooth_and_interpolate_coordinates(
                    coordinates=self.csv_data[["x", "y"]].to_numpy(), method='bspline',
                    polynomial_order=3, weight_smooth=0.3)
            # Yaw recomputation deferred until after smooth_speed (below) so that the
            # vx signs used to determine forward/reverse are from the final speed profile.

        if self.smooth_speed:
            # smooth_speed operates on the unsigned magnitude for the speed profile shape,
            # then the sign is restored from the original vx column sign. A deadband treats
            # |vx| < reverse_speed_threshold as forward, removing the sign-flip dithering of
            # a noisy near-stationary start (where recorded vx oscillates around zero).
            vx = self.csv_data["vx"].to_numpy()
            signs = np.sign(vx)
            signs[np.abs(vx) < self.reverse_speed_threshold] = 1.0
            smoothed_mag, _ = trajectory_utils.dynamic_smoothing_velocity(
                    0, self.smooth_speed_start_vel, self.smooth_speed_accel_limit, self.smooth_speed_time_constant,
                    self.csv_data.loc[:, ['x', 'y', 'speed']].to_numpy())
            self.csv_data["vx"] = smoothed_mag * signs

        if self.smooth_path:
            # Recompute yaw from the smoothed (x, y) using the sign of vx to determine
            # whether the robot is moving forward or backward at each waypoint:
            #   forward  (vx > 0): heading = path tangent      arctan2( dy,  dx)
            #   reverse  (vx < 0): heading = opposite tangent  arctan2(-dy, -dx)
            # At the turnaround (vx → 0) the two expressions agree, so there is no
            # discontinuity — the heading stays constant through the stop-and-reverse.
            xs = self.csv_data["x"].to_numpy()
            ys = self.csv_data["y"].to_numpy()
            vxs = self.csv_data["vx"].to_numpy()
            dx = np.diff(xs)
            dy = np.diff(ys)
            seg_signs = np.sign(vxs[:-1])   # one sign per segment
            # Same deadband as the speed sign above: near-zero vx counts as forward.
            seg_signs[np.abs(vxs[:-1]) < self.reverse_speed_threshold] = 1.0
            yaws = np.arctan2(dy * seg_signs, dx * seg_signs)
            yaws = np.append(yaws, yaws[-1])  # repeat terminal heading for last waypoint
            self.csv_data["yaw"] = yaws
            self.csv_data["qx"] = 0.0
            self.csv_data["qy"] = 0.0
            self.csv_data["qz"] = np.sin(yaws / 2.0)
            self.csv_data["qw"] = np.cos(yaws / 2.0)

        if self.smooth_yaw:
            # Unwrap the yaw values to prevent wrap-around issues during filtering
            yaws = np.unwrap(self.csv_data["yaw"].to_numpy())
            # Smooth the unwrapped yaw values using moving average
            yaws_smoothed = filters.moving_average(yaws[:, np.newaxis], window_length=9, kernel_type='gaussian').squeeze()
            # Normalize the smoothed yaw values back to [-pi, pi]
            yaws_smoothed = np.arctan2(np.sin(yaws_smoothed), np.cos(yaws_smoothed))
            self.csv_data["yaw"] = yaws_smoothed

            # Recalculate quaternions from the smoothed yaw (vectorized; avoids label-vs-position index mismatch)
            self.csv_data["qx"] = 0.0
            self.csv_data["qy"] = 0.0
            self.csv_data["qz"] = np.sin(yaws_smoothed / 2.0)
            self.csv_data["qw"] = np.cos(yaws_smoothed / 2.0)

        self.frame_ids = self.csv_data["frame_id"].tolist()
        self.xs = self.csv_data["x"].tolist()
        self.ys = self.csv_data["y"].tolist()
        self.zs = self.csv_data["z"].tolist()
        self.yaws = self.csv_data["yaw"].tolist()
        self.qxs = self.csv_data["qx"].tolist()
        self.qys = self.csv_data["qy"].tolist()
        self.qzs = self.csv_data["qz"].tolist()
        self.qws = self.csv_data["qw"].tolist()
        self.vxs = self.csv_data["vx"].tolist()
        self.vys = self.csv_data["vy"].tolist()
        # Always use signed vx: the MPC state uses vx (base_tracker.py: speed = vx),
        # so the reference must be signed too. The unsigned "speed" column is only
        # equivalent for forward-only paths and breaks reversal tracking.
        self.speeds = self.csv_data["vx"].tolist()
        self.yaw_rates = self.csv_data["omega"].tolist()

        # Parse waypoints
        self.marker_id = 0
        self.waypoint_parser()

        # Publish once
        self.publisher_callback()

        self.get_logger().info('Successfully loaded, transformed, and published waypoints. Stopping initialization timer.')
        self.init_timer.cancel()

    def transform_waypoints(self):
        """Transform each loaded waypoint into target_frame_id, in place on csv_data.

        Each pose is transformed from its own recorded ``frame_id`` to ``target_frame_id``.
        The recorder can emit mixed frames, so the transform is looked up once
        per unique source frame and applied to every row in that frame.

        Behaviour is gated on ``target_frame_id``: when unset this only warns about
        multi-frame data and returns without modifying coordinates.
        """
        source_frames = set(self.csv_data["frame_id"])
        if len(source_frames) > 1:
            self.get_logger().warning(
                f'CSV contains multiple frames {sorted(source_frames)} — the recorder '
                f'likely wrote some rows in their source frame after TF failures. '
                f'Coordinates across frames are NOT directly comparable.')

        if not self.target_frame_id:
            if len(source_frames) > 1:
                self.get_logger().warning(
                    'target_frame_id is unset — waypoints are published in their original '
                    'mixed frames without transformation. Set target_frame_id to unify them.')
            return

        # One lookup per unique source frame; reused across all rows in that frame.
        transforms = {}
        for source_frame in source_frames:
            if source_frame == self.target_frame_id:
                continue
            if self.tf_buffer.can_transform(self.target_frame_id, source_frame, self._time_zero):
                transforms[source_frame] = self.tf_buffer.lookup_transform(
                        self.target_frame_id, source_frame, self._time_zero)
            else:
                msg = f'No transform {source_frame} → {self.target_frame_id} available.'
                if self.publish_if_transform_fails:
                    self.get_logger().warning(
                        msg + f' Publishing {source_frame} rows in their source frame.')
                else:
                    raise TransformException(
                        msg + ' Set publish_if_transform_fails=True to publish anyway.')

        if not transforms:
            return  # all rows already in target frame (or unavailable + publish anyway)

        for i in self.csv_data.index:
            source_frame = self.csv_data.at[i, "frame_id"]
            trans = transforms.get(source_frame)
            if trans is None:
                continue  # already in target frame, or unavailable + publish_if_transform_fails

            pose = Pose()
            pose.position.x = float(self.csv_data.at[i, "x"])
            pose.position.y = float(self.csv_data.at[i, "y"])
            pose.position.z = float(self.csv_data.at[i, "z"])
            pose.orientation.x = float(self.csv_data.at[i, "qx"])
            pose.orientation.y = float(self.csv_data.at[i, "qy"])
            pose.orientation.z = float(self.csv_data.at[i, "qz"])
            pose.orientation.w = float(self.csv_data.at[i, "qw"])

            new_pose = tf2_geometry_msgs.do_transform_pose(pose, trans)

            self.csv_data.at[i, "x"] = new_pose.position.x
            self.csv_data.at[i, "y"] = new_pose.position.y
            self.csv_data.at[i, "z"] = new_pose.position.z
            self.csv_data.at[i, "qx"] = new_pose.orientation.x
            self.csv_data.at[i, "qy"] = new_pose.orientation.y
            self.csv_data.at[i, "qz"] = new_pose.orientation.z
            self.csv_data.at[i, "qw"] = new_pose.orientation.w
            _, _, yaw = tf_transformations.euler_from_quaternion(
                    [new_pose.orientation.x, new_pose.orientation.y,
                     new_pose.orientation.z, new_pose.orientation.w])
            self.csv_data.at[i, "yaw"] = yaw
            self.csv_data.at[i, "frame_id"] = self.target_frame_id

        self.get_logger().info(
            f'Transformed waypoints from {sorted(transforms.keys())} into '
            f'"{self.target_frame_id}".')

    def publisher_callback(self):
        """Publish all waypoint messages. Transient-local QoS handles late subscribers."""
        self.path_pub.publish(self.path_msg)
        self.speed_pub.publish(self.speed_msg)
        self.marker_pub.publish(self.marker_array_msg)

    def parse_path(self, index):
        frame_id = self.frame_ids[index]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self._parse_stamp
        pose_msg.header.frame_id = frame_id

        pose_msg.pose.position.x = self.xs[index]
        pose_msg.pose.position.y = self.ys[index]
        pose_msg.pose.position.z = self.zs[index]
        pose_msg.pose.orientation.x = self.qxs[index]
        pose_msg.pose.orientation.y = self.qys[index]
        pose_msg.pose.orientation.z = self.qzs[index]
        pose_msg.pose.orientation.w = self.qws[index]

        self.path_msg.poses.append(pose_msg)

    def parse_speed(self, index):
        """todo: switch to custom message instead"""
        self.speed_msg.data.append(float(self.speeds[index]))

    def parse_trajectory(self, index):
        self.parse_path(index)
        self.parse_speed(index)

    def parse_marker(self, index, scale=None, rgb=None, display_type='orientation'):
        """
        :param index:
        :param scale:
        :param rgb:
        :param display_type: orientation (arrow), speed (text), position (sphere)
        :return:
        """
        if rgb is None:
            rgb = [0., 1., 0.]
        if scale is None:
            scale = [0.2, 0.1, 0.05]

        marker = Marker()
        marker.header.frame_id = self.frame_ids[index]
        marker.header.stamp = self._parse_stamp
        marker.ns = 'waypoint_loader'
        marker.id = self.marker_id
        self.marker_id += 1

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        # See: http://wiki.ros.org/rviz/DisplayTypes/Marker |
        if display_type == 'orientation':
            marker_type = marker.ARROW
        elif display_type == 'speed':
            marker_type = marker.TEXT_VIEW_FACING
        elif display_type == 'position':
            marker_type = marker.SPHERE
        else:
            self.get_logger().warning(f'Unknown marker display_type "{display_type}", defaulting to orientation.')
            marker_type = marker.ARROW

        marker.type = marker_type  # SPHERE, ARROW, TEXT_VIEW_FACING
        marker.action = marker.ADD

        # Set the scale of the marker
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        # Set the color
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = self.xs[index]
        marker.pose.position.y = self.ys[index]
        marker.pose.position.z = self.zs[index]
        marker.pose.orientation.x = self.qxs[index]
        marker.pose.orientation.y = self.qys[index]
        marker.pose.orientation.z = self.qzs[index]
        marker.pose.orientation.w = self.qws[index]

        # Set text (shown when display_type='speed')
        marker.text = f"{self.speeds[index]:.2f}"

        # to automatically delete old markers
        marker.lifetime = Duration(seconds=0).to_msg()  # 0 means forever.

        self.marker_array_msg.markers.append(marker)


def main(args=None):
    rclpy.init(args=args)
    try:
        wl_node = WaypointLoaderNode()
        try:
            rclpy.spin(wl_node)
        finally:
            wl_node.destroy_node()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
