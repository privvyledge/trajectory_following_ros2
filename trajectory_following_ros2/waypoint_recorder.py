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
        * add option to save at a specific frequency or only when messages are received
        * add the option to either save Odometry or subscribe to a path topic and save
        * add option to save only if the distance (odometry) changes by a threshold

Msgs:
    Path: http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Path.html
    Marker: text and pose
"""
import os
import sys
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# TF
from tf2_ros import TransformException, LookupException
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf_transformations
import tf2_geometry_msgs

# messages
from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path


class WaypointRecorderNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for WaypointRecorderNode"""
        super(WaypointRecorderNode, self).__init__('waypoint_recorder')

        # declare parameters
        self.declare_parameter('file_path', '')
        self.declare_parameter('waypoint_source', 'odometry')
        # self.declare_parameter('save_frequency', 20.0)  # Hz. todo: remove
        self.declare_parameter('save_interval', 1.0)  # seconds
        self.declare_parameter('target_frame_id', 'odom',
                               ParameterDescriptor(description='The static frame to save the waypoints in.'))  # map
        self.declare_parameter('odom_topic', '/odometry/local')
        self.declare_parameter('pose_topic', 'pose')
        self.declare_parameter('twist_topic', 'twist')
        self.declare_parameter('path_topic', '/waypoint_recorder/path')
        self.declare_parameter('marker_topic', '/waypoint_recorder/markers')
        self.declare_parameter('save_if_transform_fails', False,
                               ParameterDescriptor(description='Should a pose be saved if there is no valid '
                                                               'transformation to the target frame.'))

        # get parameters
        self.file_path = str(self.get_parameter('file_path').value)
        self.waypoint_source = self.get_parameter('waypoint_source').value
        # self.save_frequency = self.get_parameter('save_frequency').value
        self.save_interval = self.get_parameter('save_interval').value
        self.target_frame_id = self.get_parameter('target_frame_id').value  # target_frame
        self.odom_topic = self.get_parameter('odom_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.save_if_transform_fails = self.get_parameter('save_if_transform_fails').value

        # initialize variables
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.qx, self.qy, self.qz, self.qw = 0.0, 0., 0., 0.
        self.roll, self.pitch, self.yaw = 0., 0., 0.
        self.vx, self.vy, self.vz = 0., 0., 0.
        self.speed, self.omega = 0.0, 0.0
        self.global_frame = ''
        self.msg_time = None
        self.total_time_elapsed = 0.0
        self.dt = 0.
        self.initial_message_received = False

        header = 'frame_id, total_time_elapsed, dt, x, y, z, yaw, qx, qy, qz, qw, vx, vy, speed, omega\n'
        self.waypoint_file = open(self.file_path, 'w', encoding="utf-8")
        self.get_logger().info(f'Saving to {self.file_path}. ')
        self.waypoint_file.write(header)

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Setup empty messages
        self.path_msg = Path()
        self.marker_array_msg = MarkerArray()

        # Setup publishers
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)

        # Setup subscribers
        self.odom_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_callback,
                1)
        self.pose_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                self.pose_topic,
                self.pose_callback,
                1)
        self.twist_sub = self.create_subscription(
                TwistStamped,
                self.twist_topic,
                self.twist_callback,
                1)

        # setup timers
        self.recording_timer = self.create_timer(self.save_interval, self.recording_callback)

        self.get_logger().info('waypoint_recorder started. ')

    def odom_callback(self, data):
        self.initial_message_received = True
        msg_time = data.header.stamp
        if self.msg_time is not None:
            time_delta = rclpy.time.Time.from_msg(msg_time) - rclpy.time.Time.from_msg(self.msg_time)
            delta_in_seconds = time_delta.nanoseconds / 1e+9
            self.dt = abs(delta_in_seconds)

        # wrong. Todo: should be done where the waypoints are being saved
        self.msg_time = msg_time
        self.total_time_elapsed = self.msg_time.sec + (self.msg_time.nanosec / 1e+9)

        frame_id = data.header.frame_id  # source frame

        pose = data.pose.pose
        twist = data.twist.twist

        position = pose.position
        orientation = pose.orientation
        # self.get_logger().info(f"Original pose: {[position.x, position.y, position.z]}, "
        #                     f"original orientation: {[orientation.x, orientation.y, orientation.z, orientation.w]}")

        # transform pose to a different frame
        source_frame = frame_id
        target_frame = self.target_frame_id
        try:
            now = rclpy.time.Time()  # or self.get_clock().now().to_msg()
            # note that this method blocks until the transform is available.
            # Check https://github.com/ros2/geometry2/blob/foxy/examples_tf2_py/examples_tf2_py/waits_for_transform.py
            # for non-blocking example.
            trans = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    now,
                    timeout=Duration(seconds=2.0)
            )
            self.global_frame = target_frame
            transform_translation = trans.transform.translation
            transform_orientation = trans.transform.rotation

            # transform_time = self.tf_listener.getLatestCommonTime(target_frame, source_frame)
            # temp_pose = PoseStamped()
            # temp_pose.header.frame_id = source_frame
            # temp_pose.pose = pose
            # pose_in_target_frame = self.tf_listener.transformPose(target_frame, temp_pose)

            new_pose = tf2_geometry_msgs.do_transform_pose(pose, trans)
            position = new_pose.position
            orientation = new_pose.orientation
        except (TransformException, LookupException) as e:
            self.get_logger().info(f"Could not transform from {source_frame} to {target_frame}: {e}")
            # Option1: save in default frame
            if self.save_if_transform_fails:
                self.global_frame = source_frame
            else:
                # Option2: do not save anything.
                # todo: set values as Nan
                self.global_frame = source_frame
                return
                # Option3: interpolate

        self.x, self.y, self.z = position.x, position.y, position.z
        self.qx, self.qy, self.qz, self.qw = orientation.x, orientation.y, orientation.z, orientation.w

        self.roll, self.pitch, self.yaw = tf_transformations.euler_from_quaternion([self.qx, self.qy, self.qz, self.qw])

        self.vx = twist.linear.x
        self.vy = twist.linear.y
        self.vz = twist.linear.z
        self.omega = twist.angular.z

        self.speed = np.linalg.norm([self.vx, self.vy, self.vz])
        self.publish_path(position, orientation)
        self.publish_marker(position, orientation, self.speed)

    def pose_callback(self, data):
        self.initial_message_received = True
        pass

    def twist_callback(self, data):
        self.initial_message_received = True
        pass

    def recording_callback(self):
        if self.initial_message_received:
            self.waypoint_file.write(f"{self.global_frame}, {self.total_time_elapsed}, {self.dt}, "
                                     f"{self.x}, {self.y}, {self.z}, {self.yaw}, "
                                     f"{self.qx}, {self.qy}, {self.qz}, {self.qw}, "
                                     f"{self.vx}, {self.vy}, "
                                     f"{self.speed}, {self.omega}\n")
            # todo: remove
            self.get_logger().info(f"{self.global_frame}, {self.total_time_elapsed}, {self.dt}, "
                                   f"{self.x}, {self.y}, {self.z}, {self.yaw}, "
                                   f"{self.qx}, {self.qy}, {self.qz}, {self.qw}, "
                                   f"{self.vx}, {self.vy}, "
                                   f"{self.speed}, {self.omega}\n")

    def publish_path(self, position, orientation):
        # todo: fix and test
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.global_frame
        pose_msg.pose.position = position
        pose_msg.pose.orientation = orientation

        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

    def publish_marker(self, position, orientation, speed, scale=None, rgb=None):
        # todo: fix and test
        if rgb is None:
            rgb = [0., 1., 0.]
        if scale is None:
            scale = [1., 1., 1.]

        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'waypoint_recorder'
        marker.id = 1268

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        # See: http://wiki.ros.org/rviz/DisplayTypes/Marker |
        marker.type = marker.SPHERE  # SPHERE, ARROW, TEXT_VIEW_FACING
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
        marker.pose.position = position
        marker.pose.orientation = orientation

        # Set text
        marker.text = f"{speed}"

        # to automatically delete old markers
        marker.lifetime = Duration(seconds=0.01).to_msg()  # 0 means forever

        self.marker_array_msg.markers.append(marker)
        self.marker_pub.publish(self.marker_array_msg)


def main(args=None):
    rclpy.init(args=args)
    wr_node = WaypointRecorderNode()
    rclpy.spin(wr_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    wr_node.waypoint_file.close()
    wr_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
