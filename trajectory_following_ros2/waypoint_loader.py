"""
Load path from waypoints and set QoS profile to transient.

Todo:
    switch to pandas [done]
    Check if the file exists [done: no need]
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
import os
import sys
import csv
import threading
import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

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
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension  # todo: remove

# Custom messages
from trajectory_following_ros2.utils.Trajectory import Trajectory


class WaypointLoaderNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for WaypointLoaderNode"""
        super(WaypointLoaderNode, self).__init__('waypoint_loader')

        # declare parameters
        self.declare_parameter('file_path', '')
        self.declare_parameter('path_topic', '/waypoint_loader/path')
        self.declare_parameter('speed_topic', '/waypoint_loader/speed')
        self.declare_parameter('marker_topic', '/waypoint_loader/markers')
        self.declare_parameter('target_frame_id', '',
                               ParameterDescriptor(description='The static frame to save the waypoints in.'))  # map
        self.declare_parameter('publish_if_transform_fails', False,
                               ParameterDescriptor(description='Should a poses be published in the default frame '
                                                               'if there is no valid '
                                                               'transformation to the target frame.'))

        # get parameters
        self.file_path = str(self.get_parameter('file_path').value)
        self.path_topic = self.get_parameter('path_topic').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.target_frame_id = self.get_parameter('target_frame_id').value
        self.publish_if_transform_fails = self.get_parameter('publish_if_transform_fails').value

        # Load waypoints (csv_data: frame_id, time_delta (dt), x, y, z, yaw, qx, qy, qz, qw, speed)
        self.csv_data = self.get_waypoints_from_csv(self.file_path)
        # # optional smoothing
        # self.trajectory_instance = Trajectory(search_index_number=10,
        #                                       goal_tolerance=self.distance_tolerance,
        #                                       stop_speed=self.speed_tolerance
        #                                       )
        # self.csv_data[:, 2:4] = self.trajectory_instance.smooth_and_interpolate_coordinates(
        #         coordinates=self.csv_data[:, 2:4],
        #         method='moving_average',
        #         window_length=9,
        #         kernel_type='box')
        # self.csv_data[:, 5] = self.trajectory_instance.smooth_yaw(self.csv_data[:, 5])
        # self.csv_data[:, 10] = self.trajectory_instance.calc_speed_profile(self.csv_data[:, 2], self.csv_data[:, 3],
        #                                                                    self.csv_data[:, 5], self.desired_speed)
        # # to smooth speed. todo: use an optimization function or let the MPC do it
        # # todo: don't use Bspline interpolation for speed as it does not necessarily respect the kinematics of a car
        # # todo: instead use an optimization function for the velocity profile
        # interpolation_function, b_spline_coefficients = scipy.interpolate.splprep(
        #         [np.arange(self.speeds.shape[0]), self.speeds], s=0.3,
        #         k=3)  # we use splprep to handle nonunique and not strictly ascending x
        # _, self.speeds = scipy.interpolate.splev(b_spline_coefficients, interpolation_function)

        # todo: just do the unpacking where needed
        self.frame_ids = self.csv_data["frame_id"].tolist()
        self.total_time_elapsed = self.csv_data["total_time_elapsed"].tolist()
        self.dts = self.csv_data["dt"].tolist()
        self.xs = self.csv_data["x"].tolist()
        self.ys = self.csv_data["y"].tolist()
        self.zs = self.csv_data["z"].tolist()
        self.yaws = self.csv_data["yaw"].tolist()
        self.qxs = self.csv_data["qx"].tolist()
        self.qys = self.csv_data["qy"].tolist()
        self.qzs = self.csv_data["qz"].tolist()
        self.qws = self.csv_data["qw"].tolist()
        # self.waypoints = self.csv_data.loc[:, "x":"yaw"]
        self.vxs = self.csv_data["vx"].tolist()
        self.vys = self.csv_data["vy"].tolist()
        self.speeds = self.csv_data["speed"].tolist()
        self.yaw_rates = self.csv_data["omega"].tolist()

        # Setup transformations to transform pose from one frame to another. todo
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Setup empty messages
        self.path_msg = Path()
        self.marker_array_msg = MarkerArray()
        self.speed_msg = Float32MultiArray()
        # self.speed_msg.layout.data_offset = 0
        # self.speed_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        # # dim[0] is the vertical dimension of your matrix
        # self.speed_msg.layout.dim[0].label = "rows"
        # self.speed_msg.layout.dim[0].size = self.vxs.shape[0]
        # self.speed_msg.layout.dim[0].stride = self.vxs.shape[0]
        # # dim[1] is the horizontal dimension of your matrix
        # self.speed_msg.layout.dim[1].label = "columns"
        # self.speed_msg.layout.dim[1].size = 1
        # self.speed_msg.layout.dim[1].stride = 1

        # Parse waypoints
        self.marker_id = 0
        self.waypoint_parser()

        # Setup publishers
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos_profile=latching_qos)
        self.speed_pub = self.create_publisher(Float32MultiArray, self.speed_topic, qos_profile=latching_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, qos_profile=latching_qos)

        # Setup timers
        self.publisher_callback()

        self.get_logger().info('waypoint_loader started. ')

    def get_waypoints_from_csv(self, path_to_csv):
        waypoints_df = pd.read_csv(path_to_csv, skipinitialspace=True)
        csv_data = waypoints_df
        return csv_data

    def waypoint_parser(self):
        """
        Should run once to add the waypoints to Path and MarkerArray messages.
        """
        # self.path_msg = Path()
        # self.path_msg.header.stamp = rospy.Time.now()
        # self.path_msg.header.frame_id = "map"
        #
        # for i in range(self.waypoints.shape[0]):
        #     loc = PoseStamped()
        #     loc.header.stamp = rospy.Time.now()
        #     loc.header.frame_id = "map"
        #     loc.pose.position.x = self.waypoints[i, 0]
        #     loc.pose.position.y = self.waypoints[i, 1]
        #     loc.pose.position.z = 0
        #     self.path_msg.poses.append(loc)

        # self.path_msg = Path()
        # self.marker_array_msg = MarkerArray()
        for row in range(self.csv_data.shape[0]):
            # waypoint: frame_id, time_delta (dt), x, y, z, yaw, qx, qy, qz, qw, speed
            if self.target_frame_id:
                # first transform then redeclare
                self.frame_ids = self.target_frame_id  # move to waypoint replanner node
                # self.waypoints[:, 0] = new_xs

            self.parse_trajectory(index=row)
            self.parse_marker(index=row, display_type='orientation')  # orientation, speed, position

    def transform_waypoints(self):
        """
        Todo: Transform the waypoints to another frame. Move to waypoint_replanner node
        This usually should be done by the controller but can optionally be done here.
        """
        return

    def publisher_callback(self):
        """
        Should run indefinitely based on the specified interval.
        """
        # self.waypoint_parser(self.waypoints)
        self.path_pub.publish(self.path_msg)
        self.speed_pub.publish(self.speed_msg)
        self.marker_pub.publish(self.marker_array_msg)

    def parse_path(self, index):
        current_time = self.get_clock().now().to_msg()
        frame_id = self.frame_ids[index]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = frame_id

        pose_msg.pose.position.x = self.xs[index]
        pose_msg.pose.position.y = self.ys[index]
        pose_msg.pose.position.z = self.zs[index]
        pose_msg.pose.orientation.x = self.qxs[index]
        pose_msg.pose.orientation.y = self.qys[index]
        pose_msg.pose.orientation.z = self.qzs[index]
        pose_msg.pose.orientation.w = self.qws[index]

        self.path_msg.header.stamp = current_time
        self.path_msg.header.frame_id = frame_id
        self.path_msg.poses.append(pose_msg)

    def parse_speed(self, index):
        """
        todo: switch to custom message instead
        :param index:
        :return:
        """
        self.speed_msg.data.append(float(self.vxs[index]))

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
        marker.header.stamp = self.get_clock().now().to_msg()  # waypoint[1]
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
            print("Unsupported marker type. Defaulting to orientation")
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

        # Set text
        marker.text = f"{self.vxs[index]}"

        # to automatically delete old markers
        marker.lifetime = Duration(seconds=0).to_msg()  # 0 means forever.

        self.marker_array_msg.markers.append(marker)


def main(args=None):
    rclpy.init(args=args)
    wl_node = WaypointLoaderNode()
    rclpy.spin(wl_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    wl_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
