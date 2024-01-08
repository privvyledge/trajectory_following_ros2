import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# libraries
import numpy as np
import time
from threading import Lock

# TF
from tf2_ros import TransformBroadcaster
import tf_transformations

# messages
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String, Header, Float32MultiArray, Float64
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist


class Twist2Ackermann(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for ClassName"""
        super(Twist2Ackermann, self).__init__('twist_to_ackermann')

        # declare parameters
        self.declare_parameter('wheelbase', 0.256, ParameterDescriptor(description='The wheelbase of the car.'))
        self.declare_parameter('twist_topic', '/cmd_vel')
        self.declare_parameter('ackermann_cmd_topic', '/drive')
        self.declare_parameter('frame_id', 'base_link',
                               ParameterDescriptor(description='The commands reference frame.'))
        self.declare_parameter('cmd_angle_instead_rotvel', False)

        # get parameters
        self.WHEELBASE = self.get_parameter('wheelbase').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.cmd_angle_instead_rotvel = self.get_parameter('cmd_angle_instead_rotvel').value

        # initialize variables

        # Setup subscribers
        self.twist_sub = self.create_subscription(
                Twist,
                self.twist_topic,
                self.twist_callback,
                1)
        # self.twist_sub  # prevent unused variable warning

        # Setup publishers
        self.ackermann_cmd_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_cmd_topic, 1)
        timer_period = 0.05
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('twist_to_ackermann node started. ')

    def timer_callback(self):
        # used for testing. todo: remove
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = self.frame_id
        ackermann_cmd.drive.steering_angle = 0.2  # steering_angle
        ackermann_cmd.drive.speed = 0.0  # longitudinal_velocity
        # self.get_logger().info(f'Speed: {0.0}, angle: {0.2}. ')

        # self.ackermann_cmd_pub.publish(ackermann_cmd)

    def twist_callback(self, msg):
        # todo: add saturation
        longitudinal_velocity = msg.linear.x  # msg.twist.linear.x for TwistStamped

        # if cmd_angle_instead_rotvel is true, the rotational velocity is already converted in the C++ node to angle.
        # in this case this script only needs to do the msg conversion from twist to Ackermann drive
        if self.cmd_angle_instead_rotvel:
            steering_angle = msg.angular.z
        else:
            yaw_rate = msg.angular.z
            steering_angle = self.yaw_rate_to_steering_angle(longitudinal_velocity, yaw_rate)

        ######### Begin saturation
        if longitudinal_velocity > 3.0:
            self.get_logger().info('Saturating speed. ')
            longitudinal_velocity = 3.0
        if longitudinal_velocity < -3.0:
            self.get_logger().info('Saturating speed. ')
            longitudinal_velocity = -3.0

        if steering_angle > 0.4:
            self.get_logger().info('Saturating steering_angle. ')
            steering_angle = 0.4
        if steering_angle < -0.4:
            self.get_logger().info('Saturating steering_angle. ')
            steering_angle = -0.4
        ######### End saturation

        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = self.frame_id
        ackermann_cmd.drive.steering_angle = steering_angle
        ackermann_cmd.drive.speed = longitudinal_velocity

        self.ackermann_cmd_pub.publish(ackermann_cmd)

    def yaw_rate_to_steering_angle(self, longitudinal_speed, desired_yaw_rate):
        if desired_yaw_rate == 0 or longitudinal_speed == 0:
            return 0.

        radius = longitudinal_speed / desired_yaw_rate
        steering_angle = math.atan2(self.WHEELBASE, radius)  # math.atan(self.WHEELBASE / radius)
        return steering_angle


def main(args=None):
    rclpy.init(args=args)
    t2a_node = Twist2Ackermann()
    rclpy.spin(t2a_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    # t2a_node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
