"""
Path and trajectory following kinematic model.

Todo:
    * don't hardcode model into do-mpc, instead use vectors
    * setup do-mpc to return the mpc object only.
    * setup do-mpc update
    * setup do-mpc warmstart
    * setup parameter type
    * setup point stabilization, i.e go to goal from RVIz pose
    * change model input to velocity instead of acceleration or use a model
    * speed up do-mpc
    * load compiled model
"""
import math
import time
import os
import sys
import csv
import threading
import numpy as np
import scipy

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
from std_msgs.msg import Float32, Float64
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped, TwistStamped, AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

# Custom messages
from trajectory_following_ros2.do_mpc.model import BicycleKinematicModel
from trajectory_following_ros2.simulator.do_mpc.do_mpc_simulator import Simulator


class KinematicDoMPCSimulationNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for KinematicCoupledMPCNode"""
        super(KinematicDoMPCSimulationNode, self).__init__('kinematic_do_mpc_simulator')
        # declare parameters
        self.declare_parameter('model_type', "continuous")  # todo: also augmented, etc
        self.declare_parameter('csv_columns', (0, 2, 3, 5, 10))  # what columns to load if loading from a CSV file
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('min_steer', -30.0,
                               ParameterDescriptor(
                                       description='The minimum lateral command (steering) to apply.'))  # in degrees
        self.declare_parameter('max_steer', 30.0)  # in degrees

        # MPC Parameters
        self.declare_parameter('max_steer_rate', 60 / 0.17)  # in degrees
        self.declare_parameter('max_speed', 100.5)  # in m/s
        self.declare_parameter('min_speed', -100.5)  # in m/s
        self.declare_parameter('max_accel', 30.0)  # in m/s^2
        self.declare_parameter('max_decel', -30.0)  # in m/s^2, i.e min_accel
        self.declare_parameter('saturate_input',
                               True)  # whether or not to saturate the input sent to the car. MPC will still use the constraints even if this is false.
        self.declare_parameter('n_states', 4)  # todo: remove and get from import model
        self.declare_parameter('n_inputs', 2)  # todo: remove and get from import model

        self.declare_parameter('odom_topic', '/vehicle/odometry/filtered')
        self.declare_parameter('acceleration_topic', '/vehicle/accel/filtered')
        self.declare_parameter('ackermann_cmd_topic', '/drive')
        self.declare_parameter('debug', False)  # displays solver output

        # get parameters
        self.update_rate = self.get_parameter('update_rate').value

        self.WHEELBASE = self.get_parameter('wheelbase').value  # [m] wheelbase of the vehicle
        self.MAX_STEER_ANGLE = math.radians(self.get_parameter('max_steer').value)
        self.MIN_STEER_ANGLE = math.radians(self.get_parameter('min_steer').value)
        self.MAX_STEER_RATE = math.radians(self.get_parameter('max_steer_rate').value)
        self.MAX_SPEED = self.get_parameter('max_speed').value
        self.MIN_SPEED = self.get_parameter('min_speed').value
        self.MAX_ACCEL = self.get_parameter('max_accel').value
        self.MAX_DECEL = self.get_parameter('max_decel').value
        self.saturate_input = self.get_parameter('saturate_input').value

        self.NX = self.get_parameter('n_states').value
        self.NU = self.get_parameter('n_inputs').value

        # initialize variables
        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value

        self.debug = self.get_parameter('debug').value

        self.dt = self.sample_time = 1 / self.update_rate  # [s] sample time. # todo: get from ros Duration
        self.msg_time_prev = 0.0  # time when image is received
        self.msg_time = 0.0

        # Setup mode
        self.acc_cmd, self.delta_cmd, self.velocity_cmd = 0.0, 0.0, 0.0
        self.zk = np.zeros((self.NX, 1))  # x, y, psi, velocity, psi
        self.uk = np.array([[self.acc_cmd, self.delta_cmd]]).T
        self.u_prev = np.zeros((self.NU, 1))

        # State of the vehicle.
        self.initial_pose_received = False
        self.initial_accel_received = False
        self.x = 0.0  # todo: remove
        self.y = 0.0  # todo: remove
        self.yaw = 0.0  # todo: remove
        self.speed = 0.0  # todo: remove
        self.yaw_rate = 0.0  # todo: remove
        self.acceleration = 0.0  # todo: remove
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))  # todo: remove
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))  # todo: remove
        self.location = [self.x, self.y]  # todo: remove

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_frame_id = ''  # the frame ID of the pose state received. odom, map

        # Initialize Simulator
        self.solution_time = 0.0
        self.run_count = 0
        self.vehicle_model = BicycleKinematicModel(length=self.WHEELBASE, width=0.192, sample_time=self.sample_time)
        self.vehicle_model.wp_id = 0  # done because of model tvp. Todo: remove
        self.vehicle_model.reference_path = [[0., 0., 0., 0.]]  # done because of model tvp. Todo: remove
        self.simulator = Simulator(self.vehicle_model, sample_time=self.sample_time)
        self.simulator = self.simulator.simulator
        self.simulator.x0 = self.zk

        # Setup subscribers (ackermann cmd or steering+accel topics)
        self.ackermann_cmd_sub = self.create_subscription(AckermannDriveStamped,
                                                          self.ackermann_cmd_topic,
                                                          self.ackermann_cmd_callback, 1)

        # Setup publishers (odometry and acceleration).
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 1)
        self.accel_pub = self.create_publisher(AccelWithCovarianceStamped, self.acceleration_topic, 1)

        # setup purepursuit timer
        self.simulation_timer = self.create_timer(self.sample_time, self.update_timer_callback)

        self.get_logger().info('kinematic_do_mpc_simulator node started. ')

    def ackermann_cmd_callback(self, data):
        self.u_prev[:, :] = self.uk
        self.msg_time_prev = self.msg_time
        frame_id = data.header.frame_id
        self.msg_time = data.header.stamp

        # msg_dt = (self.msg_time - self.msg_time_prev).nanoseconds

        self.delta_cmd = data.drive.steering_angle
        self.velocity_cmd = data.drive.speed
        if data.drive.acceleration != 0.0:
            self.acc_cmd = data.drive.acceleration
        else:
            self.acc_cmd = (self.velocity_cmd - self.speed) / self.sample_time  # i.e dv/dt
        self.uk[0, 0] = self.acc_cmd
        self.uk[1, 0] = self.delta_cmd

    def update_timer_callback(self):
        # Acts as zero order hold if this callback is faster than input callback rate
        uk = self.uk.copy()
        self.zk[:, :] = self.simulator.make_step(uk)
        self.x = self.zk[0, 0]  # self.x += self.vel * math.cos(self.yaw) * self.dt
        self.y = self.zk[1, 0]  # self.x += self.vel * math.sin(self.yaw) * self.dt
        self.speed = self.zk[2, 0]  # self.speed += acc * dt
        self.yaw = self.zk[3, 0]  # self.yaw += self.speed * math.tan(uk[1, 0]) / self.WHEELBASE * dt

        self.yaw_rate = (self.speed / self.WHEELBASE) * math.tan(uk[1, 0])
        self.acceleration = self.speed / self.sample_time

        self.publish_odometry()
        self.publish_acceleration()

        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))

    def publish_odometry(self):
        """
        Todo: transform coordinates
        :param data:
        :return:
        """
        odom_msg = Odometry()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.header.stamp = self.get_clock().now().to_msg()

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        odom_quaternion = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        odom_msg.pose.pose.orientation.x = odom_quaternion[0]
        odom_msg.pose.pose.orientation.y = odom_quaternion[1]
        odom_msg.pose.pose.orientation.z = odom_quaternion[2]
        odom_msg.pose.pose.orientation.w = odom_quaternion[3]

        odom_msg.twist.twist.linear.x = self.speed
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0

        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = self.yaw_rate  # todo: test

        self.odom_pub.publish(odom_msg)

    def publish_acceleration(self):
        accel_msg = AccelWithCovarianceStamped()
        accel_msg.header.frame_id = 'base_link'

        accel_msg.accel.accel.linear.x = self.acceleration
        accel_msg.accel.accel.angular.z = self.yaw_rate / self.sample_time
        # accel_msg.accel.covariance = [0., 0., 0., 0., 0., 0.]


def main(args=None):
    rclpy.init(args=args)
    kdsim_node = KinematicDoMPCSimulationNode()
    rclpy.spin(kdsim_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    kdsim_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
