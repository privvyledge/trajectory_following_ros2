"""
Path and trajectory following kinematic model.

Todo:
    * publish odom transform [done]
    * add global and robot frames
    * add clock publisher using python time, e.g see Carla ros-bridge bridge.py
    * improve the accuracy of the simulator
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
from tf2_ros import TransformBroadcaster
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
from trajectory_following_ros2.simulator.acados.acados_simulator import Simulator


class KinematicAcadosMPCSimulationNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for KinematicCoupledMPCNode"""
        super(KinematicAcadosMPCSimulationNode, self).__init__('kinematic_acados_simulator')
        # declare parameters
        # todo: transform from/to the global and robot frames
        self.declare_parameter('robot_frame', 'base_link',
                               ParameterDescriptor(description='The frame attached to the car. '
                                                               'The relative/local frame. '
                                                               'Usually the ground projection of the center '
                                                               'of the rear axle of a car or the center of gravity '
                                                               'of a differential robot. '
                                                               'Actuation commands, speed and acceleration are '
                                                               'relative to this frame.'
                                                               'E.g base_link, base_footprint, '
                                                               'ego_vehicle (Carla). '
                                                               'Obstacle positions could be specified '
                                                               'relative to this frame or the global frame.'))
        self.declare_parameter('global_frame', 'odom',
                               ParameterDescriptor(description='The global/world/map frame. '
                                                               'This frame is static and ideally its origin should '
                                                               'not change during the lifetime of motion. '
                                                               'Position errors are usually calculated relative '
                                                               'to this frame, e.g X, Y, Psi '
                                                               'for MPC, purepursuit, etc. '
                                                               'Target/goal positions are also specified here.'
                                                               'Usually the ground projection of the center '
                                                               'of the rear axle of a car or the center of gravity '
                                                               'of a differential robot. '
                                                               'Obstacle positions could be specified '
                                                               'relative to this frame or the robots frame.'
                                                               'E.g odom, map. '
                                                               'ROS2 Nav2 local costmaps are usually '
                                                               'in this frame, i.e "odom", '
                                                               'odometry messages are also in the "odom" frame '
                                                               'whereas waypoints and goal poses are usually '
                                                               'specified in the "map" or "gnss" frame so it makes '
                                                               'sense to transform the goal points (waypoints) to the '
                                                               '"odom" frame. '
                                                               'Since Carla does not use the odom frame, '
                                                               'set to "map", '
                                                               'otherwise use "odom".'))  # global frame: odom, map
        self.declare_parameter('model_type', "continuous")  # todo: also augmented, etc
        # self.declare_parameter('csv_columns', (0, 2, 3, 5, 10))  # what columns to load if loading from a CSV file
        self.declare_parameter('update_rate', 50.0)
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

        # todo: could get from rviz or a topic
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('initial_speed', 0.0)
        self.declare_parameter('initial_yaw_rate', 0.0)
        self.declare_parameter('initial_acceleration', 0.0)

        # get parameters
        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
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

        # State of the vehicle.
        self.initial_pose_received = False
        self.initial_accel_received = False
        self.x = self.get_parameter('initial_x').value
        self.y = self.get_parameter('initial_y').value
        self.yaw = self.get_parameter('initial_yaw').value
        self.speed = self.get_parameter('initial_speed').value
        self.yaw_rate = self.get_parameter('initial_yaw_rate').value
        self.acceleration = self.get_parameter('initial_yaw_rate').value
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        self.location = [self.x, self.y]  # or self.rear_x, self.rear_y

        # Setup mode
        self.acc_cmd, self.delta_cmd, self.velocity_cmd = 0.0, 0.0, 0.0
        self.zk = np.zeros((self.NX, 1))  # x, y, psi, velocity, psi
        self.zk[0, 0] = self.x
        self.zk[1, 0] = self.y
        self.zk[2, 0] = self.speed
        self.zk[3, 0] = self.yaw
        self.uk = np.array([[self.acc_cmd, self.delta_cmd]]).T
        self.u_prev = np.zeros((self.NU, 1))

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_frame_id = ''  # the frame ID of the pose state received. odom, map

        # Setup transform publisher, e.g odom to base_link
        self.br = TransformBroadcaster(self)

        # Initialize Simulator
        self.solution_time = 0.0
        self.run_count = 0
        # self.vehicle_model = BicycleKinematicModel(length=self.WHEELBASE, width=0.192, sample_time=self.sample_time)

        self.simulator = Simulator(vehicle=None, sample_time=self.sample_time, generate=True, build=True,
                                   with_cython=True, integrator_config_file="kinematic_bicycle_acados_integrator.json",
                                   code_export_directory="c_generated_code_integrator")
        self.simulator = self.simulator.acados_integrator
        self.simulator.set("u", self.uk.flatten())
        self.simulator.set("x", self.zk.flatten())
        self.sim_status = self.simulator.solve()

        # Setup subscribers (ackermann cmd or steering+accel topics)
        self.ackermann_cmd_sub = self.create_subscription(AckermannDriveStamped,
                                                          self.ackermann_cmd_topic,
                                                          self.ackermann_cmd_callback, 1)

        # Setup publishers (odometry and acceleration).
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 1)
        self.accel_pub = self.create_publisher(AccelWithCovarianceStamped, self.acceleration_topic, 1)

        # setup simulation callback timer
        self.simulation_timer = self.create_timer(self.sample_time, self.update_timer_callback)

        self.get_logger().info('kinematic_acados_simulator node started. ')

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
        xk = self.zk.copy()

        nx = xk.shape[0]  # todo: remove
        nu = uk.shape[0]  # todo: remove

        self.simulator.set("x", xk.flatten())
        self.simulator.set("u", uk.flatten())
        self.simulator.set("p", np.array([
            0.256,  # wheelbase
            *np.zeros(nx),  # zref
            *np.zeros(nu),  # uref
            *xk.flatten(),  # zk
            *self.u_prev.flatten(),  # u_prev
        ]))  # optional
        self.sim_status = self.simulator.solve()
        self.zk[:, :] = self.simulator.get("x").reshape(-1, 1)

        self.x = self.zk[0, 0]  # self.x += self.vel * math.cos(self.yaw) * self.dt
        self.y = self.zk[1, 0]  # self.x += self.vel * math.sin(self.yaw) * self.dt
        self.speed = self.zk[2, 0]  # self.speed += acc * dt
        self.yaw = self.zk[3, 0]  # self.yaw += self.speed * math.tan(uk[1, 0]) / self.WHEELBASE * dt

        self.yaw_rate = (self.speed / self.WHEELBASE) * math.tan(uk[1, 0])
        self.acceleration = self.speed / self.sample_time

        self.publish_transform()
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
        odom_msg.header.frame_id = self.global_frame
        odom_msg.child_frame_id = self.robot_frame
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
        odom_msg.twist.twist.angular.z = self.yaw_rate

        self.odom_pub.publish(odom_msg)

    def publish_acceleration(self):
        accel_msg = AccelWithCovarianceStamped()
        accel_msg.header.frame_id = self.robot_frame

        accel_msg.accel.accel.linear.x = self.acceleration
        accel_msg.accel.accel.angular.z = self.yaw_rate / self.sample_time
        # accel_msg.accel.covariance = [0., 0., 0., 0., 0., 0.]
        self.accel_pub.publish(accel_msg)

    def publish_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.global_frame
        t.child_frame_id = self.robot_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        tf_quaternion = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        t.transform.rotation.x = tf_quaternion[0]
        t.transform.rotation.y = tf_quaternion[1]
        t.transform.rotation.z = tf_quaternion[2]
        t.transform.rotation.w = tf_quaternion[3]
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    kasim_node = KinematicAcadosMPCSimulationNode()
    rclpy.spin(kasim_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    kasim_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
