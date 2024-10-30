"""
Sources:
    https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/pure_pursuit/pure_pursuit.py
    https://github.com/thomasfermi/Algorithms-for-Automated-Driving/blob/master/code/solutions/control/pure_pursuit.py

    ROS RPP: https://github.com/ros-planning/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/src/regulated_pure_pursuit_controller.cpp

PP Pipeline:
    Get waypoints from CSV (or Path message from ROS topic)
    Find the path point closest to the vehicle
    Find the goal/target point
    Transform the goal point to vehicle coordinates
    Get desired speed (optional)
    Calculate the curvature and request the vehicle to set the steering to that curvature
    Update the vehicles position (no need cause localization will handle it)
    Saturate steering
    Publish steering

Todo:
    Use path message instead of loading from CSV [done]
    Add derivative in PI controller [done]
    Increase carla simulation sample size and time
    Test live with F1/10
    Add timeout to queue get and put blocking methods
    Update purepursuit class with the latest changes
    Switch to Purepursuit class
    Switch to PID class
    Move PID speed controller to separate node
    Test debug and purepursuit queues and speedup
    setup Queues for actuation commands, odom, accel, etc or thread locks
    Test Queues with smaller timeouts
    Add a flag for stamped or unstamped message
    Make speed command optional
    Rename topics to purepursuit namespace
    Test adaptive lookahead
    Implement transformations
    Implement collision avoidance
    Implement regulated pp
    Add costmap and other ROS stuff
    Subscribe to path topic instead of fetching from a CSV
    Test updating states and setting faster rates
    Create goal and progress checker nodes that take into account goal tolerance (optionally lookahead distance) and speed tolerance to update the next point.(This is because get_target_point can get stuck if the desired velocity is 0 and the next waypoint is farther than the lookahead distance)
    Move purepursuit methods to a separate class
    Optimize
    Test adaptive lookahead
"""
import math
import time
import numpy as np
import os
import sys
import csv
import queue
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.executors import ExternalShutdownException

# TF
from tf2_ros import TransformException, LookupException
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf_transformations
import tf2_geometry_msgs

# messages
from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32, Float64, Float32MultiArray
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped, TwistStamped, AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

# Custom messages.
from trajectory_following_ros2.utils.Trajectory import Trajectory
import trajectory_following_ros2.utils.filters as filters
import trajectory_following_ros2.utils.trajectory_utils as trajectory_utils


class PurePursuitNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for PurePursuitNode"""
        super(PurePursuitNode, self).__init__('purepursuit')
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
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('speed_tolerance', 0.5)  # todo: might need to refactor
        self.declare_parameter('lookahead_distance', 0.4)  # (0.4) [m] lookahead distance
        self.declare_parameter('min_lookahead', 0.3)
        self.declare_parameter('max_lookahead', 0.452 + 10.0)
        self.declare_parameter('adaptive_lookahead_gain', 0.4)
        self.declare_parameter('use_adaptive_lookahead', False)
        self.declare_parameter('speedup_first_lookup', True)
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('min_steer', -23.0,
                               ParameterDescriptor(description='The minimum lateral command (steering) to apply.'))  # in degrees
        self.declare_parameter('max_steer', 23.0)  # in degrees
        self.declare_parameter('path_topic', '/trajectory/path')
        self.declare_parameter('speed_topic', '/trajectory/speed')
        self.declare_parameter('odom_topic', '/vehicle/odometry/filtered')
        self.declare_parameter('acceleration_topic', '/vehicle/accel/filtered')
        self.declare_parameter('ackermann_cmd_topic', '/drive')
        self.declare_parameter('publish_twist_topic', True)
        self.declare_parameter('twist_topic', '/cmd_vel')

        '''Speed stuff'''
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('speed_Kp', 2.0)  # uses PID if P gain > 0
        self.declare_parameter('speed_Ki', 0.2)
        self.declare_parameter('speed_Kd', 0.0)
        self.declare_parameter('desired_speed', 0.0)

        # get parameter values
        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.control_rate = self.get_parameter('control_rate').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.speed_tolerance = self.get_parameter('speed_tolerance').value
        self.desired_lookahead = self.get_parameter('lookahead_distance').value
        self.MIN_LOOKAHEAD = self.get_parameter('min_lookahead').value
        self.MAX_LOOKAHEAD = self.get_parameter('max_lookahead').value
        self.speedup_first_lookup = self.get_parameter('speedup_first_lookup').value

        self.WHEELBASE = self.get_parameter('wheelbase').value  # [m] wheelbase of the vehicle
        self.MAX_STEER_ANGLE = math.radians(self.get_parameter('max_steer').value)
        self.MIN_STEER_ANGLE = math.radians(self.get_parameter('min_steer').value)
        self.MAX_SPEED = self.get_parameter('max_speed').value

        # initialize variables
        self.use_adaptive_lookahead = self.get_parameter('use_adaptive_lookahead').value
        self.lookahead_gain = self.get_parameter('max_lookahead').value  # look forward gain

        self.path_topic = self.get_parameter('path_topic').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value
        self.publish_twist_topic = self.get_parameter('publish_twist_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value

        if not self.use_adaptive_lookahead:
            self.lookahead_gain = 0.0
        self.lookahead = self.desired_lookahead

        self.speed_Kp = self.get_parameter('speed_Kp').value  # speed proportional gain
        self.speed_Ki = self.get_parameter('speed_Ki').value  # speed integral gain
        self.speed_Kd = self.get_parameter('speed_Kd').value  # speed derivative gain
        self.speed_Pterm = 0.0
        self.speed_Iterm = 0.0
        self.speed_Dterm = 0.0  # derivative term
        self.speed_last_error = None
        self.speed_error = 0.0

        self.dt = self.sample_time = 1 / self.control_rate  # [s] sample time. # todo: get from ros Duration

        # State of the vehicle.
        self.initial_pose_received = False
        self.initial_accel_received = False
        self.path_received = False
        self.desired_speed_received = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.direction = 1.0
        self.yaw_rate = 0.0
        self.acceleration = 0.0
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        self.location = [self.x, self.y]
        self.cumulative_distance = 0.0
        self.alpha = 0.0  # yaw error
        self.desired_curvature = 0.0
        self.desired_steering_angle = 0.0
        self.desired_yaw_rate = 0.0

        # setup waypoint variables
        self.stop_flag = False
        self.desired_speed = self.target_speed = self.get_parameter(
            'desired_speed').value  # self.speeds[self.target_ind] todo: get from speeds above or some other message
        self.final_idx = None
        self.final_goal = None
        self.target_ind = 0  # current_idx. 0 or None
        self.goal = None  #  self.target_point
        self.dist_arr = None

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_frame_id = ''  # the frame ID of the pose state received. odom, map. todo: transform <-> self.global frame

        # initialize trajectory class
        self.trajectory = Trajectory(search_index_number=10,
                                     goal_tolerance=self.goal_tolerance,
                                     stop_speed=self.speed_tolerance
                                     )
        self.trajectory_initialized = False

        self.waypoints = None
        self.des_yaw_list = None
        self.speeds = None
        self.reference_path = None  # todo: remove

        # Setup queues for multi-thread access
        self.mutex = threading.Lock()  # or threading.RLock()
        queue_size = 5
        self.location_queue = queue.Queue(maxsize=queue_size)
        self.orientation_queue = queue.Queue(maxsize=queue_size)
        self.yaw_queue = queue.Queue(maxsize=queue_size)
        self.goal_queue = queue.Queue(maxsize=queue_size)

        # todo: remove
        self.purepursuit_dt = None
        self.debug_dt = None

        # ReentrantCallbackGroup() or MutuallyExclusiveCallbackGroup()
        self.subscription_group = ReentrantCallbackGroup()
        self.purepursuit_group = ReentrantCallbackGroup()
        self.debug_group = ReentrantCallbackGroup()  # can use either

        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # Setup subscribers
        self.path_sub = self.create_subscription(
                Path,
                self.path_topic,
                self.path_callback,
                qos_profile=latching_qos,
                # callback_group=self.subscription_group
        )  # to get the desired/reference states of the robot
        self.speed_sub = self.create_subscription(
                Float32MultiArray,
                self.speed_topic,
                self.desired_speed_callback,
                qos_profile=latching_qos,
                # callback_group=self.subscription_group
        )  # to get the desired/reference states of the robot
        self.odom_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_callback,
                1,
                callback_group=self.subscription_group
        )  # to get the current state of the robot
        #self.pose_sub = self.create_subscription(
        #        PoseWithCovarianceStamped,
        #        self.pose_topic,
        #        self.pose_callback,
        #        1,
        #        callback_group=self.subscription_group)
        self.accel_sub = self.create_subscription(
                AccelWithCovarianceStamped,
                self.acceleration_topic,
                self.acceleration_callback,
                1,
                callback_group=self.subscription_group
        )

        # Setup publishers. Todo: publish purepursuit path
        self.ackermann_cmd_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_cmd_topic, 1)
        self.steer_pub = self.create_publisher(Float32, '/purepursuit/des_steer', 1)
        self.speed_pub = self.create_publisher(Float32, '/purepursuit/des_speed', 1)
        self.yaw_rate_pub = self.create_publisher(Float32, '/purepursuit/des_yaw_rate', 1)
        if self.publish_twist_topic:
            self.twist_cmd_pub = self.create_publisher(TwistStamped, self.twist_topic, 1)

        self.purepursuit_goal_pub = self.create_publisher(PointStamped, '/purepursuit/goal_point', 1)
        self.purepursuit_lookahead_pub = self.create_publisher(PointStamped, '/purepursuit/lookahead_point', 1)
        self.purepursuit_path_pub = self.create_publisher(Path, '/purepursuit/path', 1)

        # publish the internally estimated state of the car
        self.purepursuit_odom_pub = self.create_publisher(Odometry, '/purepursuit/current_state', 1)
        self.purepursuit_pose_pub = self.create_publisher(PoseStamped, '/purepursuit/current_pose', 1)

        # setup purepursuit timer
        self.purepursuit_timer = self.create_timer(self.sample_time, self.purepursuit_callback,
                                                   callback_group=self.purepursuit_group)
        self.debug_timer = self.create_timer(0.25, self.publish_debug_topics,
                                             callback_group=self.debug_group)  # todo: setup separate publishing dt

        self.get_logger().info('purepursuit node started. ')

        # if self.initial_pose_received:
        #     self.get_target_point()

    def update_queue(self, data_queue, data, overwrite_if_full=True):
        # todo: move to utils function to avoid repeating in different nodes
        if data_queue.full():
            if overwrite_if_full:
                # acquire lock
                if data_queue.full():
                    drop = data_queue.get(block=True)  # remove one item
                else:
                    data_queue.put(data, block=True)
                # release the lock
        else:
            data_queue.put(data, block=True)

    def path_callback(self, data):
        """
        Todo: add flag to remove duplicates, smooth path, speed, yaw, etc.
        :param data:
        :return:
        """
        path_frame_id = data.header.frame_id
        path_time = data.header.stamp

        frame_id_list = []
        coordinate_list = []
        yaw_list = []
        for pose_msg in data.poses:
            node_time = pose_msg.header.stamp
            node_frame_id = pose_msg.header.frame_id
            frame_id_list.append(node_frame_id)

            node_position = pose_msg.pose.position
            node_x = node_position.x
            node_y = node_position.y
            node_z = node_position.z
            coordinate_list.append([node_x, node_y])

            node_orientation_quaternion = pose_msg.pose.orientation
            node_qx = node_orientation_quaternion.x
            node_qy = node_orientation_quaternion.y
            node_qz = node_orientation_quaternion.z
            node_qw = node_orientation_quaternion.w

            _, _, node_yaw = tf_transformations.euler_from_quaternion([node_qx, node_qy, node_qz, node_qw])
            yaw_list.append(node_yaw)

        self.waypoints = np.array(coordinate_list)
        self.des_yaw_list = np.array(yaw_list)
        self.frame_id_list = frame_id_list
        self.path_received = True
        self.final_idx = self.waypoints.shape[0] - 1
        self.final_goal = self.waypoints[self.final_idx, :]  # self.waypoints[-1, :]

        self.goal = self.waypoints[self.target_ind, :]  #  self.target_point
        self.update_queue(self.goal_queue, self.goal)
        self.dist_arr = np.zeros(len(self.waypoints))  # array of distances used to check closes path

        self.trajectory.current_index = self.target_ind
        self.trajectory.trajectory = np.zeros(
                (self.waypoints.shape[0], len(self.trajectory.trajectory_keys)))
        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['x']] = self.waypoints[:, 0]
        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['y']] = self.waypoints[:, 1]

        self.trajectory.trajectory[:,
        self.trajectory.trajectory_key_to_column['yaw']] = self.des_yaw_list

        # self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['dt']] = csv_df.dt.values
        # self.trajectory.trajectory[:,
        # self.trajectory.trajectory_key_to_column['total_time_elapsed']] = csv_df.total_time_elapsed.values

        cdists = trajectory_utils.cumulative_distance_along_path(
                self.trajectory.trajectory[:, [self.trajectory.trajectory_key_to_column['x'],
                                               self.trajectory.trajectory_key_to_column['y']]])
        curvs_all = trajectory_utils.calculate_curvature_all(cdists,
                                                             self.trajectory.trajectory[:,
                                                             self.trajectory.trajectory_key_to_column['yaw']],
                                                             smooth=True)

        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['curvature']] = curvs_all
        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['cum_dist']] = cdists

    def desired_speed_callback(self, data):
        """
        Todo: add flag to remove duplicates, smooth path, speed, yaw, etc.
        :param data:
        :return:
        """
        speed_list = []
        for speed in data.data:
            speed_list.append(speed)
        self.speeds = np.array(speed_list)

        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['speed']] = self.speeds
        # self.trajectory.trajectory[:,
        # self.trajectory.trajectory_key_to_column['omega']] = csv_df.omega.values  # todo

        self.desired_speed_received = True

    def get_distance(self, location, waypoint):
        """
        Compute the Euclidean distance between an array of point(s) (waypoints) and the current location
        then return the lowest.
        """
        distance = np.linalg.norm(location - waypoint, axis=-1)
        # if len(waypoint.shape) > 1:
        #     distance = np.sqrt((waypoint[:, 0] - location[0]) ** 2 + (waypoint[:, 1] - location[1]) ** 2)
        # else:
        #     distance = np.sqrt((waypoint[0] - location[0]) ** 2 + (waypoint[1] - location[1]) ** 2)
        return distance

    def find_angle(self, v1, v2):
        """
        find the angle between two vectors. Used to get alpha
        """
        # method 1
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        angle = np.arctan2(sinang, cosang)

        '''
        # method 2. Same as above
        v1_u = v1 / np.linalg.norm(v1)  # find the unit vector
        v2_u = v1 / np.linalg.norm(v1)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        '''
        return angle

    def odom_callback(self, data):
        self.state_frame_id = data.header.frame_id
        # todo: transform to the desired global frame. self.global_frame

        # msg_time = data.header.stamp
        # time_delta = self.get_clock().now() - rclpy.time.Time.from_msg(msg_time)
        # delta_in_seconds = time_delta.nanoseconds
        #self.get_logger().info(f"Odom callback time delta: {delta_in_seconds}")
        pose = data.pose.pose
        twist = data.twist.twist

        position = pose.position
        orientation = pose.orientation

        # todo: transform pose to waypoint frame or transform waypoints to odom frame

        # get current pose
        self.x, self.y = position.x, position.y
        self.location = [self.x, self.y]
        self.update_queue(self.location_queue, self.location)
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        _, _, self.yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
        self.yaw_rate = twist.angular.z

        # get current speed
        self.vx = twist.linear.x
        self.vy = twist.linear.y
        self.vz = twist.linear.z
        self.omega = twist.angular.z

        try:
            self.direction = self.vx / abs(self.vx)
        except ZeroDivisionError:
            self.direction = 1.0

        # self.speed = self.direction * np.linalg.norm([self.vx, self.vy, self.vz])
        self.speed = self.vx

        if not self.initial_pose_received:
            self.initial_pose_received = True

    def acceleration_callback(self, data):
        self.initial_accel_received = True
        accel_frame_id = data.header.frame_id

        linear_acceleration = data.accel.accel.linear
        angular_acceleration = data.accel.accel.angular
        acceleration_covariance = data.accel.covariance

        longitudinal_acc, lateral_acc = linear_acceleration.x, linear_acceleration.y
        roll_acceleration, pitch_acceleration, yaw_acceleration = angular_acceleration.x, \
                                                                  angular_acceleration.y, \
                                                                  angular_acceleration.z
        _, _, yaw_rate = roll_acceleration * self.dt, pitch_acceleration * self.dt, yaw_acceleration * self.dt
        self.acceleration = longitudinal_acc

    def purepursuit_callback(self):
        # self.get_logger().info(f"##### Entered purepursuit callback")
        # start_time = time.time()
        # current_time = self.get_clock().now()
        # if self.purepursuit_dt is not None:
        #     time_delta = current_time - self.purepursuit_dt
        #     time_delta = time_delta.nanoseconds / 1e+9
        #     self.get_logger().info(f"##### Purepursuit Time delta: {time_delta}")
        #
        # self.purepursuit_dt = current_time

        if self.path_received and self.desired_speed_received and not self.trajectory_initialized:
            # todo: account for no speed topic but desired speed != 0
            self.reference_path = np.array(
                    [self.waypoints[:, 0], self.waypoints[:, 1], self.speeds, self.des_yaw_list]).T
            self.trajectory_initialized = True

        if self.initial_pose_received and self.path_received:
            self.stop_flag = self.trajectory.check_goal(self.x, self.y, self.speed,
                                                        self.final_goal, self.target_ind, self.final_idx)

            # self.stop_flag = self.stop(self.target_ind, self.final_idx,
            #                            self.location, self.final_goal,
            #                            self.goal_tolerance)

            if not self.stop_flag:
                ref_traj = self.get_target_point()  # self.get_target_point_old()
                if ref_traj is None:
                    self.get_logger().info("Reached end of trajectory.")
                    self.publish_command(0.0, 0.0)
                    return

                self.get_desired_steering_angle()
                # self.speed_error = self.desired_speed - self.speed
                #self.get_logger().info(f"Speed (before) error: {self.speed_error}, desired: {self.desired_speed}, actual: {self.speed}")
                target_speed = 0.0
                if self.desired_speed_received and self.desired_speed == 0.0:
                    # get the speed from the CSV/topic
                    target_speed = self.speeds[self.target_ind]
                else:
                    # overwrite the speed from the CSV file or topic with the desired speed
                    target_speed = self.desired_speed  # todo: add direction from velocity profile

                speed_cmd = target_speed
                if abs(self.speed_Kp) > 0.0:
                    speed_cmd = self.set_speed(target_speed, self.speed)

                #self.get_logger().info(f"Speed (after) error: {self.speed_error}, desired: {self.desired_speed}, actual: {self.speed}")
                #self.get_logger().info(f"Desired steering angle: {self.desired_steering_angle}")
                if not (self.MAX_SPEED >= speed_cmd >= -self.MAX_SPEED):
                    # self.get_logger().info(f"Desired speed of {speed_cmd} is higher than limit set. Saturating...")
                    pass
                speed_cmd = np.clip(speed_cmd, -self.MAX_SPEED, self.MAX_SPEED)
                # self.get_logger().info(f"Sending cmd speed: {speed_cmd}, steering: {self.desired_steering_angle}, idx: {self.target_ind}")
                self.publish_command(self.desired_steering_angle, speed_cmd)

            else:
                self.get_logger().info("Stop conditions met.")
                self.publish_command(0.0, 0.0)

        # self.get_logger().info(f"##### Left purepursuit callback after {abs(time.time() - start_time)} seconds")

    def get_desired_steering_angle(self):
        """
        """
        '''Calculate the yaw error'''
        # self.alpha = math.atan2(ty - self.rear_y, tx - self.rear_x) - self.yaw
        # v1 = target_point - [self.x, self.y]
        # v2 = [np.cos(self.yaw), np.sin(self.yaw)]
        # alpha = find_angle(v1, v2)
        self.alpha = math.atan2(self.goal[1] - self.y, self.goal[0] - self.x) - self.yaw

        ''' (Optional) Get cross-track error: 
        the lateral distance between the heading vector and the goal point in body frame'''
        crosstrack_error = math.sin(self.alpha) * self.lookahead
        # crosstrack_error = self.goal[1] - self.y  # todo: calculate in body frame

        '''(Optional) Calculate turn radius, r = 1 / curvature'''
        turn_radius = self.lookahead ** 2 / (2 * abs(crosstrack_error))

        '''Get the curvature'''
        self.desired_curvature = (2.0 * math.sin(self.alpha)) / self.lookahead
        # self.desired_curvature = (2.0 / (self.lookahead ** 2)) * crosstrack_error
        # self.desired_curvature = 1 / turn_radius
        # self.desired_curvature = 2.0 * abs(crosstrack_error) / self.lookahead ** 2

        '''Get the steering angle'''
        # desired_steering_angle = math.atan(self.desired_curvature * self.WHEELBASE)
        desired_steering_angle = math.atan2(2.0 * self.WHEELBASE * math.sin(self.alpha), self.lookahead)
        # desired_steering_angle = math.atan2(2.0 * abs(crosstrack_error) * self.WHEELBASE, self.lookahead ** 2)
        # desired_steering_angle = math.atan2(2.0 * abs(crosstrack_error), self.lookahead ** 2)  # Might be wrong

        if not (self.MAX_STEER_ANGLE >= desired_steering_angle >= self.MIN_STEER_ANGLE):
            # self.get_logger().info(f"Desired steering of {desired_steering_angle} is higher than limit set. Saturating...")
            pass

        self.desired_steering_angle = np.clip(desired_steering_angle, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE)

        '''Get the yaw rate'''
        self.desired_yaw_rate = (2.0 * self.desired_speed * math.sin(self.alpha)) / self.lookahead
        return self.desired_steering_angle

    def update_states(self, acceleration, steering_angle):
        """
        Vehicle kinematic motion model (in the global frame), here we are using simple bicycle model
        :param acceleration: float, acceleration
        :param steering_angle: float, heading control.

        Todo: either use this or from Odom callback maybe by checking the time of the last odom message. Should be used for prediction
        """
        #self.x += self.speed * math.cos(self.yaw) * self.dt
        #self.y += self.speed * math.sin(self.yaw) * self.dt
        #self.yaw += self.speed * math.tan(steering_angle) / self.WHEELBASE * self.dt
        #self.speed += acceleration * self.dt
        ## if x, and y are relative to CoG
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        self.location = [self.x, self.y]
        # self.update_queue(self.location_queue, self.location)

        # Publish vehicle kinematic state
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.global_frame
        odom.child_frame_id = self.robot_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        odom_quaternion = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation.x = odom_quaternion[0]
        odom.pose.pose.orientation.y = odom_quaternion[1]
        odom.pose.pose.orientation.z = odom_quaternion[2]
        odom.pose.pose.orientation.w = odom_quaternion[3]

        odom.twist.twist.linear.x = self.speed
        odom.twist.twist.angular.z = self.yaw_rate
        self.purepursuit_odom_pub.publish(odom)
        # Todo: publish the internally estimated state of the car (self.purepursuit_pose_pub)

    def update_lookahead(self):
        """Update the lookahead distance based on speed"""
        if self.use_adaptive_lookahead:
            lookahead = self.lookahead_gain * self.speed + self.desired_lookahead
            self.lookahead = np.clip(lookahead, self.MIN_LOOKAHEAD, self.MAX_LOOKAHEAD)
            return

    def get_target_point_old(self):
        """
        Get the next look ahead point
        :param pos: list, vehicle position
        :return: list, target point

        Todo: use a better algorithm to find the target point, e.g line-circle intersection
         (https://github.com/thomasfermi/Algorithms-for-Automated-Driving/blob/master/code/solutions/control/get_target_point.py)
         (https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit)
        """
        self.update_lookahead()
        # todo: get acceleration from Kalman Filter node
        self.update_states(acceleration=self.acceleration, steering_angle=self.desired_steering_angle)

        # To speed up nearest point search, doing it at only first time.
        if (self.target_ind == 0) and self.speedup_first_lookup:
            distances = self.get_distance(self.location, self.waypoints)
            self.target_ind = np.argmin(distances)
            self.speedup_first_lookup = False  # so this condition does not repeat
        else:
            self.dist_arr = self.get_distance(self.location, self.waypoints)
            # Use points greater than lookahead as heuristics
            goal_candidates = np.where((self.dist_arr > self.lookahead))[0]
            if len(goal_candidates) > 0:
                indices_greater_than_last = np.where(goal_candidates >= self.target_ind)[0]

                if len(indices_greater_than_last) > 0:
                    goal_candidates = goal_candidates[indices_greater_than_last]
                    self.target_ind = np.min(goal_candidates)

        self.goal = self.get_point(self.target_ind)
        self.update_queue(self.goal_queue, self.goal)
        #self.get_logger().info(f"Get target point callback... Target_ind: {self.target_ind}, goal point: {self.goal}")

    def get_target_point(self):
        # update trajectory state and reference
        # get the robots current state, waypoint index and normalize the angle from -pi to pi or [0, 2PI). todo: update state in purepursuit and MPC node
        self.trajectory.current_index = self.target_ind
        self.trajectory.state[0, self.trajectory.state_key_to_column['x']] = self.x
        self.trajectory.state[0, self.trajectory.state_key_to_column['y']] = self.y
        self.trajectory.state[0, self.trajectory.state_key_to_column['speed']] = self.speed
        self.trajectory.state[0, self.trajectory.state_key_to_column['yaw']] = self.yaw
        self.trajectory.state[0, self.trajectory.state_key_to_column['omega']] = self.omega
        self.trajectory.state[
            0, self.trajectory.state_key_to_column['curvature']] = trajectory_utils.calculate_curvature_single(
                self.trajectory.trajectory[:,
                [self.trajectory.trajectory_key_to_column['x'],
                 self.trajectory.trajectory_key_to_column['y']]],
                goal_index=self.target_ind)
        # self.trajectory.state[0, self.trajectory.state_key_to_column['cum_dist']] = self.cumulative_distance

        self.update_lookahead()
        # todo: get acceleration from Kalman Filter node
        # self.update_states(acceleration=self.acceleration, steering_angle=self.desired_steering_angle)

        self.target_ind, ref_traj, _, _, _ = self.trajectory.calc_ref_trajectory(
            state=None, trajectory=None, current_index=None,
            dt=0.02, prediction_horizon=50,
            lookahead_time=1.0, lookahead=self.lookahead,
            num_points_to_interpolate=50)
        self.goal = self.get_point(self.target_ind)
        self.update_queue(self.goal_queue, self.goal)

        return ref_traj

    def get_point(self, index):
        point = self.waypoints[index]
        return point

    def stop(self, current_idx, final_idx, current_location, final_waypoint, tolerance):
        stop = False
        distance = self.get_distance(current_location, final_waypoint)
        if current_idx >= final_idx and distance <= tolerance:
            stop = True
            self.get_logger().info("Stop conditions met.")
        return stop

    def set_speed(self, target, measurement):
        # use PID here
        self.speed_error = target - measurement

        self.speed_Pterm = self.speed_Kp * self.speed_error
        self.speed_Iterm += self.speed_error * self.speed_Ki * self.dt
        # todo: clip integral to avoid integral windup

        if self.speed_last_error is not None:
            self.speed_Dterm = self.speed_Kd * (self.speed_error - self.speed_last_error) / self.dt

        self.speed_last_error = self.speed_error

        speed_cmd = self.speed_Pterm + self.speed_Iterm + self.speed_Dterm
        # todo: clip total and add ramping
        # self.get_logger().info(f"Desired speed: {target}, "
        #                        f"current_speed: {measurement}, "
        #                        f"speed cmd: {speed_cmd}, "
        #                        f"Speed error: {self.speed_error}, "
        #                        f"P term: {self.speed_Pterm}, I term: {self.speed_Iterm}, D term: {self.speed_Dterm}")

        return speed_cmd

    def publish_command(self, steering_angle, longitudinal_velocity):
        # Publish Ackermann command
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = self.robot_frame  # self.frame_id
        ackermann_cmd.drive.steering_angle = steering_angle
        ackermann_cmd.drive.speed = longitudinal_velocity
        self.ackermann_cmd_pub.publish(ackermann_cmd)

        steer_msg = Float32()
        steer_msg.data = float(steering_angle)
        self.steer_pub.publish(steer_msg)
        self.speed_pub.publish(Float32(data=float(longitudinal_velocity)))
        self.yaw_rate_pub.publish(Float32(data=float(self.yaw_rate)))

    def publish_twist(self, lateral_velocity=0.0):
        # # todo: also add an option to publish Twist() message and get bool parameter to select type
        # twist_stamped_cmd = TwistStamped()
        # twist_stamped_cmd.header.stamp = self.get_clock().now().to_msg()
        # twist_stamped_cmd.header.frame_id = self.robot_frame  # self.frame_id
        # twist_stamped_cmd.twist.linear.x = self.velocity_cmd
        # twist_stamped_cmd.twist.linear.y = lateral_velocity
        # twist_stamped_cmd.twist.angular.z = (self.velocity_cmd / self.WHEELBASE) * math.tan(self.delta_cmd)
        # self.twist_cmd_pub.publish(twist_stamped_cmd)
        return

    def publish_debug_topics(self):
        """
        Publish markers and path.
        Todo: could transform poses into other frames
        """
        timestamp = self.get_clock().now().to_msg()
        frame_id = self.robot_frame

        # self.get_logger().info(f"##### Entered debug callback")
        # start_time = time.time()
        # current_time = self.get_clock().now()
        # if self.debug_dt is not None:
        #     time_delta = current_time - self.debug_dt
        #     time_delta = time_delta.nanoseconds / 1e+9
        #     self.get_logger().info(f"##### debug Time delta: {time_delta}")
        #
        # self.debug_dt = current_time

        if self.initial_pose_received and self.path_received:
            goal = None
            location = None
            # get items from queues
            if not self.goal_queue.empty():
                goal = self.goal_queue.get(block=True)

            if not self.location_queue.empty():
                location = self.location_queue.get(block=True)

            # publish the position of the current lookahead.
            # todo: either publish this directly to the robots frame, i.e base link or add to the robots global pose and publish in the global frame.

            if location is not None:
                # todo: refactor
                point_msg = PointStamped()
                point_msg.header.stamp = timestamp
                point_msg.header.frame_id = self.robot_frame  # self.robot_frame or self.global_frame. Todo: publishing to global frame does not account for orientation, transform first, else use base_link
                point_msg.point.x = self.lookahead  # self.x  # self.lookahead, self.lookahead + self.x, location[0] + self.lookahead
                point_msg.point.y = 0.0  # self.y  # 0.0 or self.y, location[1]
                self.purepursuit_lookahead_pub.publish(point_msg)

            if goal is not None:
                # publish the position of the current goal point
                point_msg = PointStamped()
                point_msg.header.stamp = timestamp
                point_msg.header.frame_id = self.global_frame
                point_msg.point.x, point_msg.point.y = goal  # self.goal
                self.purepursuit_goal_pub.publish(point_msg)

            # Publish a few points in the path left
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = self.global_frame

            start_idx = self.target_ind
            end_idx = self.target_ind + 30
            end_idx = min(end_idx, len(self.waypoints))
            for index in range(start_idx, end_idx):
                waypoint_pose = PoseStamped()
                waypoint_pose.header.stamp = timestamp
                waypoint_pose.header.frame_id = self.global_frame
                waypoint_pose.pose.position.x, waypoint_pose.pose.position.y = self.waypoints[index, :]
                waypoint_pose.pose.orientation.z = self.des_yaw_list[index]
                path_msg.poses.append(waypoint_pose)
            self.purepursuit_path_pub.publish(path_msg)

        # self.get_logger().info(f"##### Left debug callback after {abs(time.time() - start_time)} seconds")


def main(args=None):
    rclpy.init(args=args)
    try:
        pp_node = PurePursuitNode()
        executor = MultiThreadedExecutor(num_threads=4)  # todo: could get as an argument or initialize in the class to get as a parameter
        executor.add_node(pp_node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            #pp_node.waypoint_file.close()   # for waypoint node
            pp_node.destroy_node()

    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)

    finally:
        rclpy.shutdown()

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    # pp_node.waypoint_file.close()
    # pp_node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
