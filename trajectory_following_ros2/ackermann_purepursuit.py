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
    Use path message instead of loading from CSV
    Add a flag for stamped or unstamped message
    Add derivative in PI controller
    Make speed command optional
    Move PID speed controller to separate node
    Add adaptive lookahead
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
import numpy as np
import os
import sys
import csv
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
from std_msgs.msg import Float32, Float64
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped, TwistStamped, AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped


class PurePursuitNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for PurePursuitNode"""
        super(PurePursuitNode, self).__init__('purepursuit')
        # declare parameters
        self.declare_parameter('file_path', "/f1tenth_ws/data/test_waypoints.csv")
        self.declare_parameter('path_source', 'file')  # from csv file (file) or ros path message (path)
        self.declare_parameter('csv_columns', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))  # what columns to load if loading from a CSV file
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('goal_tolerance', 0.5)
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
        self.declare_parameter('odom_topic', '/vehicle/odometry/filtered')
        self.declare_parameter('acceleration_topic', '/vehicle/accel/filtered')
        self.declare_parameter('ackermann_cmd_topic', '/drive')

        '''Speed stuff'''
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('speed_Kp', 1.0)
        self.declare_parameter('speed_Ki', 0.1)
        self.declare_parameter('speed_Kd', 0.0)
        self.declare_parameter('desired_speed', 0.7)

        # get parameters
        self.file_path = self.get_parameter('file_path').value
        self.path_source = self.get_parameter('path_source').value
        self.csv_columns = self.get_parameter('csv_columns').value

        self.control_rate = self.get_parameter('control_rate').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
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

        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value


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
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.acceleration = 0.0
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        self.location = [self.x, self.y]
        self.alpha = 0.0  # yaw error
        self.desired_curvature = 0.0
        self.desired_steering_angle = 0.0
        self.desired_yaw_rate = 0.0

        # Get waypoints
        if self.path_source == 'file':
            # todo: rename self.waypoints to self.path
            self.frame_id_list, self.waypoints, self.des_yaw_list, self.speeds = self.get_waypoints_from_csv(
                    self.file_path, columns=self.csv_columns)
            self.dist_arr = np.zeros(len(self.waypoints))

        # setup waypoint variables
        self.stop_flag = False
        self.last_idx = None  # self.waypoints.shape[0] - 1
        self.final_idx = len(self.waypoints) - 1
        self.final_goal = self.waypoints[-1, :]
        self.desired_speed = self.target_speed = self.get_parameter('desired_speed').value  # self.speeds[self.target_ind] todo: get from speeds above or some other message
        self.goal = self.target_point = [0., 0.]

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_frame_id = ''  # the frame ID of the pose state received. odom, map

        # Setup subscribers
        self.odom_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_callback,
                1)  # to get the current state of the robot
        #self.pose_sub = self.create_subscription(
        #        PoseWithCovarianceStamped,
        #        self.pose_topic,
        #        self.pose_callback,
        #        1)
        self.accel_sub = self.create_subscription(
                AccelWithCovarianceStamped,
                self.acceleration_topic,
                self.acceleration_callback,
                1)

        # Setup publishers. Todo: publish purepursuit path
        self.ackermann_cmd_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_cmd_topic, 1)
        self.steer_pub = self.create_publisher(Float32, '/purepursuit/des_steer', 1)
        self.speed_pub = self.create_publisher(Float32, '/purepursuit/des_speed', 1)
        self.yaw_rate_pub = self.create_publisher(Float32, '/purepursuit/des_yaw_rate', 1)

        self.purepursuit_goal_pub = self.create_publisher(PointStamped, '/purepursuit/goal_point', 1)
        self.purepursuit_lookahead_pub = self.create_publisher(PointStamped, '/purepursuit/lookahead_point', 1)
        self.purepursuit_path_pub = self.create_publisher(Path, '/purepursuit/path', 1)

        # publish the internally estimated state of the car
        self.purepursuit_odom_pub = self.create_publisher(Odometry, '/purepursuit/current_state', 1)
        self.purepursuit_pose_pub = self.create_publisher(PoseStamped, '/purepursuit/current_pose', 1)

        # setup purepursuit timer
        self.purepursuit_timer = self.create_timer(self.sample_time, self.purepursuit_callback)
        self.debug_timer = self.create_timer(self.sample_time, self.publish_debug_topics)

        self.get_logger().info('purepursuit node started. ')

        if self.initial_pose_received:
            self.get_target_point()

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
        self.initial_pose_received = True
        self.state_frame_id = data.header.frame_id
        # msg_time = data.header.stamp
        # time_delta = self.get_clock().now() - rclpy.time.Time.from_msg(msg_time)
        # delta_in_seconds = time_delta.nanoseconds
        #print(f"Odom callback time delta: {delta_in_seconds}")
        pose = data.pose.pose
        twist = data.twist.twist

        position = pose.position
        orientation = pose.orientation

        # todo: transform pose to waypoint frame or transform waypoints to odom frame

        # get current pose
        self.x, self.y = position.x, position.y
        self.location = [self.x, self.y]
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        _, _, self.yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
        self.yaw_rate = twist.angular.z

        # get current speed
        self.vx = twist.linear.x
        self.vy = twist.linear.y
        self.vz = twist.linear.z
        self.omega = twist.angular.z

        #self.speed = np.linalg.norm([self.vx, self.vy, self.vz])  # todo: multiply by direction
        self.speed = twist.linear.x

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
        self.get_target_point()
        self.stop_flag = self.stop(self.target_ind, self.final_idx, self.location, self.final_goal, self.goal_tolerance)
        if self.initial_pose_received and not self.stop_flag:
            self.get_desired_steering_angle()
            # self.speed_error = self.desired_speed - self.speed
            #print(f"Speed (before) error: {self.speed_error}, desired: {self.desired_speed}, actual: {self.speed}")
            target_speed = self.speeds[self.last_idx]
            target_speed = (target_speed / abs(target_speed)) * self.desired_speed  # todo: catch ZeroDivisionError
            speed_cmd = self.set_speed(target_speed, self.speed)  #self.set_speed(self.desired_speed, self.speed)
            #print(f"Speed (after) error: {self.speed_error}, desired: {self.desired_speed}, actual: {self.speed}")
            #print(f"Desired steering angle: {self.desired_steering_angle}")
            if not (self.MAX_SPEED >= speed_cmd >= -self.MAX_SPEED):
                print(f"Desired speed of {speed_cmd} is higher than limit set. Saturating...")
            speed_cmd = np.clip(speed_cmd, -self.MAX_SPEED, self.MAX_SPEED)
            print(f"Sending speed cmd of {speed_cmd}")
            self.publish_command(self.desired_steering_angle, speed_cmd)

    def get_desired_steering_angle(self):
        """
        Todo: saturate actuation commands
        """
        '''Calculate the yaw error'''
        # self.alpha = math.atan2(ty - self.rear_y, tx - self.rear_x) - self.yaw
        # v1 = target_point - [self.x, self.y]
        # v2 = [np.cos(self.yaw), np.sin(self.yaw)]
        # alpha = find_angle(v1, v2)
        self.alpha = math.atan2(self.target_point[1] - self.y, self.target_point[0] - self.x) - self.yaw

        ''' (Optional) Get cross-track error: 
        the lateral distance between the heading vector and the goal point in body frame'''
        crosstrack_error = math.sin(self.alpha) * self.lookahead
        # crosstrack_error = self.target_point[1] - self.y  # todo: calculate in body frame

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
            print(f"Desired steering of {desired_steering_angle} is higher than limit set. Saturating...")

        self.desired_steering_angle = np.clip(desired_steering_angle, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE)

        '''Get the yaw rate'''
        self.desired_yaw_rate = (2.0 * self.desired_speed * math.sin(self.alpha)) / self.lookahead
        return self.desired_steering_angle

    def update_states(self, acceleration, steering_angle):
        """
        Vehicle kinematic motion model (in the global frame), here we are using simple bicycle model
        :param acceleration: float, acceleration
        :param steering_angle: float, heading control.

        Todo: either use this or from Odom callback maybe by checking the time of the last odom message.
        """
        self.x += self.speed * math.cos(self.yaw) * self.dt
        self.y += self.speed * math.sin(self.yaw) * self.dt
        self.yaw += self.speed * math.tan(steering_angle) / self.WHEELBASE * self.dt
        self.speed += acceleration * self.dt
        # if x, and y are relative to CoG
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        self.location = [self.x, self.y]

        # Publish vehicle kinematic state
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
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

    def get_target_point(self):
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
        if (self.last_idx is None) and self.speedup_first_lookup:
            distances = self.get_distance(self.location, self.waypoints)
            self.last_idx = np.argmin(distances)
        else:
            self.dist_arr = self.get_distance(self.location, self.waypoints)
            # Use points greater than lookahead as heuristics
            goal_candidates = np.where((self.dist_arr > self.lookahead))[0]
            if len(goal_candidates) > 0:
                indices_greater_than_last = np.where(goal_candidates >= self.last_idx)[0]

                if len(indices_greater_than_last) > 0:
                    goal_candidates = goal_candidates[indices_greater_than_last]
                    self.last_idx = np.min(goal_candidates)

        self.target_ind = self.last_idx
        self.target_point = self.get_point(self.target_ind)

    def get_point(self, index):
        point = self.waypoints[index]
        return point

    def stop(self, current_idx, final_idx, current_location, final_waypoint, tolerance):
        stop = False
        distance = self.get_distance(current_location, final_waypoint)
        if current_idx >= final_idx and distance <= tolerance:
            stop = True
            print("Stop conditions met.")
        return stop

    def set_speed(self, target, measurement):
        # todo: tune PID gains
        # use PID here
        self.speed_error = target - measurement

        self.speed_Pterm = self.speed_Kp * self.speed_error
        self.speed_Iterm += self.speed_error * self.speed_Ki * self.dt

        if self.speed_last_error is not None:
            self.speed_Dterm = (self.speed_error - self.speed_last_error)/self.dt*self.speed_Kd

        self.speed_last_error = self.speed_error

        #speed_cmd = self.speed_Pterm + self.speed_Iterm + self.speed_Dterm
        #speed_cmd = self.target_speed
        speed_cmd = target
        #  speed_cmd = self.speeds[self.target_ind]
        #print(f"Desired speed: {target}, current_speed: {measurement}, speed cmd: {speed_cmd}, Speed error: {self.speed_error}, P term: {self.speed_Pterm}, I term: {self.speed_Iterm}, D term: {self.speed_Dterm}")

        return speed_cmd

    def publish_command(self, steering_angle, longitudinal_velocity):
        # Publish Ackermann command
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = 'base_link'  # self.frame_id
        ackermann_cmd.drive.steering_angle = steering_angle
        ackermann_cmd.drive.speed = longitudinal_velocity
        self.ackermann_cmd_pub.publish(ackermann_cmd)

        steer_msg = Float32()
        steer_msg.data = float(steering_angle)
        self.steer_pub.publish(steer_msg)
        self.speed_pub.publish(Float32(data=float(longitudinal_velocity)))
        self.yaw_rate_pub.publish(Float32(data=float(self.yaw_rate)))

    def publish_debug_topics(self):
        """
        Publish markers and path.
        Todo: could transform poses into other frames
        """
        timestamp = self.get_clock().now().to_msg()
        frame_id = 'base_link'

        # publish the position of the current lookahead
        point_msg = PointStamped()
        point_msg.header.stamp = timestamp
        point_msg.header.frame_id = 'base_link'
        point_msg.point.x = self.lookahead
        self.purepursuit_lookahead_pub.publish(point_msg)

        # publish the position of the current goal point
        point_msg = PointStamped()
        point_msg.header.stamp = timestamp
        point_msg.header.frame_id = 'odom'
        point_msg.point.x, point_msg.point.y = self.target_point
        self.purepursuit_goal_pub.publish(point_msg)

        # Publish a few points in the path left
        path_msg = Path()
        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = "odom"

        start_idx = self.last_idx
        end_idx = self.last_idx + 30
        end_idx = min(end_idx, len(self.waypoints))
        for index in range(start_idx, end_idx):
            waypoint_pose = PoseStamped()
            waypoint_pose.header.stamp = timestamp
            waypoint_pose.header.frame_id = 'odom'
            waypoint_pose.pose.position.x, waypoint_pose.pose.position.y = self.waypoints[index, :]
            waypoint_pose.pose.orientation.z = self.des_yaw_list[index]
            path_msg.poses.append(waypoint_pose)
        self.purepursuit_path_pub.publish(path_msg)

    def get_waypoints_from_csv(self, path_to_csv, columns=None):
        """
        frame_id, timestamp, x, y, z, yaw angle, qx, qy, qz, qw, vx, vy, speed, omega
        frame_id_list, self.waypoints, des_yaw_list, speed
        """

        # to import as a struct with column indexing
        csv_array = np.genfromtxt(path_to_csv, delimiter=',', skip_header=1, usecols=columns, dtype=float)[35:468, :]
        # csv_struct = np.genfromtxt(path_to_csv, delimiter=',', names=True, usecols=columns, dtype=float)

        frame_ids = ['odom'] * len(csv_array)  # todo: get from CSV or don't hardcode
        waypoints_array = csv_array[:, 3:5]
        des_yaw_list = csv_array[:, 6]
        des_speed_list = csv_array[:, 11]
        return frame_ids, waypoints_array, des_yaw_list, des_speed_list


def main(args=None):
    rclpy.init(args=args)
    pp_node = PurePursuitNode()
    rclpy.spin(pp_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    pp_node.waypoint_file.close()
    pp_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
