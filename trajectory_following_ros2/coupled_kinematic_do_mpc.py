"""
Path and trajectory following kinematic model.

Tips:
    1. ROS simple commander (https://navigation.ros.org/commander_api/index.html | https://github.com/ros-planning/navigation2/tree/humble/nav2_simple_commander)
    2. Goal action example (https://github.com/ros-planning/navigation2/blob/humble/nav2_msgs/action/FollowPath.action)
    3. Example smoothing (https://github.com/ros-planning/navigation2/blob/humble/nav2_msgs/action/SmoothPath.action)
    4. ROS2 Nav2 MPC Example (https://github.com/ros-planning/navigation2/tree/humble/nav2_mppi_controller)
        i. Path utilities: https://github.com/ros-planning/navigation2/blob/humble/nav2_mppi_controller/src/path_handler.cpp
        ii. Controller: https://github.com/ros-planning/navigation2/blob/humble/nav2_mppi_controller/src/controller.cpp
    5. ROS2 Nav2 Controller plugin explanation (https://navigation.ros.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html)
Todo:
    * don't hardcode model into do-mpc, instead use vectors
    * setup do-mpc to return the mpc object only.
    * setup do-mpc update
    * setup parameter types (https://github.com/ros2/rclpy/blob/5285e61f02ab87f8b84f11ee33e608ff118a8f06/rclpy/test/test_parameter.py). Types (https://roboticsbackend.com/ros2-rclpy-parameter-callback/#Check_the_parameters_type)
        * See saturate_inputs parameter below
    * setup point stabilization, i.e go to goal from RVIz pose
    * set default reference speed to 0 and set velocity cost weight to close to 0 [done]
    * Fix/detect time jump, e.g when using ROSBags (https://github.com/ros2/geometry2/issues/606#issuecomment-1589927587 | https://github.com/ros2/geometry2/pull/608#discussion_r1229877511)
    * change model input to velocity instead of acceleration or use a model
    * speed up do-mpc
    * install HSL MA27 solver
    * load compiled model
    * get goal/reference states using ROS actions
    * update goals using ROS actions (goal checker, etc should be done via actions). See (2) above
    * use ROS services or action servers to smooth stuff. See (3) above
    * get global frame from costmap and transform to odom frame in odom callback. Also transform goal/path frame to common frame
        i. Get costmap: global (map frame), local (odom frame). Either is good but local is recommended since it also includes dynamic obstacles
        ii. Get path and path frame_id: e.g map frame (whatever the paths global frame id is)
        iii. Get robots states and frame id from odom callback: pose (odom), speed (base_link)
        iv. Convert the robots pose to costmap frame
        v. Transform paths (reference) to costmap frame_id. Speed should already be in the robots frame_id
        vi. Perform goal checking in costmap frame (could also be done in robots local frame 'base_link')
        vii. Run MPC (or any controller) in its desired frame
    * draw the path taken by the robot in RViz
    * switch stop condition to use ROS Goal checker or create mine (https://navigation.ros.org/configuration/packages/nav2_controller-plugins/simple_goal_checker.html | https://navigation.ros.org/plugins/index.html#waypoint-task-executors)
    * Switch to custom message or actions to specify Path + speeds
    * Move main MPC code to separate package and as a submodule to this ROS repo instead
    * Wait for messages (e.g path, speed, states) before calling or use actions
"""
import math
import time
import os
import sys
import csv
import queue
import threading
import numpy as np
import scipy
from scipy import interpolate

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.executors import ExternalShutdownException

# TF
from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf_transformations
import tf2_geometry_msgs

# messages
from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32, Float64, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped, TwistStamped, AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

# Custom messages. todo: probably remove Vehicle and trajectory stuff but import casadi model instead
from trajectory_following_ros2.utils.Trajectory import Trajectory
import trajectory_following_ros2.utils.filters as filters
import trajectory_following_ros2.utils.trajectory_utils as trajectory_utils
from trajectory_following_ros2.do_mpc.mpc_setup import initialize_mpc_problem


class KinematicCoupledDoMPCNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for KinematicCoupledDoMPCNode"""
        super(KinematicCoupledDoMPCNode, self).__init__('kinematic_coupled_do_mpc_controller')
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
        self.declare_parameter('ode_type', "continuous_kinematic_coupled")  # todo: also augmented, etc
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('debug_frequency', 4.0)
        self.declare_parameter('distance_tolerance', 0.2)  # todo: move to goal checker node
        self.declare_parameter('speed_tolerance', 0.5)  # todo: move to goal checker node
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('min_steer', -27.0,
                               ParameterDescriptor(
                                       description='The minimum lateral command (steering) to apply.'))  # in degrees
        self.declare_parameter('max_steer', 27.0)  # in degrees

        # MPC Parameters
        self.declare_parameter('max_steer_rate', 60 / 0.17)  # in degrees
        self.declare_parameter('max_speed', 1.5)  # in m/s
        self.declare_parameter('min_speed', -1.5)  # in m/s
        self.declare_parameter('max_accel', 3.0)  # in m/s^2
        self.declare_parameter('max_decel', -3.0)  # in m/s^2, i.e min_accel
        self.declare_parameter('saturate_input',
                               True)  # whether or not to saturate the input sent to the car. MPC will still use the constraints even if this is false.
        self.declare_parameter('allow_reversing',
                               True)  # whether or not to allow reversing. Will set min_speed as 0 if False
        self.declare_parameter('n_states', 4)  # todo: remove and get from import model
        self.declare_parameter('n_inputs', 2)  # todo: remove and get from import model
        self.declare_parameter('horizon', 25)
        self.declare_parameter('prediction_time', 0.0)
        self.declare_parameter('max_iter', 15)  # it usually doesnt take more than 15 iterations
        self.declare_parameter('termination_condition',
                               1e-6)  # iteration finish param. todo: pass to mpc initializer solver options

        # tips for tuning weights (https://www.mathworks.com/help/mpc/ug/tuning-weights.html)
        self.declare_parameter('R', [0.01, 0.01])
        self.declare_parameter('Rd', [10., 100.])
        self.declare_parameter('Q', [1.0, 1.0, 1.0, 0.01])
        self.declare_parameter('Qf', [0.04, 0.04, 0.1, 0.01])

        self.declare_parameter('path_topic', '/trajectory/path')  # todo: replace with custom message or action
        self.declare_parameter('speed_topic', '/trajectory/speed')  # todo: replace with custom message or action
        self.declare_parameter('odom_topic', '/odometry/local')
        self.declare_parameter('acceleration_topic', '/accel/local')
        self.declare_parameter('ackermann_cmd_topic', '/drive')
        self.declare_parameter('publish_twist_topic', True)
        self.declare_parameter('twist_topic', '/cmd_vel')

        self.declare_parameter('desired_speed', 0.0)  # todo: get from topic/trajectory message
        self.declare_parameter('loop', False)  # todo: remove and pass to waypoint publisher
        self.declare_parameter('n_ind_search', 10)  # the number of points to check for when searching for the closest
        self.declare_parameter('smooth_yaw', False)
        self.declare_parameter('debug', False)  # displays solver output
        self.declare_parameter('generate_mpc_model', True)  # generate and build model
        self.declare_parameter('build_with_cython', True)  # whether to use cython (recommended) or ctypes. not used. todo: remove
        self.declare_parameter('model_directory',
                               '/f1tenth_ws/src/trajectory_following_ros2/data/model')  # don't use absolute paths

        # get parameter values
        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.control_rate = self.get_parameter('control_rate').value
        self.debug_frequency = self.get_parameter('debug_frequency').value
        self.distance_tolerance = self.get_parameter('distance_tolerance').value
        self.speed_tolerance = self.get_parameter('speed_tolerance').value

        self.WHEELBASE = self.get_parameter('wheelbase').value  # [m] wheelbase of the vehicle
        self.MAX_STEER_ANGLE = math.radians(self.get_parameter('max_steer').value)
        self.MIN_STEER_ANGLE = math.radians(self.get_parameter('min_steer').value)
        self.MAX_STEER_RATE = math.radians(self.get_parameter('max_steer_rate').value)
        self.MAX_SPEED = self.get_parameter('max_speed').value
        self.MIN_SPEED = self.get_parameter('min_speed').value
        self.MAX_ACCEL = self.get_parameter('max_accel').value
        self.MAX_DECEL = self.get_parameter('max_decel').value
        self.saturate_input = self.get_parameter('saturate_input').value
        self.allow_reversing = self.get_parameter('allow_reversing').value

        if not self.allow_reversing:
            self.MIN_SPEED = 0.0

        self.NX = self.get_parameter('n_states').value
        self.NU = self.get_parameter('n_inputs').value
        self.R = np.diag(self.get_parameter('R').value)
        self.Rd = np.diag(self.get_parameter('Rd').value)
        self.Q = np.diag(self.get_parameter('Q').value)
        self.Qf = self.get_parameter('Qf').value
        if self.Qf is not None:
            self.Qf = np.diag(self.get_parameter('Qf').value)
        else:
            self.Qf = self.Q
        self.horizon = int(self.get_parameter('horizon').value)
        self.prediction_time = self.get_parameter('prediction_time').value

        self.max_iter = int(self.get_parameter('max_iter').value)
        self.termination_condition = self.get_parameter('termination_condition').value

        # initialize variables
        self.ode_type = self.get_parameter('ode_type').value
        self.path_topic = self.get_parameter('path_topic').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value
        self.publish_twist_topic = self.get_parameter('publish_twist_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value

        self.desired_speed = self.get_parameter('desired_speed').value
        self.loop = self.get_parameter('loop').value
        self.n_ind_search = self.get_parameter('n_ind_search').value
        self.smooth_yaw = self.get_parameter('smooth_yaw').value
        self.debug = self.get_parameter('debug').value
        self.generate_mpc_model = self.get_parameter('generate_mpc_model').value

        self.model_type = 'continuous' if 'continuous' in self.ode_type else 'discrete'
        
        self.dt = self.sample_time = 1 / self.control_rate  # [s] sample time. # todo: get from ros Duration
        if (self.prediction_time is None) or (self.prediction_time == 0.0):
            self.prediction_time = self.sample_time * self.horizon

        # initialize trajectory class
        self.trajectory = Trajectory(search_index_number=10,
                                     goal_tolerance=self.distance_tolerance,
                                     stop_speed=self.speed_tolerance
                                     )

        self.acc_cmd, self.delta_cmd, self.velocity_cmd = 0.0, 0.0, 0.0
        self.zk = np.zeros((self.NX, 1))  # x, y, psi, velocity, psi
        self.uk = np.array([[self.acc_cmd, self.delta_cmd]]).T
        self.u_prev = np.zeros((self.NU, 1))

        self.xref = np.zeros((self.horizon + 1, self.NX))
        self.uref = np.zeros((self.horizon, self.NU))

        # State of the vehicle.
        self.initial_pose_received = False
        self.initial_accel_received = False
        self.path_received = False
        self.desired_speed_received = False
        self.mpc_built = False
        self.mpc_initialized = False
        self.trajectory_initialized = False
        self.x = 0.0  # todo: remove
        self.y = 0.0  # todo: remove
        self.yaw = 0.0  # todo: remove
        self.speed = 0.0  # todo: just use vx
        self.omega = 0.0  # todo: remove
        self.direction = 1.0
        self.yaw_rate = 0.0  # todo: remove
        self.acceleration = 0.0  # todo: remove
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))  # todo: remove
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))  # todo: remove
        self.location = [self.x, self.y]  # todo: remove
        self.cumulative_distance = 0.0
        self.alpha = 0.0  # yaw error  # todo: remove
        self.desired_curvature = 0.0  # todo: remove
        self.desired_steering_angle = 0.0  # todo: remove
        self.desired_yaw_rate = 0.0  # todo: remove

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_frame_id = ''  # the frame ID of the pose state received. odom, map

        self.path = None
        self.des_yaw_list = None
        self.speeds = None

        # setup waypoint variables. todo: make mpc not need the path to be initialized
        self.reference_path = None  # todo: remove
        self.current_idx = 0
        self.target_point = None
        self.final_idx = None
        self.final_goal = None
        self.final_goal_reached = False
        self.stop_flag = False

        # Initialize MPC
        self.mpc_reference_states = np.zeros((self.horizon + 1, self.NX))
        self.mpc_reference_inputs = np.zeros((self.horizon, self.NU))
        self.mpc_predicted_states = np.zeros((self.horizon + 1, self.NX))
        self.mpc_predicted_inputs = np.zeros((self.horizon, self.NU))
        self.solution_time = 0.0
        self.solver_iteration_count = 0
        self.run_count = 0
        self.solution_status = False
        self.Controller = None

        self.constraint = None
        self.model = None
        self.controller = None
        self.ocp = None

        # Setup queues for multi-thread access
        self.mutex = threading.Lock()  # or threading.RLock()
        queue_size = 5
        self.location_queue = queue.Queue(maxsize=queue_size)
        self.mpc_reference_states_queue = queue.Queue(maxsize=queue_size)
        self.mpc_predicted_states_queue = queue.Queue(maxsize=queue_size)
        self.goal_queue = queue.Queue(maxsize=queue_size)

        # ReentrantCallbackGroup() or MutuallyExclusiveCallbackGroup()
        self.subscription_group = ReentrantCallbackGroup()
        self.mpc_group = ReentrantCallbackGroup()
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
        # self.pose_sub = self.create_subscription(
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
                callback_group=self.subscription_group)

        # Setup publishers.
        self.ackermann_cmd_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_cmd_topic, 1)
        self.steer_pub = self.create_publisher(Float32, '/mpc/des_steer', 1)
        self.speed_pub = self.create_publisher(Float32, '/mpc/des_speed', 1)
        self.yaw_rate_pub = self.create_publisher(Float32, '/mpc/des_yaw_rate', 1)
        if self.publish_twist_topic:
            self.twist_cmd_pub = self.create_publisher(TwistStamped, self.twist_topic, 1)

        self.mpc_goal_pub = self.create_publisher(PointStamped, '/mpc/goal_point', 1)
        self.waypoint_path_pub = self.create_publisher(Path, '/mpc/path_remaining', 1)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc/predicted_path', 1)
        self.mpc_reference_path_pub = self.create_publisher(Path, '/mpc/reference_path', 1)

        # publish the internally estimated state of the car
        self.mpc_odom_pub = self.create_publisher(Odometry, '/mpc/current_state', 1)
        self.mpc_pose_pub = self.create_publisher(PoseStamped, '/mpc/current_pose', 1)

        # setup mpc timer.
        self.mpc_timer = self.create_timer(self.sample_time, self.mpc_callback,
                                           callback_group=self.mpc_group)
        if self.debug_frequency > 0:
            self.debug_timer = self.create_timer(1 / self.debug_frequency, self.publish_debug_topics,
                                                 callback_group=self.debug_group)

        self.setup_mpc_model()
        self.get_logger().info('kinematic_coupled_do_mpc_controller node started. ')

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

        self.path = np.array(coordinate_list)
        self.des_yaw_list = np.array(yaw_list)
        self.frame_id_list = frame_id_list
        self.path_received = True
        self.final_idx = self.path.shape[0] - 1
        self.final_goal = self.path[self.final_idx, :]

        self.target_point = self.path[self.current_idx, :]
        self.update_queue(self.goal_queue, self.target_point[0:2])

        self.trajectory.trajectory = np.zeros(
                (self.path.shape[0], len(self.trajectory.trajectory_keys)))
        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['x']] = self.path[:, 0]
        self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['y']] = self.path[:, 1]

        self.trajectory.trajectory[:,
        self.trajectory.trajectory_key_to_column['yaw']] = self.des_yaw_list

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

    def setup_mpc_model(self):
        _, self.Controller, _ = initialize_mpc_problem(reference_path=self.reference_path,
                                                       horizon=self.horizon,
                                                       sample_time=self.sample_time,
                                                       Q=self.Q, R=self.R,
                                                       Qf=self.Qf, Rd=self.Rd,
                                                       wheelbase=self.WHEELBASE,
                                                       delta_min=np.degrees(self.MIN_STEER_ANGLE),
                                                       delta_max=np.degrees(self.MAX_STEER_ANGLE),
                                                       vel_min=self.MIN_SPEED,
                                                       vel_max=self.MAX_SPEED,
                                                       ay_max=4.0, acc_min=self.MAX_DECEL,
                                                       acc_max=self.MAX_ACCEL,
                                                       max_iterations=self.max_iter,
                                                       tolerance=self.termination_condition,
                                                       suppress_ipopt_output=True, model_type=self.model_type,
                                                       warmstart=True, quad_prog_mode=True,
                                                       compile_model=self.generate_mpc_model, jit_compilation=False)
        self.mpc_built = True
        self.get_logger().info("Finished setting up the MPC.")

    def initialize_mpc_solver(self):
        # warmstart
        self.Controller.mpc.x0 = self.zk
        self.Controller.mpc.set_initial_guess()
        # self.Controller.get_control(self.zk)  # todo: test

        self.mpc_initialized = True
        self.get_logger().info("######################## MPC initialized ##################################")

    def odom_callback(self, data):
        """
        Todo: transform coordinates
        todo: add flag to convert to and from center of mass to rear axle
        :param data:
        :return:
        """
        # update location
        self.state_frame_id = data.header.frame_id
        child_frame_id = data.child_frame_id
        pose = data.pose.pose
        twist = data.twist.twist

        position = pose.position
        orientation = pose.orientation

        # todo: transform pose to waypoint frame or transform waypoints to odom frame

        # get current pose
        self.x, self.y = position.x, position.y

        # self.location = [self.x, self.y]
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        _, _, self.yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
        self.yaw_rate = twist.angular.z

        # get current speed
        self.vx = twist.linear.x
        self.vy = twist.linear.y
        self.vz = twist.linear.z
        self.omega = twist.angular.z

        try:
            self.direction = self.vx / abs(
                self.vx)  # todo: remove noise as this leads to noisy data at high frequencies
        except ZeroDivisionError:
            self.direction = 1.0

        # self.speed = self.direction * np.linalg.norm([self.vx, self.vy, self.vz])  # todo: refactor as direction can be noisy so migh
        self.speed = self.vx

        # todo: use flag to convert from the center of mass <--> rear axle
        # great tutorial: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
        # # if x, and y are relative to CoG
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        # self.location = [self.rear_x, self.rear_x_y]  # [self.x, self.y]
        # todo: to convert from rear axle to cog see new eagles misc report

        #self.zk[0, 0] = self.x
        #self.zk[1, 0] = self.y
        #self.zk[2, 0] = self.speed
        #self.zk[3, 0] = self.yaw

        if not self.initial_pose_received:
            # if self.mpc_initialized: todo: remove from here
            #    # warmstart
            #    self.Controller.mpc.x0 = self.zk
            #    self.Controller.mpc.set_initial_guess()

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

    def mpc_callback(self):
        # todo: refactor this method
        if self.mpc_built and not self.mpc_initialized and self.initial_pose_received:
            # initialize the MPC solver with the current state if it is already built
            num_tries = 5  # 1, 5, 10
            for _ in range(num_tries):
                # a kind of hacky way to fix not finding a solution the first time the solver is called
                self.initialize_mpc_solver()

        if self.mpc_initialized and self.trajectory_initialized:
            # break out of loop (optional)
            self.stop_flag = self.trajectory.check_goal(self.x, self.y, self.speed, self.final_goal,
                                                        self.current_idx, self.path.shape[0])

            if self.stop_flag:
                self.get_logger().info("Final goal reached.")
                self.delta_cmd = 0.0
                self.acc_cmd = 0.0
                self.velocity_cmd = 0.0
                self.publish_command()
                return

            # copy to prevent overwriting/race conditions due to odom callback update. Todo: Use thread locks instead
            x = self.x  # float(self.zk[0, 0])
            y = self.y
            vel = self.speed
            psi = self.yaw
            omega = self.omega

            # update trajectory state and reference
            # get the robots current state, waypoint index and normalize the angle from -pi to pi or [0, 2PI). todo: update state in purepursuit and MPC node
            self.trajectory.current_index = self.current_idx
            self.trajectory.state[0, self.trajectory.state_key_to_column['x']] = x
            self.trajectory.state[0, self.trajectory.state_key_to_column['y']] = y
            self.trajectory.state[0, self.trajectory.state_key_to_column['speed']] = vel
            self.trajectory.state[0, self.trajectory.state_key_to_column['yaw']] = psi
            self.trajectory.state[0, self.trajectory.state_key_to_column['omega']] = omega
            self.trajectory.state[
                0, self.trajectory.state_key_to_column['curvature']] = trajectory_utils.calculate_curvature_single(
                    self.trajectory.trajectory[:,
                    [self.trajectory.trajectory_key_to_column['x'],
                     self.trajectory.trajectory_key_to_column['y']]],
                    goal_index=self.current_idx)
            self.trajectory.state[0, self.trajectory.state_key_to_column['cum_dist']] = self.cumulative_distance

            # get target point and update reference. todo: setup interpolation of all states and references for the horizon
            # xref, self.current_idx, _ = self.trajectory.calc_ref_trajectory(self.x,
            #                                                                 self.y,
            #                                                                 self.path[:, 0],
            #                                                                 self.path[:, 1],
            #                                                                 self.des_yaw_list, 0,
            #                                                                 self.speeds, 1.,
            #                                                                 self.current_idx,
            #                                                                 n_states=self.NX,
            #                                                                 horizon_length=self.horizon,
            #                                                                 dt=self.sample_time,
            #                                                                 lookahead=self.distance_tolerance)

            target_speed = None
            if self.desired_speed_received and self.desired_speed == 0.0:
                # get the speed from the CSV/topic
                target_speed = self.speeds[self.current_idx]
            elif abs(self.desired_speed) > 0.0:
                # overwrite the speed from the CSV file or topic with the desired speed
                target_speed = self.desired_speed  # todo: add direction from velocity profile

            self.current_idx, reference_traj, _, _, _ = self.trajectory.calc_ref_trajectory(
                    state=None, trajectory=None, current_index=None,
                    dt=self.sample_time, prediction_horizon=self.horizon,
                    lookahead_time=1.0, lookahead=30.0,  # lookahead=pp_controller.lookahead
                    num_points_to_interpolate=self.horizon, target_speed=target_speed
            )

            if reference_traj is None:
                self.get_logger().info("No more waypoints. Final goal reached.")
                self.delta_cmd = 0.0
                self.acc_cmd = 0.0
                self.velocity_cmd = 0.0
                self.publish_command()
                return

            self.xref[:, :] = np.array(
                    [reference_traj['x_ref'], reference_traj['y_ref'],
                     reference_traj['vel_ref'], reference_traj['yaw_ref']]).T
            # self.xref[:, :] = self.xref[0, :]  # to only use the target point

            # self.trajectory.current_index = self.current_idx

            # break out of loop (optional)
            # self.stop_flag = self.trajectory.check_goal(self.x, self.y, self.speed, self.final_goal,
            #                                             self.current_idx, self.path.shape[0])

            # if self.stop_flag:
            #     # if self.current_idx >= len(self.path[:, 0]):
            #     #     self.get_logger().info("Target index out of bounds")
            #     #     return
            #
            #     self.get_logger().info("Goal reached")
            #     return

            if self.initial_pose_received and not self.stop_flag:
                '''update: reference trajectory, reference input, previous input, current state (warmstart)'''
                self.zk[0, 0] = x
                self.zk[1, 0] = y
                self.zk[2, 0] = vel
                self.zk[3, 0] = psi

                # see (https://github.com/bzarr/TUM-CONTROL/blob/main/Model_Predictive_Controller/Nominal_NMPC/NMPC_STM_acados_settings.py#L41)
                normalized_yaw_diff = trajectory_utils.normalize_angle(psi - self.xref[0, 3],
                                                                       minus_pi_to_pi=True, pi_is_negative=True,
                                                                       degrees=False)
                self.cumulative_distance += trajectory_utils.calculate_current_arc_length(
                        self.speed, normalized_yaw_diff, self.sample_time)

                # update previous input
                self.u_prev = self.uk.copy()
                self.Controller.reference_states[:, :] = self.xref
                # self.Controller.vehicle.wp_id = self.current_idx
                # self.Controller.vehicle.current_waypoint = [self.path[self.current_idx, 0],
                #                                             self.path[self.current_idx, 1],
                #                                             self.speeds[self.current_idx],
                #                                             self.des_yaw_list[self.current_idx],
                #                                             0]
                start_time = time.process_time()
                u = self.Controller.get_control(self.zk)
                dt = time.process_time() - start_time

                # unpack mpc predicted states
                time_index = -1  # -1 or self.run_count. Optionally used to index MPC solution history.
                # time_index = self.run_count  # this fails when running with multiple threads
                state_indices = range(1)  # todo: replace with self.NX when vectorized
                input_indices = range(1)  # todo: replace with self.NU when vectorized
                predicted_x_state = self.Controller.mpc.data.prediction(('_x', 'pos_x'), t_ind=time_index)[0]
                predicted_y_state = self.Controller.mpc.data.prediction(('_x', 'pos_y'), t_ind=time_index)[0]
                predicted_vel_state = self.Controller.mpc.data.prediction(('_x', 'vel'), t_ind=time_index)[0]
                predicted_psi_state = self.Controller.mpc.data.prediction(('_x', 'psi'), t_ind=time_index)[0]
                predicted_acc = self.Controller.mpc.data.prediction(('_u', 'acc'), t_ind=time_index)[0]
                predicted_delta = self.Controller.mpc.data.prediction(('_u', 'delta'), t_ind=time_index)[0]

                self.mpc_predicted_states[:, :] = np.hstack([predicted_x_state, predicted_y_state,
                                                             predicted_vel_state, predicted_psi_state])
                self.mpc_predicted_inputs[:, :] = np.hstack([predicted_acc, predicted_delta])

                self.acc_cmd = u[0, 0]
                self.delta_cmd = u[1, 0]
                '''To get the velocity command: use either of the following:
                1: Predicted velocity[k+1], i.e predicted_vel_state[1, 0]
                2: Current_velocity + integrate(acceleration), i.e current_speed + self.acc_cmd * self.sample_time
                '''
                self.velocity_cmd = predicted_vel_state[1, 0]
                # self.velocity_cmd = vel + self.acc_cmd * self.sample_time

                if self.saturate_input:
                    self.input_saturation()

                self.uk[0, 0] = self.acc_cmd
                self.uk[1, 0] = self.delta_cmd

                self.publish_command()
                if self.publish_twist_topic:
                    self.publish_twist(
                        lateral_velocity=predicted_y_state[1, 0])  # todo: test predicted_y_state[1, 0].item()

                # debugging data. Query mpc results and compare tvp ref to xref. # todo: initialize solution_dict above and modify values
                # Controller.mpc.data.prediction(('_x', 'pos_x'), t_ind=simulation_instance.count)[0]
                x_state_ref = self.Controller.mpc.data.prediction(('_tvp', 'x_ref', 0), t_ind=time_index).reshape(-1, 1)
                y_state_ref = self.Controller.mpc.data.prediction(('_tvp', 'y_ref', 0), t_ind=time_index).reshape(-1, 1)
                vel_state_ref = self.Controller.mpc.data.prediction(('_tvp', 'vel_ref', 0), t_ind=time_index).reshape(
                        -1, 1)
                psi_state_ref = self.Controller.mpc.data.prediction(('_tvp', 'psi_ref', 0), t_ind=time_index).reshape(
                        -1, 1)
                self.target_point = self.xref[0, :].tolist()  # todo: refactor
                self.xref[:, :] = np.hstack([x_state_ref, y_state_ref, vel_state_ref, psi_state_ref])  # todo: refactor

                # todo: initialize solution_dict above and modify values
                self.solution_time = self.Controller.mpc.solver_stats['t_wall_total']

                self.solver_iteration_count = self.Controller.mpc.solver_stats['iter_count']
                self.solution_status = self.Controller.mpc.solver_stats['success']
                # todo: publish solver stats

                # if self.run_count % self.control_rate == 0:
                #     self.get_logger().info(f"Accel command: {self.acc_cmd}, delta_cmd: {np.degrees(self.delta_cmd)}, run_count: {self.run_count}, Status: {self.solution_status}\n")

                self.run_count += 1

                # print every second
                if self.run_count % self.control_rate == 0:
                    self.get_logger().info(f"Accel command: {self.acc_cmd}, delta_cmd: {np.degrees(self.delta_cmd)}, run_count: {self.run_count}, Status: {self.solution_status}\n")
                if dt > 0 and self.solution_time > 0:
                    self.get_logger().info(f"Python time: {1 / dt}, do_mpc solve time: {1 / self.solution_time}, solve status: {self.solution_status}")

                # update queues for debug publishing
                self.update_queue(self.goal_queue, self.target_point[0:2])
                self.update_queue(self.mpc_reference_states_queue, self.xref)
                self.update_queue(self.mpc_predicted_states_queue, self.mpc_predicted_states)

                self.get_logger().info(f"Status: {self.solution_status}\n")

                # return

        # todo: replace with an else block and just  print that the model isn't initialized
        # todo: setup using desired_speed float instead of receiving speed array
        if self.path_received and self.desired_speed_received and not self.trajectory_initialized:
            self.reference_path = np.array([self.path[:, 0], self.path[:, 1], self.speeds, self.des_yaw_list]).T
            # relative_dts = np.array([self.sample_time] * self.path.shape[0])
            # relative_times = np.cumsum(relative_dts)

            # todo: get from csv or topics
            relative_times, relative_dts = trajectory_utils.calc_path_relative_time(self.path, self.speeds,
                                                                                    min_dt=1.0)  # min_dt=self.sample_time

            interp = interpolate.interp1d(range(len(self.path)), relative_times, kind="slinear",
                                          fill_value='extrapolate')
            relative_times = interp(range(len(self.path)))

            interp = interpolate.interp1d(range(len(self.path)), relative_dts, kind="slinear", fill_value='extrapolate')
            relative_dts = interp(range(len(self.path)))

            self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['dt']] = relative_dts
            self.trajectory.trajectory[:, self.trajectory.trajectory_key_to_column['total_time_elapsed']] = relative_times
            self.trajectory_initialized = True

    def input_saturation(self):
        # saturate inputs
        if self.saturate_input:
            self.acc_cmd = np.clip(self.acc_cmd, self.MAX_DECEL, self.MAX_ACCEL)
            self.delta_cmd = np.clip(self.delta_cmd, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE)
            self.velocity_cmd = np.clip(self.velocity_cmd, self.MIN_SPEED, self.MAX_SPEED)

    def publish_command(self):
        # filter inputs using a low-pass (or butterworth) filter. Todo

        # Publish Ackermann command
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = self.robot_frame  # self.frame_id
        ackermann_cmd.drive.steering_angle = self.delta_cmd
        # ackermann_cmd.drive.steering_angle_velocity = 0.0
        ackermann_cmd.drive.speed = self.velocity_cmd
        ackermann_cmd.drive.acceleration = self.acc_cmd
        # ackermann_cmd.drive.jerk = 0.0
        self.ackermann_cmd_pub.publish(ackermann_cmd)

        steer_msg = Float32()
        steer_msg.data = float(self.delta_cmd)
        self.steer_pub.publish(steer_msg)
        self.speed_pub.publish(Float32(data=float(self.velocity_cmd)))

    def publish_twist(self, lateral_velocity=0.0):
        # todo: also add an option to publish Twist() message and get bool parameter to select type
        twist_stamped_cmd = TwistStamped()
        twist_stamped_cmd.header.stamp = self.get_clock().now().to_msg()
        twist_stamped_cmd.header.frame_id = self.robot_frame  # self.frame_id
        twist_stamped_cmd.twist.linear.x = self.velocity_cmd
        twist_stamped_cmd.twist.linear.y = lateral_velocity
        twist_stamped_cmd.twist.angular.z = (self.velocity_cmd / self.WHEELBASE) * math.tan(self.delta_cmd)
        self.twist_cmd_pub.publish(twist_stamped_cmd)

    def publish_debug_topics(self):
        """
        Publish markers and path.
        Todo: check if mpc is initialized first
        todo: use numpy to list instead of costly for loops
        Todo: publish car path travelled as a Path message
        Todo: speedup or use thread locks to copy
        """
        if (not self.mpc_initialized) or self.run_count == 0:
            return

        timestamp = self.get_clock().now().to_msg()
        frame_id = self.robot_frame

        goal = None
        location = None
        mpc_reference_states = None
        mpc_predicted_states = None

        # get items from queues
        if not self.goal_queue.empty():
            goal = self.goal_queue.get(block=True)

        if not self.location_queue.empty():
            location = self.location_queue.get(block=True)

        if not self.mpc_predicted_states_queue.empty():
            mpc_predicted_states = self.mpc_predicted_states_queue.get(block=True)

        if not self.mpc_reference_states_queue.empty():
            mpc_reference_states = self.mpc_reference_states_queue.get(block=True)

        if goal is not None:
            # publish the position of the current goal point
            point_msg = PointStamped()
            point_msg.header.stamp = timestamp
            point_msg.header.frame_id = self.global_frame
            point_msg.point.x, point_msg.point.y = goal  # self.target_point[0:2]
            self.mpc_goal_pub.publish(point_msg)

        # Publish a few points in the path left
        path_msg = Path()
        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = self.global_frame

        # todo: remove
        start_idx = self.current_idx
        end_idx = self.final_idx  # self.current_idx + 51
        end_idx = min(end_idx, len(self.path))
        for index in range(start_idx, end_idx):
            waypoint_pose = PoseStamped()
            waypoint_pose.header.stamp = timestamp
            waypoint_pose.header.frame_id = self.global_frame
            waypoint_pose.pose.position.x, waypoint_pose.pose.position.y = self.path[index, :]
            waypoint_pose.pose.orientation.z = self.des_yaw_list[index]
            path_msg.poses.append(waypoint_pose)
        self.waypoint_path_pub.publish(path_msg)

        # Publish the points MPC is considering, i.e state reference
        if mpc_reference_states is not None:
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = self.global_frame

            for k in range(mpc_reference_states.shape[0]):
                # todo: publish velocity or publish as odometry instead
                reference_pose = PoseStamped()
                reference_pose.header.stamp = timestamp
                reference_pose.header.frame_id = self.global_frame

                reference_pose.pose.position.x, reference_pose.pose.position.y = mpc_reference_states[k, 0], \
                mpc_reference_states[k, 1]
                # self.get_logger().info(f"x: {self.xref[0, index]}, y: {self.xref[1, index]}, yaw: {self.xref[3, index]}\n")

                reference_quaternion = tf_transformations.quaternion_from_euler(0, 0, mpc_reference_states[k, 3])
                reference_pose.pose.orientation.x = reference_quaternion[0]
                reference_pose.pose.orientation.y = reference_quaternion[1]
                reference_pose.pose.orientation.z = reference_quaternion[2]
                reference_pose.pose.orientation.w = reference_quaternion[3]

                path_msg.poses.append(reference_pose)
            self.mpc_reference_path_pub.publish(path_msg)

        if mpc_predicted_states is not None:
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = self.global_frame

            # Publish the MPC predicted states
            for k in range(mpc_predicted_states.shape[0]):
                predicted_pose = PoseStamped()
                predicted_pose.header.stamp = timestamp
                predicted_pose.header.frame_id = self.global_frame
                predicted_pose.pose.position.x = mpc_predicted_states[k, 0]
                predicted_pose.pose.position.y = mpc_predicted_states[k, 1]

                predicted_quaternion = tf_transformations.quaternion_from_euler(0, 0, mpc_predicted_states[k, 3])
                predicted_pose.pose.orientation.x = predicted_quaternion[0]
                predicted_pose.pose.orientation.y = predicted_quaternion[1]
                predicted_pose.pose.orientation.z = predicted_quaternion[2]
                predicted_pose.pose.orientation.w = predicted_quaternion[3]

                path_msg.poses.append(predicted_pose)
            self.mpc_path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        mpc_node = KinematicCoupledDoMPCNode()
        executor = MultiThreadedExecutor(
            num_threads=4)  # todo: could get as an argument or initialize in the class to get as a parameter
        executor.add_node(mpc_node)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            mpc_node.destroy_node()

    except KeyboardInterrupt:
        pass

    except ExternalShutdownException:
        sys.exit(1)

    finally:
        rclpy.shutdown()

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    # mpc_node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
