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
    * setup parameter type
    * setup point stabilization, i.e go to goal from RVIz pose
    * change model input to velocity instead of acceleration or use a model
    * speed up do-mpc
    * install HSL MA27 solver
    * load compiled model
    * publish cmd_vel, i.e yaw rate
    * get goal/reference states using ROS actions
    * update goals using ROS actions (goal checker, etc should be done via actions). See (2) above
    * use ROS services or action servers to smooth stuff. See (3) above
    * add allow reversing flag
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
from std_msgs.msg import Float32, Float64, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, \
    PoseWithCovarianceStamped, PointStamped, TransformStamped, TwistStamped, AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

# Custom messages. todo: probably remove Vehicle and trajectory stuff but import casadi model instead
from trajectory_following_ros2.utils.Trajectory import Trajectory
from trajectory_following_ros2.acados.acados_settings import acados_settings


class KinematicCoupledAcadosMPCNode(Node):
    """docstring for ClassName"""

    def __init__(self, ):
        """Constructor for KinematicCoupledAcadosMPCNode"""
        super(KinematicCoupledAcadosMPCNode, self).__init__('kinematic_coupled_acados_controller')
        # declare parameters
        self.declare_parameter('ode_type', "continuous")  # todo: also augmented, etc
        self.declare_parameter('stage_cost_type', "LINEAR_LS")  # Acados specific (LINEAR_LS, NONLINEAR_LS, EXTERNAL)
        self.declare_parameter('terminal_cost_type', "LINEAR_LS")  # Acados specific (LINEAR_LS, NONLINEAR_LS, EXTERNAL)
        self.declare_parameter('model_type', "continuous")
        self.declare_parameter('control_rate', 20.0)
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
        self.declare_parameter('n_states', 4)  # todo: remove and get from import model
        self.declare_parameter('n_inputs', 2)  # todo: remove and get from import model
        self.declare_parameter('horizon', 25)
        self.declare_parameter('prediction_time')
        self.declare_parameter('max_iter', 15)  # it usually doesnt take more than 15 iterations
        self.declare_parameter('termination_condition',
                               0.1)  # iteration finish param. todo: pass to mpc initializer solver options

        # tips for tuning weights (https://www.mathworks.com/help/mpc/ug/tuning-weights.html)
        self.declare_parameter('R', [0.01, 0.01])
        self.declare_parameter('Rd', [10., 100.])
        self.declare_parameter('Q', [1.0, 1.0, 1.0, 0.01])
        self.declare_parameter('Qf', [0.04, 0.04, 0.1, 0.01])

        self.declare_parameter('path_topic', '/waypoint_loader/path')  # todo: replace with custom message or action
        self.declare_parameter('speed_topic', '/waypoint_loader/speed')  # todo: replace with custom message or action
        self.declare_parameter('odom_topic', '/vehicle/odometry/filtered')
        self.declare_parameter('acceleration_topic', '/vehicle/accel/filtered')
        self.declare_parameter('ackermann_cmd_topic', '/drive')

        self.declare_parameter('desired_speed')  # todo: get from topic/trajectory message
        self.declare_parameter('loop', False)  # todo: remove and pass to waypoint publisher
        self.declare_parameter('n_ind_search', 10)  # the number of points to check for when searching for the closest
        self.declare_parameter('smooth_yaw', False)
        self.declare_parameter('debug', False)  # displays solver output

        self.control_rate = self.get_parameter('control_rate').value
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
        self.horizon = self.get_parameter('horizon').value
        self.prediction_time = self.get_parameter('prediction_time').value

        self.max_iter = self.get_parameter('max_iter').value
        self.termination_condition = self.get_parameter('termination_condition').value

        # initialize variables
        self.ode_type = self.get_parameter('ode_type').value
        self.stage_cost_type = self.get_parameter('stage_cost_type').value
        self.terminal_cost_type = self.get_parameter('terminal_cost_type').value
        self.path_topic = self.get_parameter('path_topic').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value

        self.desired_speed = self.get_parameter('desired_speed').value
        self.loop = self.get_parameter('loop').value
        self.n_ind_search = self.get_parameter('n_ind_search').value
        self.smooth_yaw = self.get_parameter('smooth_yaw').value
        self.debug = self.get_parameter('debug').value

        self.dt = self.sample_time = 1 / self.control_rate  # [s] sample time. # todo: get from ros Duration
        if self.prediction_time is None:
            self.prediction_time = self.sample_time * self.horizon

        # initialize trajectory class
        self.trajectory_instance = Trajectory(search_index_number=10,
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
        self.mpc_initialized = False
        self.x = 0.0  # todo: remove
        self.y = 0.0  # todo: remove
        self.yaw = 0.0  # todo: remove
        self.speed = 0.0  # todo: remove
        self.direction = 1.0
        self.yaw_rate = 0.0  # todo: remove
        self.acceleration = 0.0  # todo: remove
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))  # todo: remove
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))  # todo: remove
        self.location = [self.x, self.y]  # todo: remove
        self.alpha = 0.0  # yaw error  # todo: remove
        self.desired_curvature = 0.0  # todo: remove
        self.desired_steering_angle = 0.0  # todo: remove
        self.desired_yaw_rate = 0.0  # todo: remove

        # Setup transformations to transform pose from one frame to another
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_frame_id = ''  # the frame ID of the pose state received. odom, map

        # Setup subscribers
        self.path_sub = self.create_subscription(
                Path,
                self.path_topic,
                self.path_callback,
                1)  # to get the desired/reference states of the robot
        self.speed_sub = self.create_subscription(
                Float32MultiArray,
                self.speed_topic,
                self.desired_speed_callback,
                1)  # to get the desired/reference states of the robot
        self.odom_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_callback,
                1)  # to get the current state of the robot
        # self.pose_sub = self.create_subscription(
        #        PoseWithCovarianceStamped,
        #        self.pose_topic,
        #        self.pose_callback,
        #        1)
        self.accel_sub = self.create_subscription(
                AccelWithCovarianceStamped,
                self.acceleration_topic,
                self.acceleration_callback,
                1)

        # Setup publishers.
        self.ackermann_cmd_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_cmd_topic, 1)
        self.steer_pub = self.create_publisher(Float32, '/mpc/des_steer', 1)
        self.speed_pub = self.create_publisher(Float32, '/mpc/des_speed', 1)
        self.yaw_rate_pub = self.create_publisher(Float32, '/mpc/des_yaw_rate', 1)

        self.mpc_goal_pub = self.create_publisher(PointStamped, '/mpc/goal_point', 1)
        self.waypoint_path_pub = self.create_publisher(Path, '/mpc/path_remaining', 1)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc/predicted_path', 1)
        self.mpc_reference_path_pub = self.create_publisher(Path, '/mpc/reference_path', 1)

        # publish the internally estimated state of the car
        self.mpc_odom_pub = self.create_publisher(Odometry, '/mpc/current_state', 1)
        self.mpc_pose_pub = self.create_publisher(PoseStamped, '/mpc/current_pose', 1)

        # setup mpc timer
        self.mpc_timer = self.create_timer(self.sample_time, self.mpc_callback)
        self.debug_timer = self.create_timer(self.sample_time, self.publish_debug_topics)

        self.path = None
        self.des_yaw_list = None
        self.speeds = None

        # setup waypoint variables. todo: make mpc not need the path to be initialized
        self.reference_path = None  # todo: remove
        self.current_idx = 0
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

        self.constraint = None
        self.model = None
        self.controller = None
        self.ocp = None

        self.get_logger().info('kinematic_coupled_do_mpc_controller node started. ')

    def path_callback(self, data):
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
        # self.dist_arr = np.zeros(len(self.path))  # array of distances used to check closes path
        if self.desired_speed_received:
            self.reference_path = np.array(
                    [self.path[:, 0], self.path[:, 1], self.speeds, self.des_yaw_list]).T  # todo: remove
            self.final_idx = self.path.shape[0] - 1
            self.final_goal = self.path[self.final_idx, :]

            self.path_received = True

    def desired_speed_callback(self, data):
        speed_list = []
        for speed in data.data:
            speed_list.append(speed)
        self.speeds = np.array(speed_list)
        self.desired_speed_received = True

    def initialize_mpc(self):
        # todo: initiailize build variables above
        # todo: specify file paths via ROS otherwise its stored wherever the node is called from, e.g home
        self.constraint, self.model, self.controller, self.ocp = acados_settings(Tf=self.prediction_time,
                                                                                 N=self.horizon,
                                                                                 x0=self.zk, scale_cost=True,
                                                                                 cost_module=self.stage_cost_type,
                                                                                 cost_module_e=self.terminal_cost_type,
                                                                                 generate=False, build=False,
                                                                                 with_cython=True,
                                                                                 mpc_config_file="kinematic_bicycle_acados_ocp.json",
                                                                                 code_export_directory="c_generated_code"
                                                                                 )
        # todo: set weights
        # initialize solver
        for i in range(self.horizon + 1):
            self.controller.set(i, "x", self.zk.flatten())

        for i in range(self.horizon):
            self.controller.set(i, "u", self.uk.flatten())

        # constrain x0
        self.controller.set(0, "lbx",
                            self.zk.flatten())  # can only be set if x0 was provided. Otherwise set 'x'. Raises Exception
        self.controller.set(0, "ubx", self.zk.flatten())

        # solve/get initial guess
        status = self.controller.solve()
        self.mpc_initialized = True

    def odom_callback(self, data):
        """
        Todo: transform coordinates
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
            self.direction = self.vx / abs(self.vx)
        except ZeroDivisionError:
            self.direction = 1.0

        self.speed = self.direction * np.linalg.norm([self.vx, self.vy, self.vz])
        # self.speed = self.vx

        self.zk[0, 0] = self.x
        self.zk[1, 0] = self.y
        self.zk[2, 0] = self.speed
        self.zk[3, 0] = self.yaw

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
        if self.mpc_initialized:
            # get target point and update reference. todo: setup interpolation of all states and references for the horizon
            xref, self.current_idx, _ = self.trajectory_instance.calc_ref_trajectory(self.x,
                                                                                     self.y,
                                                                                     self.path[:, 0],
                                                                                     self.path[:, 1],
                                                                                     self.des_yaw_list, 0,
                                                                                     self.speeds, 1.,
                                                                                     self.current_idx,
                                                                                     n_states=self.NX,
                                                                                     horizon_length=self.horizon,
                                                                                     dt=self.sample_time)
            self.xref[:, :] = xref.T
            # break out of loop (optional)
            self.stop_flag = self.trajectory_instance.check_goal(self.x, self.y, self.speed, self.final_goal,
                                                                 self.current_idx, self.path.shape[0])
            if self.stop_flag:
                # if self.current_idx >= len(self.path[:, 0]):
                #     print("Target index out of bounds")
                #     return

                print("Goal reached")
                return

            if self.initial_pose_received and not self.stop_flag:
                '''update: reference trajectory, reference input, previous input, current state (warmstart)'''
                # copy to prevent overwriting/race conditions due to odom callback update. Todo: Use thread locks instead
                x = self.x  # float(self.zk[0, 0])
                y = self.y
                vel = self.speed
                psi = self.yaw

                # update previous input
                self.u_prev = self.uk.copy()
                # self.controller.set(0, "u", self.u_prev.flatten())  # for now, not used by Acados
                current_speed = self.speed

                # initial condition
                self.controller.constraints_set(0, "lbx", np.array([x, y, vel, psi]))  # zk.flatten()
                self.controller.constraints_set(0, "ubx", np.array([x, y, vel, psi]))

                # update reference
                for j in range(self.horizon):
                    yref = np.array(
                            [xref[0, j], xref[1, j], xref[2, j], xref[3, j], 0, 0])  # reference state and input

                    if self.stage_cost_type in ['LINEAR_LS', 'NONLINEAR_LS']:
                        self.controller.cost_set(j, "yref", yref)  # update reference, cost_set(j, "yref", yref)

                    '''
                    set the value of parameters
                                                    wheelbase, zref, uref, z_k, u_prev
                    '''
                    self.controller.set(j, "p", np.array([0.256, *yref, *self.zk, *self.u_prev]))
                    u_prev = self.controller.get(0, "u").flatten()

                    # # warmstart. Optional as Acados does this Automatically
                    # if len(warmstart_variables) > 0 and j > 1:
                    #     # previous solution (warmstart)
                    #     self.controller.set(j, "x", warmstart_variables['z_ws'][:, j])

                # set the terminal reference
                yref_N = np.array([xref[0, self.horizon],
                                   xref[1, self.horizon],
                                   xref[2, self.horizon],
                                   xref[3, self.horizon]])  # terminal state
                if self.terminal_cost_type in ['LINEAR_LS', 'NONLINEAR_LS']:
                    self.controller.set(self.horizon, "yref", yref_N)
                self.controller.set(self.horizon,
                                  "p",
                                  np.array([0.256, *yref_N, *np.zeros(2), *self.zk, *self.u_prev]))

                # if len(warmstart_variables) > 0:
                #     self.controller.set(self.horizon,
                #                       "x",
                #                       warmstart_variables['z_ws'][:, self.horizon])

                start_time = time.process_time()
                # 0 – success, 1 – failure, 2 – maximum number of iterations reached,
                # 3 – minimum step size in QP solver reached, 4 – qp solver failed
                status = self.controller.solve()
                if status != 0:
                    # acados_solver.print_statistics()
                    print("acados returned status {} in"
                          " closed loop iteration {}.".format(status, self.run_count))
                dt = time.process_time() - start_time

                # retrieve/unpack mpc predicted states
                time_index = self.run_count  # self.run_count or -1. Optionally used to index MPC solution history
                state_indices = range(1)  # todo: replace with self.NX when vectorized
                input_indices = range(1)  # todo: replace with self.NU when vectorized

                xk_pred = self.controller.get(0, "x")
                x_next_pred = self.controller.get(1, "x")
                u = self.controller.get(0, "u")

                solution_dict = {'z_mpc': [], 'u_mpc': [], 'z_ref': [], 'sl_mpc': []}  # todo: initialize above
                for i in range(self.horizon + 1):
                    x_s = self.controller.get(i, 'x')
                    # print(pred_)
                    solution_dict['z_mpc'].append(x_s)
                    solution_dict['sl_mpc'].append([self.controller.get(i, "sl"), self.controller.get(i, "su")])
                    # solution_dict['z_ref'].append(acados_solver.get(i, 'yref'))  # Acados doesn't return yref

                    if i < self.horizon:
                        u_s = self.controller.get(i, 'u')
                        solution_dict['u_mpc'].append(u_s)

                solution_dict['z_mpc'] = np.array(solution_dict['z_mpc']).T
                solution_dict['u_mpc'] = np.array(solution_dict['u_mpc']).T

                cost = self.controller.get_cost()
                # print(f'Cost: {cost}')
                time_taken = self.controller.get_stats('time_tot')
                num_iter = self.controller.get_stats('sqp_iter')
                stats = self.controller.get_stats('statistics')
                solve_time_ = self.controller.get_stats('time_tot')
                time_qp_solution_ = self.controller.get_stats('time_qp')
                time_linearization_ = self.controller.get_stats('time_lin')
                time_integrator_ = self.controller.get_stats('time_sim')
                # acados_solver.print_statistics()

                warmstart_variables = {'z_ws': solution_dict['z_mpc'], 'u_ws': solution_dict['u_mpc'],
                                       'sl_ws': solution_dict['sl_mpc']}
                predicted_x_state = solution_dict['z_mpc'][0, :]
                predicted_y_state = solution_dict['z_mpc'][1, :]
                predicted_vel_state = solution_dict['z_mpc'][2, :]
                predicted_psi_state = solution_dict['z_mpc'][3, :]
                predicted_acc = solution_dict['u_mpc'][0, :]
                predicted_delta = solution_dict['u_mpc'][1, :]
                self.mpc_predicted_states[:, :] = np.hstack([predicted_x_state, predicted_y_state,
                                                             predicted_vel_state, predicted_psi_state])
                self.mpc_predicted_inputs[:, :] = np.hstack([predicted_acc, predicted_delta])

                self.acc_cmd = u[0]
                self.delta_cmd = u[1]
                '''To get the velocity command: use either of the following:
                1: Predicted velocity[k+1], i.e predicted_vel_state[1, 0]
                2: Current_velocity + integrate(acceleration), i.e current_speed + self.acc_cmd * self.sample_time
                '''
                self.velocity_cmd = predicted_vel_state[1]
                self.publish_command()

                # debugging data. Query mpc results and compare tvp ref to xref. # todo: initialize solution_dict above and modify values
                x_state_ref = xref[0, :].reshape(-1, 1)  # solution_dict['z_ref'][:, 0]
                y_state_ref = xref[1, :].reshape(-1, 1)  # solution_dict['z_ref'][:, 1]
                vel_state_ref = xref[2, :].reshape(-1, 1)  # solution_dict['z_ref'][:, 2]
                psi_state_ref = xref[3, :].reshape(-1, 1)  # solution_dict['z_ref'][:, 3]
                self.xref[:, :] = np.hstack([x_state_ref, y_state_ref, vel_state_ref, psi_state_ref])  # todo: refactor
                self.target_point = self.xref[0, :].tolist()  # todo: refactor

                # todo: initialize solution_dict above and modify values
                self.solution_time = self.Controller.mpc.solver_stats['t_wall_total']

                self.solver_iteration_count = self.Controller.mpc.solver_stats['iter_count']
                self.solution_status = self.Controller.mpc.solver_stats['success']

                self.run_count += 1
                return

        if self.path_received and self.desired_speed_received and not self.mpc_initialized:
            self.initialize_mpc()

    def publish_command(self):
        # filter inputs using a low-pass (or butterworth) filter. Todo

        # saturate inputs
        if self.saturate_input:
            self.acc_cmd = np.clip(self.acc_cmd, self.MAX_DECEL, self.MAX_ACCEL)
            self.delta_cmd = np.clip(self.delta_cmd, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE)
            self.velocity_cmd = np.clip(self.velocity_cmd, self.MIN_SPEED, self.MAX_SPEED)

        self.uk[0, 0] = self.acc_cmd
        self.uk[1, 0] = self.delta_cmd

        # Publish Ackermann command
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = 'base_link'  # self.frame_id
        ackermann_cmd.drive.steering_angle = self.delta_cmd
        ackermann_cmd.drive.speed = self.velocity_cmd
        self.ackermann_cmd_pub.publish(ackermann_cmd)

        steer_msg = Float32()
        steer_msg.data = float(self.delta_cmd)
        self.steer_pub.publish(steer_msg)
        self.speed_pub.publish(Float32(data=float(self.velocity_cmd)))

    def publish_debug_topics(self):
        """
        Publish markers and path.
        Todo: check if mpc is initialized first
        """
        if (not self.mpc_initialized) or self.run_count == 0:
            return

        timestamp = self.get_clock().now().to_msg()
        frame_id = 'base_link'

        # publish the position of the current goal point
        point_msg = PointStamped()
        point_msg.header.stamp = timestamp
        point_msg.header.frame_id = 'odom'
        point_msg.point.x, point_msg.point.y = self.target_point[0:2]
        self.mpc_goal_pub.publish(point_msg)

        # Publish a few points in the path left
        path_msg = Path()
        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = "odom"

        # todo: remove
        start_idx = self.current_idx
        end_idx = self.final_idx  # self.current_idx + 51
        end_idx = min(end_idx, len(self.path))
        for index in range(start_idx, end_idx):
            waypoint_pose = PoseStamped()
            waypoint_pose.header.stamp = timestamp
            waypoint_pose.header.frame_id = 'odom'
            waypoint_pose.pose.position.x, waypoint_pose.pose.position.y = self.path[index, :]
            waypoint_pose.pose.orientation.z = self.des_yaw_list[index]
            path_msg.poses.append(waypoint_pose)
        self.waypoint_path_pub.publish(path_msg)

        # Publish the points MPC is considering, i.e state reference
        path_msg = Path()
        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = "odom"

        for k in range(self.xref.shape[0]):
            # todo: publish velocity or publish as odometry instead
            reference_pose = PoseStamped()
            reference_pose.header.stamp = timestamp
            reference_pose.header.frame_id = 'odom'

            reference_pose.pose.position.x, reference_pose.pose.position.y = self.xref[k, 0], self.xref[k, 1]
            # print(f"x: {self.xref[0, index]}, y: {self.xref[1, index]}, yaw: {self.xref[3, index]}\n")

            reference_quaternion = tf_transformations.quaternion_from_euler(0, 0, self.xref[k, 3])
            reference_pose.pose.orientation.x = reference_quaternion[0]
            reference_pose.pose.orientation.y = reference_quaternion[1]
            reference_pose.pose.orientation.z = reference_quaternion[2]
            reference_pose.pose.orientation.w = reference_quaternion[3]

            path_msg.poses.append(reference_pose)
        self.mpc_reference_path_pub.publish(path_msg)

        # Publish the MPC predicted states
        for k in range(self.mpc_predicted_states.shape[0]):
            predicted_pose = PoseStamped()
            predicted_pose.header.stamp = timestamp
            predicted_pose.header.frame_id = 'odom'
            predicted_pose.pose.position.x = self.mpc_predicted_states[k, 0]
            predicted_pose.pose.position.y = self.mpc_predicted_states[k, 1]

            predicted_quaternion = tf_transformations.quaternion_from_euler(0, 0, self.mpc_predicted_states[k, 3])
            predicted_pose.pose.orientation.x = predicted_quaternion[0]
            predicted_pose.pose.orientation.y = predicted_quaternion[1]
            predicted_pose.pose.orientation.z = predicted_quaternion[2]
            predicted_pose.pose.orientation.w = predicted_quaternion[3]

            path_msg.poses.append(predicted_pose)
        self.mpc_path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    mpc_node = KinematicCoupledAcadosMPCNode()
    rclpy.spin(mpc_node)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    mpc_node.waypoint_file.close()  # todo: remove
    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
