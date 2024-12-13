"""
(optionally) Load waypoints
(optionally) Remap waypoint path topics to the one expected by the nodes
Set directory paths
Set default parameters
Choose MPC type
Remap nav2 path topic to

Todo:
    * add purepursuit
    * add MPC groups
    * rename from MPC as it includes all controllers now
    * setup node name arguments
"""

import os
import yaml
import pathlib

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import PushRosNamespace
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml, ReplaceString


def generate_launch_description():
    trajectory_following_ros2_pkg_prefix = get_package_share_directory('trajectory_following_ros2')

    # Create the launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    log_level = LaunchConfiguration('log_level')
    robot_frame = LaunchConfiguration('robot_frame')
    global_frame = LaunchConfiguration('global_frame')
    frequency = LaunchConfiguration('frequency')
    publish_twist_topic = LaunchConfiguration('publish_twist_topic')
    wheelbase = LaunchConfiguration('wheelbase')
    ode_type = LaunchConfiguration('ode_type')
    load_waypoints = LaunchConfiguration('load_waypoints')
    waypoints_csv = LaunchConfiguration('waypoints_csv')

    # Constraints
    saturate_inputs = LaunchConfiguration('saturate_inputs')
    allow_reversing = LaunchConfiguration('allow_reversing')
    max_speed = LaunchConfiguration('max_speed')
    min_speed = LaunchConfiguration('min_speed')
    max_accel = LaunchConfiguration('max_accel')
    max_decel = LaunchConfiguration('max_decel')
    max_steer = LaunchConfiguration('max_steer')
    min_steer = LaunchConfiguration('min_steer')
    max_steer_rate = LaunchConfiguration('max_steer_rate')
    desired_speed = LaunchConfiguration('desired_speed')

    # # MPC parameters
    mpc_toolbox = LaunchConfiguration('mpc_toolbox')
    horizon = LaunchConfiguration('horizon')
    sample_time = LaunchConfiguration('sample_time')
    prediction_time = LaunchConfiguration('prediction_time')
    R_diagonal = LaunchConfiguration('R_diagonal')
    Rd_diagonal = LaunchConfiguration('Rd_diagonal')
    Q_diagonal = LaunchConfiguration('Q_diagonal')
    Qf_diagonal = LaunchConfiguration('Qf_diagonal')
    scale_cost = LaunchConfiguration('scale_cost')
    max_iterations = LaunchConfiguration('max_iterations')
    termination_condition = LaunchConfiguration('termination_condition')
    stage_cost_type = LaunchConfiguration('stage_cost_type')
    terminal_cost_type = LaunchConfiguration('terminal_cost_type')
    generate_mpc_model = LaunchConfiguration('generate_mpc_model')
    build_with_cython = LaunchConfiguration('build_with_cython')
    model_directory = LaunchConfiguration('model_directory')
    debug_frequency = LaunchConfiguration('debug_frequency')

    # # Trajectory/goal parameters
    distance_tolerance = LaunchConfiguration('distance_tolerance')
    speed_tolerance = LaunchConfiguration('speed_tolerance')

    # #  Topics
    odom_topic = LaunchConfiguration('odom_topic', default="/odometry/local")
    ackermann_cmd_topic = LaunchConfiguration('ackermann_cmd_topic', default="/drive")
    twist_topic = LaunchConfiguration('twist_topic', default="/cmd_vel")
    acceleration_topic = LaunchConfiguration('acceleration_topic', default="/accel/local")
    path_topic = LaunchConfiguration('path_topic', default="/trajectory/path")
    speed_topic = LaunchConfiguration('speed_topic', default="/trajectory/speed")

    # Declare default launch arguments
    config_file_path = os.path.join(trajectory_following_ros2_pkg_prefix, 'config/mpc_parameters.yaml')
    waypoints_csv_path = os.path.join(trajectory_following_ros2_pkg_prefix, 'data/waypoints.csv')
    mpc_model_path = os.path.join(trajectory_following_ros2_pkg_prefix, 'data/mpc')

    # declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true')
    params_file_la = DeclareLaunchArgument(
            'params_file',
            default_value=config_file_path,
            description='Path to config file for localization nodes'
    )
    robot_frame_la = DeclareLaunchArgument(
            'robot_frame',
            default_value='base_link',
            description='The frame attached to the car. '
                        'The relative/local frame. '
                        'Usually the ground projection of the center '
                        'of the rear axle of a car or the center of gravity '
                        'of a differential robot. '
                        'Actuation commands, speed and acceleration are '
                        'relative to this frame.'
                        'E.g base_link, base_footprint, '
                        'ego_vehicle (Carla). '
                        'Obstacle positions could be specified '
                        'relative to this frame or the global frame.'
    )
    global_frame_la = DeclareLaunchArgument(
            'global_frame',
            default_value='odom',
            description='The global/world/map frame. '
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
                        'otherwise use "odom".'
    )
    frequency_la = DeclareLaunchArgument(
            'frequency',
            default_value='50.0',
            description='Controller frequency.'
    )
    publish_twist_topic_la = DeclareLaunchArgument(
            'publish_twist_topic',
            default_value='True',
            description='Whether or not to publish as a TwistStamped topic.'
    )
    wheelbase_la = DeclareLaunchArgument(
            'wheelbase',
            default_value='0.256',
            description='The cars wheelbase.'
    )
    ode_type_la = DeclareLaunchArgument(
            'ode_type',
            default_value='discrete_kinematic_coupled',
            description='The type of ode. '
                        'Examples: '
                        '   continuous_kinematic_coupled, continuous_kinematic_coupled_augmented'
                        '   discrete_kinematic_coupled, discrete_kinematic_coupled_augmented, '
                        '   discrete_dynamic_decoupled '
                        'Options: continuous/discrete, kinematic/dynamic, coupled/decoupled, augmented.'
    )
    load_waypoints_la = DeclareLaunchArgument(
            'load_waypoints',
            default_value='True',
            description='Whether to load waypoints from a file an publish.'
    )
    waypoints_csv_la = DeclareLaunchArgument(
            'waypoints_csv',
            default_value=waypoints_csv_path,
            description='Path to the waypoints csv file.'
    )

    saturate_inputs_la = DeclareLaunchArgument(
            'saturate_inputs',
            default_value='True',
            description='Whether to saturate the inputs before sending to the actuators.'
                        'Not necessary for MPC nodes as the optimal solution should already do this.'
    )
    allow_reversing_la = DeclareLaunchArgument(
            'allow_reversing',
            default_value='True',
            description='Whether or not to allow reversing. Will set min_speed as 0 if False'
    )
    max_speed_la = DeclareLaunchArgument(
            'max_speed',
            default_value='1.5',
            description='Maximum speed in m/s allowed.'
    )
    min_speed_la = DeclareLaunchArgument(
            'min_speed',
            default_value='-1.5',
            description='Maximum speed in m/s allowed.'
    )
    max_accel_la = DeclareLaunchArgument(
            'max_accel',
            default_value='3.0',
            description='Maximum acceleration in m/s^2 allowed.'
    )
    max_decel_la = DeclareLaunchArgument(
            'max_decel',
            default_value='-3.0',
            description='Maximum deceleration (or minimum acceleration) in m/s^2 allowed.'
    )

    max_steer_la = DeclareLaunchArgument(
            'max_steer',
            default_value='27.0',
            description='Maximum steering angle in degrees allowed.'
    )
    min_steer_la = DeclareLaunchArgument(
            'min_steer',
            default_value='-27.0',
            description='Maximum steering angle in degrees allowed.'
    )
    max_steer_rate_la = DeclareLaunchArgument(
            'max_steer_rate',
            default_value='352.9411764706',  # 60 / 0.17
            description='Maximum steering angle rate in degrees/s allowed.'
    )

    desired_speed_la = DeclareLaunchArgument(
            'desired_speed',
            default_value='0.0',
            description='Used to set target speed or override trajectory speed..'
    )

    mpc_toolbox_la = DeclareLaunchArgument(
            'mpc_toolbox',
            default_value='acados',
            description='MPC toolbox to use. acados, do_mpc, casadi.'
    )
    horizon_la = DeclareLaunchArgument(
            'horizon',
            default_value='15',
            description='MPC horizon length.'
    )
    sample_time_la = DeclareLaunchArgument(
            'sample_time',
            default_value='0.05',
            description='Sample time.'
    )
    prediction_time_la = DeclareLaunchArgument(
            'prediction_time',
            default_value='1.5',
            description='Time in seconds to lookahead. prediction_time = sample_time * horizon'
    )
    R_diagonal_la = DeclareLaunchArgument(
            'R_diagonal',
            default_value='[0.01, 0.01]',
            description='List containing the diagonal for the R matrix.'
    )
    Rd_diagonal_la = DeclareLaunchArgument(
            'Rd_diagonal',
            default_value='[10., 100.]',
            description='List containing the diagonal for the Rd matrix.'
    )
    Q_diagonal_la = DeclareLaunchArgument(
            'Q_diagonal',
            default_value='[1.0, 1.0, 1.0, 0.01]',
            description='List containing the diagonal for the Q matrix.'
    )
    Qf_diagonal_la = DeclareLaunchArgument(
            'Qf_diagonal',
            default_value='[0.04, 0.04, 0.1, 0.01]',
            description='List containing the diagonal for the Qf matrix.'
    )
    scale_cost_la = DeclareLaunchArgument(
            'scale_cost',
            default_value='False',
            description='Whether to scale the cost by (horizon / final time).'
    )
    max_iterations_la = DeclareLaunchArgument(
            'max_iterations',
            default_value='15',
            description='Maximum MPC solver iterations.'
    )
    termination_condition_la = DeclareLaunchArgument(
            'termination_condition',
            default_value='0.001',
            description='MPC solver termination tolerance. '
                        'Larger values lead to earlier termination but less accurate results.'
    )
    stage_cost_type_la = DeclareLaunchArgument(
            'stage_cost_type',
            default_value='NONLINEAR_LS',
            description='Stage cost type. LINEAR_LS, NONLINEAR_LS, EXTERNAL'
    )
    terminal_cost_type_la = DeclareLaunchArgument(
            'terminal_cost_type',
            default_value='NONLINEAR_LS',
            description='Terminal cost type. LINEAR_LS, NONLINEAR_LS, EXTERNAL'
    )
    generate_mpc_model_la = DeclareLaunchArgument(
            'generate_mpc_model',
            default_value='False',
            description='Whether to generate the MPC model.'
    )
    build_with_cython_la = DeclareLaunchArgument(
            'build_with_cython',
            default_value='True',
            description='Whether to build the model with cython (faster) or ctypes.'
    )
    model_directory_la = DeclareLaunchArgument(
            'model_directory',
            default_value=mpc_model_path,
            description='Path to the built/generated model.'
    )
    distance_tolerance_la = DeclareLaunchArgument(
            'distance_tolerance',
            default_value='0.2',
            description='Distance tolerance for arrival.'
    )
    speed_tolerance_la = DeclareLaunchArgument(
            'speed_tolerance',
            default_value='0.5',
            description='Speed tolerance for arrival.'
    )
    declare_log_level_cmd = DeclareLaunchArgument(
            'log_level', default_value='info',
            description='log level')

    odom_topic_la = DeclareLaunchArgument(
            'odom_topic',
            default_value=odom_topic
    )

    ackermann_cmd_topic_la = DeclareLaunchArgument(
            'ackermann_cmd_topic',
            default_value=ackermann_cmd_topic
    )

    twist_topic_la = DeclareLaunchArgument(
            'twist_topic',
            default_value=twist_topic
    )

    acceleration_topic_la = DeclareLaunchArgument(
            'acceleration_topic',
            default_value=acceleration_topic
    )

    path_topic_la = DeclareLaunchArgument(
            'path_topic',
            default_value=path_topic
    )

    speed_topic_la = DeclareLaunchArgument(
            'speed_topic',
            default_value=speed_topic
    )

    debug_frequency_la = DeclareLaunchArgument(
            'debug_frequency',
            default_value='4.0',
            description='The rate at which to publish debugging/visualization topics. If <= 0, no topics are published.'
    )

    # Create Launch Description
    ld = LaunchDescription(
            [declare_use_sim_time_cmd, params_file_la,
             robot_frame_la, global_frame_la,
             frequency_la, publish_twist_topic_la, wheelbase_la, ode_type_la,
             load_waypoints_la, waypoints_csv_la,
             saturate_inputs_la, allow_reversing_la, max_speed_la, min_speed_la, max_accel_la, max_decel_la,
             max_steer_la, min_steer_la, max_steer_rate_la, desired_speed_la,
             mpc_toolbox_la, horizon_la, sample_time_la, prediction_time_la,
             R_diagonal_la, Rd_diagonal_la, Q_diagonal_la, Qf_diagonal_la, scale_cost_la,
             max_iterations_la, termination_condition_la,
             stage_cost_type_la, terminal_cost_type_la,
             generate_mpc_model_la, build_with_cython_la, model_directory_la,
             distance_tolerance_la, speed_tolerance_la,
             declare_log_level_cmd,
             odom_topic_la, ackermann_cmd_topic_la, twist_topic_la, acceleration_topic_la, path_topic_la,
             speed_topic_la, debug_frequency_la]
    )

    # Load Nodes
    waypoint_loader_node = Node(
            condition=IfCondition(load_waypoints),
            package='trajectory_following_ros2',
            executable='waypoint_loader',
            name='waypoint_loader_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'file_path': waypoints_csv},
                # {'global_frame': global_frame}  # todo
            ],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=[
                ('/waypoint_loader/path', path_topic),
                ('/waypoint_loader/speed', speed_topic),
            ]
    )

    # todo: use SetParameter and Group Action to avoid repeating MPC parameters.
    acados_mpc_node = Node(
            condition=LaunchConfigurationEquals('mpc_toolbox', 'acados'),
            package='trajectory_following_ros2',
            executable='coupled_kinematic_acados',
            name='acados_mpc_node',
            output='screen',
            parameters=[
                params_file,
                {'use_sim_time': use_sim_time},
                {
                    'robot_frame': robot_frame,
                    'global_frame': global_frame,
                },
                {'control_rate': frequency},
                {'publish_twist_topic': publish_twist_topic},
                {'wheelbase': wheelbase},
                {'ode_type': ode_type},
                {'max_speed': max_speed},
                {'min_speed': min_speed},
                {'max_accel': max_accel},
                {'max_decel': max_decel},
                {'max_steer': max_steer},
                {'min_steer': min_steer},
                {'max_steer_rate': max_steer_rate},
                {'desired_speed': desired_speed},
                {'saturate_inputs': saturate_inputs},
                {'allow_reversing': allow_reversing},
                {'horizon': horizon},
                {'sample_time': sample_time},
                {'prediction_time': prediction_time},
                {'R': R_diagonal},
                {'Rd': Rd_diagonal},
                {'Q': Q_diagonal},
                {'Qf': Qf_diagonal},
                {'scale_cost': scale_cost},
                {'max_iter': max_iterations},
                {'termination_condition': termination_condition},
                {'generate_mpc_model': generate_mpc_model},
                {'build_with_cython': build_with_cython},
                {'model_directory': model_directory},
                {'stage_cost_type': stage_cost_type},
                {'terminal_cost_type': terminal_cost_type},
                {'distance_tolerance': distance_tolerance},
                {'speed_tolerance': speed_tolerance},
                {
                    'odom_topic': odom_topic,
                    'ackermann_cmd_topic': ackermann_cmd_topic,
                    'twist_topic': twist_topic,
                    'acceleration_topic': acceleration_topic,
                    'path_topic': path_topic,
                    'speed_topic': speed_topic,
                },
                {'debug_frequency': debug_frequency},
            ],
            arguments=['--ros-args', '--log-level', log_level],
            # remappings=[
            #     ('/waypoint_loader/path', '/trajectory/path'),
            #     ('/waypoint_loader/speed', '/trajectory/speed'),
            # ]
    )

    do_mpc_node = Node(
            condition=LaunchConfigurationEquals('mpc_toolbox', 'do_mpc'),
            package='trajectory_following_ros2',
            executable='coupled_kinematic_do_mpc',
            name='do_mpc_node',
            output='screen',
            parameters=[
                params_file,
                {'use_sim_time': use_sim_time},
                {
                    'robot_frame': robot_frame,
                    'global_frame': global_frame,
                },
                {'control_rate': frequency},
                {'publish_twist_topic': publish_twist_topic},
                {'wheelbase': wheelbase},
                {'ode_type': ode_type},
                {'max_speed': max_speed},
                {'min_speed': min_speed},
                {'max_accel': max_accel},
                {'max_decel': max_decel},
                {'max_steer': max_steer},
                {'min_steer': min_steer},
                {'max_steer_rate': max_steer_rate},
                {'desired_speed': desired_speed},
                {'saturate_inputs': saturate_inputs},
                {'allow_reversing': allow_reversing},
                {'horizon': horizon},
                {'sample_time': sample_time},
                {'prediction_time': prediction_time},
                {'R': R_diagonal},
                {'Rd': Rd_diagonal},
                {'Q': Q_diagonal},
                {'Qf': Qf_diagonal},
                {'scale_cost': scale_cost},
                {'max_iter': max_iterations},
                {'termination_condition': termination_condition},
                {'generate_mpc_model': generate_mpc_model},
                {'build_with_cython': build_with_cython},
                {'model_directory': model_directory},
                # {'stage_cost_type': stage_cost_type},
                # {'terminal_cost_type': terminal_cost_type},
                {'distance_tolerance': distance_tolerance},
                {'speed_tolerance': speed_tolerance},
                {
                    'odom_topic': odom_topic,
                    'ackermann_cmd_topic': ackermann_cmd_topic,
                    'twist_topic': twist_topic,
                    'acceleration_topic': acceleration_topic,
                    'path_topic': path_topic,
                    'speed_topic': speed_topic,
                },
                {'debug_frequency': debug_frequency},
            ],
            arguments=['--ros-args', '--log-level', log_level],
            # remappings=[
            #     ('/waypoint_loader/path', '/trajectory/path'),
            #     ('/waypoint_loader/speed', '/trajectory/speed'),
            # ]
    )

    custom_purepursuit_node = Node(
            package='trajectory_following_ros2',
            executable='purepursuit',
            name=f'purepursuit_node',
            output='screen',
            parameters=[
                {
                    'use_sim_time': use_sim_time,
                    'robot_frame': 'base_link',
                    'global_frame': 'map',
                    # 'file_path': waypoints_csv_path,  # todo
                    # 'path_source': 'file',
                    'control_rate': 20.0,
                    'goal_tolerance': 5.0,
                    'lookahead_distance': 9.0,  # 10.0
                    'min_lookahead': 4.35,
                    'max_lookahead': 15.0,
                    'adaptive_lookahead_gain': 4.0,
                    'use_adaptive_lookahead': False,
                    'wheelbase': 2.87528,
                    'max_steer': 69.99999284118222,
                    'min_steer': -69.99999284118222,
                    'max_steer_rate': 352.9411764706,
                    'max_speed': 9.0,
                    # 'min_speed': '-10.5',
                    'speed_Kp': 2.0,
                    'speed_Ki': 0.2,
                    'speed_Kd': 0.0,
                    'desired_speed': 8.9,  # 8.9
                    'odom_topic': odom_topic,
                    'ackermann_cmd_topic': ackermann_cmd_topic,
                    # f'/ackermann_cmd_{role_name_string}' or f'/carla/{role_name_string}/ackermann_cmd'
                    'twist_topic': twist_topic,  # f'/carla/{role_name_string}/twist'
                    'acceleration_topic': acceleration_topic,  # f'/carla/{role_name_string}/twist'
                    'path_topic': path_topic,
                    'speed_topic': speed_topic,
                    'speedup_first_lookup': True,
                }
            ]
    )

    casadi_mpc_node = None

    load_nodes = GroupAction(
            actions=[
                # PushRosNamespace(LaunchConfiguration('chatter_ns')),
                # SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),
                waypoint_loader_node,
                acados_mpc_node,
                do_mpc_node,
                #custom_purepursuit_node,
                # casadi_mpc_node,
            ])

    # Add the actions to launch all of the mpc nodes
    ld.add_action(load_nodes)

    return ld
