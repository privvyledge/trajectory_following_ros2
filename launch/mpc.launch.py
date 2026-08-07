"""
(optionally) Load waypoints
(optionally) Remap waypoint path topics to the one expected by the nodes
Set directory paths
Set default parameters
Choose MPC type
Remap nav2 path topic to

"""

import os
from typing import List

import numpy as np

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import (Node, PushRosNamespace, SetParametersFromFile, SetParameter,
                                SetRemap)
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    trajectory_following_ros2_pkg_prefix = get_package_share_directory('trajectory_following_ros2')

    # Create the launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_namespace = LaunchConfiguration('use_namespace', default=False)
    namespace = LaunchConfiguration('namespace', default='')
    load_visualizer = LaunchConfiguration('load_visualizer', default=False)
    viz_spawn_viewer = LaunchConfiguration('viz_spawn_viewer', default=True)
    viz_backend = LaunchConfiguration('viz_backend', default='both')
    viz_recording_path = LaunchConfiguration('viz_recording_path', default='')
    viz_serve_web = LaunchConfiguration('viz_serve_web', default=False)
    viz_web_port = LaunchConfiguration('viz_web_port', default=9090)
    viz_web_open_browser = LaunchConfiguration('viz_web_open_browser', default=True)
    viz_actuator_feedback_topic = LaunchConfiguration('viz_actuator_feedback_topic', default='')
    viz_reference_cmd_topic = LaunchConfiguration('viz_reference_cmd_topic', default='')
    params_file = LaunchConfiguration('params_file')
    load_params_from_file = LaunchConfiguration('load_params_from_file', default=True)
    load_params_from_args = LaunchConfiguration('load_params_from_args', default=True)
    log_level = LaunchConfiguration('log_level')
    robot_frame = LaunchConfiguration('robot_frame')
    global_frame = LaunchConfiguration('global_frame')
    waypoint_target_frame = LaunchConfiguration('waypoint_target_frame')
    namespaced_tf = LaunchConfiguration('namespaced_tf')
    frequency = LaunchConfiguration('frequency')
    publish_twist_topic = LaunchConfiguration('publish_twist_topic')
    wheelbase = LaunchConfiguration('wheelbase')
    ode_type = LaunchConfiguration('ode_type')
    use_opti = LaunchConfiguration('use_opti')
    num_obstacles = LaunchConfiguration('num_obstacles')
    obstacle_topic = LaunchConfiguration('obstacle_topic')
    forward_escape_enabled = LaunchConfiguration('forward_escape_enabled')
    forward_escape_speed = LaunchConfiguration('forward_escape_speed')
    merge_obstacle_sources = LaunchConfiguration('merge_obstacle_sources', default=False)
    obstacle_source_topics = LaunchConfiguration('obstacle_source_topics')
    obstacle_source_qos = LaunchConfiguration('obstacle_source_qos')
    obstacle_gate_radius = LaunchConfiguration('obstacle_gate_radius')
    discrete_model_type = LaunchConfiguration('discrete_model_type')
    discrete_integration_method = LaunchConfiguration('discrete_integration_method')
    load_waypoints = LaunchConfiguration('load_waypoints')
    waypoints_csv = LaunchConfiguration('waypoints_csv')
    resample_spacing = LaunchConfiguration('resample_spacing')

    # Constraints
    saturate_input = LaunchConfiguration('saturate_input')
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
    control_type = LaunchConfiguration('control_type')
    horizon = LaunchConfiguration('horizon')
    sample_time = LaunchConfiguration('sample_time')
    prediction_time = LaunchConfiguration('prediction_time')
    R_diagonal = LaunchConfiguration('R_diagonal')
    Rd_diagonal = LaunchConfiguration('Rd_diagonal')
    Q_diagonal = LaunchConfiguration('Q_diagonal')
    Qf_diagonal = LaunchConfiguration('Qf_diagonal')
    slack_weights_input_rate = LaunchConfiguration('slack_weights_input_rate')
    slack_scale_input_rate = LaunchConfiguration('slack_scale_input_rate')
    slack_upper_bound_input_rate = LaunchConfiguration('slack_upper_bound_input_rate')
    scale_cost = LaunchConfiguration('scale_cost')
    max_iterations = LaunchConfiguration('max_iterations')
    termination_condition = LaunchConfiguration('termination_condition')
    stage_cost_type = LaunchConfiguration('stage_cost_type')
    terminal_cost_type = LaunchConfiguration('terminal_cost_type')
    generate_mpc_model = LaunchConfiguration('generate_mpc_model')
    build_with_cython = LaunchConfiguration('build_with_cython')
    code_gen_directory = LaunchConfiguration('code_gen_directory')
    debug_frequency = LaunchConfiguration('debug_frequency')

    # # Trajectory/goal parameters
    distance_tolerance = LaunchConfiguration('distance_tolerance')
    speed_tolerance = LaunchConfiguration('speed_tolerance')
    arclength_index_advance = LaunchConfiguration('arclength_index_advance')
    projection_window = LaunchConfiguration('projection_window')
    solver_log_file = LaunchConfiguration('solver_log_file')

    # #  Topics
    odom_topic = LaunchConfiguration('odom_topic', default="odometry/local")
    ackermann_cmd_topic = LaunchConfiguration('ackermann_cmd_topic', default="drive")
    twist_topic = LaunchConfiguration('twist_topic', default="cmd_vel")
    acceleration_topic = LaunchConfiguration('acceleration_topic', default="accel/local")
    path_topic = LaunchConfiguration('path_topic', default="trajectory/path")
    speed_topic = LaunchConfiguration('speed_topic', default="trajectory/speed")

    # Per-platform / per-backend overlay config (stacked on top of the base file).
    # Stack order (highest wins): weights > platform > base mpc_parameters.yaml.
    # Empty (default) = skip the overlay (legacy single-file behaviour).
    platform = LaunchConfiguration('platform', default='')
    weights = LaunchConfiguration('weights', default='')
    platforms_dir = os.path.join(trajectory_following_ros2_pkg_prefix, 'config', 'platforms')
    weights_dir = os.path.join(trajectory_following_ros2_pkg_prefix, 'config', 'weights')

    # Declare default launch arguments
    config_file_path = os.path.join(trajectory_following_ros2_pkg_prefix, 'config/mpc_parameters.yaml')
    waypoints_csv_path = os.path.join(trajectory_following_ros2_pkg_prefix, 'data/carla_waypoints.csv')
    mpc_model_path = os.path.join(trajectory_following_ros2_pkg_prefix, 'data/mpc')

    # declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true')
    use_namespace_la = DeclareLaunchArgument(
            'use_namespace', default_value=use_namespace,
            description='Use namespace if true. ')
    namespace_la = DeclareLaunchArgument(
            'namespace', default_value=namespace,
            description='Namespace for the nodes')
    params_file_la = DeclareLaunchArgument(
            'params_file',
            default_value=config_file_path,
            description='Path to config file for localization nodes'
    )
    load_params_from_file_la = DeclareLaunchArgument(
            'load_params_from_file',
            default_value=load_params_from_file,
            description='Load params from the parameter file if True.')
    load_params_from_args_la = DeclareLaunchArgument(
            'load_params_from_args',
            default_value=load_params_from_args,
            description='Load params from command line arguments if True. CLIs override YAML.')
    platform_la = DeclareLaunchArgument(
            'platform',
            default_value='',
            description="Platform overlay name (e.g. 'f1tenth', 'carla') -> "
                        'config/platforms/<platform>.yaml. Holds vehicle physical params '
                        '(wheelbase, steer/speed limits, frames, topics). Empty = skip overlay. '
                        'Applied AFTER individual launch args, so it overrides them.')
    weights_la = DeclareLaunchArgument(
            'weights',
            default_value='',
            description="Weight overlay name (e.g. 'f1tenth_casadi', 'carla_do_mpc') -> "
                        'config/weights/<weights>.yaml. Holds Q/R/Rd/Qf + horizon + solver '
                        'config + speed policy. Empty = skip overlay. Highest precedence.')
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
    namespaced_tf_la = DeclareLaunchArgument(
            'namespaced_tf',
            default_value='True',
            description='Under use_namespace, remap /tf and /tf_static onto the namespaced '
                        '<ns>/tf and <ns>/tf_static (the Nav2 convention). tf2 builds its '
                        'listener on the ABSOLUTE /tf, which a PushRosNamespace does NOT '
                        'redirect, so without this a namespaced controller never sees a robot '
                        'that publishes /<ns>/tf. That failure is silent, not loud: '
                        'base_tracker.odom_callback warns once on the failed lookup and then '
                        'tracks the RAW odom pose as though it were already in global_frame, '
                        'so a map-frame route runs healthy-looking but offset by the whole '
                        'map->odom correction. No effect when use_namespace is False (the '
                        'relative name resolves back to /tf). Set False only for a stack that '
                        'namespaces its nodes while keeping one global TF tree.'
    )
    waypoint_target_frame_la = DeclareLaunchArgument(
            'waypoint_target_frame',
            default_value=global_frame,
            description='TF frame waypoint_loader transforms the loaded CSV into, i.e. the '
                        'frame the published path carries. It MUST equal global_frame: the '
                        'controller awaits a TF from the path frame to global_frame before it '
                        'will accept a path, and tf2 subscribes to the ABSOLUTE /tf, so under '
                        'use_namespace that lookup cannot see a namespaced /<ns>/tf. Defaults '
                        'to global_frame so the two cannot silently disagree; the loader\'s own '
                        'default is "map", which does not match this file\'s "odom" default. '
                        'Set both to "map" for a run with a global localizer.'
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
    use_opti_la = DeclareLaunchArgument(
            'use_opti',
            default_value='False',
            description='CasADi only: use the Opti-stack formulation (KinematicMPCCasadiOpti) '
                        'instead of the function-based NLP. Restart-only. Applied AFTER the '
                        'platform/weights overlays so the launch arg is authoritative (it is a '
                        'formulation selector, not a weight).'
    )
    num_obstacles_la = DeclareLaunchArgument(
            'num_obstacles',
            default_value='0',
            description='Number of obstacle keep-out constraints to bake into the OCP. '
                        'Restart-only (sizes the solver). Applied AFTER the platform/weights '
                        'overlays so the launch arg is authoritative — it is a structural '
                        'launch-time choice, not a weight, so a weights file must not silently '
                        'override the requested count. Obstacle runs also need '
                        'generate_mpc_model:=true.'
    )
    obstacle_topic_la = DeclareLaunchArgument(
            'obstacle_topic',
            default_value='fake_obstacles/object_array',
            description='ObjectArray topic the controller constrains against. Defaults to '
                        'the fake_obstacle_publisher topic; point it at a real perception '
                        'feed to run against live detections. For CARLA use '
                        '/carla/merged_obstacles (dynamic actors merged with static level '
                        'geometry: light poles, sign posts, traffic-light masts); '
                        '/carla/ego_vehicle/objects is the dynamic-actors-only predecessor. '
                        'Use an EGO-SCOPED feed — a world-scoped '
                        'one reports the ego vehicle itself and would make the car its own '
                        'keep-out. Detections must already be in global_frame; they are not '
                        'TF-transformed. Applied AFTER the platform/weights overlays so the '
                        'launch arg is authoritative.'
    )
    forward_escape_enabled_la = DeclareLaunchArgument(
            'forward_escape_enabled', default_value='False',
            description='Enable bounded forward-only recovery from a stationary '
                        'detour-bound stop. Defaults off; CARLA obstacle overlays opt in.')
    forward_escape_speed_la = DeclareLaunchArgument(
            'forward_escape_speed', default_value='1.0',
            description='Maximum forward speed (m/s) of the bounded escape reference.')
    merge_obstacle_sources_la = DeclareLaunchArgument(
            'merge_obstacle_sources',
            default_value='False',
            description='Launch the obstacle_aggregator node, which merges the '
                        'obstacle_source_topics feeds into one and gates them to '
                        'obstacle_gate_radius around the vehicle. Point obstacle_topic at '
                        'its output (obstacles/object_array by default). Off by default, so '
                        'single-source setups are untouched.'
    )
    obstacle_source_topics_la = DeclareLaunchArgument(
            'obstacle_source_topics',
            default_value="['/carla/merged_obstacles']",
            description='ObjectArray topics the aggregator merges (Python-list syntax). '
                        'Only used when merge_obstacle_sources:=True.'
    )
    obstacle_source_qos_la = DeclareLaunchArgument(
            'obstacle_source_qos',
            default_value="['volatile']",
            description="Per-source durability, parallel to obstacle_source_topics: "
                        "'volatile' or 'transient_local'. A transient_local source is "
                        'treated as latched static content — it is picked up on late join '
                        'and never aged out by source_timeout.'
    )
    obstacle_gate_radius_la = DeclareLaunchArgument(
            'obstacle_gate_radius',
            default_value='0.0',
            description='Metres around the vehicle the aggregator forwards obstacles from; '
                        '0.0 = forward everything. Sized from the worst-case horizon reach. '
                        'This is what keeps a map-scale feed (CARLA publishes ~1235 objects) '
                        'from being deserialized in the controller process, where it '
                        'saturates the GIL and dilates the control tick.'
    )
    discrete_model_type_la = DeclareLaunchArgument(
            'discrete_model_type',
            default_value='nonlinear',
            description='CasADi discrete formulation model form: '
                        "'nonlinear' (full ODE, default) or 'ltv' "
                        '(forward-Jacobian/Taylor linearization). '
                        'Only used for discrete ode_type values.'
    )
    discrete_integration_method_la = DeclareLaunchArgument(
            'discrete_integration_method',
            default_value='rk4',
            description='CasADi discrete-model discretization scheme: '
                        "'rk4' (default) or 'euler'. Only these explicit "
                        "schemes work for the MPC NLP. Ignored when "
                        "discrete_model_type='ltv'."
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

    resample_spacing_la = DeclareLaunchArgument(
            'resample_spacing',
            default_value='0.0',
            description='waypoint_loader: resample the route at this uniform arc-length '
                        'spacing (m) before smoothing; 0.0 = publish the recorded sampling '
                        'unchanged. Smoothing alone does NOT even out spacing, and a gap '
                        'wider than the reference projection can step over stalls the '
                        'reference index.'
    )

    saturate_input_la = DeclareLaunchArgument(
            'saturate_input',
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
            description='MPC toolbox to use. acados, do_mpc, casadi.',
            choices=['acados', 'casadi', 'do_mpc', 'none']
    )
    control_type_la = DeclareLaunchArgument(
            'control_type',
            default_value='mpc',
            description='Type of control to use. mpc, purepursuit',
            choices=['mpc', 'purepursuit']
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
    slack_weights_input_rate_la = DeclareLaunchArgument(
            'slack_weights_input_rate',
            default_value='[1.0, 1.0]',
            description='List containing the diagonal for the slack weights input rate.'
    )
    slack_scale_input_rate_la = DeclareLaunchArgument(
            'slack_scale_input_rate',
            default_value='[1.0, 1.0]',
            description='List containing the diagonal for the slack scale input rate.'
    )
    slack_upper_bound_input_rate_la = DeclareLaunchArgument(
            'slack_upper_bound_input_rate',
            default_value=f'[{np.inf}, {np.inf}]',
            description='List containing the diagonal for the slack upper bound input rate.'
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
            default_value='EXTERNAL',
            description='Stage cost type. EXTERNAL (default), NONLINEAR_LS, LINEAR_LS. '
                        'Only EXTERNAL carries the Rd input-rate penalty and enables obstacles.'
    )
    terminal_cost_type_la = DeclareLaunchArgument(
            'terminal_cost_type',
            default_value='EXTERNAL',
            description='Terminal cost type. EXTERNAL (default), NONLINEAR_LS, LINEAR_LS.'
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
    code_gen_directory_la = DeclareLaunchArgument(
            'code_gen_directory',
            default_value=mpc_model_path,
            description='Directory for generated/compiled solver code: acados C-code + '
                        'compiled model, or CasADi JIT artifacts. Unified across backends.'
    )
    distance_tolerance_la = DeclareLaunchArgument(
            'distance_tolerance',
            default_value='0.2',
            description='Distance tolerance for arrival.'
    )
    solver_log_file_la = DeclareLaunchArgument(
            'solver_log_file',
            default_value='',
            description='Append one CSV row per solve to this file (empty = disabled). '
                        'Same argument closed_loop_sim.launch.py accepts; it is the '
                        'primary per-tick diagnostic for a hardware run, where there is '
                        'no simulator log to fall back on.'
    )
    arclength_index_advance_la = DeclareLaunchArgument(
            'arclength_index_advance',
            default_value='True',
            description='Reference-index advance mode. True = along-track '
                        '(arc-length) projection: the index advances with longitudinal '
                        'progress even when the vehicle is held laterally off the line '
                        '(obstacle swerve), never freezing at a standoff. False = legacy '
                        'Euclidean distance-gate.'
    )
    projection_window_la = DeclareLaunchArgument(
            'projection_window',
            default_value='5.0',
            description='Forward arc-length span (m) of the arc-length projection '
                        'window. Kept short so the projection cannot leap to '
                        'end-of-path points sitting near the start on a closed loop; '
                        'must be < loop length and > one tick of travel.'
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

    load_visualizer_la = DeclareLaunchArgument(
            'load_visualizer',
            default_value='False',
            description='Launch the Rerun trajectory visualizer node. Requires rerun-sdk (pip install rerun-sdk).'
    )
    viz_spawn_viewer_la = DeclareLaunchArgument(
            'viz_spawn_viewer',
            default_value='True',
            description='Spawn the Rerun viewer process automatically when the visualizer starts.'
    )
    viz_backend_la = DeclareLaunchArgument(
            'viz_backend',
            default_value='both',
            description="Visualization backend: 'rerun' | 'native' (matplotlib) | 'both'. Pick by "
                        'where the pixels are drawn, not by habit. On a headless machine choose '
                        "'rerun': the native backend has no display to render into, warns "
                        '"Starting a Matplotlib GUI outside of the main thread will likely fail", '
                        'and over ssh renders through X11 forwarding at the cost of a full core.'
    )
    viz_recording_path_la = DeclareLaunchArgument(
            'viz_recording_path',
            default_value='',
            description='Path to save a .rrd Rerun recording file. Empty = no file saved.'
    )
    viz_serve_web_la = DeclareLaunchArgument(
            'viz_serve_web',
            default_value='False',
            description='Serve the Rerun web viewer (WebGPU in the browser) instead of the '
                        'native viewer. Renders in-browser, bypassing the native Vulkan path '
                        'broken on WSL2. Open http://localhost:<viz_web_port> in a browser. '
                        'Single-sink on rerun <0.23, so mutually exclusive with viz_recording_path.'
    )
    viz_web_port_la = DeclareLaunchArgument(
            'viz_web_port',
            default_value='9090',
            description='HTTP port for the Rerun web viewer (used when viz_serve_web:=true).'
    )
    viz_web_open_browser_la = DeclareLaunchArgument(
            'viz_web_open_browser',
            default_value='True',
            description='Auto-open the system browser at the web viewer URL (used when viz_serve_web:=true).'
    )
    viz_actuator_feedback_topic_la = DeclareLaunchArgument(
            'viz_actuator_feedback_topic',
            default_value='',
            description='AckermannDriveStamped topic carrying actual hardware actuator state (optional).'
    )
    viz_reference_cmd_topic_la = DeclareLaunchArgument(
            'viz_reference_cmd_topic',
            default_value='',
            description='AckermannDriveStamped topic carrying reference commands from a driver dataset (optional).'
    )

    # Create Launch Description
    ld = LaunchDescription(
            [declare_use_sim_time_cmd, use_namespace_la, namespace_la, params_file_la,
             load_params_from_file_la, load_params_from_args_la,
             platform_la, weights_la,
             robot_frame_la, global_frame_la, waypoint_target_frame_la, namespaced_tf_la,
             frequency_la, publish_twist_topic_la, wheelbase_la, ode_type_la,
             use_opti_la, num_obstacles_la, obstacle_topic_la,
             forward_escape_enabled_la, forward_escape_speed_la,
             merge_obstacle_sources_la, obstacle_source_topics_la,
             obstacle_source_qos_la, obstacle_gate_radius_la,
             discrete_model_type_la, discrete_integration_method_la,
             load_waypoints_la, waypoints_csv_la, resample_spacing_la,
             saturate_input_la, allow_reversing_la, max_speed_la, min_speed_la, max_accel_la, max_decel_la,
             max_steer_la, min_steer_la, max_steer_rate_la, desired_speed_la,
             mpc_toolbox_la, control_type_la, horizon_la, sample_time_la, prediction_time_la,
             R_diagonal_la, Rd_diagonal_la, Q_diagonal_la, Qf_diagonal_la,
             slack_weights_input_rate_la, slack_scale_input_rate_la, slack_upper_bound_input_rate_la,
             scale_cost_la,
             max_iterations_la, termination_condition_la,
             stage_cost_type_la, terminal_cost_type_la,
             generate_mpc_model_la, build_with_cython_la, code_gen_directory_la,
             distance_tolerance_la, speed_tolerance_la,
             arclength_index_advance_la, projection_window_la, solver_log_file_la,
             declare_log_level_cmd,
             odom_topic_la, ackermann_cmd_topic_la, twist_topic_la, acceleration_topic_la, path_topic_la,
             speed_topic_la, debug_frequency_la,
             load_visualizer_la, viz_spawn_viewer_la, viz_backend_la,
             viz_recording_path_la,
             viz_serve_web_la, viz_web_port_la, viz_web_open_browser_la,
             viz_actuator_feedback_topic_la, viz_reference_cmd_topic_la]
    )

    common_parameters = {
        # 'use_sim_time': use_sim_time,
        # 'robot_frame': robot_frame,
        # 'global_frame': global_frame,
        # 'control_rate': frequency,
        # 'publish_twist_topic': publish_twist_topic,
        # 'wheelbase': wheelbase,
        # 'ode_type': ode_type,
        # 'max_speed': max_speed,
        # 'min_speed': min_speed,
        # 'max_accel': max_accel,
        # 'max_decel': max_decel,
        # 'max_steer': max_steer,
        # 'min_steer': min_steer,
        # 'max_steer_rate': max_steer_rate,
        # 'desired_speed': desired_speed,
        # 'saturate_input': saturate_input,
        # 'allow_reversing': allow_reversing,
        # 'horizon': horizon,
        # 'sample_time': sample_time,
        # 'prediction_time': prediction_time,
        'slack_weights_input_rate': slack_weights_input_rate,
        'slack_scale_input_rate': slack_scale_input_rate,
        'slack_upper_bound_input_rate': slack_upper_bound_input_rate,
        # NOTE: the cost weights are NOT set here. A node-level
        # `parameters=` entry outranks every SetParametersFromFile in the group, so
        # putting them here made `weights:=<file>` silently dead for Q/R/Rd/Qf while
        # the same file's horizon/max_iter/solver entries applied normally — the
        # controller ran the launch-arg defaults (Q = [1, 1, 1, 0.01]) on hardware
        # while sim, which uses closed_loop_sim.launch.py and no such dict, ran the
        # tuned values. They are SetParameter entries below instead, placed before
        # the overlays so the documented stack (weights > platform > args > base)
        # actually holds.
        # 'scale_cost': scale_cost,
        # 'max_iter': max_iterations,
        # 'termination_condition': termination_condition,
        # 'generate_mpc_model': generate_mpc_model,
        # 'build_with_cython': build_with_cython,
        # 'code_gen_directory': code_gen_directory,
        # 'stage_cost_type': stage_cost_type,
        # 'terminal_cost_type': terminal_cost_type,
        # 'distance_tolerance': distance_tolerance,
        # 'speed_tolerance': speed_tolerance,
        # 'odom_topic': odom_topic,
        # 'ackermann_cmd_topic': ackermann_cmd_topic,
        # 'twist_topic': twist_topic,
        # 'acceleration_topic': acceleration_topic,
        # 'path_topic': path_topic,
        # 'speed_topic': speed_topic,
        # 'debug_frequency': debug_frequency
    }  # use Set Parameter and GroupAction below

    # Load Nodes
    waypoint_loader_node = Node(
            condition=IfCondition(load_waypoints),
            package='trajectory_following_ros2',
            executable='waypoint_loader',
            name='waypoint_loader_node',
            output='screen',
            # parameters=[
            #     {'use_sim_time': use_sim_time},
            #     {'file_path': waypoints_csv},
            # ],
            arguments=['--ros-args', '--log-level', log_level],
            # remappings=[
            #     ('waypoint_loader/path', path_topic),
            #     ('waypoint_loader/speed', speed_topic),
            #     ('waypoint_loader/markers', 'trajectory/markers'),
            # ]
    )

    acados_mpc_node = Node(
            condition=LaunchConfigurationEquals('mpc_toolbox', 'acados'),
            package='trajectory_following_ros2',
            executable='coupled_kinematic_acados',
            name='acados_mpc_node',
            output='screen',
            parameters=[
                # params_file,
                common_parameters,
            ],
            arguments=['--ros-args', '--log-level', log_level],
            # remappings=[
            #     ('/waypoint_loader/path', '/trajectory/path'),
            #     ('/waypoint_loader/speed', '/trajectory/speed'),
            # ]
    )

    casadi_mpc_node = Node(
            condition=LaunchConfigurationEquals('mpc_toolbox', 'casadi'),
            package='trajectory_following_ros2',
            executable='coupled_kinematic_casadi',
            name='casadi_mpc_node',
            output='screen',
            parameters=[
                # params_file,
                common_parameters,
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
                # params_file,
                common_parameters,
            ],
            arguments=['--ros-args', '--log-level', log_level],
            # remappings=[
            #     ('/waypoint_loader/path', '/trajectory/path'),
            #     ('/waypoint_loader/speed', '/trajectory/speed'),
            # ]
    )

    custom_purepursuit_node = Node(
            condition=LaunchConfigurationEquals('control_type', 'purepursuit'),
            package='trajectory_following_ros2',
            executable='purepursuit',
            name='purepursuit_node',
            output='screen',
            parameters=[
                # params_file,
                common_parameters,
            ],
    )

    # Merges N obstacle feeds into the single topic the controller subscribes to, and
    # gates them to obstacle_gate_radius. Its own process, deliberately: the per-message
    # deserialization of a map-scale feed is what starves the control loop's GIL.
    obstacle_aggregator_node = Node(
            condition=IfCondition(merge_obstacle_sources),
            package='trajectory_following_ros2',
            executable='obstacle_aggregator',
            name='obstacle_aggregator',
            output='screen',
            parameters=[{
                'input_topics': ParameterValue(obstacle_source_topics, value_type=List[str]),
                'input_qos': ParameterValue(obstacle_source_qos, value_type=List[str]),
                # The controller's feed IS this node's output, so the two can never be
                # pointed at different topics by accident. Do not list obstacle_topic in
                # obstacle_source_topics — that would feed the node its own output.
                'output_topic': ParameterValue(obstacle_topic, value_type=str),
                'ego_gate_radius': ParameterValue(obstacle_gate_radius, value_type=float),
                'odom_topic': ParameterValue(odom_topic, value_type=str),
            }],
            arguments=['--ros-args', '--log-level', log_level],
    )

    visualizer_node = Node(
            condition=IfCondition(load_visualizer),
            package='trajectory_following_ros2',
            executable='trajectory_visualizer',
            name='trajectory_visualizer_node',
            output='screen',
            parameters=[{
                'viz_backend': ParameterValue(viz_backend, value_type=str),
                'spawn_viewer': ParameterValue(viz_spawn_viewer, value_type=bool),
                'recording_path': ParameterValue(viz_recording_path, value_type=str),
                'serve_web': ParameterValue(viz_serve_web, value_type=bool),
                'web_port': ParameterValue(viz_web_port, value_type=int),
                'web_open_browser': ParameterValue(viz_web_open_browser, value_type=bool),
                'actuator_feedback_topic': ParameterValue(
                    viz_actuator_feedback_topic, value_type=str),
                'reference_cmd_topic': ParameterValue(viz_reference_cmd_topic, value_type=str),
            }],
            # Visualization is best-effort: keep its CPU priority below the control
            # loop so full-figure redraws cannot starve the controller tick.
            prefix='nice -n 10',
    )

    load_nodes = GroupAction(
            actions=[
                PushRosNamespace(
                        condition=IfCondition(use_namespace),
                        namespace=namespace
                ),
                # Follow the namespace onto the TF topics. tf2 constructs its listener on
                # the ABSOLUTE /tf, which PushRosNamespace does not redirect, so without
                # these two a namespaced controller silently never receives the transform
                # a robot publishes on /<ns>/tf. Relative names, so they resolve to
                # /<ns>/tf under the push and back to /tf without it -- which is why the
                # condition is on namespaced_tf alone and this is a no-op unnamespaced.
                # Must precede every node in the group: SetRemap applies to what follows.
                SetRemap(src='/tf', dst='tf', condition=IfCondition(namespaced_tf)),
                SetRemap(src='/tf_static', dst='tf_static', condition=IfCondition(namespaced_tf)),
                # Set common parameters. todo: test passing a list of names and values instead of separate SetParameter
                SetParametersFromFile(params_file, condition=IfCondition(load_params_from_file)),
                SetParameter(name='use_sim_time', value=use_sim_time),
                SetParameter(name='robot_frame', value=robot_frame, condition=IfCondition(load_params_from_args)),
                SetParameter(name='global_frame', value=global_frame, condition=IfCondition(load_params_from_args)),
                SetParameter(name='control_rate', value=frequency, condition=IfCondition(load_params_from_args)),
                SetParameter(name='publish_twist_topic', value=publish_twist_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='wheelbase', value=wheelbase, condition=IfCondition(load_params_from_args)),
                SetParameter(name='ode_type', value=ode_type, condition=IfCondition(load_params_from_args)),
                SetParameter(name='discrete_model_type', value=discrete_model_type, condition=IfCondition(load_params_from_args)),
                SetParameter(name='discrete_integration_method', value=discrete_integration_method, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_speed', value=max_speed, condition=IfCondition(load_params_from_args)),
                SetParameter(name='min_speed', value=min_speed, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_accel', value=max_accel, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_decel', value=max_decel, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_steer', value=max_steer, condition=IfCondition(load_params_from_args)),
                SetParameter(name='min_steer', value=min_steer, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_steer_rate', value=max_steer_rate, condition=IfCondition(load_params_from_args)),
                SetParameter(name='desired_speed', value=desired_speed, condition=IfCondition(load_params_from_args)),
                SetParameter(name='saturate_input', value=saturate_input, condition=IfCondition(load_params_from_args)),
                SetParameter(name='allow_reversing', value=allow_reversing, condition=IfCondition(load_params_from_args)),
                SetParameter(name='horizon', value=horizon, condition=IfCondition(load_params_from_args)),
                SetParameter(name='sample_time', value=sample_time, condition=IfCondition(load_params_from_args)),
                SetParameter(name='prediction_time', value=prediction_time, condition=IfCondition(load_params_from_args)),
                # NOTE: use_opti is set AFTER the overlays (see below) so the launch arg wins;
                # it is a formulation selector, not a weight, and must not be overridable here.
                SetParameter(name='solver_type', value='quad', condition=IfCondition(load_params_from_args)),
                SetParameter(name='solver', value='qrqp', condition=IfCondition(load_params_from_args)),
                SetParameter(name='normalize_yaw_error', value=True, condition=IfCondition(load_params_from_args)),
                # Cost weights and input-rate slacks. These sit here, not in a
                # node-level `parameters=` dict, so a weights/platform overlay can
                # still override them (a node-level entry outranks every overlay).
                SetParameter(name='R', value=ParameterValue(R_diagonal, value_type=List[float]),
                             condition=IfCondition(load_params_from_args)),
                SetParameter(name='Rd', value=ParameterValue(Rd_diagonal, value_type=List[float]),
                             condition=IfCondition(load_params_from_args)),
                SetParameter(name='Q', value=ParameterValue(Q_diagonal, value_type=List[float]),
                             condition=IfCondition(load_params_from_args)),
                SetParameter(name='Qf', value=ParameterValue(Qf_diagonal, value_type=List[float]),
                             condition=IfCondition(load_params_from_args)),
                # slack_* stay node-level: no platform/weights file sets them, and the
                # slack_upper_bound default is [inf, inf], which SetParameter cannot
                # coerce from its string form.
                SetParameter(name='slack_objective_is_quadratic', value=False, condition=IfCondition(load_params_from_args)),
                SetParameter(name='scale_cost', value=scale_cost, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_iter', value=max_iterations, condition=IfCondition(load_params_from_args)),
                SetParameter(name='termination_condition', value=termination_condition, condition=IfCondition(load_params_from_args)),
                SetParameter(name='generate_mpc_model', value=generate_mpc_model, condition=IfCondition(load_params_from_args)),
                SetParameter(name='build_with_cython', value=build_with_cython, condition=IfCondition(load_params_from_args)),
                SetParameter(name='code_gen_directory', value=code_gen_directory, condition=IfCondition(load_params_from_args)),
                SetParameter(name='stage_cost_type', value=stage_cost_type, condition=IfCondition(load_params_from_args)),
                SetParameter(name='terminal_cost_type', value=terminal_cost_type, condition=IfCondition(load_params_from_args)),
                SetParameter(name='distance_tolerance', value=distance_tolerance, condition=IfCondition(load_params_from_args)),
                SetParameter(name='speed_tolerance', value=speed_tolerance, condition=IfCondition(load_params_from_args)),
                SetParameter(name='arclength_index_advance', value=arclength_index_advance, condition=IfCondition(load_params_from_args)),
                SetParameter(name='projection_window', value=projection_window, condition=IfCondition(load_params_from_args)),
                # Skipped entirely when empty rather than passed as an empty string: rcl
                # rejects a bare `-p solver_log_file:=` ("Couldn't parse parameter override
                # rule") and the node dies before it starts, so the disabled-by-default case
                # took down every launch that did not name a log file. The node's own default
                # is '' == disabled, so not setting it is the correct way to express that.
                SetParameter(name='solver_log_file', value=solver_log_file,
                             condition=IfCondition(PythonExpression(
                                     ["'", load_params_from_args, "'.lower() == 'true' and '",
                                      solver_log_file, "' != ''"]))),
                SetParameter(name='odom_topic', value=odom_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='ackermann_cmd_topic', value=ackermann_cmd_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='twist_topic', value=twist_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='acceleration_topic', value=acceleration_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='path_topic', value=path_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='speed_topic', value=speed_topic, condition=IfCondition(load_params_from_args)),
                SetParameter(name='marker_topic', value='trajectory/markers', condition=IfCondition(load_params_from_args)),
                SetParameter(name='debug_frequency', value=debug_frequency, condition=IfCondition(load_params_from_args)),

                # PurePursuit parameters
                SetParameter(name='goal_tolerance', value=distance_tolerance, condition=IfCondition(load_params_from_args)),
                SetParameter(name='lookahead_distance', value=9.0, condition=IfCondition(load_params_from_args)),
                SetParameter(name='min_lookahead', value=4.35, condition=IfCondition(load_params_from_args)),
                SetParameter(name='max_lookahead', value=15.0, condition=IfCondition(load_params_from_args)),
                SetParameter(name='use_adaptive_lookahead', value=False, condition=IfCondition(load_params_from_args)),
                SetParameter(name='adaptive_lookahead_gain', value=4.0, condition=IfCondition(load_params_from_args)),
                SetParameter(name='speed_Kp', value=2.0, condition=IfCondition(load_params_from_args)),
                SetParameter(name='speed_Ki', value=0.2, condition=IfCondition(load_params_from_args)),
                SetParameter(name='speed_Kd', value=0.0, condition=IfCondition(load_params_from_args)),
                SetParameter(name='speedup_first_lookup', value=True, condition=IfCondition(load_params_from_args)),

                # Obstacle Avoidance parameters
                # NOTE: num_obstacles is set AFTER the overlays (see below) so the launch
                # arg wins; it must not be pinned here or in a weights file.
                SetParameter(name='ego_radius', value=1.0, condition=IfCondition(load_params_from_args)),
                # obstacle_topic is NOT set here — it is applied AFTER the overlays (see
                # below) so the launch arg wins, exactly like num_obstacles. It used to be
                # pinned to the fake-publisher topic at this point with no launch argument
                # at all, which made a real perception feed (e.g. a CARLA ros-bridge
                # /carla/<role>/objects) unreachable from the launch line.
                SetParameter(name='obstacle_collision_avoidance_method', value="euclidean", condition=IfCondition(load_params_from_args)),
                SetParameter(name='forward_escape_enabled',
                             value=ParameterValue(forward_escape_enabled, value_type=bool),
                             condition=IfCondition(load_params_from_args)),
                SetParameter(name='forward_escape_speed',
                             value=ParameterValue(forward_escape_speed, value_type=float),
                             condition=IfCondition(load_params_from_args)),

                # Waypoint Parameters
                SetParameter(name='file_path', value=waypoints_csv, condition=IfCondition(load_params_from_args)),
                # Only waypoint_loader declares target_frame_id; the controllers do not
                # auto-declare from overrides, so this is a no-op for them.
                SetParameter(name='target_frame_id', value=waypoint_target_frame,
                             condition=IfCondition(load_params_from_args)),
                # Only waypoint_loader declares this one too.
                SetParameter(name='resample_spacing',
                             value=ParameterValue(resample_spacing, value_type=float),
                             condition=IfCondition(load_params_from_args)),

                # # Remap common topics
                # SetRemap(src='trajectory/path', dst=path_topic),
                # SetRemap(src='trajectory/speed', dst=speed_topic),
                # SetRemap(src='odometry/local', dst=odom_topic),
                # SetRemap(src='accel/local', dst=acceleration_topic),
                # SetRemap(src='drive', dst=ackermann_cmd_topic),
                # SetRemap(src='cmd_vel', dst=twist_topic),
                # SetRemap(src='fake_obstacles/object_array', dst='fake_obstacles/object_array'),
                # SetRemap(src='mpc/des_steer', dst='mpc/des_steer'),

                # Per-platform / per-backend overlays (applied LAST so they override the
                # base file AND the individual launch-arg defaults above).
                # Stack (highest wins): weights > platform > args > base.
                SetParametersFromFile(
                        [platforms_dir, '/', platform, '.yaml'],
                        condition=IfCondition(PythonExpression(["'", platform, "' != ''"]))),
                SetParametersFromFile(
                        [weights_dir, '/', weights, '.yaml'],
                        condition=IfCondition(PythonExpression(["'", weights, "' != ''"]))),

                # Formulation selector applied LAST so the `use_opti:=...` launch arg is
                # authoritative over any overlay (mirrors closed_loop_sim.launch.py).
                SetParameter(name='use_opti', value=use_opti, condition=IfCondition(load_params_from_args)),
                # num_obstacles sizes the OCP (restart-only). Also applied LAST so an
                # explicit `num_obstacles:=N` wins over a weights file that would otherwise
                # pin the count (value_type=int coerces the launch-arg string to integer).
                SetParameter(name='num_obstacles',
                             value=ParameterValue(num_obstacles, value_type=int),
                             condition=IfCondition(load_params_from_args)),
                # Which topic the CONTROLLER takes detections from. Also applied LAST: the
                # feed is an explicit launch choice and a weights/platform file must not
                # silently redirect it. Point it at an ego-scoped perception topic to run
                # against live detections; a world-scoped one reports the ego itself and
                # would make the vehicle its own keep-out.
                SetParameter(name='obstacle_topic', value=obstacle_topic,
                             condition=IfCondition(load_params_from_args)),

                # Load nodes
                waypoint_loader_node,
                obstacle_aggregator_node,
                acados_mpc_node,
                casadi_mpc_node,
                do_mpc_node,
                custom_purepursuit_node,
                visualizer_node,
            ])

    # Add the actions to launch all of the mpc nodes
    ld.add_action(load_nodes)

    return ld
