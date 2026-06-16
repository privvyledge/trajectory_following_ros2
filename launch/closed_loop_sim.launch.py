"""
Self-contained closed-loop simulation + visualization test.

Brings up, in one shot, a hardware-free loop for exercising a controller and the
visualizer backends:

  1. static_transform_publisher  map -> odom (identity)
       The waypoint_loader publishes the Path in `map`; the controller's
       `global_frame` is `odom`, so the path frame differs from the global
       frame and the controller performs an async TF lookup (map -> odom).
       This static broadcaster is what that lookup resolves against.
  2. waypoint_loader             loads `waypoints_csv` (in the `map` frame),
       remapped onto the controller's `trajectory/path` + `trajectory/speed`.
  3. do-mpc simulator            closes the loop: consumes `drive`, publishes
       odometry on `odometry/local` and broadcasts odom -> base_link.
  4. coupled_kinematic_casadi    the controller (defaults to NLP + IPOPT).
  5. trajectory_visualizer       native and/or rerun visualization backends.

The controller loads config/mpc_parameters.yaml for the tuned cost weights and
speed/curvature policy, then layers the explicit solver overrides below on top
(solver_type=nlp, solver=ipopt, max_iter). Node-default weights alone (Rd=[10,100])
over-penalize steering rate and diverge within ~10% of a lap, so the YAML is loaded.

Example:
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py viz_backend:=both

  # WSL2 Vulkan workaround #1 (current): force the lavapipe software ICD for the
  # native viewer.
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py viz_backend:=rerun \
      vulkan_icd:=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

  # WSL2 Vulkan workaround #2 (browser/WebGPU): serve the Rerun web viewer — no
  # native Vulkan involved. Open http://localhost:9090 in a (Windows) browser.
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py viz_backend:=rerun \
      viz_serve_web:=true

  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      solver_type:=quad solver:=qrqp max_iter:=30 viz_backend:=rerun
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_prefix = get_package_share_directory('trajectory_following_ros2')
    default_waypoints = os.path.join(pkg_prefix, 'data', 'waypoints.csv')

    # ---- Launch configurations ----------------------------------------------
    waypoints_csv = LaunchConfiguration('waypoints_csv')
    global_frame = LaunchConfiguration('global_frame')
    robot_frame = LaunchConfiguration('robot_frame')
    map_frame = LaunchConfiguration('map_frame')
    odom_topic = LaunchConfiguration('odom_topic')

    ode_type = LaunchConfiguration('ode_type')
    solver_type = LaunchConfiguration('solver_type')
    solver = LaunchConfiguration('solver')
    max_iter = LaunchConfiguration('max_iter')

    viz_backend = LaunchConfiguration('viz_backend')
    viz_spawn_viewer = LaunchConfiguration('viz_spawn_viewer')
    viz_recording_path = LaunchConfiguration('viz_recording_path')
    viz_video_path = LaunchConfiguration('viz_video_path')
    viz_serve_web = LaunchConfiguration('viz_serve_web')
    viz_web_port = LaunchConfiguration('viz_web_port')
    viz_web_open_browser = LaunchConfiguration('viz_web_open_browser')
    vulkan_icd = LaunchConfiguration('vulkan_icd')

    initial_x = LaunchConfiguration('initial_x')
    initial_y = LaunchConfiguration('initial_y')
    initial_yaw = LaunchConfiguration('initial_yaw')
    initial_speed = LaunchConfiguration('initial_speed')
    step_on_command = LaunchConfiguration('step_on_command')
    simulator = LaunchConfiguration('simulator')

    # ---- Declare launch arguments -------------------------------------------
    declare_args = [
        DeclareLaunchArgument(
            'waypoints_csv', default_value=default_waypoints,
            description='Absolute path to the waypoints CSV (in the `map` frame).'),
        DeclareLaunchArgument(
            'global_frame', default_value='odom',
            description="Controller + simulator global frame."),
        DeclareLaunchArgument(
            'robot_frame', default_value='base_link',
            description='Vehicle body frame.'),
        DeclareLaunchArgument(
            'map_frame', default_value='map',
            description='Frame the waypoints are expressed in (parent of the static TF).'),
        DeclareLaunchArgument(
            'odom_topic', default_value='odometry/local',
            description='Odometry topic the simulator publishes and the controller/viz consume.'),
        DeclareLaunchArgument(
            'ode_type', default_value='discrete_kinematic_coupled',
            description='CasADi formulation: discrete_kinematic_coupled | continuous_kinematic_coupled | ...'),
        DeclareLaunchArgument(
            'solver_type', default_value='nlp',
            description='CasADi solver_type: nlp | quad | conic.'),
        DeclareLaunchArgument(
            'solver', default_value='ipopt',
            description='CasADi solver: ipopt | qrqp | osqp.'),
        DeclareLaunchArgument(
            'max_iter', default_value='200',
            description='Iteration budget. >=100 for IPOPT (node default 15 is too low for IPOPT).'),
        DeclareLaunchArgument(
            'viz_backend', default_value='native',
            description="Visualization backend: native | rerun | both."),
        DeclareLaunchArgument(
            'viz_spawn_viewer', default_value='true',
            description='Spawn the rerun viewer window (set false on headless WSL; use viz_recording_path).'),
        DeclareLaunchArgument(
            'viz_recording_path', default_value='',
            description='Optional .rrd output path for the rerun backend (e.g. /tmp/traj.rrd). '
                        'On rerun <0.23 a live viewer and a recording cannot run at once '
                        '(single-sink); the recording is kept and the viewer skipped. For a '
                        'live view while recording, set viz_spawn_viewer:=false and open the '
                        'file in a separate viewer (it tails as it grows): rerun /tmp/traj.rrd. '
                        'rerun >=0.23 tees both automatically.'),
        DeclareLaunchArgument(
            'viz_video_path', default_value='',
            description='Optional video output path for the native (matplotlib) backend '
                        '(e.g. /tmp/traj.mp4 via ffmpeg, or /tmp/traj.gif via Pillow).'),
        DeclareLaunchArgument(
            'viz_serve_web', default_value='false',
            description='Serve the Rerun web viewer (WebGPU in the browser) instead of '
                        'spawning the native viewer. Renders in-browser, bypassing the '
                        'native Vulkan path that is broken on WSL2 — no vulkan_icd '
                        'workaround needed. Open http://localhost:<viz_web_port> in a '
                        'browser. Requires viz_backend:=rerun (or both).'),
        DeclareLaunchArgument(
            'viz_web_port', default_value='9090',
            description='HTTP port for the Rerun web viewer (used when viz_serve_web:=true).'),
        DeclareLaunchArgument(
            'viz_web_open_browser', default_value='true',
            description='Auto-open the system browser at the web viewer URL '
                        '(used when viz_serve_web:=true).'),
        DeclareLaunchArgument(
            'vulkan_icd', default_value='',
            description='Path to a Vulkan ICD JSON to force for the rerun viewer. '
                        'Empty = system default. On WSL2 the dzn adapter lacks R32Float; '
                        'use lavapipe (software): '
                        '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json'),
        DeclareLaunchArgument('initial_x', default_value='0.0',
                              description='Simulator initial x.'),
        DeclareLaunchArgument('initial_y', default_value='0.0',
                              description='Simulator initial y.'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0',
                              description='Simulator initial yaw (rad).'),
        DeclareLaunchArgument('initial_speed', default_value='0.0',
                              description='Simulator initial speed (m/s).'),
        DeclareLaunchArgument(
            'step_on_command', default_value='false',
            description='Lockstep the simulator to the controller: advance physics '
                        'one dt only when a new drive command arrives, instead of '
                        'free-running with zero-order hold. Use to reproduce '
                        'fixed-step MATLAB/Simulink runs or to test a slow controller.'),
        DeclareLaunchArgument(
            'simulator', default_value='do_mpc',
            description='Simulator backend: do_mpc | acados. Selects which simulator '
                        'node closes the loop.'),
    ]

    # ---- 1. Static transform: map -> odom (identity) ------------------------
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', map_frame, '--child-frame-id', global_frame,
        ],
        output='screen',
    )

    # ---- 2. Waypoint loader -------------------------------------------------
    waypoint_loader_node = Node(
        package='trajectory_following_ros2',
        executable='waypoint_loader',
        name='waypoint_loader',
        output='screen',
        parameters=[{
            'file_path': waypoints_csv,
            'target_frame_id': map_frame,
        }],
        remappings=[
            ('waypoint_loader/path', 'trajectory/path'),
            ('waypoint_loader/speed', 'trajectory/speed'),
        ],
    )

    # ---- 3. Simulator (do-mpc) ----------------------------------------------
    # Both backends take the same parameters; `simulator` picks which one runs.
    simulator_params = [{
        'odom_topic': odom_topic,
        'global_frame': global_frame,
        'robot_frame': robot_frame,
        'initial_x': initial_x,
        'initial_y': initial_y,
        'initial_yaw': initial_yaw,
        'initial_speed': initial_speed,
        'step_on_command': step_on_command,
    }]
    dompc_simulator_node = Node(
        condition=LaunchConfigurationEquals('simulator', 'do_mpc'),
        package='trajectory_following_ros2',
        executable='kinematic_dompc_simulator',
        name='kinematic_dompc_simulator',
        output='screen',
        parameters=simulator_params,
    )
    acados_simulator_node = Node(
        condition=LaunchConfigurationEquals('simulator', 'acados'),
        package='trajectory_following_ros2',
        executable='kinematic_acados_simulator',
        name='kinematic_acados_simulator',
        output='screen',
        parameters=simulator_params,
    )

    # ---- 4. Controller: coupled_kinematic_casadi (NLP + IPOPT) --------------
    controller_node = Node(
        package='trajectory_following_ros2',
        executable='coupled_kinematic_casadi',
        name='kinematic_coupled_casadi_controller',
        output='screen',
        parameters=[
            # Load the tuned cost weights + speed/curvature policy. Without this the
            # controller falls back to node-default weights (Rd=[10,100] over-penalizes
            # steering rate), which can't follow curvature and diverges within ~10% of a
            # lap. The dict below overrides the YAML where they overlap.
            os.path.join(pkg_prefix, 'config', 'mpc_parameters.yaml'),
            {
                'global_frame': global_frame,
                'robot_frame': robot_frame,
                'odom_topic': odom_topic,
                'ode_type': ode_type,
                'solver_type': solver_type,
                'solver': solver,
                'max_iter': max_iter,
            },
        ],
    )

    # ---- 5. Visualizer ------------------------------------------------------
    # Built in an OpaqueFunction so vulkan_icd can be resolved to a string and
    # VK_ICD_FILENAMES injected ONLY when non-empty (an empty value would make
    # the Vulkan loader find no ICDs at all, so the default path stays untouched).
    def _make_visualizer_node(context, *_args, **_kwargs):
        icd = vulkan_icd.perform(context)
        additional_env = {'VK_ICD_FILENAMES': icd} if icd else None
        return [Node(
            package='trajectory_following_ros2',
            executable='trajectory_visualizer',
            name='trajectory_visualizer',
            output='screen',
            additional_env=additional_env,
            parameters=[{
                'viz_backend': viz_backend,
                'odom_topic': odom_topic,
                'spawn_viewer': viz_spawn_viewer,
                'recording_path': viz_recording_path,
                'native_video_path': viz_video_path,
                'serve_web': viz_serve_web,
                'web_port': viz_web_port,
                'web_open_browser': viz_web_open_browser,
            }],
        )]

    return LaunchDescription(declare_args + [
        static_tf_node,
        waypoint_loader_node,
        dompc_simulator_node,
        acados_simulator_node,
        controller_node,
        OpaqueFunction(function=_make_visualizer_node),
    ])
