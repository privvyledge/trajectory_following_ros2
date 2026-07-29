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
  4. controller                  selected by `control_type` (mpc | purepursuit)
       and, for MPC, `mpc_toolbox` (casadi | acados | do_mpc). Defaults to the
       casadi NLP + IPOPT controller. For Pure Pursuit set
       `control_type:=purepursuit mpc_toolbox:=none`.
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

  # Discrete-model matrix (gate-2 6a/6b): exercise each form/discretization.
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      discrete_model_type:=nonlinear discrete_integration_method:=rk4   # default
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      discrete_model_type:=nonlinear discrete_integration_method:=euler
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      discrete_model_type:=ltv   # true LTV since 2026-07-02: A/B/G at per-stage z_op/u_op
                                 # (shifted warm start); pairs with solver_type:=qp for a
                                 # direct one-QP-per-tick qpsol path (no obstacles/opti)

  # Verify JIT artifacts land in code_gen_directory, not the launch cwd.
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py code_gen_directory:=/tmp/cg

  # Select the controller backend (gate-2 6c/6d/6e/6f):
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py use_opti:=true      # 6c CasADi Opti
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py mpc_toolbox:=acados # 6d acados (ERK)
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      mpc_toolbox:=acados integrator_type:=DISCRETE                                   # 6d-ext acados DISCRETE
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py mpc_toolbox:=do_mpc # 6e do-mpc
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      control_type:=purepursuit mpc_toolbox:=none                                     # 6f Pure Pursuit

  # Per-platform / per-backend weight tuning (config/platforms + config/weights):
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      platform:=f1tenth weights:=f1tenth_casadi
  ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
      mpc_toolbox:=do_mpc platform:=carla weights:=carla_do_mpc
  # Overlay stack (highest wins): sim frames/topics > weights > platform >
  # launch-arg solver config > base mpc_parameters.yaml. The sim re-applies its own
  # frames/topics last, so a platform file's odom_topic/global_frame does NOT
  # repoint the controller off the simulator.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def _overlay_files(context, platforms_dir, weights_dir):
    """Resolve platform/weights launch args to overlay YAML paths (skip empties).

    Returns the list in stack order: [platform.yaml?, weights.yaml?]. Empty list
    when neither is set (legacy single-file behaviour).
    """
    platform_name = LaunchConfiguration('platform').perform(context)
    weights_name = LaunchConfiguration('weights').perform(context)
    overlays = []
    if platform_name:
        overlays.append(os.path.join(platforms_dir, platform_name + '.yaml'))
    if weights_name:
        overlays.append(os.path.join(weights_dir, weights_name + '.yaml'))
    return overlays


def generate_launch_description():
    pkg_prefix = get_package_share_directory('trajectory_following_ros2')
    default_waypoints = os.path.join(pkg_prefix, 'data', 'waypoints.csv')

    # ---- Launch configurations ----------------------------------------------
    waypoints_csv = LaunchConfiguration('waypoints_csv')
    global_frame = LaunchConfiguration('global_frame')
    robot_frame = LaunchConfiguration('robot_frame')
    map_frame = LaunchConfiguration('map_frame')
    odom_topic = LaunchConfiguration('odom_topic')

    mpc_toolbox = LaunchConfiguration('mpc_toolbox')
    control_type = LaunchConfiguration('control_type')

    # Per-platform / per-backend overlay config (stacked on top of mpc_parameters.yaml).
    # Stack order (highest wins): sim frames/topics > weights > platform > launch-arg
    # solver config > base mpc_parameters.yaml. Frames/topics win last because this is a
    # sim harness — a platform file (e.g. carla.yaml sets odom_topic=/carla/...) must not
    # repoint the controller off the simulator's odometry/local + odom/base_link frames.
    # Empty (default) = skip the overlay (legacy single-file behaviour).
    platform = LaunchConfiguration('platform')
    weights = LaunchConfiguration('weights')
    platforms_dir = os.path.join(pkg_prefix, 'config', 'platforms')
    weights_dir = os.path.join(pkg_prefix, 'config', 'weights')

    num_obstacles = LaunchConfiguration('num_obstacles')
    ode_type = LaunchConfiguration('ode_type')
    discrete_model_type = LaunchConfiguration('discrete_model_type')
    discrete_integration_method = LaunchConfiguration('discrete_integration_method')
    use_opti = LaunchConfiguration('use_opti')
    solver_type = LaunchConfiguration('solver_type')
    solver = LaunchConfiguration('solver')
    max_iter = LaunchConfiguration('max_iter')
    integrator_type = LaunchConfiguration('integrator_type')
    stage_cost_type = LaunchConfiguration('stage_cost_type')
    terminal_cost_type = LaunchConfiguration('terminal_cost_type')
    generate_mpc_model = LaunchConfiguration('generate_mpc_model')
    code_gen_directory = LaunchConfiguration('code_gen_directory')
    arclength_index_advance = LaunchConfiguration('arclength_index_advance')
    projection_window = LaunchConfiguration('projection_window')
    delay_compensation_enabled = LaunchConfiguration('delay_compensation_enabled')
    estimated_delay = LaunchConfiguration('estimated_delay')
    delay_compensation_method = LaunchConfiguration('delay_compensation_method')
    solver_log_file = LaunchConfiguration('solver_log_file')
    solver_failure_mode = LaunchConfiguration('solver_failure_mode')
    solver_failure_hold_count = LaunchConfiguration('solver_failure_hold_count')
    solver_failure_hold_time = LaunchConfiguration('solver_failure_hold_time')
    solver_failure_zero_on_saturation = LaunchConfiguration(
        'solver_failure_zero_on_saturation')
    acados_failure_dump_file = LaunchConfiguration('acados_failure_dump_file')

    obstacle_topic = LaunchConfiguration('obstacle_topic')
    footprint_topic = LaunchConfiguration('footprint_topic')
    viz_ego_radius = LaunchConfiguration('viz_ego_radius')
    viz_safe_distance = LaunchConfiguration('viz_safe_distance')
    vehicle_length = LaunchConfiguration('vehicle_length')
    vehicle_width = LaunchConfiguration('vehicle_width')
    vehicle_height = LaunchConfiguration('vehicle_height')
    footprint_rear_axle_offset = LaunchConfiguration('footprint_rear_axle_offset')

    viz_backend = LaunchConfiguration('viz_backend')
    launch_visualizer = LaunchConfiguration('launch_visualizer')
    viz_spawn_viewer = LaunchConfiguration('viz_spawn_viewer')
    viz_recording_path = LaunchConfiguration('viz_recording_path')
    viz_video_path = LaunchConfiguration('viz_video_path')
    viz_serve_web = LaunchConfiguration('viz_serve_web')
    viz_web_port = LaunchConfiguration('viz_web_port')
    viz_web_open_browser = LaunchConfiguration('viz_web_open_browser')
    rerun_spatial_frequency = LaunchConfiguration('rerun_spatial_frequency')
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
            'platform', default_value='',
            description="Platform overlay name (e.g. 'f1tenth', 'carla') -> "
                        'config/platforms/<platform>.yaml. Vehicle physical params '
                        '(wheelbase, steer/speed limits). Empty = skip overlay. NOTE: this '
                        "sim harness re-applies its own frames/topics on top, so a platform's "
                        'global_frame/robot_frame/odom_topic do NOT repoint the loop.'),
        DeclareLaunchArgument(
            'weights', default_value='',
            description="Weight overlay name (e.g. 'f1tenth_casadi', 'carla_do_mpc') -> "
                        'config/weights/<weights>.yaml. Q/R/Rd/Qf + horizon + solver config '
                        '+ speed policy. Empty = skip overlay. Overrides the launch-arg solver '
                        'config (solver_type/solver/max_iter/discrete_*) below.'),
        DeclareLaunchArgument(
            'obstacle_topic', default_value='fake_obstacles/object_array',
            description='Topic where fake obstacles are published (ObjectArray).'),
        DeclareLaunchArgument(
            'footprint_topic', default_value='',
            description='Topic where Nav2 footprint is published (PolygonStamped).'),
        DeclareLaunchArgument(
            'viz_ego_radius', default_value='0.15',
            description='Ego vehicle radius (m) for visualizer margin circles.'),
        DeclareLaunchArgument(
            'viz_safe_distance', default_value='0.15',
            description='Safety margin distance (m) for visualizer margin circles. '
                        'F1/10-realistic default; the drawn margin ring is '
                        'viz_ego_radius + viz_safe_distance. Raise for larger platforms.'),
        DeclareLaunchArgument(
            'vehicle_length', default_value='0.58',
            description='Vehicle physical footprint length (m).'),
        DeclareLaunchArgument(
            'vehicle_width', default_value='0.31',
            description='Vehicle physical footprint width (m).'),
        DeclareLaunchArgument(
            'vehicle_height', default_value='0.12',
            description='Vehicle physical footprint height (m).'),
        DeclareLaunchArgument(
            'footprint_rear_axle_offset', default_value='0.19',
            description='Distance (m) from rear axle to footprint geometric center.'),

        DeclareLaunchArgument(
            'mpc_toolbox', default_value='casadi',
            choices=['acados', 'casadi', 'do_mpc', 'none'],
            description='Which MPC controller node to launch (when control_type=mpc). '
                        'Set to `none` (and control_type=purepursuit) to run Pure Pursuit only.'),
        DeclareLaunchArgument(
            'control_type', default_value='mpc',
            choices=['mpc', 'purepursuit'],
            description='Controller family: `mpc` launches the mpc_toolbox node; '
                        '`purepursuit` launches the geometric Pure Pursuit node. To run '
                        'purepursuit alone, also set mpc_toolbox:=none.'),
        DeclareLaunchArgument(
            'ode_type', default_value='discrete_kinematic_coupled',
            description='CasADi formulation: discrete_kinematic_coupled | continuous_kinematic_coupled | ...'),
        DeclareLaunchArgument(
            'discrete_model_type', default_value='nonlinear',
            description='CasADi discrete model form (ode_type=discrete_*): nonlinear | ltv. '
                        'Restart-only param; ignored by the continuous formulation.'),
        DeclareLaunchArgument(
            'discrete_integration_method', default_value='rk4',
            description='CasADi discretization for discrete_model_type=nonlinear: rk4 | euler. '
                        'Ignored when discrete_model_type=ltv (Euler by construction).'),
        DeclareLaunchArgument(
            'use_opti', default_value='false',
            description='CasADi only: use the Opti-stack formulation (KinematicMPCCasadiOpti) '
                        'instead of the function-based NLP. Restart-only.'),
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
            'num_obstacles', default_value='0',
            description='Number of obstacles the controller constrains against (casadi/acados). '
                        '0 disables obstacle avoidance. Requires an ObjectArray publisher on '
                        'the obstacle_topic (e.g. the fake_obstacle_publisher node). Applied '
                        'AFTER the weights overlay, so the launch arg is authoritative — it '
                        'sizes the OCP and a weights file must not silently override the count. '
                        'Obstacle runs also need generate_mpc_model:=true.'),
        DeclareLaunchArgument(
            'integrator_type', default_value='ERK',
            description='acados only: OCP integrator. ERK (default) | DISCRETE. '
                        'Changing it regenerates the acados C-code.'),
        DeclareLaunchArgument(
            'stage_cost_type', default_value='EXTERNAL',
            description='acados only: stage cost module. EXTERNAL (default) | NONLINEAR_LS | '
                        'LINEAR_LS. Only EXTERNAL carries the Rd input-rate penalty (LS costs '
                        'silently drop it); EXTERNAL is also required for obstacle/CBF constraints.'),
        DeclareLaunchArgument(
            'terminal_cost_type', default_value='EXTERNAL',
            description='acados only: terminal cost module. EXTERNAL (default) | NONLINEAR_LS | '
                        'LINEAR_LS. Keep it EXTERNAL alongside stage_cost_type for the Rd penalty.'),
        DeclareLaunchArgument(
            'generate_mpc_model', default_value='true',
            description='acados only: regenerate and compile the OCP model. Required after '
                        'changing structural settings such as num_obstacles, horizon, or '
                        'ego_disc_offsets. Set false to reuse an already-built compatible model.'),
        DeclareLaunchArgument(
            'code_gen_directory',
            default_value=os.path.join(pkg_prefix, 'data', 'casadi_codegen'),
            description='Directory for CasADi JIT artifacts (jit_tmp.c, tmp_*.o/.so). '
                        'Keeps generated code out of the launch cwd; empty string = cwd (legacy).'),
        DeclareLaunchArgument(
            'viz_backend', default_value='native',
            description="Visualization backend: native | rerun | both."),
        DeclareLaunchArgument(
            'launch_visualizer', default_value='true',
            description='Launch trajectory_visualizer. Set false for cadence isolation.'),
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
            'rerun_spatial_frequency', default_value='5.0',
            description='Maximum per-stream Rerun spatial logging rate in Hz. Time-series '
                        'logging is not throttled; set <=0 to disable the spatial cap.'),
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
        DeclareLaunchArgument(
            'arclength_index_advance', default_value='True',
            description='Reference-index advance mode (all controllers). True = '
                        'along-track (arc-length) projection: the index advances with '
                        'longitudinal progress even when the vehicle is held laterally '
                        'off the line (obstacle swerve), never freezing at a standoff. '
                        'False = legacy Euclidean distance-gate.'),
        DeclareLaunchArgument(
            'projection_window', default_value='5.0',
            description='Forward arc-length span (m) of the arc-length projection '
                        'window. Kept short so the projection cannot leap to '
                        'end-of-path points sitting near the start on a closed loop; '
                        'must be < loop length and > one tick of travel.'),
        DeclareLaunchArgument(
            'delay_compensation_enabled', default_value='false',
            description='Replace x0 with an RK4-propagated state (over estimated_delay) '
                        'before each solve (all MPC controllers). Compensates sensor + '
                        'solve + actuator latency.'),
        DeclareLaunchArgument(
            'estimated_delay', default_value='0.0',
            description='Total round-trip delay (s) propagated forward when '
                        'delay_compensation_enabled is true (0 = no propagation).'),
        DeclareLaunchArgument(
            'delay_compensation_method', default_value='forward_simulation',
            description='Delay-compensation method (currently forward_simulation).'),
        DeclareLaunchArgument(
            'solver_log_file', default_value='',
            description='If set, append one per-solve stats row (status, commands, '
                        'obstacle selection/side/clearance) to this CSV path '
                        '(empty = disabled).'),
        DeclareLaunchArgument(
            'solver_failure_mode', default_value='hold_last',
            description="Solver failure action: 'zero' or bounded 'hold_last'."),
        DeclareLaunchArgument(
            'solver_failure_hold_count', default_value='1',
            description='Maximum consecutive failures allowed to hold the last command; '
                        '0 disables the count gate.'),
        DeclareLaunchArgument(
            'solver_failure_hold_time', default_value='0.1',
            description='Maximum wall seconds since the last successful command publication '
                        'for hold_last; 0 disables the time gate.'),
        DeclareLaunchArgument(
            'solver_failure_zero_on_saturation', default_value='false',
            description='When true, never hold a command at an accel, steering, or speed '
                        'limit. Default false: a legitimately saturated last-good command '
                        '(max steer mid-corner, max-speed cruise) is safer to hold through '
                        'a transient failure than to zero.'),
        DeclareLaunchArgument(
            'acados_failure_dump_file', default_value='',
            description='acados only: write the first hard-failure inputs and solver stats '
                        'to this .npz file before recovery reset (empty = disabled).'),
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
        # Simulator-fidelity knobs (0.0 = ideal sim, the default). Set non-zero to
        # stress-test controller robustness: Gaussian odometry noise (std devs) and
        # first-order actuator lag (time constants, s). Realistic F1/10 stress values
        # are ~0.02 m / 0.05 m/s / 0.02 rad noise and ~0.08 s / 0.15 s lag.
        'noise_std_x': 0.0,
        'noise_std_y': 0.0,
        'noise_std_v': 0.0,
        'noise_std_psi': 0.0,
        'steering_time_constant': 0.0,
        'acceleration_time_constant': 0.0,
    }]
    # Built in an OpaqueFunction so the platform overlay's PHYSICAL vehicle params
    # (wheelbase, steer/speed/accel limits) can be resolved and applied to the
    # simulator too. Without this the simulator uses its own defaults (wheelbase
    # 0.256 m + max_steer 30 deg, F1/10) while the controller uses the platform's
    # values, so any non-F1/10 platform drives a controller/plant model mismatch:
    # the plant over-rotates and clips steering vs the plan, and tracking diverges.
    # Only the physical keys the simulator declares are pulled from the platform
    # file — NOT the whole file: the platform also carries use_sim_time, frames, and
    # command topics that would repoint the sim off this harness (there is no /clock
    # here, so use_sim_time:=True would freeze the sim). The extracted physical dict
    # is merged on top of the sim's own param dict, which keeps its frames/topics/
    # initial pose (the platform sets none of the physical keys the sim would then
    # need to re-assert).

    def _make_simulator_nodes(context, *_args, **_kwargs):
        import yaml
        physical_keys = {
            'wheelbase', 'max_steer', 'min_steer', 'max_steer_rate',
            'max_speed', 'min_speed', 'max_accel', 'max_decel',
        }
        platform_name = LaunchConfiguration('platform').perform(context)
        sim_phys = {}
        if platform_name:
            platform_file = os.path.join(platforms_dir, platform_name + '.yaml')
            if os.path.exists(platform_file):
                with open(platform_file) as fh:
                    doc = yaml.safe_load(fh) or {}
                params = (doc.get('/**', {}) or {}).get('ros__parameters', {}) or {}
                sim_phys = {k: v for k, v in params.items() if k in physical_keys}
        sim_params = [{**simulator_params[0], **sim_phys}]
        return [
            Node(
                condition=LaunchConfigurationEquals('simulator', 'do_mpc'),
                package='trajectory_following_ros2',
                executable='kinematic_dompc_simulator',
                name='kinematic_dompc_simulator',
                output='screen',
                parameters=sim_params,
            ),
            Node(
                condition=LaunchConfigurationEquals('simulator', 'acados'),
                package='trajectory_following_ros2',
                executable='kinematic_acados_simulator',
                name='kinematic_acados_simulator',
                output='screen',
                parameters=sim_params,
            ),
        ]

    # ---- 4. Controller -------------------------------------------------------
    # Selected by `control_type` (mpc | purepursuit) and, for MPC, `mpc_toolbox`
    # (casadi | acados | do_mpc). Each node loads config/mpc_parameters.yaml for the
    # tuned cost weights + speed/curvature policy, then layers backend-specific
    # overrides on top. Without the YAML the controller falls back to node-default
    # weights (Rd=[10,100] over-penalizes steering rate), which can't follow curvature
    # and diverges within ~10% of a lap.
    params_file = os.path.join(pkg_prefix, 'config', 'mpc_parameters.yaml')

    # Frame/topic overrides every controller needs (the YAML targets hardware topics).
    # These are applied AFTER the platform/weights overlays so a platform file cannot
    # repoint the loop off the simulator (see the overlay note at the top). use_sim_time
    # is forced False: this harness runs on wall clock with no /clock publisher, but
    # platforms/carla.yaml sets use_sim_time:True (correct only for the real ros-bridge).
    sim_controller_params = {
        'global_frame': global_frame,
        'robot_frame': robot_frame,
        'odom_topic': odom_topic,
        'use_sim_time': False,
    }

    # Launch-arg solver/model config, applied BEFORE the overlays so a weights file
    # (which sets solver_type/solver/max_iter/discrete_*) overrides these defaults.
    casadi_solver_params = {
        'ode_type': ode_type,
        'discrete_model_type': discrete_model_type,
        'discrete_integration_method': discrete_integration_method,
        'solver_type': solver_type,
        'solver': solver,
        'max_iter': max_iter,
    }
    # `use_opti` and `num_obstacles` are structural launch-time choices, not weights,
    # so they are applied in the tail_dict (AFTER the platform/weights overlays) — an
    # explicit `use_opti:=...` / `num_obstacles:=...` launch arg must win. num_obstacles
    # sizes the OCP (its parameter block and keep-out constraint count), so a weights
    # file must not be able to silently override the requested count. (The other solver
    # knobs above stay before the overlays so a per-backend weights file can still tune
    # solver_type/solver/max_iter/discrete_* on purpose.)
    acados_solver_params = {
        'integrator_type': integrator_type,
        'stage_cost_type': stage_cost_type,
        'terminal_cost_type': terminal_cost_type,
        'max_iter': max_iter,
    }
    do_mpc_solver_params = {
        'max_iter': max_iter,
    }

    # Controllers are built inside an OpaqueFunction so the platform/weights overlay
    # files can be resolved to paths and inserted into each node's `parameters` list
    # at the right precedence. ROS 2 applies a parameters list left-to-right (later
    # wins), so the stack is:
    #   base mpc_parameters.yaml -> launch-arg solver config -> platform.yaml ->
    #   weights.yaml -> sim frames/topics (+ code_gen_directory).
    def _make_controller_nodes(context, *_args, **_kwargs):
        overlays = _overlay_files(context, platforms_dir, weights_dir)

        # Backend-agnostic reference-index params (base_tracker) applied to every
        # controller. Placed BEFORE the overlays so a weights file may still pin them
        # per-platform (e.g. a shorter projection_window for a small loop); with no
        # weights override the launch-arg default takes effect.
        # Wrap in ParameterValue with an explicit value_type: a bare LaunchConfiguration
        # in a param dict resolves to a STRING, and a string override against a node's
        # typed (bool/float) declared param — which mpc_parameters.yaml sets — is dropped,
        # so the launch arg would silently have no effect (e.g. delay compensation staying
        # off despite delay_compensation_enabled:=true).
        reference_params = {
            'arclength_index_advance': ParameterValue(arclength_index_advance, value_type=bool),
            'projection_window': ParameterValue(projection_window, value_type=float),
            'delay_compensation_enabled': ParameterValue(
                delay_compensation_enabled, value_type=bool),
            'estimated_delay': ParameterValue(estimated_delay, value_type=float),
            'delay_compensation_method': delay_compensation_method,
            'solver_log_file': solver_log_file,
        }
        # Safety/diagnostic flags are explicit launch choices and therefore follow
        # platform/weight overlays rather than being silently overridden by them.
        failure_policy_params = {
            'solver_failure_mode': solver_failure_mode,
            'solver_failure_hold_count': ParameterValue(
                solver_failure_hold_count, value_type=int),
            'solver_failure_hold_time': ParameterValue(
                solver_failure_hold_time, value_type=float),
            'solver_failure_zero_on_saturation': ParameterValue(
                solver_failure_zero_on_saturation, value_type=bool),
        }

        def _params(solver_dict, tail_dict):
            return ([params_file, reference_params, solver_dict] + overlays
                    + [failure_policy_params, tail_dict])

        return [
            Node(
                condition=LaunchConfigurationEquals('mpc_toolbox', 'casadi'),
                package='trajectory_following_ros2',
                executable='coupled_kinematic_casadi',
                name='kinematic_coupled_casadi_controller',
                output='screen',
                parameters=_params(
                    casadi_solver_params,
                    {**sim_controller_params, 'code_gen_directory': code_gen_directory,
                     'use_opti': use_opti, 'num_obstacles': num_obstacles}),
            ),
            Node(
                condition=LaunchConfigurationEquals('mpc_toolbox', 'acados'),
                package='trajectory_following_ros2',
                executable='coupled_kinematic_acados',
                name='kinematic_coupled_acados_controller',
                output='screen',
                parameters=_params(
                    acados_solver_params,
                    {**sim_controller_params, 'code_gen_directory': code_gen_directory,
                     'num_obstacles': num_obstacles,
                     'generate_mpc_model': ParameterValue(
                         generate_mpc_model, value_type=bool),
                     'acados_failure_dump_file': acados_failure_dump_file}),
            ),
            Node(
                condition=LaunchConfigurationEquals('mpc_toolbox', 'do_mpc'),
                package='trajectory_following_ros2',
                executable='coupled_kinematic_do_mpc',
                name='kinematic_coupled_do_mpc_controller',
                output='screen',
                parameters=_params(do_mpc_solver_params, dict(sim_controller_params)),
            ),
            Node(
                condition=LaunchConfigurationEquals('control_type', 'purepursuit'),
                package='trajectory_following_ros2',
                executable='purepursuit',
                name='purepursuit_controller',
                output='screen',
                parameters=_params({}, dict(sim_controller_params)),
            ),
        ]

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
            # Visualization is best-effort: keep its CPU priority below the control
            # loop so full-figure redraws cannot starve the 20 Hz controller tick on
            # a core-limited box.
            prefix='nice -n 10',
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
                'rerun_spatial_frequency': ParameterValue(
                    rerun_spatial_frequency, value_type=float),
                'obstacle_topic': obstacle_topic,
                'footprint_topic': footprint_topic,
                'viz_ego_radius': viz_ego_radius,
                'viz_safe_distance': viz_safe_distance,
                'vehicle_length': vehicle_length,
                'vehicle_width': vehicle_width,
                'vehicle_height': vehicle_height,
                'footprint_rear_axle_offset': footprint_rear_axle_offset,
            }],

        )]

    return LaunchDescription(declare_args + [
        static_tf_node,
        waypoint_loader_node,
        OpaqueFunction(function=_make_simulator_nodes),
        OpaqueFunction(function=_make_controller_nodes),
        OpaqueFunction(
            function=_make_visualizer_node,
            condition=IfCondition(launch_visualizer)),
    ])
