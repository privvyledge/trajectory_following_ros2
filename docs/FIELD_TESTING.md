# Field Testing Guide — Jetson (F1/10) and CARLA

Copy-pasteable commands for bringing this package up on a Jetson (F1/10 car) and in
CARLA, focused on the **CasADi (function-based NLP, no Opti)** and **acados** backends,
with and without obstacle avoidance.

Everything below assumes:
- `~/ros2_ws` is the workspace root (the directory containing `src/`).
- This package is at `~/ros2_ws/src/trajectory_following_ros2`.
- acados and CasADi are already installed for the Python interpreter you build with.

## Table of Contents

- [0. One-time setup](#0-one-time-setup)
- [1. Concepts you need before running anything](#1-concepts-you-need-before-running-anything)
- [2. Flow A — record waypoints](#2-flow-a--record-waypoints)
- [3. Flow B — replay waypoints only](#3-flow-b--replay-waypoints-only)
- [4. Flow C — closed-loop simulation (no hardware)](#4-flow-c--closed-loop-simulation-no-hardware)
- [5. Flow D — Jetson / F1-10 on the real car](#5-flow-d--jetson--f110-on-the-real-car)
- [6. Flow E — CARLA](#6-flow-e--carla)
- [7. Obstacle avoidance](#7-obstacle-avoidance)
- [8. Verification checklist and diagnostics](#8-verification-checklist-and-diagnostics)
- [9. Command reference tables](#9-command-reference-tables)

---

## 0. One-time setup

### 0.1 Build

The package is `ament_python`. If the nodes run against a **Python venv**, build with
`python3 -m colcon`, not bare `colcon` — a bare `colcon` stamps system-python shebangs
into the installed scripts and venv-only dependencies become invisible at runtime.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash          # adjust distro

# If using a venv, activate it FIRST so `python3` is the venv interpreter.
# source ~/venvs/ros_venv/bin/activate

python3 -m colcon build --packages-select trajectory_following_ros2 --symlink-install
source install/setup.bash
```

`--symlink-install` means later **Python-only** edits are live without a rebuild.
Rebuild when you change `setup.py`, launch files, or `config/*.yaml` (those are copied
into `share/`, not symlinked).

### 0.2 Dependencies not in `package.xml`

```bash
# Core (from the workspace root, against the venv/interpreter you build with)
python3 -m pip install -c src/trajectory_following_ros2/constraints.txt \
    casadi scipy numpy pandas

# Obstacle avoidance needs derived_object_msgs (a ROS package, install via apt/rosdep):
sudo apt install ros-humble-derived-object-msgs

# Launch-file helpers
sudo apt install ros-humble-nav2-common

# Optional: live visualization
python3 -m pip install -c src/trajectory_following_ros2/constraints.txt rerun-sdk matplotlib
```

`constraints.txt` pins `numpy<2` and `rerun-sdk<0.23`. Those pins are load-bearing:
`rerun-sdk >= 0.23` hard-requires `numpy >= 2`, but the MPC backends need numpy 1.x.
Always install with `-c constraints.txt`.

### 0.3 acados environment

acados needs its shared libraries on the loader path in **every** shell that runs an
acados node. Put this in `~/.bashrc` on the Jetson:

```bash
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
```

Verify before you fight a launch failure:

```bash
python3 -c "import acados_template; print(acados_template.__file__)"
ls $ACADOS_SOURCE_DIR/lib/libacados.so
```

### 0.4 First-run code generation

Both backends generate code on first launch into `code_gen_directory` (default
`<share>/data/mpc`):

| Backend | What is generated | First-run cost |
|---|---|---|
| acados | C code + compiled (optionally Cython) OCP solver | ~1 min |
| CasADi | JIT artifacts (`jit_tmp.c`, `tmp_*.o/.so`) | ~seconds |

`generate_mpc_model:=true` forces regeneration. **You must regenerate whenever the OCP
structure changes** — horizon, model form, `num_obstacles`, cost type, slack weights,
`enforce_input_rate_constraint`. Reference-side params (`distance_tolerance`,
`max_lateral_accel`, `Q`/`R`/`Rd`/`Qf`) do **not** need regeneration.

Make the directory writable and pre-create it on the Jetson:

```bash
mkdir -p ~/ros2_ws/install/trajectory_following_ros2/share/trajectory_following_ros2/data/mpc
```

---

## 1. Concepts you need before running anything

### 1.1 The parameter overlay stack

Most parameters resolve in this order, **highest wins**:

```
config/weights/<weights>.yaml   >   config/platforms/<platform>.yaml   >   launch args   >   config/mpc_parameters.yaml
```

The overlays are applied late inside the launch file, so a value pinned in a weights YAML
usually beats a launch argument of the same name. One consequence that will bite you:

- `weights:=f1tenth_acados_obstacle` pins `horizon: 25` and `max_iter: 20` (it deliberately
  does **not** pin `num_obstacles` — that is a launch-applied structural override).

Structural exceptions are applied *after* the overlays so their launch values stay
authoritative: `use_opti`, `num_obstacles`, and `generate_mpc_model`.

### 1.1b Undeclared launch arguments are silently ignored

**Not every node parameter has a matching launch argument**, and `ros2 launch` does **not**
error on an argument that was never declared — it accepts it and does nothing. Verified:

```bash
$ ros2 launch trajectory_following_ros2 mpc.launch.py mpc_toolbox:=none control_type:=mpc loop:=0
[INFO] [launch]: All log files can be found below ...
[INFO] [launch]: Default logging verbosity is set to INFO
# no error — and `loop` was never applied
```

So a typo (`waypoint_csv:=` instead of `waypoints_csv:=`) or a param-that-isn't-an-arg
(`loop:=0`) fails **silently**, and you get default behaviour while believing you
configured something. A *declared* argument with an invalid value does error properly
(`control_type:=none` → `Valid options are: ['mpc', 'purepursuit']`), so the check only
exists for args the launch file knows about.

Always confirm what a launch file actually accepts:

```bash
ros2 launch trajectory_following_ros2 mpc.launch.py --show-args
ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py --show-args
```

Node-params-only (no launch arg on `mpc.launch.py`) — set these in a YAML overlay, or with
`ros2 param set` if hot-reloadable (§5.5):

| Param | Set via |
|---|---|
| `loop` | YAML / `ros2 param set` (hot-reloadable) |
| `num_obstacles`, `safe_distance` | weights YAML (restart-only) |
| `integrator_type` | weights YAML (restart-only; launch arg on `closed_loop_sim.launch.py` only) |
| `solver_type`, `solver` | weights YAML (restart-only; launch args on `closed_loop_sim.launch.py` only) |
| `solver_log_file` | weights YAML (restart-only) |
| `actuator_feedback_topic` | YAML |
| `delay_compensation_enabled`, `estimated_delay`, `delay_compensation_method` | YAML (launch args on `closed_loop_sim.launch.py` only) |
| `max_lateral_accel`, `min_reference_speed`, `use_speed_profile` | YAML / `ros2 param set` (hot-reloadable) |

Name mismatches to watch (launch arg ≠ node param):

| Launch arg (`mpc.launch.py`) | Node param |
|---|---|
| `max_iterations` | `max_iter` |
| `frequency` | `control_rate` |
| `waypoints_csv` | `file_path` |

`loop` specifically: `0` = stop at goal (default, from `config/mpc_parameters.yaml`),
`-1` = loop indefinitely, `N > 0` = run N laps then stop. Hot-reloadable:
`ros2 param set /acados_mpc_node loop -1`.

### 1.2 Platform files (vehicle physics — shared across backends)

| File | Vehicle | Key values |
|---|---|---|
| `config/platforms/f1tenth.yaml` | F1/10 car | `wheelbase: 0.256`, `max_speed: 1.5`, `max_steer: 27°`, `distance_tolerance: 0.2`, `use_sim_time: False` |
| `config/platforms/carla.yaml` | CARLA ego | `wheelbase: 2.87528`, `max_speed: 10.5`, `max_steer: 70°`, `distance_tolerance: 5.0`, `use_sim_time: True`, `global_frame: map`, `robot_frame: ego_vehicle`, `odom_topic: /carla/ego_vehicle/odometry` |

### 1.3 Weights files (cost + horizon + solver — per backend)

| File | Backend | Obstacles | Notes |
|---|---|---|---|
| `f1tenth_casadi.yaml` | CasADi | no | |
| `f1tenth_acados.yaml` | acados | no | pins SQP + `max_iter: 20` |
| `f1tenth_casadi_obstacle.yaml` | CasADi | **yes** | `quad`/`qrqp` sqpmethod, exact nonconvex keep-out |
| `f1tenth_casadi_qp_obstacle.yaml` | CasADi | **yes** | `qp`/`qrqp` LTV, *linearized* keep-out — weaker head-on |
| `f1tenth_acados_obstacle.yaml` | acados | **yes** | exact nonlinear `con_h`, `max_iter: 20` (time budget — do not raise to "fix" status-2), two-disc ego cover |
| `carla_casadi.yaml` | CasADi | no | tuned Town01: CTE mean 0.032 m / max 0.298 m |
| `carla_acados.yaml` | acados | no | tuned Town01: CTE mean 0.068 m / max 0.465 m |
| `carla_acados_obstacle.yaml` | acados | **yes** | carla_acados tuning + obstacle block scaled to the CARLA ego (two 1.5 m discs) — see [§7.4](#74-carla--obstacles) |
| `f1tenth_opti.yaml` | CasADi Opti | no | not covered here (you are running `use_opti:=false`) |

### 1.4 `distance_tolerance` is the corner-cutting lever, not a goal tolerance

It doubles as the **reference anchor distance**: the reference horizon is built starting
from the first waypoint that far *ahead* of the vehicle. Too large and the reference
anchors past a corner apex and the car cuts the chord. On CARLA Town01 (R≈12 m hairpins)
dropping it 5.0 → 2.5 cut corner CTE from ~2.5 m to ~0.5 m — more than any weight change.

Floors (below these, preview is too short and route progress stalls):

| Backend | Floor | Recommended |
|---|---|---|
| CasADi `quad`/`qrqp` | ~2.5 (CARLA) | 2.5 |
| acados SQP | ~2.0 (CARLA; 1.8 fails) | 2.5 |
| F1/10 (either) | — | 0.2 (platform default) |

### 1.5 The two launch files

| Launch file | What it starts | Use for |
|---|---|---|
| `mpc.launch.py` | controller (+ `waypoint_loader` if `load_waypoints:=true`, + visualizer if `load_visualizer:=true`) | **Real hardware and CARLA** |
| `closed_loop_sim.launch.py` | static map→odom TF + `waypoint_loader` + a simulator node + controller | **Hardware-free testing** |

They do **not** have identical arguments. Most importantly, `num_obstacles` is a launch
argument on `closed_loop_sim.launch.py` but **not** on `mpc.launch.py` (see §7).

---

## 2. Flow A — record waypoints

Drive the route manually (teleop / CARLA autopilot / RC transmitter) while
`waypoint_recorder` logs odometry to CSV.

```bash
source ~/ros2_ws/install/setup.bash

ros2 run trajectory_following_ros2 waypoint_recorder --ros-args \
    -p file_path:=$HOME/waypoints_$(date +%m%d%Y).csv \
    -p odom_topic:=/odometry/filtered \
    -p target_frame_id:=map \
    -p save_interval:=0.05 \
    -p publish_markers:=true
```

Stop with `Ctrl-C` — the file is written incrementally, so a `Ctrl-C` keeps what you drove.

**`save_interval` is the one to get right.** The reference generator and the curvature
filter both want dense, evenly spaced points. The 0.1 s (10 Hz) default is adequate for
the F1/10 but coarse at CARLA speeds — spacing is `save_interval × speed`:

| Platform | Speed | `save_interval` | Resulting spacing |
|---|---|---|---|
| F1/10 | ~1.5 m/s | `0.1` (default) → `0.05` for margin | ~15 cm → ~7 cm |
| CARLA ego | ~9 m/s | `0.05` → `0.02` for tight corners | ~45 cm → ~18 cm |

Other parameters worth knowing:

| Parameter | Default | Meaning |
|---|---|---|
| `target_frame_id` | `map` | TF frame the recorded poses are transformed into. **Must match the `global_frame` you replay with.** |
| `stale_odom_timeout` | `0.5` | Skip writes when the odom topic goes silent (preserves dwell time at intentional stops) |
| `min_distance` | `0.0` (off) | Distance gate — suppresses stop dwell time. Leave off unless you want that. |
| `marker_max_speed` | `10.0` | Colour scale ceiling for the velocity-arrow trail |

CSV columns produced: `frame_id, total_time_elapsed, dt, x, y, z, yaw, qx, qy, qz, qw, vx, vy, speed, omega`.

The recorded `vx` is **signed** — the MPC state uses `vx`, so reverse segments record as
negative and replay as reverse. That is intended.

CARLA recording (drive with the ros-bridge manual control or autopilot):

```bash
ros2 run trajectory_following_ros2 waypoint_recorder --ros-args \
    -p file_path:=$HOME/carla_town01_$(date +%m%d%Y).csv \
    -p odom_topic:=/carla/ego_vehicle/odometry \
    -p target_frame_id:=map \
    -p save_interval:=0.02 \
    -p use_sim_time:=true
```

---

## 3. Flow B — replay waypoints only

Sanity-check a CSV before putting a controller behind it. `waypoint_loader` publishes
`Path` + speeds on **transient-local (latched)** QoS, so subscribers that start later
still receive them.

```bash
ros2 run trajectory_following_ros2 waypoint_loader --ros-args \
    -p file_path:=$HOME/waypoints_07152026.csv \
    -p target_frame_id:=map \
    -p smooth_path:=true \
    -p smooth_speed:=true
```

Inspect:

```bash
ros2 topic echo /waypoint_loader/path --once | head -40
ros2 topic echo /waypoint_loader/speed --once | head -20
rviz2 -d ~/ros2_ws/src/trajectory_following_ros2/rviz/mpc.rviz
```

Under `mpc.launch.py` the loader publishes on the remapped `path_topic` /
`speed_topic` (defaults `trajectory/path`, `trajectory/speed`), not `waypoint_loader/*`.

Speed-smoothing params with a per-vehicle calibration table live next to their
declarations in `waypoint_loader.py`: `smooth_speed_start_vel`, `smooth_speed_accel_limit`,
`smooth_speed_time_constant`, `reverse_speed_threshold`.

---

## 4. Flow C — closed-loop simulation (no hardware)

Run this **before** every hardware session — same controller, same params, simulated
vehicle. One command brings up static TF + loader + simulator + controller.

### 4.1 CasADi, no obstacles

```bash
ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
    mpc_toolbox:=casadi \
    use_opti:=false \
    platform:=f1tenth \
    weights:=f1tenth_casadi \
    simulator:=do_mpc \
    waypoints_csv:=$HOME/waypoints_07152026.csv \
    viz_backend:=native
```

### 4.2 acados, no obstacles

```bash
ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
    mpc_toolbox:=acados \
    platform:=f1tenth \
    weights:=f1tenth_acados \
    simulator:=acados \
    waypoints_csv:=$HOME/waypoints_07152026.csv \
    stage_cost_type:=EXTERNAL \
    terminal_cost_type:=EXTERNAL \
    viz_backend:=native
```

> `generate_mpc_model` is an authoritative structural launch argument. Use `True` whenever
> the generated OCP structure changes (for example obstacle count, ego-disc offsets, or
> cost structure), then use `False` on later launches to reuse that build.

### 4.3 With obstacles

Obstacles need a publisher on `fake_obstacles/object_array`. Start the controller first,
then in a second terminal:

```bash
# Terminal 1 — acados + obstacle weights (num_obstacles is the launch arg; the
# weights YAML supplies the disc/slack/max_iter obstacle config but not the count)
ros2 launch trajectory_following_ros2 closed_loop_sim.launch.py \
    mpc_toolbox:=acados \
    platform:=f1tenth \
    weights:=f1tenth_acados_obstacle \
    simulator:=acados \
    num_obstacles:=1 \
    waypoints_csv:=$HOME/waypoints_07152026.csv

# Terminal 2 — one 0.3 m sphere at (3.0, 0.0) in the odom frame
ros2 run trajectory_following_ros2 fake_obstacle_publisher --ros-args \
    -p obstacle_x:="[3.0]" \
    -p obstacle_y:="[0.0]" \
    -p obstacle_radius:="[0.3]" \
    -p frame_id:=odom
```

CasADi equivalent — swap `mpc_toolbox:=casadi weights:=f1tenth_casadi_obstacle
use_opti:=false simulator:=do_mpc`.

### 4.4 Simulation-fidelity knobs

The simulator defaults to a perfect vehicle. Before trusting a sim result, add noise and
lag — a controller that passes clean sim can still stall on the real car:

```bash
ros2 node list   # confirm the name first — it differs per launch file (§9.5)

ros2 param set /kinematic_acados_simulator noise_std_x 0.01
ros2 param set /kinematic_acados_simulator noise_std_y 0.01
ros2 param set /kinematic_acados_simulator noise_std_psi 0.005
ros2 param set /kinematic_acados_simulator steering_time_constant 0.05
ros2 param set /kinematic_acados_simulator acceleration_time_constant 0.1
```

Available: `steering_time_constant`, `acceleration_time_constant`, `noise_std_x`,
`noise_std_y`, `noise_std_v`, `noise_std_psi`, plus `initial_x/y/yaw/speed`.

---

## 5. Flow D — Jetson / F1-10 on the real car

### 5.1 Pre-flight

```bash
source ~/ros2_ws/install/setup.bash

# Odometry alive and fresh? The controller zero-commands after 0.5 s of silence.
ros2 topic hz /odometry/filtered

# TF chain complete? A broken chain makes the controller hang on startup, not error.
ros2 run tf2_ros tf2_echo map base_link

# Drive topic reaching the VESC/actuator?
ros2 topic info /drive
```

> **Safety:** keep the car on blocks (wheels off the ground) for the first launch of any
> new config. Verify the commands on `/drive` look sane before putting it on the floor.

### 5.2 CasADi (no Opti), no obstacles

```bash
ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=casadi \
    control_type:=mpc \
    use_opti:=false \
    platform:=f1tenth \
    weights:=f1tenth_casadi \
    load_waypoints:=true \
    waypoints_csv:=$HOME/waypoints_07152026.csv \
    global_frame:=odom \
    robot_frame:=base_link \
    odom_topic:=/odometry/filtered \
    ackermann_cmd_topic:=/drive \
    log_level:=info
```

The weights file pins `solver_type: quad`, `solver: qrqp`,
`discrete_model_type: nonlinear`, `discrete_integration_method: rk4`,
`slack_objective_is_quadratic: True`.

> `slack_objective_is_quadratic: True` is **required** on the `qp` path and strongly
> recommended on `quad`. A LINEAR slack cost leaves zero rows in the QP Hessian; raw qrqp
> then "drops bounds for regularity" and can return **success with the steering box
> violated**. Do not remove it from an obstacle or `qp` weights file.

### 5.3 acados, no obstacles

```bash
ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=acados \
    control_type:=mpc \
    platform:=f1tenth \
    weights:=f1tenth_acados \
    load_waypoints:=true \
    waypoints_csv:=$HOME/waypoints_07152026.csv \
    global_frame:=odom \
    robot_frame:=base_link \
    odom_topic:=/odometry/filtered \
    ackermann_cmd_topic:=/drive \
    stage_cost_type:=EXTERNAL \
    terminal_cost_type:=EXTERNAL \
    generate_mpc_model:=true
```

`integrator_type` (`ERK` default | `DISCRETE`) is **not** a launch argument on
`mpc.launch.py` — it is a launch arg on `closed_loop_sim.launch.py` only. Passing
`integrator_type:=ERK` here is one of the silent no-ops from §1.1b. To change it on
hardware, set it in the weights YAML (restart-only; needs `generate_mpc_model: True`
since it changes the OCP). `ERK` is the node default and is what you want unless you are
driving acados CBF or symbolic-`x[k+1]` obstacle constraints, which need `DISCRETE`.

After the first successful build, pass `generate_mpc_model:=False` to skip the recompile
on later launches. Keep the generated directory paired with the same OCP structure.

**Keep `stage_cost_type` and `terminal_cost_type` at `EXTERNAL`.** It is the only cost
type that applies the tuned `Rd` rate penalty and the only one that enables obstacle/CBF
constraints. `NONLINEAR_LS`/`LINEAR_LS` silently drop `Rd` and the steering jitters. The
node logs a WARN if you pick anything else. EXTERNAL selects `hessian_approx=EXACT` and
costs ~5 ms/solve, well inside budget.

### 5.4 Delay compensation (real hardware)

Sensor lag + solve time + actuator deadtime. Measure, then set. **This is a
`closed_loop_sim.launch.py` argument only** — under `mpc.launch.py` set it as a node
parameter via the weights YAML, or use `ros2 param set` at runtime (it is not
hot-reloadable in the sense that matters — restart is cleaner):

```yaml
# in config/weights/f1tenth_acados.yaml
delay_compensation_enabled: true
delay_compensation_method: forward_simulation
estimated_delay: 0.06     # seconds — measure yours
```

The base tracker then RK4-propagates `x0` forward by `estimated_delay` before calling the
solver, and logs a throttled `Delay compensation (N ms): x0 <raw> -> <propagated>` line.
All backends benefit transparently.

### 5.5 Runtime tuning without a restart

These are hot-reloadable — tune while the car idles, no rebuild:

```bash
# Under mpc.launch.py the CasADi node is `casadi_mpc_node`; under
# closed_loop_sim.launch.py the SAME node is `kinematic_coupled_casadi_controller`.
# Confirm first — see §9.5:
ros2 node list

ros2 param set /casadi_mpc_node distance_tolerance 0.25
ros2 param set /casadi_mpc_node max_lateral_accel 2.0
ros2 param set /casadi_mpc_node Q "[100.0, 100.0, 100.0, 8.0]"
ros2 param set /casadi_mpc_node Rd "[10.0, 10.0]"
ros2 param set /casadi_mpc_node max_speed 1.0
```

Hot-reloadable: `Q`, `R`, `Rd`, `Qf`, `use_bryson_weights` + all `max_error_*` /
`bryson_max_*`, `distance_tolerance`, `speed_tolerance`, `max_speed`, `min_speed`,
`max_accel`, `max_decel`, `desired_speed`, `loop`, `use_speed_profile`,
`max_lateral_accel`, `min_reference_speed`, `arclength_index_advance`,
`projection_window`, `wheelbase`, `robot_frame`, `global_frame`.

Restart-only: `horizon`, `sample_time`, `max_steer_rate`, `max_iter`,
`termination_condition`, `saturate_input`, `allow_reversing`, `num_obstacles`,
`ego_radius`, `safe_distance`, `integrator_type`, `stage_cost_type`,
`terminal_cost_type`, `solver_type`, `solver`, `use_opti`, `discrete_model_type`.

`Q`/`R`/`Rd`/`Qf` are length-validated — a wrong-length array is rejected
(`ros2 param set` reports failure) and the previous weights are kept.

A `ros2 param set` of an **unrecognized** parameter returns failure rather than being
silently ignored. If a set fails, read the error — it is telling you something.

---

## 6. Flow E — CARLA

### 6.1 Bring up CARLA + ros-bridge

```bash
# Terminal 1 — simulator
cd /opt/carla-simulator && ./CarlaUE4.sh -quality-level=Low

# Terminal 2 — ros-bridge with an ego vehicle
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py \
    town:=Town01 \
    timeout:=30
```

Confirm the topic names against your bridge version before launching the controller —
`config/platforms/carla.yaml` assumes `/carla/ego_vehicle/odometry` and `/drive`:

```bash
ros2 topic list | grep carla
ros2 topic hz /carla/ego_vehicle/odometry
```

### 6.2 Spawn at the route start — this is not optional

The reference index anchor starts from path index 0 and the projection window
(`projection_window`, default 5.0 m) searches only that far ahead of it. **If the ego
spawns away from the route start, the controller cannot acquire the path.** Move the ego
to the first CSV row's pose before launching the controller, or set
`arclength_index_advance:=False` (legacy Euclidean mode) for a mid-path start.

Read the first waypoint:

```bash
head -2 "$(ros2 pkg prefix trajectory_following_ros2)/share/trajectory_following_ros2/data/carla_town01_moving.csv"
```

### 6.3 CARLA + CasADi (tuned)

```bash
ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=casadi \
    control_type:=mpc \
    use_opti:=false \
    platform:=carla \
    weights:=carla_casadi \
    load_waypoints:=true \
    waypoints_csv:="$(ros2 pkg prefix trajectory_following_ros2)/share/trajectory_following_ros2/data/carla_town01_moving.csv" \
    use_sim_time:=true \
    robot_frame:=ego_vehicle \
    global_frame:=map \
    frequency:=20.0 \
    wheelbase:=2.87528 \
    odom_topic:=/carla/ego_vehicle/odometry \
    ackermann_cmd_topic:=/drive \
    load_visualizer:=true
```

The explicit CARLA values make the command self-describing and provide an immediate
sanity check in the printed process arguments. They intentionally match
`config/platforms/carla.yaml`, which remains the authoritative platform profile and also
sets the CARLA speed, acceleration, steering, and tolerance limits.

Expected on the tuned Town01 route: CTE mean 0.032 m, median 0.000 m, max 0.298 m
(the two R≈11–12 m hairpins).

### 6.4 CARLA + acados (tuned)

```bash
ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=acados \
    control_type:=mpc \
    platform:=carla \
    weights:=carla_acados \
    load_waypoints:=true \
    waypoints_csv:="$(ros2 pkg prefix trajectory_following_ros2)/share/trajectory_following_ros2/data/carla_town01_moving.csv" \
    stage_cost_type:=EXTERNAL \
    terminal_cost_type:=EXTERNAL \
    use_sim_time:=true \
    robot_frame:=ego_vehicle \
    global_frame:=map \
    frequency:=20.0 \
    wheelbase:=2.87528 \
    odom_topic:=/carla/ego_vehicle/odometry \
    ackermann_cmd_topic:=/drive \
    load_visualizer:=true
```

Expected: CTE mean 0.068 m, median 0.001 m, max 0.465 m.

`carla_acados.yaml` pins `generate_mpc_model: True` — the first launch spends ~1 min in
C codegen. Edit the YAML to `False` after a successful build.

---

## 7. Obstacle avoidance

### 7.1 How obstacles are handled — read this first

Two behaviours you should know about before wiring a real perception stack in:

**Obstacles are NOT filtered by heading or velocity.** `base_tracker._obstacle_callback`
computes a plain Euclidean distance from the vehicle to each object and sorts by it:

```python
dist = np.linalg.norm(np.array(pos[:2]) - np.array([self.x, self.y]))
obstacles.append({'state': [pos[0], pos[1], radius], 'distance': dist})
self.obstacles = sorted(obstacles, key=lambda o: o['distance'])
```

There is no dot product against heading, no forward/behind test, no velocity gate. The
controller then takes the **first `num_obstacles` entries** of that sorted list. So an
obstacle 1 m *behind* the car outranks one 5 m *ahead*, and with `num_obstacles: 1` it
will crowd the relevant one out of the OCP entirely. This is fine with the
`fake_obstacle_publisher` and a handful of static objects; it is **not** fine with a live
perception feed that reports everything around the vehicle. Either filter upstream
(publish only forward objects) or raise `num_obstacles` enough to cover the clutter —
each obstacle adds constraints and solve time, and `num_obstacles` is restart-only
(it changes the OCP structure, so `generate_mpc_model:=true`).

**Obstacles are NOT transformed through TF.** The callback reads `obj.pose.position`
directly and compares it against the vehicle odometry. `data.header.frame_id` is
ignored entirely. So:

- Obstacles **must already be in the controller's `global_frame`** (`odom` by default,
  `map` for CARLA).
- Objects in a sensor frame (`laser`, `velodyne`, `camera_link`, …) will be interpreted
  as global coordinates and place a phantom keep-out near the origin. There is no
  warning — the numbers are silently wrong.
- If your perception publishes in a sensor frame, you need a **transform node between
  perception and the controller** that looks up `sensor_frame → global_frame` and
  republishes the `ObjectArray`. That node does not exist in this package yet.
- In the closed-loop sim, `map → odom` is an identity TF, so map/waypoint coordinates can
  be used directly as `odom` coordinates.

Also note: the callback reads `self.x`/`self.y` without acquiring the state mutex (there
is a `todo` marker at the site). Harmless at current rates, but it is a known race.

### 7.2 Keep-out geometry

The per-obstacle constraint enforces:

```
dist(ego_ref_point, obstacle_center) >= ego_radius + obstacle_radius + safe_distance
```

| Parameter | Where declared | Default | Notes |
|---|---|---|---|
| `ego_radius` | `base_tracker.py` | `-1.0` (→ Carla-sized fallback) | restart-only |
| `safe_distance` | `base_tracker.py` | `0.5` | restart-only |
| `obstacle_radius` | from the message | — | SPHERE: `dimensions[0]` |

**For F1/10 keep the total small.** The obstacle weights files use `ego_radius: 0.15` +
`safe_distance: 0.15` → a ~0.5 m keep-out with a 0.2 m obstacle. The old 0.5 m
`safe_distance` default made an ~0.85 m bubble that swallowed the path and deadlocked
avoidance near the line.

At `v≈0` there is no yaw authority to steer around a fully on-path blocker
(`∂ψ̇/∂δ = (v/L)sec²δ = 0`), so stopping short of an occupied corridor is expected
behaviour, not a bug.

### 7.3 F1/10 obstacle runs

**`mpc.launch.py` has no `num_obstacles` launch argument** — it hardcodes
`num_obstacles: 0` and `ego_radius: 1.0`. Passing `num_obstacles:=1` will fail as an
unknown launch argument. Obstacles on hardware come from the **weights YAML overlay**
(which is applied last and wins).

acados — exact nonlinear keep-out, strongest for head-on / on-path:

```bash
# Terminal 1
ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=acados \
    platform:=f1tenth \
    weights:=f1tenth_acados_obstacle \
    load_waypoints:=true \
    waypoints_csv:=$HOME/waypoints_07152026.csv \
    global_frame:=odom \
    odom_topic:=/odometry/filtered \
    ackermann_cmd_topic:=/drive \
    stage_cost_type:=EXTERNAL \
    terminal_cost_type:=EXTERNAL \
    generate_mpc_model:=true

# Terminal 2 — obstacle in the ODOM frame (== global_frame)
ros2 run trajectory_following_ros2 fake_obstacle_publisher --ros-args \
    -p obstacle_x:="[3.0]" -p obstacle_y:="[0.0]" -p obstacle_radius:="[0.2]" \
    -p frame_id:=odom -p publish_rate:=10.0
```

CasADi — exact sqpmethod path:

```bash
ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=casadi \
    use_opti:=false \
    platform:=f1tenth \
    weights:=f1tenth_casadi_obstacle \
    load_waypoints:=true \
    waypoints_csv:=$HOME/waypoints_07152026.csv \
    global_frame:=odom \
    odom_topic:=/odometry/filtered \
    ackermann_cmd_topic:=/drive \
    generate_mpc_model:=true
```

Multiple obstacles — equal-length arrays, and bump `num_obstacles` in the weights YAML to
match:

```bash
ros2 run trajectory_following_ros2 fake_obstacle_publisher --ros-args \
    -p obstacle_x:="[3.0, 6.5, 9.0]" \
    -p obstacle_y:="[0.0, -0.4, 0.5]" \
    -p obstacle_radius:="[0.2, 0.2, 0.3]" \
    -p frame_id:=odom
```

Which CasADi obstacle path to use:

| Weights file | Path | Keep-out | Head-on |
|---|---|---|---|
| `f1tenth_casadi_obstacle` | `quad`/`qrqp` sqpmethod, nonlinear model | **exact** squared distance | strong — **use this** |
| `f1tenth_casadi_qp_obstacle` | `qp`/`qrqp` LTV | linearized + slacked | weak (gradient vanishes head-on), but ~0.4–2.9 ms |

`ltv` + `num_obstacles > 0` is rejected for every `solver_type` except `qp` — obstacle
constraints are nonconvex. `solver_type: qp` + CBF is rejected outright; **CBF is
IPOPT-only** (non-convex, incompatible with QP solvers).

Slack lever: `slack_weights_obstacle_avoidance` (default `[1000.0]`). Raising it hardens
avoidance — try `1000 → 10000 → 100000` to force progressively larger deviation.
Restart-only, and it changes the OCP, so pair with `generate_mpc_model:=true`.

### 7.4 CARLA + obstacles

Use `config/weights/carla_acados_obstacle.yaml` — the tuned `carla_acados.yaml` set plus
the obstacle block scaled to the CARLA ego. Key values and why:

| Param | Value | Why |
|---|---|---|
| `ego_radius` / `ego_disc_offsets` | `1.5` / `[0.18, 2.53]` | Two discs covering a ~4.69 × 1.85 m body with ~1.0 m rear overhang (the two half-rectangle circumcircles, same construction as the F1/10 file). A single rear-axle disc lets the front of the car clip an obstacle at *positive* reported clearance. If your ego blueprint differs, recompute per the comment in the file. |
| `safe_distance` | `0.8` | Vehicle-scale comfort margin (F1/10 uses 0.15). Also the band the braking envelope can spend during recovery. Raise it to start detours earlier and carry more speed. |
| `max_iter` | `20` (+ `termination_condition: 0.1`) | A **time budget**, not a convergence knob: near a nonconvex keep-out the SQP limit-cycles and eats any budget; the budget-limited status-2 iterate is accepted RTI-style. Do not raise it to "fix" status-2 floods — worst-case solve time scales with the cap. |
| `obstacle_slack_weight` / `_quadratic_weight` | `1000` / `10000` | Linear term keeps shallow margin use expensive; quadratic dominates deep penetration while keeping a recovery path from an already-infeasible start. |

`num_obstacles` is **not** pinned in the file — pass it at launch (it is a structural
override applied after the overlays). `ego_disc_offsets`, `num_obstacles`, and the slack
weights all change the generated OCP → pair any change with `generate_mpc_model:=true`.

Rebuild (YAML files are copied into `share/`, not symlinked), then launch:

```bash
cd ~/ros2_ws && python3 -m colcon build --packages-select trajectory_following_ros2 --symlink-install
source install/setup.bash

ros2 launch trajectory_following_ros2 mpc.launch.py \
    mpc_toolbox:=acados platform:=carla weights:=carla_acados_obstacle \
    load_waypoints:=true \
    waypoints_csv:=$HOME/ros2_ws/src/trajectory_following_ros2/data/carla_town01_moving.csv \
    stage_cost_type:=EXTERNAL terminal_cost_type:=EXTERNAL use_sim_time:=true

# CARLA's global_frame is `map`, so publish obstacles in map coordinates:
ros2 run trajectory_following_ros2 fake_obstacle_publisher --ros-args \
    -p obstacle_x:="[120.0]" -p obstacle_y:="[2.0]" -p obstacle_radius:="[1.0]" \
    -p frame_id:=map -p use_sim_time:=true
```

Pick the x/y from a point on your recorded route — read the CSV and place the obstacle
near (or exactly on) a waypoint you want to test against. Two obstacles whose keep-outs
leave a corridor narrower than the vehicle are supported: the reference projection merges
them into one enclosing circle and routes around the group (the OCP keeps the true
circles), so expect a visibly wider detour than the drawn per-obstacle keep-outs — that
is intentional, not a bug.

The **braking envelope** (`obstacle_braking_envelope`, bool node param, default `True`,
hot-reloadable) is the predictive safety layer: it compares stopping room (reaction
tick + actuation lag at `max_decel`) against measured physical clearance, latches a hold
on first fire, and releases when the margin recovers. **Its right setting is
scale-dependent**, and the two platforms' A/Bs came out opposite ways:

- **F1/10: leave it on.** With it off on the two-obstacle course the reactive layers
  only fire *after* intrusion — physical overlaps returned and the run stalled. The
  f1tenth obstacle YAMLs keep the default `True`.
- **CARLA: the obstacle weights YAMLs set it `false` (advisory).** The envelope
  extrapolates straight-line motion, so it cannot see the planned swerve; at ~9 m/s its
  `v²/2·max_decel` term fires ~17 m out on every pass, and because the reference
  projection grazes the keep-out boundary the latch cannot release mid-pass — the car
  creeps past every obstacle at ~1–2 m/s. A full-route A/B (6 obstacles,
  `carla_town01_moving`) showed off = equal-or-better executed clearance (acados
  +0.7999 vs +0.7998 m; CasADi +0.797 vs **+0.325** m — the brake-lurch cycle actually
  degraded tracking) with 0 overlaps and a 20–26 % faster run. The failure-triggered
  layers (`unsafe_iterate`, `obstacle_solve_failure`, `failure_policy`) stay active, and
  advisory mode still logs every margin/clearance diagnostic to the CSV.
- **Caveat — moving actors:** that A/B used static obstacles. For a first live CARLA run
  with moving vehicles/pedestrians, consider re-enabling it for the session
  (`ros2 param set <controller> obstacle_braking_envelope true`) until the solver's
  handling of reported obstacle motion is validated at speed.

Note `max_decel` and the actuation lag are what the envelope's stopping-room math
assumes: keep `max_decel` honest for the CARLA ego.

For post-run analysis, set `solver_log_file` in the weights YAML (§8.2) — the CSV
carries per-tick clearance, safety-stop reason, and proposed-vs-applied commands.

To use CARLA's **own** actors instead of the fake publisher, the ros-bridge publishes
`derived_object_msgs/ObjectArray` on `/carla/ego_vehicle/objects`. Pass it as a launch
argument on either launch file:

```bash
    obstacle_topic:=/carla/ego_vehicle/objects
```

Use the **ego-scoped** topic, not the world-scoped `/carla/objects`: the latter reports
the ego vehicle itself, which would make the car its own keep-out and stop it dead.

Setting `obstacle_topic:` in the weights YAML also works and stays supported — the launch
argument is applied after the overlays, so it wins if both are set.

With a live feed the reported object **count varies** as actors spawn and despawn, unlike
the fixed-length fake publisher. Nothing extra is required for that, but it is the
condition that used to crash obstacle ranking, so on the first live run confirm the
controller survives an actor appearing or disappearing mid-route rather than assuming it.

**Verify the frame first** — §7.1 applies in full here:

```bash
ros2 topic echo /carla/ego_vehicle/objects --once | grep -m2 frame_id
```

If that reports anything other than `map` (CARLA's `global_frame`), the positions will be
silently misinterpreted and you need a transform node in between. Also note that feed
reports **all** actors including ones behind the ego, and the nearest-N sort does not care
about direction — filter upstream or raise `num_obstacles`.

### 7.5 What normal obstacle logs look like

- acados `status=2` blips while negotiating a keep-out are **expected**, not failures. Near
  a nonconvex keep-out the SQP limit-cycles on the stationarity residual while every inner
  QP succeeds; the near-converged iterate is usable and the adapter accepts it when finite.
- On a hard failure (status 1/3/4 or a non-finite status-2) the adapter resets the solver
  and re-seeds from the reference. Without that, one NaN iterate would latch failure
  permanently, since acados warm-starts from its own internal memory.
- A reference point that lands inside a keep-out is projected onto the keep-out boundary
  with per-obstacle side hysteresis, so the car picks a side and stays with it instead of
  flip-flopping head-on. Debug topics show the **raw** reference; the projection happens on
  a copy.
- Expected on-path result: ~0.5–0.6 m detour, rejoin downstream, 0 failure latches.
- `N obstacle keep-outs merged into 1 projection circle(s)` INFO when two keep-outs pinch
  a too-narrow corridor — the reference routes around the group while the OCP keeps the
  true circles. Expected at close obstacle pairs.
- **Safety interventions** print a throttled WARN plus an action line, with the reason:
  - `braking_envelope` — the predictive stopping-room envelope fired. Its action is a
    bounded decel that *keeps the solver's steering* (not a hard zero), and it latches:
    one fire arms a hold that releases when the keep-out margin recovers. Bursts of a few
    consecutive ticks around an obstacle pass are normal; a hold that never releases is
    not.
  - `unsafe_iterate` — the solver's returned plan would put an ego disc in *physical*
    overlap; hard-zeroed. An in-contact vehicle's reverse escape is still accepted
    (recovery iterates that go no deeper than an already-overlapping stage 0 pass).
  - `obstacle_solve_failure` — any failed solve with an obstacle selected; hard-zeroed
    (never bridged with the last command).
  - Occasional singles are fine. With the envelope **on**, `unsafe_iterate` /
    `obstacle_solve_failure` should be rare — they are the after-the-fact layers. Frequent
    hard-zeros usually mean the envelope was disabled or `max_decel` is optimistic.
- If `num_obstacles > 0` and the obstacle feed goes silent (publisher died), the node
  warns every 5 s. A run with no detections and every clearance column `nan` is a run
  that never saw an obstacle — not a pass.

---

## 8. Verification checklist and diagnostics

### 8.1 Is it working?

```bash
# Commands going out
ros2 topic echo /drive

# Solve time — must stay under 1/control_rate (50 ms at 20 Hz)
ros2 topic echo /mpc/solve_time

# Reference vs prediction
ros2 topic echo /mpc/goal_point --once
ros2 topic hz /mpc/predicted_path

# Rough per-backend solve-time expectation (F1/10, horizon 25):
#   acados ~2 ms | CasADi quad/qrqp ~3.5-6 ms | CasADi qp/qrqp ~0.4-3 ms | do-mpc ~21 ms
```

### 8.2 Per-solve CSV log

Backend-agnostic, opt-in, restart-only. One row per solve at the single `solver.solve()`
call site, so it captures **every** tick including the zero-command fallback ones.

`solver_log_file` is a **node parameter and not a launch argument**, so under
`ros2 launch` you cannot pass it on the command line — add it to the weights YAML you are
launching with:

```yaml
# in config/weights/f1tenth_acados.yaml — then rebuild (YAMLs are copied, not symlinked)
    solver_log_file: '/home/<user>/solve_log.csv'   # '' = disabled
```

Or run the controller node directly, bypassing the launch file:

```bash
ros2 run trajectory_following_ros2 coupled_kinematic_acados --ros-args \
    --params-file ~/ros2_ws/install/trajectory_following_ros2/share/trajectory_following_ros2/config/mpc_parameters.yaml \
    --params-file ~/ros2_ws/install/trajectory_following_ros2/share/trajectory_following_ros2/config/platforms/f1tenth.yaml \
    --params-file ~/ros2_ws/install/trajectory_following_ros2/share/trajectory_following_ros2/config/weights/f1tenth_acados.yaml \
    -p solver_log_file:=$HOME/solve_log_$(date +%H%M).csv
```

(`--params-file` is last-wins, which is exactly the overlay order the launch file applies.
You must start `waypoint_loader` separately in this mode.)

Columns, in groups:

- **Solve:** `wall_time, ref_idx, solve_time_ms, status, is_optimal,
  consecutive_failures, accel_cmd, steering_cmd, velocity_cmd, error`
- **Obstacle:** `n_selected, sel_id, sel_side, sel_min_clearance, sel_ego_clearance` —
  `sel_min_clearance` is the minimum over the *planned* horizon (the constraint is
  slacked, so a plan dipping into the margin is expected); `sel_ego_clearance` is where
  the vehicle measurably *was* this tick. Both are worst-of-all-ego-discs against the
  keep-out (`ego_radius + obstacle_radius + safe_distance`).
- **Safety:** `safety_stop, safety_reason, safety_obstacle_id, physical_clearance,
  closing_speed, stopping_room` — `physical_clearance` is body-to-body (excludes
  `safe_distance`; negative = real overlap), recorded every tick, not just on
  interventions. `safety_reason` ∈ {`braking_envelope`, `unsafe_iterate`,
  `obstacle_solve_failure`, `failure_policy`}.
- **Applied:** `applied_accel, applied_steering, applied_speed` — what was actually
  published. `*_cmd` above are what the solver *proposed*; they differ on every tick an
  intervention fired, so judge command traces on `applied_*`.
- **Tick timing:** `tick_interval_ms, reference_ms, obstacle_ms, solver_wall_ms,
  pre_log_ms, previous_log_write_ms` — report cadence (p50/p95/max `tick_interval_ms`)
  before quoting solver stats.

Quick pass/fail read of a run: physical-overlap rows (`physical_clearance < 0`) must be
0; `n_selected` must equal the expected obstacle count throughout (a silent feed voids
the run); check the longest consecutive `safety_stop` run for hold deadlocks.

To see the solver's own per-iteration output instead, set `suppress_solver_output:=false`
(CasADi).

### 8.3 Record a run for offline analysis

```bash
ros2 bag record -o run_$(date +%m%d_%H%M) \
    /odometry/filtered /drive /trajectory/path /trajectory/speed \
    /mpc/predicted_path /mpc/reference_path /mpc/goal_point /mpc/solve_time \
    /fake_obstacles/object_array /tf /tf_static
```

Note: replaying a bag with a backward clock jump resets lap progress when `loop != 0`.

### 8.4 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Car doesn't move, no errors | Path or speeds never arrived | `ros2 topic echo /trajectory/path --once`. Check `load_waypoints:=true` and that loader QoS is transient-local. |
| `Stale odometry, skipping solve` | Odom gap > 0.5 s | Fix the odom source. Controller zero-commands and holds. |
| Launch hangs on startup, no log | Broken TF chain — `wait_for_transform_async` waits forever | `ros2 run tf2_ros tf2_echo <global_frame> <robot_frame>` |
| Zero commands after ~5 solves | 5 consecutive non-optimal solves → safety fallback | Genuine infeasibility. Obstacle on the vehicle? Check `/mpc/solve_time` and the solver log. Note: a low `max_iter` alone will **not** trip this — budget-limited returns count as optimal. |
| Steering jitters (acados) | Cost type is not `EXTERNAL` → `Rd` silently dropped | `stage_cost_type:=EXTERNAL terminal_cost_type:=EXTERNAL` |
| Silently stalls at a sharp corner under noise, index frozen, no failure flag | `nlp_solver_type: SQP_RTI` — completes in clean sim, fails with noise + lag | Use `SQP` (the node default). |
| Corner cutting | `distance_tolerance` anchors the reference past the apex | Lower it toward the backend floor (§1.4); also lower `max_lateral_accel` |
| Route progress stalls, car creeps to a stop at a corner | `distance_tolerance` below the backend floor — preview too short | Raise it. **Do not** work around it by inflating it elsewhere. |
| "Final goal reached" on lap 1 of a closed loop | `projection_window` ≥ loop length — the anchor leaps to end-of-path points near the start | Lower `projection_window` below the loop length (keep it above one tick's `v·dt`) |
| Obstacle ignored / phantom obstacle near origin | Objects published in a sensor frame; no TF applied | Publish in `global_frame` (§7.1) |
| Relevant obstacle missing from the OCP | Nearest-N sort picked one behind the car | Filter upstream or raise `num_obstacles` (§7.1) |
| CasADi `conic process failed` at standstill | Opti + bare QP solver: singular QP at `v=0` | You are running `use_opti:=false` — not applicable. If you must use Opti, pair with `solver:=ipopt`. |
| `rerun` import fails inside a node | Built with bare `colcon` → system-python shebang | Rebuild with `python3 -m colcon` (§0.1) |
| acados param/YAML change has no effect | Weights overlay applied last, wins over launch args | Edit the YAML (§1.1) |
| OCP change ignored | Reused stale generated C code | `generate_mpc_model:=true` |

---

## 9. Command reference tables

### 9.1 Args shared by both launch files

| Argument | Default | Values |
|---|---|---|
| `mpc_toolbox` | `acados` (mpc) / `casadi` (sim) | `acados` \| `casadi` \| `do_mpc` \| `none` |
| `control_type` | `mpc` | `mpc` \| `purepursuit` |
| `platform` | `''` | `f1tenth` \| `carla` |
| `weights` | `''` | any `config/weights/<name>.yaml` basename |
| `waypoints_csv` | `data/carla_waypoints.csv` (mpc) / `data/waypoints.csv` (sim) | path |
| `use_opti` | `false` | `true` \| `false` — **authoritative over overlays** |
| `ode_type` | `discrete_kinematic_coupled` | `discrete_kinematic_coupled` \| `continuous_kinematic_coupled` \| `discrete_kinematic_coupled_augmented` |
| `discrete_model_type` | `nonlinear` | `nonlinear` \| `ltv` |
| `discrete_integration_method` | `rk4` | `rk4` \| `euler` |
| `stage_cost_type` / `terminal_cost_type` | `EXTERNAL` | `EXTERNAL` \| `NONLINEAR_LS` \| `LINEAR_LS` |
| `code_gen_directory` | `<share>/data/mpc` | path |
| `global_frame` | `odom` | frame |
| `robot_frame` | `base_link` | frame |
| `odom_topic` | `odometry/local` | topic |
| `arclength_index_advance` | `True` | `True` \| `False` |
| `projection_window` | `5.0` | metres — **< loop length** |

### 9.2 `mpc.launch.py` only

| Argument | Default | Notes |
|---|---|---|
| `log_level` | `info` | `debug` \| `info` \| `warn` \| `error` |
| `load_waypoints` | `True` | launches `waypoint_loader` |
| `use_sim_time` | `false` | not an arg on `closed_loop_sim.launch.py` |
| `max_iterations` | `15` | node param is `max_iter`; `closed_loop_sim` uses `max_iter` directly |
| `load_params_from_file` | `True` | loads `config/mpc_parameters.yaml` |
| `load_params_from_args` | `True` | enables the `SetParameter` block |
| `load_visualizer` | `False` | launches `trajectory_visualizer` |
| `frequency` | `50.0` | → `control_rate` node param (platform files override to 20.0) |
| `generate_mpc_model` | `False` | overridden by most weights YAMLs |
| `ackermann_cmd_topic` | `drive` | |
| `distance_tolerance` | `0.2` | overridden by platform/weights |
| `use_namespace` / `namespace` | `False` / `''` | |

### 9.3 `closed_loop_sim.launch.py` only

| Argument | Default | Notes |
|---|---|---|
| `simulator` | `do_mpc` | `do_mpc` \| `acados` |
| `num_obstacles` | `0` | **a launch arg here, not in `mpc.launch.py`** |
| `initial_x/y/yaw/speed` | `0.0` | spawn pose — restart a killed sim at the last live pose |
| `viz_backend` | `native` | `rerun` \| `native` \| `both` |
| `delay_compensation_enabled` | `false` | with `estimated_delay`, `delay_compensation_method` |
| `solver_type` | `nlp` | `nlp` \| `quad` \| `qp` \| `conic` |
| `solver` | `ipopt` | `ipopt` \| `qrqp` \| `osqp` \| `qpoases` |
| `max_iter` | `200` | semantics vary by solver path — see CLAUDE.md |
| `integrator_type` | `ERK` | `ERK` \| `DISCRETE` (acados) |
| `step_on_command` | `false` | |

### 9.4 Valid CasADi `(solver_type, solver)` combos

Validated at startup — an invalid pair raises `ValueError` with the supported matrix.

| `solver_type` | `solver` | `max_iter` means | Safe range |
|---|---|---|---|
| `quad` | `qrqp` | outer SQP iterations | 10–30 (YAMLs use 200; safe) |
| `nlp` | `ipopt` | IPOPT interior-point iterations | 100–2000 |
| `quad` | `ipopt` | inner IPOPT iters per QP | 100–500 |
| `nlp`/`conic` | `osqp` | ADMM steps | 500–4000 |
| `qp` | `qrqp`/`osqp`/`qpoases` | **unused** — bound via `qp_inner_max_iter` | n/a |

`qp` requires `discrete_model_type=ltv` + a discrete `ode_type` + `use_opti=False`, and
needs `slack_objective_is_quadratic: True`.

### 9.5 Node names for `ros2 param set`

**The two launch files name the same nodes differently.** Always confirm with
`ros2 node list` before a `ros2 param set` — a set against a wrong node name fails, and in
a hurry that reads like the parameter was rejected.

| Node | via `mpc.launch.py` | via `closed_loop_sim.launch.py` | via `ros2 run` |
|---|---|---|---|
| CasADi controller | `casadi_mpc_node` | `kinematic_coupled_casadi_controller` | `kinematic_coupled_casadi_controller` |
| acados controller | `acados_mpc_node` | `kinematic_coupled_acados_controller` | `kinematic_coupled_acados_controller` |
| do-mpc controller | `do_mpc_node` | `kinematic_coupled_do_mpc_controller` | `kinematic_coupled_do_mpc_controller` |
| Pure Pursuit | `purepursuit_node` | `purepursuit_controller` | — |
| Waypoint loader | `waypoint_loader_node` | `waypoint_loader` | `waypoint_loader` |
| Visualizer | `trajectory_visualizer_node` | `trajectory_visualizer` | `trajectory_visualizer` |
| acados simulator | — | `kinematic_acados_simulator` | `kinematic_acados_simulator` |
| do-mpc simulator | — | `kinematic_dompc_simulator` | `kinematic_do_mpc_simulator` |
| Obstacle publisher | — | — | `fake_obstacle_publisher` |

Note the do-mpc simulator: the launch file renames it to `kinematic_dompc_simulator`, but
run directly it is `kinematic_do_mpc_simulator` (underscore between `do` and `mpc`).

---

## Related documents

- `CLAUDE.md` — architecture overview, parameter semantics, backend internals
- `docs/GOTCHAS.md` — non-obvious footguns (unit conversions, launch wiring, solver quirks)
