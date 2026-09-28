# Non-obvious Gotchas

Surprising behaviours and footguns in this codebase. Read this before debugging unit mismatches, launch wiring, or solver cost configuration.

## Contents

- [Steering angle units](#steering-angle-units)
- [`max_steer_rate` unit conversion](#max_steer_rate-unit-conversion)
- [`frequency` launch arg → `control_rate` node param](#frequency-launch-arg--control_rate-node-param)
- [Simulator uses `update_rate`, not `control_rate`](#simulator-uses-update_rate-not-control_rate)
- [`control_type=purepursuit` vs `mpc_toolbox`](#control_typepurepursuit-vs-mpc_toolbox)
- [`speed` uses `vx`, not vector magnitude](#speed-uses-vx-not-vector-magnitude)
- [acados cost-type switching silently drops `Rd`](#acados-stage_cost_typeterminal_cost_type-switching-silently-drops-rd)
- [Heading error must be wrapped with `atan2`, not `fmod`](#heading-error-must-be-wrapped-with-atan2-not-fmod)
- [`save_interval` must be matched to platform speed](#save_interval-must-be-matched-to-platform-speed)
- [`do_mpc` must be installed at user level](#do_mpc-must-be-installed-at-user-level-not-venv-only)
- [`wait_for_transform_async` hangs on broken TF](#wait_for_transform_async-hangs-indefinitely-on-broken-tf)
- [Parameter vector ordering — cost weights after obstacle params](#parameter-vector-ordering--cost-weights-after-obstacle-params-technical-debt)
- [Obstacles are not TF-transformed](#obstacles-are-not-tf-transformed)
- [Obstacle relevance is measured against the horizon, not the vehicle](#obstacle-relevance-is-measured-against-the-horizon-not-the-vehicle)
- [Undeclared launch arguments are silently ignored](#undeclared-launch-arguments-are-silently-ignored)
- [Node names differ between the two launch files](#node-names-differ-between-the-two-launch-files)
- [One logging helper cannot serve two severities](#one-logging-helper-cannot-serve-two-severities)
- [The keep-out disc does not cover the vehicle body](#the-keep-out-disc-does-not-cover-the-vehicle-body)
- [A silent obstacle feed looks exactly like a clean run](#a-silent-obstacle-feed-looks-exactly-like-a-clean-run)

## Steering angle units

`min_steer`/`max_steer` are declared and stored in **degrees** in `config/mpc_parameters.yaml` and `_declare_common_parameters()`. `_read_parameters()` converts them to radians via `math.radians()` and stores them as `self.MIN_STEER_ANGLE`/`self.MAX_STEER_ANGLE` (radians). However, CasADi backends receive them back as degrees — `coupled_kinematic_casadi._init_solver()` calls `np.degrees(self.MIN_STEER_ANGLE)` before passing to the solver. The acados backend uses radians (inherits via `self.MAX_STEER_ANGLE` directly). When writing a new backend, check which unit convention the underlying solver expects.

## `max_steer_rate` unit conversion

Like `min_steer`/`max_steer`, `max_steer_rate` is declared and configured in **deg/s** (default `60/0.17 ≈ 353 deg/s`). `_read_parameters()` converts it to rad/s via `math.radians()` and stores as `self.MAX_STEER_RATE`. Backends receive the rad/s value.

## `frequency` launch arg → `control_rate` node param

In `mpc.launch.py` the launch argument is declared as `frequency` (default `50.0`), but it maps to the ROS parameter named `control_rate` via `SetParameter(name='control_rate', value=frequency, ...)`. If you query or set this parameter at runtime, use `control_rate`, not `frequency`.

## Simulator uses `update_rate`, not `control_rate`

`BaseSimulator` declares `update_rate` (default `50.0` Hz) as its timer frequency parameter. Controller nodes use `control_rate`. Keep them consistent when testing closed-loop.

## `control_type=purepursuit` vs `mpc_toolbox`

The two launch args are independent selectors. `control_type=purepursuit` launches the purepursuit node regardless of `mpc_toolbox`. The MPC nodes (acados/casadi/do_mpc) are gated on `mpc_toolbox` only; they will also launch if `control_type=mpc` is set together with the matching `mpc_toolbox` value. There is no guard preventing both a purepursuit node and an MPC node from running simultaneously if args are mis-combined.

## `speed` uses `vx`, not vector magnitude

`odom_callback` sets `self.speed = twist.linear.x` (longitudinal component only). For reversing, `vx` must be negative in the odometry — the controller infers direction from its sign. A localizer that always reports positive speed magnitude will break reversing.

## acados `stage_cost_type`/`terminal_cost_type` switching silently drops `Rd`

Switching from `EXTERNAL` to `LINEAR_LS` or `NONLINEAR_LS` via the ROS parameters requires no code changes — `acados_settings.py` branches at build time and `AcadosSolverAdapter.solve()` gates `cost_set('yref', ...)` on the active type. However, the `Rd` input rate penalty is **only included in `EXTERNAL`** cost (via `u_prev` in the parameter vector). `LINEAR_LS` and `NONLINEAR_LS` have no mechanism to inject `u_prev` into the LS residual without augmenting `y`, so switching away from `EXTERNAL` silently removes the rate cost. Check `acados_settings.py` lines 208–219 (`EXTERNAL` branch) vs lines 194–206 (`NONLINEAR_LS` branch) to see the difference.

## Heading error must be wrapped with `atan2`, not `fmod`

The yaw tracking error `psi - psi_ref` must be wrapped to `[-pi, pi]` before it enters the cost, or it explodes to ~2*pi whenever the reference heading crosses ±pi (vehicle travelling in -x). The unwrapped error sends the solver the long-way-around gradient, manifesting as steering saturation and IPOPT timeouts.

Two ways to get this wrong:

- **`casadi.fmod(psi - psi_ref + pi, 2*pi) - pi`** — CasADi's `fmod` follows C semantics (result takes the **sign of the dividend**, range `(-2*pi, 2*pi)`), so this trick is wrong whenever `psi - psi_ref < -pi`. This bit both the CasADi backends and `do_mpc/model.py`. The correct, C1-smooth form is **`atan2(sin(e), cos(e))`** — see `casadi/_casadi_base.py::_wrap_angle`. (Note: NumPy/Python `%` follows the *divisor* sign, so `(e + pi) % (2*pi) - pi` in host-side `utils/trajectory_utils.py::normalize_angle` is correct — the bug is specific to `casadi.fmod` in the solver graph.)

- **acados plain subtraction.** acados forms the cost residual itself, so the wrap is applied differently per cost type (all three are now handled; code comments mark each branch):
  - **`EXTERNAL`** (default) and **`NONLINEAR_LS`** — wrapped *in-solver*: the psi component of the cost expression is `atan2(sin(psi - psi_ref), cos(psi - psi_ref))`, using the `psi_ref` carried in the `zref` parameter. For `NONLINEAR_LS` this means `AcadosSolverAdapter.solve()` must set the **psi entry of `yref` to 0** (the residual is already formed in the expression — see `acados_settings.py` and the adapter).
  - **`LINEAR_LS`** — *cannot* wrap in-solver (residual `Vx·x + Vu·u − yref` is linear by construction). Instead the adapter pre-unwraps the psi reference trajectory (`np.unwrap`) and shifts it by a multiple of 2π to sit within ±π of the current heading, so `psi − psi_ref` is already wrapped. This is robust as long as the predicted heading doesn't diverge more than π from the reference mid-horizon (always true for a tracking MPC with a sane horizon).

## `save_interval` must be matched to platform speed

The default is **0.1 s (10 Hz)** (`waypoint_recorder.py`), which is adequate for an F1/10
but coarse at full-size speeds — spacing is `save_interval × speed`, so the same setting
that gives 15 cm on an F1/10 gives 2.5 m in CARLA. Too coarse is too sparse for MPC
lookahead and makes the curvature filter noisy. Recommended recording rates by platform:

| Platform | `save_interval` | spacing at typical speed |
|---|---|---|
| F1/10 — indoor/campus | 0.1 s (10 Hz) | ~0.3 m at 3 m/s |
| F1/10 — outdoor, faster | 0.1 s (10 Hz) | ~0.5 m at 5 m/s |
| Full-size urban (≤50 km/h) | 0.2 s (5 Hz) | ~2 m at 10 m/s |
| Full-size highway / CARLA | 0.1 s (10 Hz) | ~2.5 m at 25 m/s |

`min_distance` (default 0.0 = disabled) can cap density further, but it suppresses dwell time at intentional stops (stop signs, pickups). Use `stale_odom_timeout` alone when the route includes stops — a stopped vehicle still publishes odom so writes continue, while a dead topic is correctly skipped.

## `do_mpc` must be installed at user level, not venv-only

`pip install do_mpc --user` is required because the hardcoded `PYTHONPATH` in `~/.bashrc` (set by acados and ROS) overwrites the venv's `site-packages` contribution. Installing into a venv only will result in `ModuleNotFoundError` at runtime even though the venv is activated. CasADi works because it was pre-installed at `~/.local`; acados works via its own `LD_LIBRARY_PATH` + path setup.

## `wait_for_transform_async` hangs indefinitely on broken TF

`path_callback` uses `await tf_buffer.wait_for_transform_async(...)` (with `Time()`) to prevent the one-shot transient-local path message from being lost when TF isn't ready at startup. The trade-off: `wait_for_transform_async` with `Time()` has no built-in timeout. If the TF chain is permanently broken — e.g. a typo in `global_frame`, or `robot_localization` never starts — the coroutine suspends in the executor's event loop indefinitely with no further output after the initial `INFO` log. The controller appears to receive no path and never initializes, with no error message. **Symptom:** node starts, path subscription fires once (visible in `ros2 topic echo`), control loop never begins, no further logs. **Workaround:** wrap with `asyncio.wait_for(tf_buffer.wait_for_transform_async(...), timeout=30.0)` and/or cache `self._pending_path_msg` so the path can be retried once TF comes up.

## Parameter vector ordering — cost weights after obstacle params (technical debt)

In both CasADi backends (`kinematic_mpc_casadi.py`, `discrete_kinematic_mpc_casadi.py`) and acados (`acados_settings.py`), cost weight symbols (`Q`, `R`, `Rd`, `Qf`) are appended to the parameter vector *after* obstacle params. This is because obstacles (Item 3) landed before runtime weights (Item 8) in the refactor. Logically weights should come first (solver-permanent vs. transient per-solve). When restructuring for Fatrop/HPIPM or adding a new backend, move cost weights before obstacle states and update all indexing in those three files.
## acados obstacle constraint: never use `1e15` for the "infinite" upper bound — and `ush` is *not* an upper bound

The obstacle keep-out in `acados_settings.py` is one-sided (`h = dist² − keepout² ≥ 0`), so the upper bound `uh` is nominally infinite. **Do not encode that as `1e15`.** HPIPM's interior-point barrier cannot scale a box of that magnitude, so the inner QP returns `qp_stat=2` (max-iter/failure) on *every* SQP iteration and the whole OCP reports `status=2` — even from a standstill with the obstacle far away and inactive. The car then never moves (5 consecutive suboptimal solves → `base_tracker` zero-command fallback). The failure is invariant to obstacle distance and to `max_iter` (200 QP iters still fails), which makes it look like a solver/tuning problem when it is purely bound scaling.

**Use a finite, well-scaled `uh=1e8`**, and **leave `lsh`/`ush` at their zero defaults.**

The `ush` half of this is a trap worth stating plainly: in acados **both `lsh` and `ush` are *lower* bounds on the slack variables** ("Lower bounds on slacks corresponding to soft lower/upper bounds… zeros by default" — `acados_ocp.py`), *not* upper bounds. So the original `ush=1e15` did not cap the slack at 1e15; it **forced the upper-side slack to be ≥ 1e15**, which is the far more plausible cause of the HPIPM breakdown than `uh` itself. Anything non-zero here just pushes a slack variable away from 0 for no benefit; slacks are unbounded above by default, which is exactly what a soft constraint wants.

An earlier revision set `ush=1e6` alongside `uh=1e8` and recorded that "both bounds must be reduced — lowering only one still fails." That was measured with both changed together, and is superseded: `uh=1e8` with `ush` at its **0 default** converges and *improves* tracking. Measured on an offline on-path-obstacle hairpin rollout of the real acados adapter (900 ticks): `ush=1e6` → 0 failures, reference index reaches 1521; `ush=0` → 0 failures, index reaches **1751** (more of the course covered in the same ticks) at the same 0.593 m max CTE.

Keep `hpipm_mode='SPEED'`; `'ROBUST'` makes it worse (reports infeasible, status 4). Diagnose with `controller.get_stats('qp_stat')` and `('residuals')` (`[stat, eq, ineq, comp]`): a converged cost/dynamics (`stat`,`eq` ~1e-14) with `ineq`/`comp` frozen is the bound-scaling signature.

Separately, with the obstacle `con_h` active, raise the SQP `max_iter` (e.g. 20→40): ~20 leaves a short *consecutive* non-convergent cluster while the obstacle first enters the horizon during the standstill launch, which can approach the `_consecutive_failures ≥ 5` fallback gate; 40 breaks the cluster into isolated single ticks the near-converged iterate absorbs.

## acados silently ignores unknown OCP attributes — a mis-set option is a no-op, not an error

`AcadosOcp`, `AcadosOcpConstraints`, `AcadosOcpCost` and `AcadosModel` are plain Python objects with no `__slots__` and no unknown-field validation. Assigning a field this acados version does not have — a typo, or an option added in a newer release — **silently creates a dead attribute**. Nothing raises, the OCP builds, the solver runs, and the option simply never exists. The code generator reads its field list from `acados_layout.json`, so anything absent there is invisible.

This bit once already: a soft stage-0 input-rate constraint was written as `ocp.model.con_h_expr_0` + `constraints.lh_0`/`idxsh_0` + `cost.zl_0`/`Zl_0`. **Stage-0 nonlinear constraints do not exist in the pinned acados** (`AcadosModel` defines only `con_h_expr` and `con_h_expr_e`; `acados_layout.json` has no `nh_0`), so every one of those assignments was a no-op while the node logged `acados input-rate constraint: ON` and the docs recorded it as verified — "OCP compiles + solves with the stage-0 rate slack" is exactly what a silently-dropped constraint looks like. The rate limit now lives in `AcadosSolverAdapter._apply_input_rate_bound`, which tightens the stage-0 input box at runtime via `constraints_set(0, 'lbu'/'ubu', …)` — no codegen, no version dependency.

**Before trusting any new `ocp.*` option, prove it exists:**

```bash
python3 -c "from acados_template import AcadosOcp; print(hasattr(AcadosOcp().model, 'con_h_expr_0'))"
grep -c nh_0 $(python3 -c "import acados_template,os;print(os.path.dirname(acados_template.__file__))")/acados_layout.json
```

And verify the *effect*, not just that it ran: diff the generated C, or check that the constraint actually binds. If you later upgrade to an acados with stage-0 `h` support, note that setting `con_h_expr_0` also **overrides stage 0's `con_h_expr`** — which carries the obstacle keep-out — so the two cannot simply coexist.

## acados on-path obstacle: `status=2` is a usable iterate, and the reference must be projected out of the keep-out

An obstacle sitting **on** the reference line (e.g. a hairpin apex waypoint) makes the acados SQP return `status=2` (max iterations) on every tick it is in the horizon. This is a **limit cycle, not a blow-up**: every inner HPIPM QP succeeds (`qp_stat=0`) and the iterate stays feasible (`res_eq`/`res_ineq` ~1e-4), but with the default `FIXED_STEP` globalization the stationarity residual oscillates O(1) forever around the nonconvex keep-out. Neither `MERIT_BACKTRACKING`/funnel globalization, quadratic obstacle slack (`Zl>0`), nor warm-start resets fix it — do not burn time on those knobs.

Three things are required (all shipped): the first two get the car around the obstacle (either alone fails), the third keeps one bad tick from becoming a permanent stop:

1. **Accept budget-limited iterates.** `AcadosSolverAdapter.solve()` treats `status==2` as `is_optimal=True` when the returned `u`/`x` sequences are finite (RTI-style — the near-converged iterate is usable; identical policy to the CasADi adapter's `Maximum_Iterations_Exceeded` handling). Statuses 1/3/4 still count as failures and feed the 5-consecutive-failure zero-command fallback. `SolverResult.status` still carries the raw acados status for logs.
2. **Project the horizon reference out of the keep-out** (`trajectory_utils.project_reference_out_of_keepouts`, called inside the adapter on a *copy* of `xref` — debug topics keep the raw reference). A reference threading the bubble makes "park at the bubble edge" a genuine local optimum; at v=0 steering has no yaw authority so the optimizer can never see the detour. The projection sweeps in-bubble points onto the keep-out boundary arc. **Side hysteresis is essential:** head-on, the entry/exit angles are ~π apart, so the shortest-way arc direction is knife-edge and flips tick-to-tick (the car then chases an alternating left/right target and parks). The helper keeps the previously chosen go-around side whenever the sweep is ambiguous (`|sweep| > π/2`); the adapter feeds the returned `side_hints` back each tick.
3. **Reset the solver after a hard failure.** A hard failure (e.g. status 4 with `QP solver returned error status 3` — HPIPM NaN detection, observed once at ~1.5 m/s on the second pass of an on-path obstacle) leaves a NaN iterate in the acados solver memory. acados warm-starts every solve from that memory, so without a reset every subsequent solve re-linearizes at the poisoned iterate and fails at SQP iteration 1 indefinitely (`QP solver returned error status 3 in SQP iteration 1, QP iteration 1` on every tick) — the vehicle latches into the zero-command fallback permanently even though a cold-started solve succeeds. On any failed solve the adapter therefore calls `controller.reset()` (zeroes the primal/dual iterate and QP-solver memory) and re-seeds the stages from the current state + the keep-out-projected reference with zero inputs, and it sanitizes non-finite results so NaN never reaches the published command, the `u_prev` echo, or the predicted-path debug topic.

Verified offline (closed-loop rollout on the real `data/waypoints.csv` hairpin): full course through all three keep-out clusters, 0 failure latches, ~0.6 m apex deviation (parity with CasADi `quad/qrqp`); offset-obstacle case still completes.

## Obstacles are not TF-transformed

A property of `base_tracker._obstacle_callback` that is safe with the
`fake_obstacle_publisher` and a couple of static objects, but wrong the moment a live
perception feed is wired in:

**No TF.** The callback reads `obj.pose.position` directly and compares it against the
vehicle odometry. `data.header.frame_id` is never read. Obstacles must therefore already
be in the controller's `global_frame` (`odom` default, `map` for CARLA). An `ObjectArray`
published in a sensor frame (`laser`, `velodyne`, `camera_link`) is silently interpreted
as global coordinates — a phantom keep-out lands near the origin and the real obstacle is
invisible. There is no warning and no frame check. A perception stack that publishes in a
sensor frame needs a transform node between it and the controller; that node does not
exist in this package. (In the closed-loop sim `map → odom` is identity, so map/waypoint
coordinates work directly as `odom` coordinates.)

**Sizing `num_obstacles` still matters.** Ranking (see below) picks the right obstacles
when the slots are contended, but each slot costs constraints and solve time, and
`num_obstacles` is restart-only — it changes the OCP structure, so raising it needs
`generate_mpc_model:=true`. Size it above the expected clutter; ranking only decides who
wins once you are already over budget.

## Obstacle relevance is measured against the horizon, not the vehicle

Ranking obstacles by distance to the ego is wrong, and was the shipped behaviour until
2026-07-15: an obstacle 1 m *behind* outranked one 5 m *ahead*, and at `num_obstacles=1`
it evicted the relevant one from the OCP entirely.

`_select_obstacles` ranks against the point set `{ego} ∪ xref[0..N]` instead, because that
is what the solve can act on — the OCP constrains stages 0..N and the keep-out projection
bends those same reference points, so an obstacle far from all of them cannot influence
the result. Consequences worth knowing:

- **No heading term exists, and none is wanted.** The horizon only runs forward, so
  "behind" falls out for free. A forward-cone test would be actively wrong: on a hairpin
  an obstacle genuinely on the path ahead can sit at 90°+ bearing and would be cut.
- **The window is `v × prediction_time`**, so it scales with speed (~1.9 m on the F1/10
  against a 0.6 m keep-out, ~10 m at CARLA speeds). There is no lookahead parameter to
  tune, by design.
- **Intruders rank by *when* the keep-out bites, not how deep.** Two obstacles both on the
  reference are both ~0 m from it, so a plain minimum-distance rank ties and can pick the
  farther one. Do not "simplify" this back to `d.min(axis=1)`.
- **Ego is in the point set on purpose.** `xref[:, 0]` is anchored `distance_tolerance`
  *ahead* of the vehicle, so the vehicle's own position is otherwise uncovered — during an
  avoidance swerve the car is off-reference by construction, and an obstacle on top of it
  must not rank as irrelevant.

Selection runs in `base_tracker._control_timer_callback` *after* the reference is built,
not in `_obstacle_callback` (which has no horizon) and not in a node override (the packing
used to be duplicated in the acados and CasADi nodes, and ran before `xref` existed).

**Still no relative-velocity gate in the ranking.** `predict_obstacle_motion` (default
`True`) propagates each obstacle over the horizon at its reported `twist`, but ranking
itself uses current positions. A zero/absent twist makes propagation identical to a static
fill, which is why the default is safe — `fake_obstacle_publisher` never sets `twist`.
Turn it off if a perception feed reports velocities too noisy to extrapolate.

## Undeclared launch arguments are silently ignored

`ros2 launch` does **not** error on a `name:=value` pair the launch file never declared —
it accepts it and does nothing. Verified:

```console
$ ros2 launch trajectory_following_ros2 mpc.launch.py mpc_toolbox:=none control_type:=mpc loop:=0
[INFO] [launch]: All log files can be found below ...
[INFO] [launch]: Default logging verbosity is set to INFO
# no error — and `loop` was never applied
```

A *declared* arg with a bad value does error properly (`control_type:=none` →
`Valid options are: ['mpc', 'purepursuit']`), so validation only exists for args the file
knows about. A typo (`waypoint_csv:=`) or a param-that-isn't-an-arg fails silently and you
get default behaviour while believing you configured something.

The two launch files do **not** accept the same arguments. Confirmed no-ops:

| Passed to | Silently ignored | Because |
|---|---|---|
| `mpc.launch.py` | `loop`, `num_obstacles`, `integrator_type`, `solver_type`, `solver`, `safe_distance`, `solver_log_file`, `simulator` | not declared there |
| `closed_loop_sim.launch.py` | `load_waypoints`, `use_sim_time`, `log_level`, `frequency`, `distance_tolerance` | not declared there |

Name mismatches (launch arg ≠ node param): `max_iterations` → `max_iter` (on
`mpc.launch.py`; `closed_loop_sim.launch.py` uses `max_iter` directly), `frequency` →
`control_rate`, `waypoints_csv` → `file_path`.

Anything in that list has to come from a YAML overlay (`platform:=` / `weights:=`). Most
overlay values beat launch arguments. Structural build choices are deliberate exceptions:
`use_opti`, `num_obstacles`, and `generate_mpc_model` are applied after the overlays so
their explicit launch values remain authoritative.

Check what a file actually accepts: `ros2 launch <pkg> <file> --show-args`. Note that
`--show-args` lists only `DeclareLaunchArgument` entries; an undeclared
`LaunchConfiguration('x', default=...)` still reads a CLI `x:=` value but won't be listed.

## Node names differ between the two launch files

The same executable gets a different node name depending on how it is started, so a
`ros2 param set` that works under one launch file fails under the other. A wrong node name
reads like a rejected parameter.

| Node | `mpc.launch.py` | `closed_loop_sim.launch.py` | `ros2 run` |
|---|---|---|---|
| CasADi controller | `casadi_mpc_node` | `kinematic_coupled_casadi_controller` | `kinematic_coupled_casadi_controller` |
| acados controller | `acados_mpc_node` | `kinematic_coupled_acados_controller` | `kinematic_coupled_acados_controller` |
| do-mpc controller | `do_mpc_node` | `kinematic_coupled_do_mpc_controller` | `kinematic_coupled_do_mpc_controller` |
| Pure Pursuit | `purepursuit_node` | `purepursuit_controller` | — |
| Waypoint loader | `waypoint_loader_node` | `waypoint_loader` | `waypoint_loader` |
| Visualizer | `trajectory_visualizer_node` | `trajectory_visualizer` | `trajectory_visualizer` |
| do-mpc simulator | — | `kinematic_dompc_simulator` | `kinematic_do_mpc_simulator` |

`ros2 node list` first. Note the do-mpc simulator in particular: the launch file renames it
to `kinematic_dompc_simulator`, but run directly it is `kinematic_do_mpc_simulator`.

## The arc-length projection anchor can ratchet onto a later pass of a self-revisiting path

`project_index_and_lookahead` picks the nearest waypoint inside a forward arc window and feeds the pick back as the next tick's floor. On a route that drives the same physical neighbourhood more than once (a small closed loop revisiting a corner), a vehicle held laterally off its local segment — an obstacle-avoidance standoff — can make a *later-pass* point the nearest candidate at the window's far edge. The monotonic anchor then advances up to a full window per tick with no real progress, cascades pass to pass to the path tail, and the controller latches a false "Final goal reached" mid-course. **No `projection_window` value prevents this** — the window slides with the anchor. The guard is the `max_advance` gate: `Trajectory.calc_nearest_index` accrues signed along-track displacement (1.5 gain on the accrual — a gate-side gain is eaten by waypoint quantization and a lag then never closes; banked surplus capped at 0.5 m) and consumes actual anchor advance, so oscillation nets to zero and a parked or thrashing vehicle cannot ratchet. A re-acquisition escape (best gated candidate farther than the whole window from the vehicle) keeps genuine localization jumps recoverable. `base_tracker` additionally refuses to latch `end_of_path` unless the normal final-goal contract passes: the vehicle must be within `distance_tolerance`, below `speed_tolerance`, and beyond the start grace distance (warn + hold zero otherwise). When placing test obstacles "off the path", validate the position against the minimum distance to the ENTIRE path — an arc-length offset along one pass can land exactly on another pass.

## A Rerun 3-D world view silently drops every 2-D archetype

`viz/rerun_helpers.build_blueprint()` pins the world view type. It is a `Spatial3DView` (the vehicle footprint is a prism drawn with `Boxes3D`/`LineStrips3D`), and a Rerun 3-D view renders **only** 3-D archetypes: `Points2D`, `Arrows2D` and `LineStrips2D` logged into it are dropped with no warning and no error — the view is simply empty. Symptoms are total: no origin/axes, no full/predicted/reference path, no goal, no obstacles, no keep-out circles, while the time-series views keep working, which reads as "the visualizer is broken" rather than "the archetypes mismatch the view". `rerun_backend.py` therefore logs all ground-plane data as 3-D at z = 0 via `_xy()` / `_strip()`. **If the world view type ever changes, every spatial log has to change with it.** Related: pinning `BlueprintPanel(state='collapsed')` removes the entity/attribute toggles outright in the web viewer.

## The Rerun visualizer needs an explicit spatial-rate budget

Measured on the two-obstacle closed-loop course (acados, F1/10, 20 Hz target, visualizer
under `nice -n 10`), unrestricted Rerun spatial logging reduced the control cadence and
introduced long stalls. `rerun_spatial_frequency` therefore limits each high-volume
spatial stream independently (vehicle pose, prediction, reference window, goal, and
obstacles); it defaults to 5 Hz. Scalar time-series and the one-shot full path remain
unthrottled. Set it to `0` only for unrestricted/offline capture. A closed-loop Rerun run
is acceptable only when the solver CSV still shows approximately 50 ms tick cadence;
use `native` if the machine cannot sustain that.

On the reproduced two-obstacle run, the default 5 Hz cap produced 50.256 ms tick p50
and 73.209 ms p95 over 491 rows; the unrestricted run measured 86.285 ms p50 and
202.528 ms p95. The remaining tail latency still matters to the braking envelope.


## One logging helper cannot serve two severities

`rclpy`'s logger caches a context per **call site** — `(function, file, line, bytecode index)` — and raises `ValueError: Logger severity cannot be changed between calls.` if that site ever logs at a second severity. A helper that funnels every level through one dispatch line is therefore a latent crash, not a convenience:

```python
getattr(self._ros_logger, 'warn' if level == 'warning' else level)(text)   # one line, all severities
```

The first `info()` through it pins the line to INFO; the first later `warning()` raises. In the acados adapter that helper was called from inside the control-timer callback, so the exception propagated out of `executor.spin()` and **killed the controller node mid-run** (exit code 1, ~250 ticks in). Nothing catches it, and it needs both severities to have actually fired, so it survives smoke tests and appears only under the condition being diagnosed. The same rule applies to the logger *name* and to the filter kwargs (`throttle_duration_sec`, `once`, `skip_first`): one call site, one severity, one name, one filter set. Give each severity its own line.

## The keep-out disc does not cover the vehicle body

`ego_radius` is measured from the **rear-axle reference point**, which is not the centre of the vehicle. With a body that extends ahead of the rear axle, a single keep-out disc on that point reports positive clearance while the front of the car is inside the obstacle: for the f1tenth footprint (0.58 x 0.31 m, rear axle 0.19 m behind the body centre) the front corner sits ~0.50 m from the reference point, so at `ego_radius: 0.15` + `safe_distance: 0.15` the solver could hold the reference point exactly on the 0.60 m boundary while the body reached ~0.10 m from a 0.3 m obstacle's centre. Measured in an acados plan: the front disc ended **0.291 m inside** the keep-out on a solve that reported clearance 0.000. Both `sel_min_clearance` and `sel_ego_clearance` measure the discs, so a positive number is only as trustworthy as the disc geometry — a run can log clean clearance and still show the footprint touching the obstacle in Rerun.

Fixed by `ego_disc_offsets`: one keep-out row per (obstacle, disc), discs placed along the heading. Prefer covering the footprint with two or three discs over inflating a single `ego_radius` to the circumscribing radius, which widens *every* detour and can deadlock a narrow corridor. **`ego_disc_offsets` sets the number of generated constraint rows, so changing it requires `generate_mpc_model:=True` — exactly like `num_obstacles`.**

## A silent obstacle feed looks exactly like a clean run

If the obstacle publisher is not running (or is on another topic), a controller with `num_obstacles > 0` produces output *indistinguishable from a healthy run*: no detections, clean tracking, low solve times, "Final goal reached" — and every obstacle column in the solver-stats CSV logs as `nan`. An obstacle-avoidance test that never saw an obstacle therefore reads as a pass. This happened on 2026-07-26: a 599-tick run whose console log looked like the best result of the session had `n_selected=0` and `sel_id=-1` on every row. `base_tracker._warn_if_obstacle_feed_silent()` now warns (throttled) both when nothing has ever arrived and when the feed goes quiet for 2 s — but when judging any obstacle run, **check `n_selected` in the CSV before reading anything else into it.**
