# Algorithmic Description of the Trajectory-Following Controller

This document specifies the control algorithm implemented in this package in the form used
by the model-predictive-control literature: a top-level receding-horizon loop (Algorithm 1)
whose subroutines are given as nested algorithms (Algorithms 2–10). Every step corresponds
to code in this repository; the mapping is given in [§9](#9-notation-to-code-map). Where the
implementation deviates from the textbook form, the deviation is stated explicitly and its
reason is given, rather than being silently idealised away.

## Contents

1. [Notation](#1-notation)
2. [System model](#2-system-model)
3. [Algorithm 1 — Receding-horizon trajectory tracking](#3-algorithm-1--receding-horizon-trajectory-tracking)
4. [Reference generation](#4-reference-generation)
   - [Algorithm 2 — Path ingestion and trajectory construction](#algorithm-2--path-ingestion-and-trajectory-construction)
   - [Algorithm 3 — Along-track reference-index advance](#algorithm-3--along-track-reference-index-advance)
   - [Algorithm 4 — Speed-aware reference generation](#algorithm-4--speed-aware-reference-generation)
   - [Algorithm 5 — Keep-out reference projection with side hysteresis](#algorithm-5--keep-out-reference-projection-with-side-hysteresis)
5. [The optimal control problem](#5-the-optimal-control-problem)
   - [Algorithm 6 — Nominal OCP](#algorithm-6--nominal-ocp)
   - [Algorithm 7 — CasADi backend](#algorithm-7--casadi-backend-solve)
   - [Algorithm 8 — acados backend](#algorithm-8--acados-backend-solve)
6. [Supervisory logic](#6-supervisory-logic)
   - [Algorithm 9 — Lap advance / termination policy](#algorithm-9--lap-advance--termination-policy)
7. [Algorithm 10 — Pure Pursuit alternative](#7-algorithm-10--pure-pursuit-alternative)
8. [Timing and concurrency](#8-timing-and-concurrency)
9. [Notation-to-code map](#9-notation-to-code-map)

---

## 1. Notation

| Symbol | Meaning | Units |
|---|---|---|
| $z = [x,\ y,\ v,\ \psi]^\top$ | vehicle state (rear-axle reference point) | m, m, m/s, rad |
| $u = [a,\ \delta]^\top$ | control input (acceleration, steering angle) | m/s², rad |
| $n_x = 4,\ n_u = 2$ | state / input dimension | — |
| $N$ | prediction horizon (default 25) | steps |
| $T_s$ | control period, $T_s = 1/f_{\text{ctrl}}$ (default 0.05) | s |
| $L$ | wheelbase (default 0.256, F1/10) | m |
| $\mathcal{P} = \{p_i\}_{i=0}^{M-1}$ | waypoint path, $p_i = (x_i, y_i, v_i, \psi_i, \kappa_i, s_i, \dots)$ | — |
| $s_i$ | cumulative arc length at waypoint $i$ | m |
| $\kappa_i$ | path curvature at waypoint $i$ | 1/m |
| $Z^{\text{ref}} \in \mathbb{R}^{n_x \times (N+1)}$ | reference trajectory over the horizon | — |
| $Q, Q_f \in \mathbb{R}^{4\times4}$ | stage / terminal state weights (diagonal) | — |
| $R, R_d \in \mathbb{R}^{2\times2}$ | input / input-rate weights (diagonal) | — |
| $\rho$ | reference-anchor distance ($=$ `distance_tolerance`, default 0.2) | m |
| $W$ | projection window (default 5.0) | m |
| $r_{\text{ego}}, r_j, d_{\text{safe}}$ | ego radius, obstacle-$j$ radius, safety margin | m |
| $\mathcal{K}_j$ | keep-out disc of obstacle $j$, radius $r_{\text{ego}} + r_j + d_{\text{safe}}$ | — |
| $k$ | index within the prediction horizon, $k \in \{0,\dots,N\}$ | — |
| $t$ | control tick index (wall clock) | — |

Superscript $\cdot^{(t)}$ denotes a quantity at control tick $t$; a hat $\hat{\cdot}$ denotes a
predicted quantity.

---

## 2. System model

All backends share one prediction model: the kinematic bicycle referenced to the **rear axle**,

$$
\dot{z} = f(z, u) =
\begin{bmatrix}
v \cos\psi \\
v \sin\psi \\
a \\
\dfrac{v}{L}\tan\delta
\end{bmatrix}.
$$

Two remarks that matter for the algorithms below.

**Loss of yaw authority at standstill.** $\partial\dot\psi/\partial\delta = (v/L)\sec^2\delta$
vanishes as $v \to 0$. Steering has no effect on heading at rest, so the linearised problem is
singular there. This is the reason a solver without a regularising step cannot pull away from
standstill, and the reason a vehicle parked at a keep-out boundary cannot see its way around
(motivating Algorithm 5).

**Discretisation.** The continuous ODE is turned into a one-step map $z_{k+1} = F(z_k, u_k)$ by
one of:

- *forward Euler:* $F = z_k + T_s f(z_k, u_k)$;
- *RK4* (default): the classical four-stage scheme, $u$ held constant over the step;
- *LTV:* a first-order Taylor expansion about a per-stage operating point
  $(z^{\text{op}}_k, u^{\text{op}}_k)$, giving $z_{k+1} = A_k z_k + B_k u_k + G_k$ — affine in
  the decision variables, hence a genuine QP (see Algorithm 7).

The "continuous" formulation offered by the CasADi node is *identical* to Euler: the collocation
constraint $z_{k+1} = z_k + f(z_k,u_k)\,T_s$ **is** the forward-Euler one-step map.

---

## 3. Algorithm 1 — Receding-horizon trajectory tracking

The top-level loop, executed on a timer at $f_{\text{ctrl}}$ Hz (default 20). Steps 8–9
constitute the receding-horizon principle: an $N$-step plan is computed, only $u_0$ is applied,
and the whole problem is re-solved next tick from the newly measured state.

```
Algorithm 1  TrackTrajectory                        [base_tracker._control_timer_callback]
────────────────────────────────────────────────────────────────────────────────────────
Input  (asynchronous, from ROS subscriptions):
        z^(t)      measured state, from odometry            [odom_callback]
        P          waypoint path                            [path_callback → Algorithm 2]
        {v_i}      reference speed profile
        O^(t)      obstacle set {(c_j, r_j)}                [_obstacle_callback]
        u_prev     last applied input (echo or actuator feedback)
Output: u_cmd = (a, δ, v_cmd) published as AckermannDriveStamped
Persistent: mpc_init, traj_init, goal_latched, n_fail, i_ref, d_cum, hints

 1  ▷ Guard: stale odometry
    if age(z^(t)) > τ_stale (= 0.5 s) then
        publish 0;  return                       ▹ fail safe, not fail silent
 2  ▷ Lazy initialisation (first tick with data)
    if ¬mpc_init ∧ odom_received then  solver.initialize(z^(t));  mpc_init ← true
    if ¬traj_init ∧ path_received ∧ speeds_ready then  InitTrajectory()   [Algorithm 2, part II]
    if ¬(mpc_init ∧ traj_init) then  return
 3  ▷ Terminal latch
    if goal_latched then  publish 0;  return
 4  z^(t) ← Snapshot()                            ▹ under mutex; atomic w.r.t. odom_callback
 5  ▷ Reference
    v_tgt ← TargetSpeed()
    (i_ref, Z^ref, status) ← CalcRefTrajectory(z^(t), P, T_s, N, v_tgt)
                                                  [Algorithms 3 + 4]
 6  ▷ Branch on the two distinct causes of an empty reference
    if status = 'lost' then                       ▹ vehicle nowhere near the path
        warn;  publish 0;  return                 ▹ hold, do NOT latch — allow re-acquisition
    end_of_path ← (Z^ref = ∅)
    at_goal     ← (d_cum ≥ 3ρ) ∧ IsGoalReached(z^(t))     ▹ grace gate: start ≈ goal on a loop
    if end_of_path ∨ at_goal then
        if AdvanceLap() then return               [Algorithm 9]  ▹ new lap: re-anchor next tick
        publish 0;  return
 7  d_cum ← d_cum + |v| cos(ψ − ψ^ref_0) · T_s    ▹ travelled arc length (grace gate)
 8  ▷ Delay compensation (optional)
    z_0 ← z^(t)
    if delay_comp ∧ τ_delay > 0 then  z_0 ← RK4(z^(t), u_prev, τ_delay, L)
 9  ▷ Solve
    Z̃^ref ← ProjectRefOutOfKeepouts(Z^ref, O^(t))          [Algorithm 5]
    r ← solver.solve(z_0, Z̃^ref, u_prev)                    [Algorithm 7 or 8]
10  ▷ Failure accounting  (three-tier degradation)
    n_fail ← 0 if r.is_optimal else n_fail + 1
    LogSolverStats(r)                             ▹ before the early return below
    if r.is_optimal then
        u_prev ← r.u_prev   (unless sourced from the drive/actuator echo)
        (a, δ, v_cmd) ← (r.accel, r.steering, r.velocity)
    else if n_fail ≥ 5 then
        publish 0;  return                        ▹ tier 3: safety stop
    else
        ▹ tier 2: hold the last good command; do not apply the degenerate iterate
11  ▷ Saturate and publish
    if saturate then  (a, δ, v_cmd) ← clip to [a_min,a_max] × [δ_min,δ_max] × [v_min,v_max]
    publish AckermannDriveStamped(a, δ, v_cmd)
12  update debug queues, cross-track error, logs
────────────────────────────────────────────────────────────────────────────────────────
```

### Design notes on Algorithm 1

**Step 6 — why `status` and not emptiness.** An empty reference has two opposite causes: the
projection has reached the path end (*done*), or every local candidate is beyond the search
radius (*lost the path*). Conflating them makes a vehicle that suffered a localisation jump
declare the run complete and park. Algorithm 3 returns the discriminating status; only the
along-track mode can produce it (the legacy Euclidean gate returns `None`, preserving its
treat-empty-as-end-of-path behaviour).

**Step 10 — the three tiers.** A single suboptimal solve is bridged by holding the last good
command rather than applying the returned iterate, because a one-off solver spike can otherwise
emit a saturated steering angle plus a hard brake and poison the loop. Only a *sustained* run of
five failures escalates to a zero command. What counts as a failure is deliberately narrow: both
backends accept budget-limited returns (max iterations / max CPU time) as usable in the
real-time-iteration sense, and count only genuinely degenerate states. Consequently a low
iteration budget alone will not trip the safety stop.

---

## 4. Reference generation

The reference is what every backend tracks, so it is generated once, backend-independently,
before the solve. Empirically this stage — not the cost weights — dominates corner-cutting
behaviour: the anchor distance $\rho$ sets how far past a corner apex the reference is placed.

### Algorithm 2 — Path ingestion and trajectory construction

```
Algorithm 2  IngestPath                                    [base_tracker.path_callback]
────────────────────────────────────────────────────────────────────────────────────────
Input:  nav_msgs/Path message D
Output: P (M × 9 trajectory array), or rejection

Part I — ingestion (asynchronous callback)
 1  if frame(D) ≠ global_frame then
        await tf.wait_for_transform_async(global_frame ← frame(D))  ▹ coroutine; latched
        (t_x, t_y, θ) ← lookup;  else (0,0,0)                        ▹ QoS delivers before TF
 2  for each pose in D:  apply the planar transform (t_x, t_y, θ) to (x,y) and ψ
 3  ▷ Validation — reject at ingestion, never downstream
    if M < 2                        then  warn; return    ▹ curvature undefined
    if ¬finite(P_new)               then  warn; return    ▹ retain previous path
    if P_new = P (elementwise)      then  return          ▹ latched re-delivery
 4  ▷ Replan handling
    if a path was already active then
        traj_init ← false;  goal_latched ← false;  n_fail ← 0;  ResetLapProgress()
        if speeds were synthesised then  discard them     ▹ length may have changed
 5  P ← P_new;  invalidate KD-tree
 6  s ← CumulativeDistance(P);   κ ← Curvature(s, ψ)      ▹ arc-length-based, then smoothed

Part II — timing initialisation (first control tick with both path and speeds)
 7  if |speeds| ≠ M then  speeds ← const(max(|v_des|, 0.1))   ▹ guards a stale speed array
 8  (τ, Δτ) ← RelativeTime(P, speeds);  write dt, total_time_elapsed, speed columns into P
 9  ApplySpeedPolicy();  traj_init ← true
10  warn if loop ≠ 0 ∧ ‖p_0 − p_{M−1}‖ > 2ρ                ▹ non-closed path with looping
11  warn if ρ ≥ diam(P)                                     ▹ scale mismatch: "done" on tick 1
────────────────────────────────────────────────────────────────────────────────────────
```

Curvature is computed against **arc length**, not waypoint index. Under dense or unevenly spaced
waypoints an index-based derivative reports large spurious curvature, which would then feed the
speed cap in Algorithm 4 and brake the vehicle for corners that do not exist.

### Algorithm 3 — Along-track reference-index advance

The controller must answer "which waypoint anchors the horizon?" Two modes exist; the
along-track projection is the default.

```
Algorithm 3  ProjectIndexAndLookahead      [trajectory_utils.project_index_and_lookahead]
────────────────────────────────────────────────────────────────────────────────────────
Input:  P (waypoints), s (cumulative arc length), position p,
        i_floor (monotonic anchor = previous projection),
        ρ (look-ahead / anchor distance), W (projection window), R_max (sanity cap)
Output: (i_target, status), status ∈ {ok, end_of_path, lost}

 1  d_i ← ‖p − p_i‖  for all i
 2  ▷ Step 1: PROJECT — nearest waypoint in a forward, arc-bounded window
    𝒲 ← { i : i ≥ i_floor  ∧  s_i − s_{i_floor} ≤ W  ∧  d_i ≤ R_max }
 3  if 𝒲 = ∅ then
        return (∅, 'lost')             ▹ i_floor always satisfies the first two terms, so an
                                          empty window means only: every candidate is beyond
                                          R_max — off the path, NOT at its end
 4  i_proj ← argmin_{i ∈ 𝒲} d_i        ▹ ignores lateral offset ⇒ advances during a swerve
 5  ▷ Step 2: LOOK AHEAD — first waypoint ρ metres of ARC beyond the projection
    𝒜 ← { i : i ≥ i_proj  ∧  s_i − s_{i_proj} ≥ ρ }
 6  if 𝒜 = ∅ then  return (∅, 'end_of_path')       ▹ projection within ρ of the tail
 7  i_floor ← i_proj                                ▹ fed back as next tick's anchor
    return (min 𝒜, 'ok')
────────────────────────────────────────────────────────────────────────────────────────
```

**Why project, then look ahead.** The legacy alternative gates on Euclidean distance from the
*vehicle*: it returns the lowest index whose distance falls in $[\rho, R_{\max}]$. That couples
index advance to lateral offset — hold the vehicle off the line (an obstacle-avoidance swerve)
and every nearby waypoint leaves the band, so the index freezes and the vehicle never makes
route progress past the obstacle. Projecting first decouples the two: the projection tracks
longitudinal progress regardless of lateral error, and the look-ahead is measured in arc length
from it.

**Why the window $W$ must be short.** Both the monotonic floor and the bounded arc span are
load-bearing on a self-returning path. A closed loop whose final waypoints sit centimetres from
its start would, under an unbounded window, let the nearest-waypoint pick leap to those tail
points the moment the anchor clears the start — and the controller reports the run complete on
lap 1. Requirement: $v\,T_s \ll W < \text{loop length}$. The algorithm assumes the vehicle
starts near $i_{\text{floor}} = 0$; a robot starting mid-path needs a wider window or the legacy
mode for first-tick acquisition.

**The role of $\rho$.** $\rho$ is simultaneously the goal tolerance and the reference-anchor
distance. It is therefore *the* corner-cutting lever: a large $\rho$ anchors the horizon past a
corner apex, so the controller aims across the chord. It is bounded below by a per-solver
robustness floor — too short a preview stalls route progress.

### Algorithm 4 — Speed-aware reference generation

```
Algorithm 4  GenerateReference    [trajectory_utils.generate_reference_trajectory_by_interpolation]
────────────────────────────────────────────────────────────────────────────────────────
Input:  P, i_target, N, T_s, v_tgt, a_lat^max, use_profile, v_min, v_max
Output: Z^ref = [x^ref; y^ref; v^ref; ψ^ref] ∈ ℝ^{4 × (N+1)}

    ▷ Branch A — speed-aware (active iff v_tgt ≠ ⊥ ∧ (a_lat^max > 0 ∨ use_profile))
 1  march ← sign(v_tgt);  cap ← min(|v_tgt|, |v_max|);  floor ← clip(v_min, 0, cap)
 2  σ ← s_{i_target}                                   ▹ arc-length cursor
 3  for h = 0 … N−1:                                   ▹ sequential: spacing and κ are COUPLED
        v_h ← cap
        if use_profile then  v_h ← min(v_h, |interp(σ; s, speed_column)|)
        if a_lat^max > 0 then
            κ ← |interp(σ; s, κ_column)|
            v_h ← min(v_h, sqrt(a_lat^max / max(κ, ε)))     ▹ lateral-accel cap, ε = 1e−3
        v_h ← clip(v_h, floor, cap)
        σ ← σ + march · T_s · v_h                      ▹ next sample is one step of TRAVEL away
 4  v_N ← v_{N−1}
 5  (x^ref, y^ref) ← interp(σ_{0:N}; s, (x, y) columns)
 6  ψ^ref ← FixAngleReference( InterpolateAngles(σ; s, ψ column), ψ^(t) )
 7  v^ref ← march · v_{0:N}
    ▷ Branch B — constant v_tgt: σ_h = s_{i_target} + h·T_s·v_tgt   (legacy)
    ▷ Branch C — no v_tgt: interpolate by recorded timestamp; v^ref ← Δs/Δt  (Pure Pursuit)
────────────────────────────────────────────────────────────────────────────────────────
```

**Why the loop is sequential.** Horizon-point spacing depends on speed, and the curvature limit
depends on the curvature at that spacing. The two are mutually dependent, so the arc-length
cursor is marched one step at a time rather than vectorised.

**Why yaw is interpolated through sin/cos.** Linear interpolation of a wrapped angle column
across the $\pm\pi$ branch cut yields a reference heading that is up to $180°$ wrong.
`InterpolateAngles` interpolates the sine and cosine and recovers the angle by `atan2`;
`FixAngleReference` then unwraps the result and shifts it onto the branch nearest the measured
heading, so the solver sees a continuous $\psi^{\text{ref}}$ and a small heading error.

**The creep floor.** $v_{\min}$ lifts a noisy near-zero recorded start so the vehicle pulls away.
It cannot trap the vehicle at the goal, because termination is owned by Algorithm 9, not by the
speed reference reaching zero.

### Algorithm 5 — Keep-out reference projection with side hysteresis

Applied to a **copy** of the reference, so the published reference-path debug topic still shows
the raw reference.

```
Algorithm 5  ProjectRefOutOfKeepouts     [trajectory_utils.project_reference_out_of_keepouts]
────────────────────────────────────────────────────────────────────────────────────────
Guard (caller):  n_obs > 0 ∧ O ≠ ∅ ∧ solver has obstacle constraints
Input:  Z^ref (4 × H), obstacle centres c_j, keep-out radii R_j, margin m, hints
Output: (Z^ref projected, hints')

 1  for each obstacle j:
 2      B ← R_j + m
        inside_k ← ‖(x^ref_k, y^ref_k) − c_j‖ < B
 3      if no k inside then  hints[j] ← 0;  continue     ▹ negotiation over; re-decide later
 4      θ_k ← atan2(y^ref_k − c_{j,y},  x^ref_k − c_{j,x})
 5      for each maximal contiguous run [i … l] of inside points:
 6          θ_in  ← θ_{i−1}  (else θ_i)                  ▹ entry angle: last outside point
            θ_out ← θ_{l+1}  (else θ_l)                  ▹ exit angle: first outside point
 7          sweep ← wrap(θ_out − θ_in);   side ← sign(sweep)
 8          ▷ Side hysteresis
            if |sweep| > π/2 ∧ hints[j] ≠ 0 ∧ side ≠ hints[j] then
                sweep ← sweep − side·2π;  side ← hints[j]   ▹ keep the committed side
            hints[j] ← side                                 ▹ clear geometry (|sweep| ≤ π/2) wins
 9          for m' = 1 … (l−i+1):                        ▹ sweep the arc from θ_in to θ_out
                θ ← θ_in + sweep · m'/(count+1)
                (x^ref, y^ref) ← c_j + B·(cos θ, sin θ)  ▹ project onto the boundary
10          recompute ψ^ref over the modified span from successive projected points
            (v^ref is left untouched)
────────────────────────────────────────────────────────────────────────────────────────
```

**Why projection is necessary.** A reference threading a keep-out hands the tracking cost a
target the constraints forbid. The closest *feasible* answer is then to park at the bubble
edge — and at $v = 0$ the model has no yaw authority (§2), so the optimiser cannot see the
detour. The vehicle deadlocks. Projecting the in-bubble points onto the boundary converts the
target into a go-around arc, which the tracking cost will follow.

**Why hysteresis is necessary.** On a dead head-on approach $\theta_{\text{in}}$ and
$\theta_{\text{out}}$ are nearly diametrically opposed, so the sign of the shortest-way sweep is
knife-edge and flips between ticks. The vehicle then chases an alternating left/right target and
again parks at the edge. When the sweep is ambiguous ($|\text{sweep}| > \pi/2$) the previously
chosen side is retained and the complementary long-way arc is swept; when the geometry clearly
dictates a side it wins and refreshes the hint.

**Why it is gated on the backend.** The projection is applied once, backend-independently, but
only when the active adapter actually constrains obstacles (duck-typed on `update_obstacles`).
Bending the reference for a backend with no obstacle constraints would fake an avoidance no
constraint enforces. Both this algorithm and the OCP constraint resolve the keep-out radius
through a single shared helper, so the enforced and the drawn/projected geometry cannot diverge.

---

## 5. The optimal control problem

### Algorithm 6 — Nominal OCP

Each tick, given $z_0$, $Z^{\text{ref}}$, and $u_{\text{prev}}$, the backends solve:

$$
\begin{aligned}
\min_{\;z_{0:N},\;u_{0:N-1},\;\varepsilon\;} \quad
& \tfrac{1}{2}\sum_{k=0}^{N-1}
\Big[
\underbrace{\|z_k - z^{\text{ref}}_k\|^2_{Q}}_{\text{tracking}}
+ \underbrace{\|u_k\|^2_{R}}_{\text{effort}}
+ \underbrace{\|u_k - u_{k-1}\|^2_{R_d}}_{\text{rate}}
+ \underbrace{\phi(\varepsilon_k)}_{\text{slack}}
\Big]
+ \tfrac{1}{2}\|z_N - z^{\text{ref}}_N\|^2_{Q_f} \\[4pt]
\text{s.t.}\quad
& z_0 = \hat{z}^{(t)} && \text{(initial state)}\\
& z_{k+1} = F(z_k, u_k), && k = 0,\dots,N-1 \quad\text{(dynamics)}\\
& u_{\min} \le u_k \le u_{\max}, && \text{(input box)}\\
& v_{\min} \le v_k \le v_{\max}, && k \ge 1 \quad\text{(speed box; stage 0 exempt)}\\
& \dot{u}_{\min} T_s \le u_k - u_{k-1} + \varepsilon_k \le \infty, && \text{(rate, softened)}\\
& \|(x_k,y_k) - c_j\|^2 + \varepsilon^{\text{obs}}_{j,k} \ge (r_{\text{ego}} + r_j + d_{\text{safe}})^2, && \forall j \quad\text{(keep-out)}\\
& \varepsilon \ge 0, && \text{(slack non-negativity)}
\end{aligned}
$$

with $u_{-1} := u_{\text{prev}}$. Only $u_0$ is applied.

**The heading term is not a plain quadratic.** Every cost path replaces the raw residual
$\psi_k - \psi^{\text{ref}}_k$ with

$$
\tilde{e}_\psi = \operatorname{atan2}\big(\sin(\psi_k - \psi^{\text{ref}}_k),\ \cos(\psi_k - \psi^{\text{ref}}_k)\big),
$$

which wraps the error to $[-\pi, \pi]$ and is $C^1$-smooth everywhere. The naive
`fmod(e + π, 2π) − π` idiom is **wrong** for $e < -\pi$ (C `fmod` takes the sign of the
dividend), returning the long-way-around error and sending the gradient the wrong direction —
observed as steering saturation whenever the reference heading crosses $\pm\pi$.

**Stage-0 speed box is exempt.** $v_0$ is pinned by the initial-state equality; a simultaneous
speed bound would conflict with it under transient sensor noise or overshoot.

**Slack quadratic vs. linear.** With a *linear* slack cost, the slack variables contribute zero
rows to the Hessian. An SQP outer loop hides this via its convexification step, but a direct QP
hands the singular Hessian to the raw QP solver, which may drop bounds "for regularity" and
return *success* with the steering box violated. A quadratic slack cost makes the Hessian
positive definite and restores exact bound compliance.

**Keep-out variants.** The squared-distance form above is used on the nonlinear paths. Two
specialisations exist:

- *Linearised distance* (LTV/QP path): the **distance**, not its square, is linearised about
  the operating point, $\sqrt{\|\cdot\|^2 + \epsilon^2}$, so the gradient has unit magnitude and
  never vanishes when the operating point sits on the obstacle centre. Linearising the squared
  distance gives a vanishing gradient there, which the slack silently absorbs — a head-on
  obstacle would produce no response.
- *Discrete control barrier function*: $h(z_{k+1}) - h(z_k) + \gamma h(z_k) \ge 0$. Nonconvex,
  hence CasADi/IPOPT only; it cannot be expressed in acados' per-stage constraint form without
  augmenting the state.

### Algorithm 7 — CasADi backend solve

```
Algorithm 7  CasAdiSolverAdapter.solve                  [coupled_kinematic_casadi.py]
────────────────────────────────────────────────────────────────────────────────────────
 1  z_0[v] ← clip(z_0[v], v_min, v_max)
 2  if warm-start empty then                    ▹ first tick, or after a degenerate solve
        z_ws ← Z^ref;  u_ws ← tile(u_prev)      ▹ cold start FROM THE REFERENCE
 3  ▷ Yaw branch alignment
    shift ← 2π · round((ψ_0 − z_ws[ψ,0]) / 2π);  z_ws[ψ,:] += shift
 4  controller.update(z_0, Z^ref, u_prev, z_ws, u_ws, [λ])
    if n_obs > 0 then  controller.update_obstacles_state(O)
 5  sol ← controller.solve()                    ▹ nlpsol / sqpmethod / qpsol per config
 6  ▷ Warm-start maintenance
    if return_status ∉ DEGENERATE then
        z_ws ← [z*_{1:N}, z*_N];  u_ws ← [u*_{1:N−1}, u*_{N−1}]   ▹ receding-horizon SHIFT
        λ_x, λ_g ← duals  if solver = ipopt  else ⊥               ▹ duals hurt sqpmethod
    else
        clear warm-start                        ▹ do not inherit a poisoned guess
 7  ▷ Acceptance policy
    is_optimal ← success ∨ status ∈ {Maximum_Iterations_Exceeded, Maximum_CpuTime_Exceeded}
    DEGENERATE = {Search_Direction_Becomes_Too_Small, Infeasible_Problem_Detected,
                  Restoration_Failed, Error_In_Step_Computation}
 8  return (u*_0, v̂_1, z*, u*, is_optimal, t_solve, status)
────────────────────────────────────────────────────────────────────────────────────────
```

Solver routing (validated against a supported-combination table at construction):

| `solver_type` | `solver` | Structure | `max_iter` governs |
|---|---|---|---|
| `nlp` | `ipopt` | interior-point NLP | IPOPT iterations |
| `quad` | `qrqp` / `osqp` / `qpoases` / `ipopt` | SQP outer loop + inner QP | outer SQP iterations |
| `qp` | `qrqp` / `osqp` / `qpoases` | **one** QP per tick (requires LTV) | unused — bound the inner QP |

The `qp` path is a true LTV-QP: because A/B/G are built from per-stage *parameter* symbols
refreshed each tick from the shifted warm start, the dynamics are affine in the decision
variables and no SQP outer loop exists — so a "search direction too small" failure is
structurally impossible. It requires `discrete_model_type = ltv`, rejects the CBF scheme
(nonconvex), and auto-disables the `atan2` yaw wrap (it breaks the quadratic-cost requirement;
the reference is already unwrapped by `FixAngleReference`).

### Algorithm 8 — acados backend solve

```
Algorithm 8  AcadosSolverAdapter.solve                  [coupled_kinematic_acados.py]
────────────────────────────────────────────────────────────────────────────────────────
 1  constraints_set(0, 'lbx'/'ubx', z_0)                ▹ pin the initial state
 2  ApplyInputRateBound(u_prev)                          [Algorithm 8a]
 3  for k = 0 … N:  set stage parameters p_k ← [L, z^ref_k, 0, z_0, u_prev, (obs), (weights)]
        NONLINEAR_LS / EXTERNAL → yref[ψ] ← 0   ▹ ψ error already wrapped IN the cost expr
        LINEAR_LS               → yref[ψ] ← unwrapped, heading-aligned ψ^ref
                                              ▹ a linear residual cannot wrap in-solver
 4  status ← controller.solve()
 5  finite ← isfinite(u_0) ∧ isfinite(z*)
    is_optimal ← finite ∧ status ∈ {0, 2}      ▹ 2 = max SQP iterations: budget-limited, not
                                                  a blow-up. Near a nonconvex keep-out the SQP
                                                  limit-cycles on the stationarity residual
                                                  while every inner QP succeeds.
 6  if ¬is_optimal then  RecoverFromFailure(z_0, Z^ref)  [Algorithm 8b]
 7  if ¬finite then  u ← 0;  z* ← tile(z_0);  v_cmd ← 0  ▹ never publish NaN; v_cmd = 0, NOT
                                                            z*[v,1] — tiling would mean
                                                            "hold speed, steer straight"
 8  return SolverResult(...)

Algorithm 8a  ApplyInputRateBound(u_prev)
 1  if rate_max = ⊥ then return
 2  u_prev ← clip(u_prev, u_min, u_max)         ▹ ⇒ the intersection below always contains
                                                  u_prev ⇒ the bound can NEVER be infeasible
 3  constraints_set(0, 'lbu', max(u_min, u_prev − rate_max·T_s))
    constraints_set(0, 'ubu', min(u_max, u_prev + rate_max·T_s))

Algorithm 8b  RecoverFromFailure(z_0, Z^ref)
 1  controller.reset()                          ▹ clears primal/dual AND QP-solver memory
 2  set(0,'x',z_0);  set(i,'x',z^ref_i) ∀i≥1;  set(i,'u',0)   ▹ reference cold start
────────────────────────────────────────────────────────────────────────────────────────
```

**Why the rate bound is a runtime box tightening (8a).** The OCP already box-bounds both inputs
at every stage, and acados permits per-stage overrides at runtime — the same mechanism the
initial-state pin uses in step 1. So the slew limit needs no code generation and no particular
acados version. It is applied at **stage 0 only**: $u_{\text{prev}}$ is a single last-applied
command, so a rate bound is physically meaningful only for the command that reaches the actuator
this tick. Interior-stage rate stays shaped by the $R_d$ cost. A true per-stage rate limit would
require augmenting the inputs into the state.

**Only the steering rate is bounded.** `max_steer_rate` is a real servo slew limit;
`max_jerk` is a comfort/weighting parameter, not an actuator limit — and acceleration is not
directly actuated here (the published command is speed + steering angle). Enforcing the jerk
bound hard at its default lets acceleration move $0.075$ m/s² per tick, taking 2 s to cross its
range, which starves the input. The node therefore passes $\dot{u}_{\max} = [\infty,\ \dot\delta_{\max}]$.

**Why recovery is mandatory (8b).** acados warm-starts every solve from its internal iterate
memory. One hard QP failure can leave a NaN iterate there, after which *every* subsequent solve
re-linearises at the poisoned point and fails at the first QP — the failure becomes permanent
even though a cold-started solve would succeed. Resetting and re-seeding makes the next tick
well-posed. The CasADi adapter recovers the same way (step 6 of Algorithm 7).

**Cost-module coupling.** The $R_d$ rate penalty exists **only** in the `EXTERNAL` cost branch;
`NONLINEAR_LS`/`LINEAR_LS` silently drop it (steering then jitters). `EXTERNAL` is also required
for obstacle/CBF constraints, and forces `hessian_approx = EXACT` (Gauss–Newton is invalid for
an external cost). It is the default for those reasons; a non-`EXTERNAL` choice emits a warning
so it is deliberate rather than a silent regression.

---

## 6. Supervisory logic

### Algorithm 9 — Lap advance / termination policy

```
Algorithm 9  AdvanceLap                                    [base_tracker._advance_lap]
────────────────────────────────────────────────────────────────────────────────────────
Parameter: loop ∈ ℤ   (0 = stop at goal; −1 = forever; n > 0 = exactly n laps)
Returns:   true ⇒ another lap (caller returns; the index re-anchors next tick)
           false ⇒ run complete (final_goal_reached latched; caller publishes 0)

 1  if loop = 0 then  goal_latched ← true;  return false
 2  n_laps ← n_laps + 1
 3  if loop = −1 ∨ n_laps < loop then
        ResetLapProgress()                     ▹ i_ref ← 0, d_cum ← 0, Trajectory.reset()
        return true
 4  goal_latched ← true;  return false
────────────────────────────────────────────────────────────────────────────────────────
```

The caller returns without solving on a lap reset: solving immediately would hand the solver a
large position error (vehicle at the path end, reference at the start) and risk extreme commands.

On a dense closed loop `IsGoalReached` rarely fires — the vehicle never stops mid-lap — so
`end_of_path` is the de-facto terminator; `IsGoalReached` catches the stopped-at-goal case on
open paths. `ResetLapProgress` is shared with the backward-clock-jump handler used during bag
replay.

---

## 7. Algorithm 10 — Pure Pursuit alternative

A geometric baseline. The node returns no solver from `_init_solver()` and **overrides**
Algorithm 1 entirely; it shares only the ROS infrastructure, the reference machinery
(Algorithms 2–4) and the terminator (Algorithm 9's conditions, inlined).

```
Algorithm 10  PurePursuitStep                    [ackermann_purepursuit._control_timer_callback]
────────────────────────────────────────────────────────────────────────────────────────
 1  stale-odom guard;  lazy trajectory init;  goal latch      ▹ mirrors Algorithm 1, steps 1–3
 2  if adaptive then  ℓ ← clip(K_ℓ · v + ℓ_0, ℓ_min, ℓ_max)   ▹ speed-scheduled look-ahead
 3  (i_ref, Z^ref) ← CalcRefTrajectory(…, N = 50, v_tgt = ⊥)  ▹ Branch C: timestamp interp
 4  terminator: end_of_path ∨ (d_cum ≥ 3ρ ∧ IsGoalReached)  ⇒  latch, publish 0, return
 5  d_cum ← d_cum + |v| cos(ψ − ψ^ref_0) · T_s
 6  g ← P[i_ref]                                              ▹ goal = the look-ahead waypoint
 7  α ← atan2(g_y − y, g_x − x) − ψ                           ▹ bearing to goal in body frame
 8  δ ← clip( atan2(2 L sin α, ℓ), δ_min, δ_max )             ▹ pure-pursuit steering law
 9  v_cmd ← PID(v_tgt − v)  if K_p ≠ 0  else v_tgt;   v_cmd ← clip(v_cmd, −v_max, v_max)
10  publish (a = 0, δ, v_cmd)
────────────────────────────────────────────────────────────────────────────────────────
```

**Two caveats, stated because they are load-bearing and non-obvious:**

1. **The goal point is not a circle intersection.** $g$ is the waypoint returned by Algorithm 3,
   whose distance ahead is set by $\rho$ (`distance_tolerance`), whereas the steering law
   divides by the *parameter* $\ell$ (`lookahead_distance`). Textbook Pure Pursuit uses one
   distance for both. Here they are independent knobs, so $\ell$ and $\rho$ must be kept
   consistent or the commanded curvature will not correspond to the geometry of the goal point.
   `Trajectory.get_next_waypoint` does compute a true circle–segment intersection and
   `calc_ref_trajectory` returns it, but this node discards it.
2. **The `PurePursuit` class is not used.** `purepursuit/PurePursuit.py` implements the same law
   (plus turn radius, curvature, and yaw rate) but nothing imports it; the node inlines steps
   7–8 above. Treat the node as authoritative.

---

## 8. Timing and concurrency

| Loop | Rate | Callback group | Notes |
|---|---|---|---|
| Control (Algorithm 1) | `control_rate`, default 20 Hz | mutually exclusive | a slow solve cannot block sensor callbacks |
| Odometry ingest | sensor-driven | reentrant | writes state under `mutex` |
| Path / speed ingest | latched (transient-local) or live | reentrant | `path_callback` is `async` (awaits TF) |
| Debug publication | `debug_frequency`, default 10 Hz | — | best-effort; `mpc/solve_time` publishes every tick |

The executor is a `MultiThreadedExecutor` sized by `executor_threads` (default 3). Shared state
between the odometry callback and the control loop is protected by a single mutex; the control
loop always reads through `_snapshot_state()` so that $(x, y, v, \psi, \omega)$ are mutually
consistent within a tick. The queues used for debug visualisation are latest-value holders, not
history buffers.

The identity $T_s = 1/f_{\text{ctrl}}$ is computed, never read: `base_tracker` does not read
`sample_time` as a ROS parameter, though backends that declare it themselves do. Keep the two
consistent in configuration.

---

## 9. Notation-to-code map

| Algorithm | Symbol / step | Code |
|---|---|---|
| 1 | whole loop | `base_tracker.py::BaseTrajectoryTracker._control_timer_callback` |
| 1 | $\tau_{\text{stale}}$ | `base_tracker.py::_STALE_ODOM_THRESHOLD_S` (0.5 s) |
| 1 | step 8, RK4 | `trajectory_utils.py::predict_state_rk4` |
| 1 | step 10, tiers | `_consecutive_failures`, threshold 5 |
| 2 | ingestion | `base_tracker.py::path_callback` |
| 2 | Part II | `base_tracker.py::_init_trajectory` |
| 2 | curvature, arc length | `trajectory_utils.py::calculate_curvature_all`, `cumulative_distance_along_path` |
| 3 | projection + look-ahead | `trajectory_utils.py::project_index_and_lookahead` |
| 3 | legacy gate | `trajectory_utils.py::find_closest_waypoints` |
| 3 | dispatch, $i_{\text{floor}}$ | `Trajectory.py::calc_nearest_index` (`previous_index`) |
| 3 | $\rho$, $R_{\max}$, $W$ | `Trajectory.GOAL_DIS` (= `distance_tolerance`), `MAXIMUM_SEARCH_RADIUS` (25 m), `projection_window` (5 m) |
| 4 | generation | `trajectory_utils.py::generate_reference_trajectory_by_interpolation` |
| 4 | yaw handling | `trajectory_utils.py::interpolate_angles`, `fix_angle_reference` |
| 4 | policy push | `base_tracker.py::_apply_speed_policy` |
| 5 | projection | `trajectory_utils.py::project_reference_out_of_keepouts` |
| 5 | caller + gate | `base_tracker.py::_project_reference_out_of_keepouts`, `_solver_has_obstacle_constraints` |
| 5 | $r_{\text{ego}}$ resolution | `trajectory_utils.py::resolve_ego_radius` (shared with the OCP and the visualiser) |
| 6 | interface | `backends/base_solver.py::BaseSolver`, `SolverResult` |
| 6 | CasADi cost | `casadi/_casadi_base.py::objective_function_setup`, `_wrap_angle` |
| 6 | CasADi constraints | `casadi/kinematic_mpc_casadi.py::constraints_setup` |
| 6 | acados cost/constraints | `acados/acados_settings.py::acados_settings` |
| 6 | model $f$ | `casadi/kinematic_bicycle_model.py`, `acados/kinematic_model.py` |
| 6 | discretisation | `utils/integrators.py::make_discrete_dynamics` |
| 6 | Bryson weights | `utils/mpc_weight_utils.py::bryson_weights` |
| 7 | adapter | `coupled_kinematic_casadi.py::CasAdiSolverAdapter.solve` |
| 7 | solver construction | `casadi/kinematic_mpc_casadi.py::setup_solver` |
| 8 | adapter | `coupled_kinematic_acados.py::AcadosSolverAdapter.solve` |
| 8a | rate bound | `AcadosSolverAdapter._apply_input_rate_bound` |
| 8b | recovery | `AcadosSolverAdapter._recover_from_failure` |
| 9 | lap policy | `base_tracker.py::_advance_lap`, `_reset_lap_progress` |
| 9 | goal test | `Trajectory.py::is_goal_reached` |
| 10 | Pure Pursuit | `ackermann_purepursuit.py::_control_timer_callback`, `_pid_speed` |

Defaults quoted throughout are the node-level parameter defaults; `config/mpc_parameters.yaml`
and the per-platform overlays in `config/weights/` override them, and launch arguments override
those.
