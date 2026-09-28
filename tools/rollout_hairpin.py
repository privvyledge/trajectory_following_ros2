"""Offline repro of the acados on-path obstacle stall (hairpin idx ~430-530).

Faithful to the failing sim:
  - reference from the real Trajectory class + waypoints.csv (arc-length index advance,
    distance_tolerance=0.2, speed profile + a_lat cap, min_ref_speed=0.3, v_max=1.5)
  - OCP built by acados_settings with f1tenth_acados_obstacle.yaml params
  - adapter p-vector layout, acados-internal warm start (no per-tick reseeding)
  - base_tracker failure semantics: hold last good cmd on isolated failure,
    zero cmd + u_prev-from-echo after >=5 consecutive failures
Usage: python3 tools/rollout_hairpin.py [variant]
  variant: baseline | quadslack | merit | funnel | mirror | zerowarm | bias
"""
import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
from trajectory_following_ros2.utils.Trajectory import Trajectory  # noqa: E402
import trajectory_following_ros2.utils.trajectory_utils as trajectory_utils  # noqa: E402
import trajectory_following_ros2.acados.acados_settings as ac_settings_mod  # noqa: E402
from trajectory_following_ros2.acados.acados_settings import acados_settings  # noqa: E402

# ----- config (f1tenth platform + f1tenth_acados_obstacle weights) -----
N, Tf, WB = 25, 1.25, 0.256
DT = Tf / N                       # 0.05 == 1/control_rate
Q = np.diag([100.0, 100.0, 100.0, 16.0])
Qf = np.diag([200.0, 200.0, 200.0, 40.0])
R = np.diag([0.01, 0.01])
Rd = np.diag([10.0, 10.0])
VMAX, VMIN, AMAX, AMIN = 1.5, -1.5, 3.0, -3.0
MAX_STEER = np.deg2rad(27.0)
MAX_STEER_RATE = np.deg2rad(183.346494)   # 3.2 rad/s
MAX_JERK = 1.5                            # base_tracker default (not overridden)
EGO_R, SAFE, OBS_SLACK = 0.15, 0.15, 1000.0
MAXIT = int(os.environ.get('MAXIT', '40'))
DIST_TOL = 0.2                            # GOAL_DIS == reference anchor
OBS = np.array([float(os.environ.get("OBSX", 1.74659958610849)), float(os.environ.get("OBSY", -2.41761855042133)), float(os.environ.get("OBSR", 0.3))])
KEEPOUT = EGO_R + OBS[2] + SAFE

START_IDX = 400
TICKS = 900

# ----- variant knobs (monkey-patch acados_settings module before build) -----
_orig = ac_settings_mod.AcadosOcp
patch = {}
ACCEPT_MAXITER = False
if VARIANT == 'quadslack':
    patch['quadratic_obstacle_slack'] = True
elif VARIANT == 'merit':
    patch['globalization'] = 'MERIT_BACKTRACKING'
elif VARIANT == 'funnel':
    patch['globalization'] = 'FUNNEL_L1PEN_LINESEARCH'
elif VARIANT == 'mirror':
    patch['regularize'] = 'MIRROR'
elif VARIANT == 'accept2':
    ACCEPT_MAXITER = True
elif VARIANT == 'merit_accept2':
    patch['globalization'] = 'MERIT_BACKTRACKING'
    ACCEPT_MAXITER = True
elif VARIANT == 'quadslack_accept2':
    patch['quadratic_obstacle_slack'] = True
    ACCEPT_MAXITER = True
elif VARIANT == 'refproj':
    pass
elif VARIANT == 'refproj_accept2':
    ACCEPT_MAXITER = True
elif VARIANT == 'refwarm_accept2':
    ACCEPT_MAXITER = True
REFPROJ = VARIANT.startswith('refproj')
REFWARM = (VARIANT == 'refwarm_accept2')
REAL_ADAPTER = (VARIANT == 'realadapter')  # drive the actual patched AcadosSolverAdapter


_side_hints = None


def project_reference_out_of_keepout(xref, margin=0.05):
    """Mirror of base_tracker._project_reference_out_of_keepouts.

    Delegates to the shared trajectory_utils implementation (including the per-obstacle
    go-around side hysteresis) and carries the hints across ticks exactly as the node
    does, so this rollout exercises the production reference-shaping path rather than a
    local copy of it. The projection lives in base_tracker now, not in the adapter, so
    the caller applies it before adapter.solve() — as base_tracker does.
    """
    global _side_hints
    xref, n_proj, _side_hints = trajectory_utils.project_reference_out_of_keepouts(
        xref, OBS[:2].reshape(1, 2), np.array([KEEPOUT]), margin=margin,
        side_hints=_side_hints)
    return xref, n_proj


class PatchedOcp(_orig):
    """Intercept attribute sets after acados_settings configures the OCP via __del__-free hook."""
    pass

# Simpler: post-hoc patching via wrapper around acados_settings is not possible
# (solver is created inside). Instead patch by env-driven edit: we wrap
# AcadosOcpSolver.generate/create to intercept the ocp object.


from acados_template import AcadosOcpSolver as _Solver  # noqa: E402

_real_init = _Solver.__init__


def _patched_init(self, ocp, json_file=None, **kw):
    if patch.get('quadratic_obstacle_slack') and getattr(ocp.constraints, 'idxsh', None) is not None \
            and len(np.atleast_1d(ocp.cost.zl)) > 0:
        n = len(np.atleast_1d(ocp.cost.zl))
        ocp.cost.Zl = OBS_SLACK * np.ones(n)
        ocp.cost.Zl_e = OBS_SLACK * np.ones(n)
    if 'globalization' in patch:
        ocp.solver_options.globalization = patch['globalization']
    if 'regularize' in patch:
        ocp.solver_options.regularize_method = patch['regularize']
    # env-driven sweep knobs (items a/b): GLOB=MERIT_BACKTRACKING, LM=1e-4,
    # QPIT decouples the HPIPM iteration cap from the SQP MAXIT
    if os.environ.get('GLOB'):
        ocp.solver_options.globalization = os.environ['GLOB']
    if os.environ.get('LM'):
        ocp.solver_options.levenberg_marquardt = float(os.environ['LM'])
    if os.environ.get('QPIT'):
        ocp.solver_options.qp_solver_iter_max = int(os.environ['QPIT'])
    _real_init(self, ocp, json_file=json_file, **kw)


_Solver.__init__ = _patched_init

# ----- build OCP -----
tmp = tempfile.mkdtemp(prefix=f"acados_hairpin_{VARIANT}_")
build = os.path.join(tmp, "cg")
os.makedirs(build, exist_ok=True)
cwd = os.getcwd()
os.chdir(build)
_, model, C, ocp, has_wp = acados_settings(
    Tf=Tf, N=N, x0=np.zeros(4), scale_cost=False, Q=Q, R=R, Qe=Qf, Rd=Rd, wheelbase=WB,
    vel_min=VMIN, vel_max=VMAX, acc_min=AMIN, acc_max=AMAX,
    delta_min=-MAX_STEER, delta_max=MAX_STEER,
    cost_module='EXTERNAL', cost_module_e='EXTERNAL',
    qp_solver='PARTIAL_CONDENSING_HPIPM', nlp_solver_type='SQP', integrator_type='ERK',
    generate=True, build=True, with_cython=False,
    num_iterations=MAXIT, tolerance=1e-6, mpc_config_file=os.path.join(tmp, "ocp.json"),
    code_export_directory=build, num_obstacles=1, ego_radius=EGO_R, safe_distance=SAFE,
    obstacle_slack_weight=OBS_SLACK)
os.chdir(cwd)
assert has_wp

# ----- trajectory (mirror base_tracker path_callback + _init_trajectory) -----
df = pd.read_csv(os.path.join(REPO, 'data', 'waypoints.csv'))
X = df[' x'].values
Y = df[' y'].values
YAW = df[' yaw'].values
VX = df[' vx'].values
path = np.column_stack([X, Y])
speeds = VX.copy()

traj = Trajectory(goal_tolerance=DIST_TOL)
tc = traj.trajectory_key_to_column
traj.trajectory = np.zeros((len(path), len(traj.trajectory_keys)))
traj.trajectory[:, tc['x']] = X
traj.trajectory[:, tc['y']] = Y
traj.trajectory[:, tc['yaw']] = YAW
cdists = trajectory_utils.cumulative_distance_along_path(traj.trajectory[:, [tc['x'], tc['y']]])
curvs = trajectory_utils.calculate_curvature_all(cdists, YAW, smooth=True)
traj.trajectory[:, tc['curvature']] = curvs
traj.trajectory[:, tc['cum_dist']] = cdists
rel_t, rel_dt = trajectory_utils.calc_path_relative_time(path, speeds, min_dt=1.0)
traj.trajectory[:, tc['dt']] = rel_dt
traj.trajectory[:, tc['total_time_elapsed']] = rel_t
traj.trajectory[:, tc['speed']] = speeds
# speed policy (f1tenth_acados_obstacle)
traj.use_speed_profile = True
traj.a_lat_max = 3.0
traj.min_reference_speed = 0.3
traj.max_reference_speed = VMAX
traj.arclength_index_advance = True
traj.projection_window = 5.0
# anchor mid-path (offline shortcut; sim reaches here by driving from idx 0)
traj.previous_index = START_IDX
traj.current_index = START_IDX

sc = traj.state_key_to_column


def set_state(x, y, v, psi, cum):
    traj.state[0, sc['x']] = x
    traj.state[0, sc['y']] = y
    traj.state[0, sc['speed']] = v
    traj.state[0, sc['yaw']] = psi
    traj.state[0, sc['omega']] = 0.0
    traj.state[0, sc['curvature']] = trajectory_utils.calculate_curvature_single(
        traj.trajectory[:, [tc['x'], tc['y']]], goal_index=traj.current_index)
    traj.state[0, sc['cum_dist']] = cum

# ----- bicycle integrator (plant) -----


def f_cont(x, u):
    return np.array([x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), u[0],
                     x[2] / WB * np.tan(u[1])])


def step_rk4(x, u, dt):
    k1 = f_cont(x, u)
    k2 = f_cont(x + dt / 2 * k1, u)
    k3 = f_cont(x + dt / 2 * k2, u)
    k4 = f_cont(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# ----- rollout -----


Qd, Rdi, Qed, Rdd = np.diag(Q), np.diag(R), np.diag(Qf), np.diag(Rd)
x0 = np.array([X[START_IDX], Y[START_IDX], VX[START_IDX], YAW[START_IDX]])
u_prev = np.zeros(2)
last_good = np.zeros(2)
fails = 0
raw_fails = 0
cum = 0.0
current_idx = START_IDX

adapter = None
if REAL_ADAPTER:
    from trajectory_following_ros2.coupled_kinematic_acados import AcadosSolverAdapter
    adapter = AcadosSolverAdapter(
        C, horizon=N, wheelbase=WB,
        stage_cost_type='EXTERNAL', terminal_cost_type='EXTERNAL',
        num_obstacles=1, ego_radius=EGO_R,
        has_weight_params=True, Q=Q, R=R, Qe=Qf, Rd=Rd,
        dt=DT,
        u_min=np.array([AMIN, -MAX_STEER]), u_max=np.array([AMAX, MAX_STEER]),
        # Mirrors coupled_kinematic_acados._init_solver: steering rate is bounded,
        # the acceleration rate is left to the Rd cost (RJ/RS override for sweeps).
        rate_max=(np.array([float(os.environ.get('RJ', 'inf')),
                            float(os.environ.get('RS', MAX_STEER_RATE))])
                  if os.environ.get('RATE', '1') == '1' else None))
    adapter.update_obstacles(np.tile(OBS.reshape(3, 1), (1, N + 1)))
    adapter.initialize(x0)
else:
    # adapter.initialize()
    for i in range(N + 1):
        C.set(i, 'x', x0)
        if i < N:
            C.set(i, 'u', np.zeros(2))
    C.constraints_set(0, 'lbx', x0)
    C.constraints_set(0, 'ubx', x0)
    C.solve()


def set_stage_p(j, yref, xk, up):
    p = [WB, *yref, 0.0, 0.0, *xk, *up, OBS[0], OBS[1], OBS[2], EGO_R,
         *Qd, *Rdi, *Qed, *Rdd]
    C.set(j, 'p', np.array(p))


hist = []
perf = []   # realadapter only: (tick, raw_status, sqp_iter, solve_ms, clearance, n_ref_inside)
DETAIL_TICKS = set()
for tick in range(TICKS):
    set_state(x0[0], x0[1], x0[2], x0[3], cum)
    tgt_speed = float(speeds[min(current_idx, len(speeds) - 1)])
    current_idx, ref, _, _, _ = traj.calc_ref_trajectory(
        state=None, trajectory=None, current_index=None,
        dt=DT, prediction_horizon=N, lookahead_time=1.0, lookahead=30.0,
        num_points_to_interpolate=N, target_speed=tgt_speed)
    if ref is None:
        print(f"tick {tick}: end of path (idx {current_idx})")
        break
    xref = np.array([ref['x_ref'], ref['y_ref'], ref['vel_ref'], ref['yaw_ref']])
    # REAL_ADAPTER projects here because the keep-out reference projection lives in
    # base_tracker (applied before solver.solve()), not in the adapter — mirror the node.
    if REFPROJ or REAL_ADAPTER:
        xref, _nproj = project_reference_out_of_keepout(xref)

    if tick < 16:
        print(f"TRACE tick {tick} idx={current_idx} x0={np.round(x0,5)} "
              f"u_prev={np.round(u_prev,5)} xref_sum={xref.sum():.6f}")
    if REAL_ADAPTER:
        result = adapter.solve(x0, xref, u_prev)
        if tick < 16:
            print(f"TRACE   -> raw_status={result.status} u0=({result.accel_cmd:.5f},"
                  f"{result.steering_cmd:.5f})")
        st = 0 if result.is_optimal else int(result.status)
        if st != 0 and tick < 40:
            try:
                _r = C.get_stats('residuals')
            except Exception as _e:
                _r = f'ERR {_e}'
            print(f"DBG tick {tick}: raw_status={result.status} is_optimal={result.is_optimal} "
                  f"residuals={_r}")
        try:
            sqp = int(np.asarray(C.get_stats('sqp_iter')).flatten()[0])
        except Exception:
            sqp = -1
        res = np.zeros(4)
        raw_st = int(result.status)
        solve_ms = float(result.solve_time) * 1e3
        u0 = np.array([result.accel_cmd, result.steering_cmd])
        x1 = result.x_sequence[:, 1]
        clr = np.hypot(x0[0] - OBS[0], x0[1] - OBS[1]) - KEEPOUT
        d_all = np.hypot(X - x0[0], Y - x0[1])
        cte = d_all.min()
        raw_fails = raw_fails + 1 if st != 0 else 0
        zero_cmd = False
        if st == 0:
            fails = 0
            last_good = u0.copy()
            u_apply = u0.copy()
        else:
            fails += 1
            if fails < 5:
                u_apply = last_good.copy()
            else:
                u_apply = np.zeros(2)
                zero_cmd = True
        u_apply[0] = np.clip(u_apply[0], AMIN, AMAX)
        u_apply[1] = np.clip(u_apply[1], -MAX_STEER, MAX_STEER)
        u_prev = u_apply.copy()
        ref_d = np.hypot(xref[0, :] - OBS[0], xref[1, :] - OBS[1])
        n_ref_inside = int((ref_d < KEEPOUT).sum())
        perf.append((tick, raw_st, sqp, solve_ms, clr, n_ref_inside))
        hist.append((tick, current_idx, st, sqp, fails, clr, cte, u_apply.copy(), res,
                     n_ref_inside))
        if zero_cmd:
            u_plant = np.array([np.clip((0.0 - x0[2]) / DT, AMIN, AMAX), 0.0])
        else:
            u_plant = u_apply
        cum += abs(x0[2]) * DT
        x0 = step_rk4(x0, u_plant, DT)
        if fails >= 60:
            print(f"tick {tick}: aborting after {fails} consecutive failures (stall confirmed)")
            break
        continue

    C.constraints_set(0, 'lbx', x0)
    C.constraints_set(0, 'ubx', x0)
    if REFWARM and raw_fails >= 1:
        # CasADi-adapter parity: cold-start from the reference after a degenerate solve
        for i in range(N + 1):
            C.set(i, 'x', xref[:, i])
            if i < N:
                C.set(i, 'u', np.zeros(2))
    if VARIANT == 'zerowarm' and fails >= 1:
        # cold-start from reference after a failed solve
        for i in range(N + 1):
            C.set(i, 'x', xref[:, i])
            if i < N:
                C.set(i, 'u', np.zeros(2))
    if VARIANT == 'bias' and fails >= 1:
        # lateral symmetry-break: shift warm start 0.3 m to the left of heading
        nrm = np.array([-np.sin(x0[3]), np.cos(x0[3])])
        for i in range(N + 1):
            xi = C.get(i, 'x')
            xi[0:2] += 0.3 * nrm
            C.set(i, 'x', xi)
    for j in range(N):
        set_stage_p(j, xref[:, j], x0, u_prev)
    set_stage_p(N, xref[:, N], x0, u_prev)

    st = C.solve()
    if tick < 16:
        _u0dbg = C.get(0, 'u')
        print(f"TRACE   -> raw_status={st} u0=({_u0dbg[0]:.5f},{_u0dbg[1]:.5f})")
    sqp = int(C.get_stats('sqp_iter'))
    res = C.get_stats('residuals')
    u0 = C.get(0, 'u')
    x1 = C.get(1, 'x')
    clr = np.hypot(x0[0] - OBS[0], x0[1] - OBS[1]) - KEEPOUT
    # nearest-waypoint cte
    d_all = np.hypot(X - x0[0], Y - x0[1])
    cte = d_all.min()

    zero_cmd = False
    raw_fails = raw_fails + 1 if st != 0 else 0
    if ACCEPT_MAXITER and st == 2:
        st = 0  # CasADi-adapter parity: budget-limited iterate is usable
    if st == 0:
        fails = 0
        last_good = u0.copy()
        u_apply = u0.copy()
    else:
        fails += 1
        if fails < 5:
            u_apply = last_good.copy()
        else:
            u_apply = np.zeros(2)
            zero_cmd = True
    # saturation (base_tracker)
    u_apply[0] = np.clip(u_apply[0], AMIN, AMAX)
    u_apply[1] = np.clip(u_apply[1], -MAX_STEER, MAX_STEER)
    # u_prev echo: base hears its own published command
    u_prev = u_apply.copy()

    # ref-window intrusion into keep-out
    ref_d = np.hypot(xref[0, :] - OBS[0], xref[1, :] - OBS[1])
    n_ref_inside = int((ref_d < KEEPOUT).sum())

    hist.append((tick, current_idx, st, sqp, fails, clr, cte, u_apply.copy(), res.copy(),
                 n_ref_inside))
    detail = (st != 0 and (len(DETAIL_TICKS) < 6 or fails % 25 == 0))
    if detail:
        DETAIL_TICKS.add(tick)
        print(f"--- FAIL tick {tick} idx={current_idx} status={st} sqp_iter={sqp} "
              f"fails={fails} clr={clr:.3f} v={x0[2]:.2f} ref_inside={n_ref_inside}/26 "
              f"res={np.array2string(np.asarray(res), precision=2)}")
        print(f"    u0=({u0[0]:.3f},{np.rad2deg(u0[1]):.1f}deg) u_prev=({u_prev[0]:.3f},"
              f"{np.rad2deg(u_prev[1]):.1f}deg) x1_pred={np.round(x1,3)}")
        try:
            s = np.array(C.get_stats('statistics'), dtype=float)
            hdr = "iter res_stat res_eq res_ineq res_comp qp_stat qp_iter alpha"
            print("    stats cols: " + hdr + f"  (shape {s.shape})")
            rows = s.T if s.shape[0] < s.shape[1] else s
            show = list(rows[:8]) + ([rows[-1]] if len(rows) > 8 else [])
            for r in show:
                print("    " + " ".join(f"{v:9.2e}" for v in r))
        except Exception as e:
            print(f"    (statistics unavailable: {e})")

    # plant step (sim converts a pure zero command into (0 - v)/dt braking accel)
    if zero_cmd:
        u_plant = np.array([np.clip((0.0 - x0[2]) / DT, AMIN, AMAX), 0.0])
    else:
        u_plant = u_apply
    cum += abs(x0[2]) * DT
    x0 = step_rk4(x0, u_plant, DT)
    if fails >= 60:
        print(f"tick {tick}: aborting after {fails} consecutive failures (stall confirmed)")
        break

# ----- summary -----
statuses = [h[2] for h in hist]
n_fail = sum(1 for s in statuses if s != 0)
longest = curr = 0
for s in statuses:
    curr = curr + 1 if s != 0 else 0
    longest = max(longest, curr)
min_clr = min(h[5] for h in hist)
last = hist[-1]
print(f"\n=== VARIANT={VARIANT} ===")
print(f"ticks={len(hist)} failures={n_fail} longest_consecutive={longest}")
print(f"final idx={last[1]} (start {START_IDX}); min clearance to keepout edge={min_clr:.3f} m")
print(f"final pos=({hist[-1][0]},) x={x0[:2].round(3)} v={x0[2]:.2f}")
maxcte = max(h[6] for h in hist)
print(f"max cte(nearest-wp)={maxcte:.3f} m")
if perf:
    ms = np.array([p[3] for p in perf])
    raws = np.array([p[1] for p in perf])
    sqps = np.array([p[2] for p in perf])
    # xref is already keep-out-projected here, so ref_inside is ~always 0; gate
    # "obstacle-active" on physical proximity to the keep-out edge instead.
    active = np.array([p[4] < 1.0 for p in perf])
    print(f"solve_ms: p50={np.percentile(ms,50):.2f} p90={np.percentile(ms,90):.2f} "
          f"p95={np.percentile(ms,95):.2f} p99={np.percentile(ms,99):.2f} max={ms.max():.2f}")
    print("raw status counts: " + " ".join(
        f"{s}:{int((raws==s).sum())}" for s in sorted(set(raws.tolist()))))
    print(f"sqp_iter: mean={sqps.mean():.1f} max={sqps.max()} | obstacle-active ticks: "
          f"{int(active.sum())} (sqp mean={sqps[active].mean() if active.any() else 0:.1f}, "
          f"solve_ms mean={ms[active].mean() if active.any() else 0:.2f} "
          f"max={ms[active].max() if active.any() else 0:.2f}) | clean ticks: "
          f"(sqp mean={sqps[~active].mean() if (~active).any() else 0:.1f}, "
          f"solve_ms mean={ms[~active].mean() if (~active).any() else 0:.2f} "
          f"max={ms[~active].max() if (~active).any() else 0:.2f})")
    slow = [p for p in perf if p[3] > 40.0]
    for p in slow[:12]:
        print(f"  SLOW tick{p[0]:4d} raw_st={p[1]} sqp={p[2]:3d} {p[3]:8.2f} ms "
              f"clr={p[4]:6.3f} ref_inside={p[5]}")
    if len(slow) > 12:
        print(f"  ... {len(slow) - 12} more slow ticks")
# progress trace every 25 ticks
for h in hist[::25]:
    print(f"  tick{h[0]:4d} idx={h[1]:4d} st={h[2]} sqp={h[3]:2d} fails={h[4]:3d} "
          f"clr={h[5]:6.3f} cte={h[6]:5.3f} u=({h[7][0]:6.2f},{np.rad2deg(h[7][1]):6.1f}deg)")
shutil.rmtree(tmp, ignore_errors=True)
