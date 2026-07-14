"""
Source:
    * https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    * https://docs.google.com/spreadsheets/d/1rVRycLnCyaWJLwnV47u30Vokp7vRu68og3OhlDbSjDU/edit#gid=959941492

See:
    * https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py
    * https://github.com/commaai/openpilot/blob/master/selfdrive/controls/tests/test_lateral_mpc.py
    * https://github.com/mlab-upenn/mpcc/blob/master/scripts/acados/models.py
    * https://github.com/duynamrcv/quadrotor_acados/blob/master/controller.py#L86

Perfomance Tips:
    * https://discourse.acados.org/t/solver-runs-slower-in-nvidia-jetson-tx2-platform/531/2

Todo: get actuation limit values from arguments instead of hardcoding in the imported model
"""
import numpy as np
import scipy.linalg
import casadi

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from trajectory_following_ros2.acados.kinematic_model import kinematic_model


def acados_settings(Tf, N, x0=None, scale_cost=True,
                    Q=None, R=None, Qe=None, Rd=None,
                    wheelbase=0.256,
                    vel_min=None, vel_max=None,
                    acc_min=None, acc_max=None,
                    delta_min=None, delta_max=None,
                    cost_module='external', cost_module_e='external',
                    qp_solver='PARTIAL_CONDENSING_HPIPM',
                    nlp_solver_type='SQP_RTI',
                    integrator_type='ERK',
                    qp_solver_cond_N=None,
                    generate=True, build=True, with_cython=True,
                    num_iterations=10, tolerance=1e-6,
                    mpc_config_file="kinematic_bicycle_acados_ocp.json",
                    code_export_directory="c_generated_code",
                    num_obstacles=0, ego_radius=1.0, safe_distance=0.5,
                    obstacle_slack_weight=100.0,
                    steer_rate_max=None, jerk_max=None,
                    input_rate_slack_weight=1e3):
    # generate = True  # generates the OCP and stores in the json file
    # build = True  # builds/compiles the model and stores in code_export_directory
    # the cython version is faster than bare C because there is no call overhead as opposed to the C code call overhead
    # todo: get constraints and max iterations

    # create render arguments
    ocp = AcadosOcp()

    # export model. Pass dt=Tf/N so the model carries a DISCRETE (RK4) one-step
    # map; acados only uses it when integrator_type='DISCRETE'.
    model, constraint = kinematic_model(dt=Tf / N)

    # Override constraint bounds from ROS params (fall back to kinematic_model defaults if None)
    if vel_min is not None:
        model.vel_min = vel_min
    if vel_max is not None:
        model.vel_max = vel_max
    if acc_min is not None:
        model.acc_min = acc_min
    if acc_max is not None:
        model.acc_max = acc_max
    if delta_min is not None:
        model.delta_min = delta_min
    if delta_max is not None:
        model.delta_max = delta_max

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    if model.disc_dyn_expr is not None:
        # Only consumed by acados when integrator_type='DISCRETE'; harmless otherwise.
        model_ac.disc_dyn_expr = model.disc_dyn_expr

    if num_obstacles > 0:
        _obs_states = casadi.SX.sym('obs_states', 3 * num_obstacles)
        _ego_r = casadi.SX.sym('ego_radius_p', 1)
        model_ac.p = casadi.vertcat(model.p, _obs_states, _ego_r)
    else:
        model_ac.p = model.p

    _has_weight_params = (cost_module.lower() == 'external'
                          and cost_module_e.lower() == 'external')
    if _has_weight_params:
        _Q_sym = casadi.SX.sym('Q_diag', model.x.size()[0])
        _R_sym = casadi.SX.sym('R_diag', model.u.size()[0])
        _Qe_sym = casadi.SX.sym('Qe_diag', model.x.size()[0])
        _Rd_sym = casadi.SX.sym('Rd_diag', model.u.size()[0])
        model_ac.p = casadi.vertcat(model_ac.p, _Q_sym, _R_sym, _Qe_sym, _Rd_sym)

    model_ac.name = model.name
    ocp.model = model_ac

    # # define constraint
    # ocp.model.con_h_expr = constraint.expr  # enable for soft/non linear constraints

    # dimensions
    nx = model.x.size()[0]  # number of states
    nu = model.u.size()[0]  # number of inputs
    nparams = model.p.size()[0]  # number of parameter
    ny = nx + nu  # number of outputs is the concatenation of x and u
    ny_e = nx  # number of residuals in Mayer term

    nsbx = 0  # (Default: 1) number of soft state bounds
    nh = 0  # (Default: constraint.expr.shape[0]) number of nonlinear constraints
    nsh = nh  # number of soft nonlinear constraints
    ns = nsh + nsbx  # total number of slacks at stages (1, N-1)

    # discretization
    ocp.dims.N = N  # prediction horizon (depracated)
    ocp.solver_options.N_horizon = N  # prediction horizon

    # set cost
    '''
    Tested with scale_cost = True
    Q = np.diag([100.0, 100.0, 1000.0, 0.001])), np.diag(100000.0, 100000.0, 1.0, 1.0])
    R = np.diag([1., 10.]), np.diag([1000., 10000.])
    Qe = np.diag([0.0, 0.0, 0.0, 10.0])
    '''
    # todo: initialize with nx, nu sizes
    if Q is None:
        Q = np.diag([100.0, 100.0, 1000.0, 0.001])  # np.diag(100000.0, 100000.0, 1.0, 1.0])

    # R = np.eye(nu)
    # R[0, 0] = 1e-3
    # R[1, 1] = 5e-3
    if R is None:
        R = np.diag([1., 10.])  # np.diag([1000., 10000.])

    if Qe is None:
        Qe = np.diag([1000.0, 1000.0, 1.0, 0.0001])  # terminal state  np.diag([5e0, 1e1, 1e-8, 1e-8]), np.zeros((nx, nx)), Q

    if Rd is None:
        Rd = np.diag([1., 10.])  # (optional) for external cost only

    unscale = 1.0
    if scale_cost:
        unscale = N / Tf  # rate = 1 / dt

    W = unscale * scipy.linalg.block_diag(Q, R)
    W_e = Qe / unscale

    # unpack parameters
    # wheelbase = model.p[0]

    zref = model.p[1:(nx + 1)]
    uref = model.p[(nx + 1):(nx + nu + 1)]  # model.p[(nx + nx + 1):(nx + nx + nu + 1)]

    z_k = model.p[(nx + nu + 1):(nx + nu + nx + 1)]   # model.p[(nx + 1):(nx + nx + 1)]
    u_prev = model.p[(nx + nu + nx + 1):(nx + nu + nx + nu + 1)]  # model.p[(nx + nx + nu + 1):(nx + nx + nu + nu + 1)]

    yref = casadi.vertcat(zref, uref)
    ocp.parameter_values = np.array([
        wheelbase,  # wheelbase
        *np.zeros(nx),  # zref
        *np.zeros(nu),  # uref
        *np.zeros(nx),  # zk
        *np.zeros(nu),  # u_prev
    ])
    if num_obstacles > 0:
        ocp.parameter_values = np.concatenate([
            ocp.parameter_values,
            np.ones(3 * num_obstacles) * 1000.0,  # obs_states: far away (won't constrain)
            [ego_radius],
        ])
    if _has_weight_params:
        ocp.parameter_values = np.concatenate([
            ocp.parameter_values,
            np.diag(Q), np.diag(R), np.diag(Qe), np.diag(Rd),
        ])

    '''
    LINEAR_LS:
        y = Vx @ x + Vu @ u
        cost =  0.5 * (y.T @ W @ y)
    NONLINEAR_LS:
        y = casadi.symbol(ny, 1)  # or any casadi expression, e.g with/without parameters
        cost = 0.5 * (y.T @ W @ y)
    EXTERNAL:
        y = casadi.vertcat(x, u). Optional
        cost = model.x.T @ Q @ model.x + model.u.T @ R @ model.u
    '''
    Vx = np.zeros((ny, nx))  # x matrix coefficient at intermediate shooting nodes (1 to N-1)
    Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))  # u matrix coefficient at intermediate shooting nodes (1 to N-1). todo
    # for u_k in range(nu):
    #     Vu[nx + u_k, u_k] = 1.0
    Vu[nx:, :] = np.eye(nu)

    Vx_e = np.zeros((ny_e, nx))  # x matrix coefficient for cost at terminal shooting node (N)
    Vx_e[:nx, :nx] = np.eye(nx)

    '''
    Lagrange/stage objective/cost type
    Options: "LINEAR_LS", "NONLINEAR_LS", "EXTERNAL"
    '''
    if cost_module.lower() == "LINEAR_LS".lower():
        # NOTE: the psi residual here is the linear psi - psi_ref. LINEAR_LS is
        # linear by construction (y = Vx@x + Vu@u), so the +/-pi wrap CANNOT be
        # applied in-solver as it is for NONLINEAR_LS/EXTERNAL. Instead the
        # adapter (AcadosSolverAdapter.solve) unwraps the psi reference trajectory
        # and aligns it to the current heading before cost_set('yref', ...), so
        # psi - psi_ref already lies in [-pi, pi]. See coupled_kinematic_acados.py.
        ocp.cost.cost_type = "LINEAR_LS"

        y = Vx @ model.x + Vu @ model.u
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu

        ocp.cost.W = W  # weight matrix at intermediate shooting nodes (1 to N-1)

        # set initial reference. Note: will be overwritten
        ocp.cost.yref = np.zeros(ny)  # reference at intermediate shooting nodes (1 to N-1) [ny, 1]

    elif cost_module.lower() == "NONLINEAR_LS".lower():
        ocp.cost.cost_type = "NONLINEAR_LS"

        # The psi entry is expressed as the already-wrapped heading error
        # atan2(sin(psi - psi_ref), cos(psi - psi_ref)) using the psi_ref carried
        # in the zref parameter, so acados' internal y - yref does NOT re-subtract
        # a raw psi. The adapter therefore must set the psi entry of yref to 0
        # (the position/velocity/input entries keep the normal yref reference).
        # Without this the heading error would jump to ~2*pi at +/-pi crossings.
        psi_err = casadi.atan2(casadi.sin(model.x[3] - zref[3]),
                               casadi.cos(model.x[3] - zref[3]))
        y = casadi.vertcat(
                model.x[0], model.x[1], model.x[2], psi_err,
                model.u
        )
        ocp.model.cost_y_expr = y
        ocp.cost.W = W  # weight matrix at intermediate shooting nodes (1 to N-1)

        # set initial reference. Note: will be overwritten
        ocp.cost.yref = np.zeros(ny)  # reference at intermediate shooting nodes (1 to N-1) [ny, 1]

    elif cost_module.lower() == "EXTERNAL".lower():
        ocp.cost.cost_type = "EXTERNAL"
        z_err = model.x - zref
        # Wrap the heading error (state index 3) to [-pi, pi] via atan2(sin, cos).
        # acados forms this residual by plain subtraction, so without wrapping the
        # psi term jumps to ~2*pi whenever psi - psi_ref crosses +/-pi (vehicle
        # traveling in -x), sending the solver the long-way-around gradient and
        # saturating steering. atan2(sin, cos) is correct everywhere and C1-smooth.
        z_err = casadi.vertcat(
            z_err[0], z_err[1], z_err[2],
            casadi.atan2(casadi.sin(z_err[3]), casadi.cos(z_err[3])),
        )
        u_err = model.u - uref
        u_rate_err = model.u - u_prev  # casadi.vertcat(u_rate, casadi.diff(u_dv))  # to pack with horizon
        ocp.model.cost_expr_ext_cost = (
            0.5 * unscale * (casadi.dot(_Q_sym * z_err, z_err) + casadi.dot(_R_sym * u_err, u_err))
            + 0.5 * casadi.dot(_Rd_sym * u_rate_err, u_rate_err)
        )

    else:
        raise AttributeError(f'Invalid cost type ({cost_module}) specified.')

    '''
    Mayer/terminal objective/cost type (does not have to be the same as lagrange)
    Options: "LINEAR_LS", "NONLINEAR_LS", "EXTERNAL"
    '''
    if cost_module_e.lower() == "LINEAR_LS".lower():
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.Vx_e = Vx_e
        ocp.cost.W_e = W_e  # weight matrix at terminal shooting node (N)

        # set initial reference. Note: will be overwritten
        ocp.cost.yref_e = np.zeros(ny_e)  # cost reference at terminal shooting node (N)  [nx, 1]

    elif cost_module_e.lower() == "NONLINEAR_LS".lower():
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Terminal psi entry is the wrapped heading error (see stage cost above);
        # the adapter zeros the psi entry of yref_e to match.
        psi_err_e = casadi.atan2(casadi.sin(model.x[3] - zref[3]),
                                 casadi.cos(model.x[3] - zref[3]))
        y_e = casadi.vertcat(
                model.x[0], model.x[1], model.x[2], psi_err_e,
        )
        ocp.model.cost_y_expr_e = y_e

        ocp.cost.W_e = W_e  # weight matrix at terminal shooting node (N)
        # set initial reference. Note: will be overwritten
        ocp.cost.yref_e = np.zeros(ny_e)  # cost reference at terminal shooting node (N)  [nx, 1]

    elif cost_module_e.lower() == "EXTERNAL".lower():
        ocp.cost.cost_type_e = "EXTERNAL"
        yref_e = yref[:nx]
        z_err_e = model.x - yref_e
        # Wrap the terminal heading error (index 3) to [-pi, pi]; see stage cost above.
        z_err_e = casadi.vertcat(
            z_err_e[0], z_err_e[1], z_err_e[2],
            casadi.atan2(casadi.sin(z_err_e[3]), casadi.cos(z_err_e[3])),
        )
        ocp.model.cost_expr_ext_cost_e = 0.5 / unscale * casadi.dot(_Qe_sym * z_err_e, z_err_e)

    else:
        raise AttributeError(f'Invalid cost type ({cost_module_e}) specified.')

    # ocp.cost.zl = 100 * np.ones((ns,))  # gradient wrt lower slack at intermediate shooting nodes (0 to N-1)
    # ocp.cost.zu = 100 * np.ones((ns,))  # gradient wrt upper slack at intermediate shooting nodes (0 to N-1)
    # ocp.cost.Zl = 1 * np.ones((ns,))  # diagonal of Hessian wrt lower slack at intermediate shooting nodes (0 to N-1)
    # ocp.cost.Zu = 1 * np.ones((ns,))  # diagonal of Hessian wrt upper slack at intermediate shooting nodes (0 to N-1)

    # Euclidean obstacle-avoidance nonlinear constraints (stage 0..N-1 via con_h_expr,
    # terminal stage N via con_h_expr_e).  CBF is not supported for acados because the
    # CBF condition h(k+1)-h(k)+γh(k)>=0 links two stages and cannot be expressed in
    # acados's per-stage con_h_expr without augmenting the state vector.
    if num_obstacles > 0:
        h_exprs = [
            (model.x[0] - _obs_states[3 * j]) ** 2
            + (model.x[1] - _obs_states[3 * j + 1]) ** 2
            - (_ego_r + _obs_states[3 * j + 2] + safe_distance) ** 2
            for j in range(num_obstacles)
        ]
        h_expr = casadi.vertcat(*h_exprs)
        ocp.model.con_h_expr = h_expr    # stages 0..N-1
        ocp.model.con_h_expr_e = h_expr  # stage N (con_h_expr_e uses only model.x, no model.u)

        # One-sided keep-out h = dist^2 - keepout^2 >= 0 (lower-bounded; the upper
        # side is nominally unbounded). Do NOT use 1e15 for the "infinite" upper
        # bound: HPIPM's interior-point barrier cannot scale a 1e15-magnitude box and
        # returns qp_stat=2 (max-iter) EVERY SQP iteration, so the OCP never converges
        # from standstill even with the obstacle far away and inactive. A
        # finite, well-scaled upper bound converges cleanly (2 SQP iters). The bounds
        # sit far above any realistic value yet below the HPIPM breakdown: the unseen-
        # obstacle "parked far away" fallback is dist~1000 m -> h~2e6, so uh=1e8 keeps it
        # feasible with ~50x margin; the slack only ever needs ~keepout^2 (tens of m^2),
        # so ush=1e6 is huge margin. Both are ~100x below the empirical HPIPM breakdown
        # (uh=1e10/ush=1e8 fails; uh=1e8/ush=1e6 and smaller converge in 2 SQP iters).
        _UH = 1e8    # constraint upper bound (>> max realistic dist^2, incl. far-park)
        _USH = 1e6   # slack upper bound (>> max keep-out violation)
        ocp.constraints.lh = np.zeros(num_obstacles)
        ocp.constraints.uh = np.full(num_obstacles, _UH)
        ocp.constraints.lh_e = np.zeros(num_obstacles)
        ocp.constraints.uh_e = np.full(num_obstacles, _UH)

        ocp.constraints.idxsh = np.arange(num_obstacles, dtype=int)
        ocp.constraints.idxsh_e = np.arange(num_obstacles, dtype=int)
        ocp.constraints.lsh = np.zeros(num_obstacles)
        ocp.constraints.ush = np.full(num_obstacles, _USH)
        ocp.constraints.lsh_e = np.zeros(num_obstacles)
        ocp.constraints.ush_e = np.full(num_obstacles, _USH)

        # Linear slack cost on lower slack only (zu=0: h>=0 cannot be violated from above).
        # TODO: expose Zl/Zl_e for quadratic penalty if stronger penalization is needed.
        ocp.cost.zl = obstacle_slack_weight * np.ones(num_obstacles)
        ocp.cost.zu = np.zeros(num_obstacles)
        ocp.cost.Zl = np.zeros(num_obstacles)
        ocp.cost.Zu = np.zeros(num_obstacles)
        ocp.cost.zl_e = obstacle_slack_weight * np.ones(num_obstacles)
        ocp.cost.zu_e = np.zeros(num_obstacles)
        ocp.cost.Zl_e = np.zeros(num_obstacles)
        ocp.cost.Zu_e = np.zeros(num_obstacles)

    # Soft input-rate (slew) constraint at the initial shooting node. acados forms
    # nonlinear constraints per stage and u_prev is a single parameter (the command
    # applied on the previous tick), so a rate bound is physically meaningful only at
    # stage 0: u0 is the command that reaches the actuator this tick and |u0 - u_prev|
    # is its per-period slew. Interior-stage rate is shaped by the Rd cost instead (a
    # true per-stage rate bound would require augmenting the state with the inputs).
    # Softened with slack so a tight limit degrades the solution rather than making the
    # QP infeasible — parity with the CasADi backend's slacked input-rate bound. Bounds
    # are per-step: rate_max * dt, with dt = Tf / N.
    _rate_rows = []
    _rate_lb = []
    _rate_ub = []
    _dt_step = Tf / N
    if jerk_max is not None and jerk_max > 0.0:
        _rate_rows.append(model.u[0] - u_prev[0])   # acceleration rate (jerk)
        _rate_lb.append(-jerk_max * _dt_step)
        _rate_ub.append(jerk_max * _dt_step)
    if steer_rate_max is not None and steer_rate_max > 0.0:
        _rate_rows.append(model.u[1] - u_prev[1])   # steering rate
        _rate_lb.append(-steer_rate_max * _dt_step)
        _rate_ub.append(steer_rate_max * _dt_step)
    if _rate_rows:
        _n_rate = len(_rate_rows)
        ocp.model.con_h_expr_0 = casadi.vertcat(*_rate_rows)
        ocp.constraints.lh_0 = np.array(_rate_lb)
        ocp.constraints.uh_0 = np.array(_rate_ub)
        # All rate rows soft (two-sided). Linear + quadratic penalty on both slacks so
        # the band is nearly hard yet never a source of infeasibility.
        ocp.constraints.idxsh_0 = np.arange(_n_rate, dtype=int)
        ocp.constraints.lsh_0 = np.zeros(_n_rate)
        ocp.constraints.ush_0 = np.zeros(_n_rate)
        ocp.cost.zl_0 = input_rate_slack_weight * np.ones(_n_rate)
        ocp.cost.zu_0 = input_rate_slack_weight * np.ones(_n_rate)
        ocp.cost.Zl_0 = input_rate_slack_weight * np.ones(_n_rate)
        ocp.cost.Zu_0 = input_rate_slack_weight * np.ones(_n_rate)

    # setting constraints
    ocp.constraints.constr_type = 'BGH'  # b: box/decision variables, g: dynamics, h:nonlinear constraints
    ocp.constraints.lbx = np.array([model.vel_min])  # state lower bound
    ocp.constraints.ubx = np.array([model.vel_max])  # state upper bound
    ocp.constraints.idxbx = np.array([2])  # velocity only; psi is unconstrained (continuous integrator state)

    ocp.constraints.lbu = np.array([model.acc_min, model.delta_min])  # input lower bound
    ocp.constraints.ubu = np.array([model.acc_max, model.delta_max])  # input upper bound
    ocp.constraints.idxbu = np.array([0, 1])

    # # Soft/slack state bounds on velocity (todo: enable if needed)
    # ocp.constraints.lsbx = np.zeros([nsbx])
    # ocp.constraints.usbx = np.zeros([nsbx])
    # ocp.constraints.idxsbx = np.array(range(nsbx))
    #
    # # Nonlinear constraints (todo: add rate constraints)
    # ocp.constraints.lh = np.array([model.delta_min])
    # ocp.constraints.uh = np.array([model.delta_max])
    # ocp.constraints.lsh = np.zeros(nsh)
    # ocp.constraints.ush = np.zeros(nsh)
    # ocp.constraints.idxsh = np.array(range(nsh))

    # set initial condition (optional). Note: will be overwritten
    if x0 is None:
        try:
            ocp.constraints.x0 = model.x0
        except AttributeError:
            pass

    else:
        ocp.constraints.x0 = x0

    # set QP solver and integration
    ocp.solver_options.qp_solver = qp_solver
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.nlp_solver_type = nlp_solver_type  # SQP_RTI, SQP. SQP_RTI does only one iteration while SQP solves to a certain tolerance
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'  # turns on globalization. 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
    # GAUSS_NEWTON is only valid for [NON]LINEAR_LS costs. For EXTERNAL cost acados
    # cannot form a GN Hessian (it prints a warning and silently falls back to the
    # exact cost Hessian every solve), so select EXACT explicitly. The CONVEXIFY
    # regularization set below keeps the (possibly indefinite) exact Hessian usable.
    _uses_external_cost = ('external' in (cost_module.lower(), cost_module_e.lower()))
    ocp.solver_options.hessian_approx = "EXACT" if _uses_external_cost else "GAUSS_NEWTON"
    # 'IRK' (implicit), 'ERK' (explicit), 'GNSF', 'DISCRETE', 'LIFTED_IRK'.
    # DISCRETE consumes model.disc_dyn_expr (RK4 one-step map); the continuous
    # sim_method_* options below only apply to ERK/IRK.
    ocp.solver_options.integrator_type = integrator_type
    # ocp.solver_options.collocation_type = 'EXPLICIT_RUNGE_KUTTA'  # 'GAUSS_RADAU_IIA', 'GAUSS_LEGENDRE', 'EXPLICIT_RUNGE_KUTTA'
    ocp.solver_options.hpipm_mode = 'SPEED'  # 'BALANCE', 'SPEED_ABS', 'SPEED', 'ROBUST' (Tested)
    # NO_REGULARIZE, MIRROR, PROJECT (Tested), CONVEXIFY, PROJECT_REDUC_HESS
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.reg_epsilon = 1e-4
    ocp.solver_options.print_level = 0
    if integrator_type.upper() != 'DISCRETE':
        ocp.solver_options.sim_method_num_stages = 4  # (1) RK1, (2) RK2, (4) RK4
        ocp.solver_options.sim_method_num_steps = 1  # 3, 1. Higher values improve discretization accuracy but with increased computational cost
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = num_iterations
    # ocp.solver_options.tol = tolerance  # 1e-4
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1
    ocp.solver_options.qp_solver_cond_N = qp_solver_cond_N if qp_solver_cond_N is not None else int(N / 2)
    # ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.qp_solver_iter_max = num_iterations
    # ocp.solver_options.qp_tol = tolerance  # 1e-3

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # create solver
    ocp.code_export_directory = code_export_directory
    if with_cython:
        ocp.code_export_directory = f"{ocp.code_export_directory}_cython"
        if generate:
            AcadosOcpSolver.generate(ocp, json_file=mpc_config_file)
        if build:
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True, verbose=True)
        acados_solver = AcadosOcpSolver.create_cython_solver(mpc_config_file)
    else:
        acados_solver = AcadosOcpSolver(ocp, json_file=mpc_config_file,
                                        build=build, generate=generate, verbose=True)

    return constraint, model, acados_solver, ocp, _has_weight_params


if __name__ == "__main__":
    constraint, model, acados_solver, ocp, _ = acados_settings(Tf=1.0, N=25)
    AcadosOcpSolver.generate(ocp, json_file="kinematic_bicycle_acados_ocp.json")
    # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
