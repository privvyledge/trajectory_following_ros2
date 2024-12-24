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
                    cost_module='external', cost_module_e='external',
                    generate=True, build=True, with_cython=True,
                    num_iterations=10, tolerance=1e-6,
                    mpc_config_file="kinematic_bicycle_acados_ocp.json", code_export_directory="c_generated_code"

):
    # generate = True  # generates the OCP and stores in the json file
    # build = True  # builds/compiles the model and stores in code_export_directory
    # the cython version is faster than bare C because there is no call overhead as opposed to the C code call overhead
    # todo: get constraints and max iterations

    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint = kinematic_model()

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
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
    # ocp.dims.N = N  # prediction horizon (depracated)
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
        ocp.cost.cost_type = "LINEAR_LS"

        y = Vx @ model.x + Vu @ model.u
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu

        ocp.cost.W = W  # weight matrix at intermediate shooting nodes (1 to N-1)

        # set initial reference. Note: will be overwritten
        ocp.cost.yref = np.zeros(ny)  # reference at intermediate shooting nodes (1 to N-1) [ny, 1]

    elif cost_module.lower() == "NONLINEAR_LS".lower():
        ocp.cost.cost_type = "NONLINEAR_LS"

        # could be an expression involving terms other than x, u
        y = casadi.vertcat(
                model.x,
                model.u
        )
        ocp.model.cost_y_expr = y
        ocp.cost.W = W  # weight matrix at intermediate shooting nodes (1 to N-1)

        # set initial reference. Note: will be overwritten
        ocp.cost.yref = np.zeros(ny)  # reference at intermediate shooting nodes (1 to N-1) [ny, 1]

    elif cost_module.lower() == "EXTERNAL".lower():
        ocp.cost.cost_type = "EXTERNAL"
        y = casadi.vertcat(
                model.x,
                model.u
        )
        # cost expression = model.x.T @ Q @ model.x  + model.u.T @ R @ model.u
        u_rate = model.u - u_prev
        # u_rate = casadi.vertcat(u_rate, casadi.diff(u_dv))  # to pack with horizon

        u_rate_cost = u_rate.T @ Rd @ u_rate  # doesn't work for now. Add to cost
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)  # todo: normalize angle

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

        # could be an expression involving terms other than x
        y_e = casadi.vertcat(
                model.x,
        )
        ocp.model.cost_y_expr_e = y_e

        ocp.cost.W_e = W_e  # weight matrix at terminal shooting node (N)
        # set initial reference. Note: will be overwritten
        ocp.cost.yref_e = np.zeros(ny_e)  # cost reference at terminal shooting node (N)  [nx, 1]

    elif cost_module_e.lower() == "EXTERNAL".lower():
        ocp.cost.cost_type_e = "EXTERNAL"
        y_e = casadi.vertcat(
                model.x,
        )
        yref_e = yref[:nx]
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)  # todo: normalize angle

    else:
        raise AttributeError(f'Invalid cost type ({cost_module_e}) specified.')

    # ocp.cost.zl = 100 * np.ones((ns,))  # gradient wrt lower slack at intermediate shooting nodes (0 to N-1)
    # ocp.cost.zu = 100 * np.ones((ns,))  # gradient wrt upper slack at intermediate shooting nodes (0 to N-1)
    # ocp.cost.Zl = 1 * np.ones((ns,))  # diagonal of Hessian wrt lower slack at intermediate shooting nodes (0 to N-1)
    # ocp.cost.Zu = 1 * np.ones((ns,))  # diagonal of Hessian wrt upper slack at intermediate shooting nodes (0 to N-1)

    # setting constraints
    ocp.constraints.constr_type = 'BGH'  # b: box/decision variables, g: dynamics, h:nonlinear constraints
    ocp.constraints.lbx = np.array([model.vel_min, model.psi_min])  # state lower bound
    ocp.constraints.ubx = np.array([model.vel_max, model.psi_max])  # state upper bound
    ocp.constraints.idxbx = np.array([2, 3])  # index/indices of states with constraints, else +/-casadi.infinity

    ocp.constraints.lbu = np.array([model.acc_min, model.delta_min])  # input lower bound
    ocp.constraints.ubu = np.array([model.acc_max, model.delta_max])  # input upper bound
    ocp.constraints.idxbu = np.array([0, 1])

    # # Soft/slack state bounds: to map lower and upper slack vectors onto x(T). todo: enable
    # ocp.constraints.lsbx = np.array([-2 * np.pi])  # np.zeros([nsbx])
    # ocp.constraints.usbx = np.array([2 * np.pi])  # np.zeros([nsbx])
    # ocp.constraints.idxsbx = np.array(range(nsbx))
    #
    # # nonlinear constraints as defined by the symbol/function/equation model_ac.con_h_expr = constraint.expr. todo: add rate constraints
    # ocp.constraints.lh = np.array(
    #         [
    #             model.psi_min,
    #             # model.throttle_min,
    #             # model.delta_min,
    #         ]
    # )
    # ocp.constraints.uh = np.array(
    #         [
    #             model.psi_max,
    #             # model.throttle_max,
    #             # model.delta_max,
    #         ]
    # )
    # # Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints
    # ocp.constraints.lsh = np.zeros(nsh)
    # # Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints
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
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP. SQP_RTI does only one iteration while SQP solves to a certain tolerance
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'  # turns on globalization. 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
    ocp.solver_options.hessian_approx = "EXACT"  # 'GAUSS_NEWTON', 'EXACT'.
    ocp.solver_options.integrator_type = "ERK"  # 'IRK' (implicit), 'ERK' (explicit), 'GNSF', 'DISCRETE', 'LIFTED_IRK'
    # ocp.solver_options.collocation_type = 'EXPLICIT_RUNGE_KUTTA'  # 'GAUSS_RADAU_IIA', 'GAUSS_LEGENDRE', 'EXPLICIT_RUNGE_KUTTA'
    ocp.solver_options.hpipm_mode = 'ROBUST'  # 'BALANCE', 'SPEED_ABS', 'SPEED', 'ROBUST'
    # NO_REGULARIZE, MIRROR, PROJECT, CONVEXIFY, PROJECT_REDUC_HESS
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.reg_epsilon = 1e-4
    ocp.solver_options.print_level = 0
    ocp.solver_options.sim_method_num_stages = 4  # (1) RK1, (2) RK2, (4) RK4
    ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = num_iterations
    # ocp.solver_options.tol = tolerance  # 1e-4
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1
    # ocp.solver_options.qp_solver_cond_N = 1  # int(N / 2)  # or N if scale_cost=False
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

    return constraint, model, acados_solver, ocp


if __name__ == "__main__":
    constraint, model, acados_solver, ocp = acados_settings(Tf=1.0, N=25)
    AcadosOcpSolver.generate(ocp, json_file="kinematic_bicycle_acados_ocp.json")
    # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
