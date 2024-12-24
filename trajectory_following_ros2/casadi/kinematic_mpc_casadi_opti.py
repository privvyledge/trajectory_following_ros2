"""
See:
  Good:
    * https://github.com/berlala/genesis_path_follower/blob/master/scripts/controllers/kinematic_mpc.py
    * https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py
    * https://github.com/tomcattiger1230/CasADi_MPC_MHE_Python/blob/master/MPC/sim_4_mpc_robot_tracking_mul_shooting_opt.py
  Extra:
    * https://github.com/mlab-upenn/MPC_project/blob/main/LMPC/Dubin/FTOCP.py
    * https://github.com/mlab-upenn/Dynamic_bic_mpc/blob/main/src/ndyn_ftocp.py
    * https://github.com/mlab-upenn/MPC_dyn/blob/main/src/dyn_ftocp.py
    * https://github.com/JZ76/f1tenth_simulator_two_agents/blob/master/node/MPC_red.py

Todo:
    * implement custom linearizer and discretizer (
        Linearizer:
            check my jacobian implementation
            https://web.casadi.org/python-api/#casadi.casadi.linearize
        Discretizer:
            https://web.casadi.org/python-api/#integrator
            https://github.com/uzh-rpg/high_mpc/blob/master/high_mpc/mpc/mpc.py#L285
            https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py#L157
            https://github.com/casadi/casadi/blob/main/docs/examples/python/race_car.py#L47
            https://github.com/casadi/casadi/blob/main/docs/examples/python/implicit_runge-kutta.py)
    * silence ipopt output (https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py#L184) [Done]
    * rename variables
    * replace for loops with map
    * replace model with kinematic_bicycle_model
    * Setup warmstarting and codegen (https://github.com/casadi/casadi/discussions/3434#discussioncomment-7404877)
"""
import time
import numpy as np
import casadi

from trajectory_following_ros2.casadi.kinematic_bicycle_model import KinematicBicycleModel


class KinematicMPCCasadiOpti(object):
    """docstring for ClassName"""

    def __init__(self, horizon=15, sample_time=0.02, wheelbase=0.256,
                 nx=4, nu=2, x0=None, u0=None,
                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                 warmstart=True, solver_options=None, solver_type='nlp', solver='ipopt', suppress_ipopt_output=True,
                 normalize_yaw_error=True,
                 slack_weights_u_rate=(0.0, 0.0),  # (1e-6, 1e-6)
                 slack_scale_u_rate=(1.0, 1.0),
                 slack_upper_bound_u_rate=None, #(1., np.radians(30.0)), (np.inf, np.inf)
                 slack_objective_is_quadratic=False):
        """ Constructor for KinematicMPCCasadiOpti """
        # self.vehicle = vehicle
        # self.model = vehicle.model

        self.horizon = horizon
        self.nx = nx
        self.nu = nu
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.Qf = casadi.diag(Qf)
        self.Rd = casadi.diag(Rd)
        self.P_u_rate = casadi.diag(slack_weights_u_rate) if slack_weights_u_rate is not None else casadi.DM.zeros(self.nu, self.nu)  # >= 0,should be small, for soft constraints and large for hard constraints. 0 to disable slack
        self.V_scale_u_rate = casadi.diag(slack_scale_u_rate) if slack_scale_u_rate is not None else casadi.DM.eye(self.nu)  # should be identity in most cases
        # set the positive slack upper bound
        if slack_upper_bound_u_rate is not None and isinstance(slack_upper_bound_u_rate, (list, tuple, np.ndarray)):
            if isinstance(slack_upper_bound_u_rate, np.ndarray):
                slack_upper_bound_u_rate = slack_upper_bound_u_rate.flatten().tolist()
            elif isinstance(slack_upper_bound_u_rate, tuple):
                slack_upper_bound_u_rate = list(slack_upper_bound_u_rate)
            for i in range(len(slack_upper_bound_u_rate)):
                if slack_upper_bound_u_rate[i] < 0:
                    slack_upper_bound_u_rate[i] = np.inf
        self.slack_upper_bound_u_rate = casadi.DM(slack_upper_bound_u_rate) if slack_upper_bound_u_rate is not None else [np.inf, np.inf]  # to put limits (upper bound) on the slack variables. Must be > 0. 0 implies None
        self.slack_objective_is_quadratic = slack_objective_is_quadratic  # if True, use quadratic slack objective else linear
        self.warmstart = warmstart
        self.solver_options = solver_options
        self.solver_type = solver_type
        self.solver_ = solver

        if self.solver_ in ["osqp", "qpoases"] or self.solver_type in ['quad', 'conic', 'qp']:
            self.solver_type = 'conic'  # casadi opti uses conic (or sqpmethod in newer versions of casadi) for SQPs, etc

        self.normalize_yaw_error = normalize_yaw_error  # normalizes the angle in the range [-pi, pi)

        self.Ts = sample_time
        self.T_predict = self.Ts * self.horizon  # prediction horizon, i.e T_predict / horizon = dt
        self.wheelbase = wheelbase
        self.lf = wheelbase / 2
        self.lr = self.wheelbase - self.lf
        self.width = 0.192

        self.previous_input = None  # previous input: [u_{acc, -1}, u_{df, -1}]
        self.current_state = x0  # current state:  [x_0, y_0, psi_0, v_0]

        self.x_ref = None
        self.y_ref = None
        self.v_ref = None
        self.psi_ref = None
        self.z_ref = np.zeros((self.nx, self.horizon + 1))  # reference trajectory ([x, y, vel, psi], horizon)
        self.u_ref = np.zeros((self.nu, self.horizon))  # reference input ([acc, delta], horizon)

        self.x_dv = None
        self.y_dv = None
        self.v_dv = None
        self.psi_dv = None
        self.z_dv = np.zeros((self.nx, self.horizon + 1))  # actual/predicted/openloop trajectory

        self.acc_dv = None
        self.delta_dv = None
        self.u_dv = np.array([self.acc_dv, self.delta_dv])  # actual/predicted/openloop inputs

        self.jerk_dv = None
        self.delta_rate_dv = None
        self.u_rate_dv = np.array([self.jerk_dv, self.delta_rate_dv])  # actual/predicted/openloop input rates

        self.sl_acc_dv = np.zeros(self.horizon)  # to handle cases where slack is disabled
        self.sl_delta_dv = np.zeros(self.horizon)  # to handle cases where slack is disabled
        self.sl_dv = np.array([self.sl_acc_dv, self.sl_delta_dv])  # slack variables for input rates

        self.mpc = self.initialize_mpc(horizon=self.horizon,
                                       nx=self.nx, nu=self.nu)

        # define the objective function and constraints
        self.constraints_setup(vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
                               jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound, reset=False)
        self.cost = self.objective_function_setup()

        if x0 is not None:
            self.update_initial_condition(x0[0], x0[1], x0[2], x0[3])

            # provide time-varing parameters: setpoints/references
            self.update_reference([self.Ts * (x + 1) for x in range(self.horizon)],
                                  self.horizon * [float(x0[1])],
                                  self.horizon * [float(x0[2])],
                                  self.horizon * [float(x0[3])])

        if u0 is None:
            self.update_previous_input(0., 0.)
        else:
            if isinstance(u0, np.ndarray):
                u0 = u0.flatten().tolist()
            self.update_previous_input(u0[0], u0[1])

        self.solver = self.setup_solver(self.cost, solver_options=self.solver_options, solver=self.solver_,
                                        suppress_output=suppress_ipopt_output)  # returns None when using Opti

        self.solution = self.solve()

    def initialize_mpc(self, horizon, nx, nu):
        if self.solver_type == 'nlp':
            mpc = casadi.Opti()
        else:
            mpc = casadi.Opti(self.solver_type)

        # Parameters
        self.previous_input = mpc.parameter(nu)
        self.current_state = mpc.parameter(nx)  # z_k = [x_0, y_0, v_0, psi_0]
        ''' 
        Reference trajectory. 
        The first index is the desired state at time k+1, i.e z_ref[0, :] = z_desired
        The second index selects the state element from [x_k, y_k, v_k, psi_k]
        '''
        self.z_ref = mpc.parameter(horizon, nx)
        self.x_ref = self.z_ref[:, 0]
        self.y_ref = self.z_ref[:, 1]
        self.v_ref = self.z_ref[:, 2]
        self.psi_ref = self.z_ref[:, 3]

        # Decision variables
        '''
        Actual trajectory we will follow given the optimal solution.
        The first index is the timestep k, i.e. self.z_dv[0,:] is z_0.	
        It has self.N+1 timesteps since we go from z_0, ..., z_self.N.
        Second index is the state element, as detailed below.
        '''
        self.z_dv = mpc.variable(horizon + 1, nx)
        self.x_dv = self.z_dv[:, 0]
        self.y_dv = self.z_dv[:, 1]
        self.v_dv = self.z_dv[:, 2]
        self.psi_dv = self.z_dv[:, 3]

        '''
        Control inputs used to achieve self.z_dv according to dynamics.
        The first index is the timestep k, i.e. self.u_dv[0,:] is u_0.
        The second index is the input element as detailed below.
        '''
        self.u_dv = mpc.variable(horizon, nu)

        self.acc_dv = self.u_dv[:, 0]
        self.delta_dv = self.u_dv[:, 1]

        '''
        Control input rates used to achieve self.z_dv according to dynamics.
        The first index is the timestep k, i.e. self.u_dv[0,:] is u_0.
        The second index is the input element as detailed below.
        '''
        self.u_rate_dv = mpc.variable(horizon, nu)

        self.jerk_dv = self.u_rate_dv[:, 0]
        self.delta_rate_dv = self.u_rate_dv[:, 1]

        '''
        Slack variables used to relax input rate constraints.
        Matches self.u_dv in structure but timesteps range from -1, ..., N-1.
        '''
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            self.sl_dv = mpc.variable(horizon, nu)

            self.sl_acc_dv = self.sl_dv[:, 0]
            self.sl_delta_dv = self.sl_dv[:, 1]
        return mpc

    def constraints_setup(
            self, vel_bound=None, delta_bound=None, acc_bound=None,
            jerk_bound=None, delta_rate_bound=None, reset=False
    ):
        if delta_bound is None:
            delta_bound = [-23.0, 23.0]

        if vel_bound is None:
            vel_bound = [-10.0, 10.0]

        if acc_bound is None:
            acc_bound = [-3.0, 3.0]

        if jerk_bound is None:
            jerk_bound = [-1.5, 1.5]

        if delta_rate_bound is None:
            delta_rate_bound = [-352.9411764706, 352.9411764706]

        # initial state constraints
        self.mpc.subject_to(self.x_dv[0] == self.current_state[0])
        self.mpc.subject_to(self.y_dv[0] == self.current_state[1])
        self.mpc.subject_to(self.v_dv[0] == self.current_state[2])
        self.mpc.subject_to(self.psi_dv[0] == self.current_state[3])

        # # todo: test replacing for-loops with map (https://web.casadi.org/docs/#for-loop-equivalents)
        #  Todo: replace with linear discrete model, i.e Ax + Bu + Gw
        # # todo: wrap integrator as a function
        # self.kin_bic_model = KinematicBicycleModel(nx=4, nu=2, x0=self.current_state,
        #                                            vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
        #                                            jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound,
        #                                            vehicle_parameters={'wheelbase': 0.256}, sample_time=self.Ts,
        #                                            symbol_type='SX',
        #                                            model_type='kinematic', model_name='vehicle_kinematic_model',
        #                                            discretization_method='cvodes')
        #
        # # todo: remove
        # model_to_discretize = self.kin_bic_model.substitute_symbols(self.kin_bic_model.model.f_expl_expr,
        #                                                        [self.kin_bic_model.model.params.wheelbase],
        #                                                        [self.kin_bic_model.wheelbase])
        # integration_function = self.kin_bic_model.setup_discretizer(self.kin_bic_model.model.x,
        #                                                             self.kin_bic_model.model.u,
        #                                                             ode=model_to_discretize, dt=self.Ts)
        #
        # temp_x = casadi.vertcat(self.x_dv[0], self.y_dv[0], self.v_dv[0], self.psi_dv[0])
        # temp_u = casadi.vertcat(self.acc_dv[0], self.delta_dv[0])
        # integration_function = self.kin_bic_model.setup_discretizer(temp_x, temp_u, ode=model_to_discretize,
        #                                                             dt=self.Ts)

        # state model constraints.
        for i in range(self.horizon):
            # # todo: add different integrators
            # k1 = self.f(st, con)
            # if self.INTEGRATION_MODE == "Euler":
            #     st_next_euler = st + (self.dT * k1)
            # elif self.INTEGRATION_MODE == "RK3":
            #     k2 = self.f(st + self.dT / 2 * k1, con)
            #     k3 = self.f(st + self.dT * (2 * k2 - k1), con)
            #     st_next_euler = st + self.dT / 6 * (k1 + 4 * k2 + k3)
            # elif self.INTEGRATION_MODE == "RK4":
            #     k2 = self.f(st + self.dT / 2 * k1, con)
            #     k3 = self.f(st + self.dT / 2 * k2, con)
            #     k4 = self.f(st + self.dT * k3, con)
            #     st_next_euler = st + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # dt = T / N  # length of a control interval
            # for k in range(N):  # loop over control intervals
            #     # Runge-Kutta 4 integration
            #     k1 = f(X[:, k], U[:, k])
            #     k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
            #     k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
            #     k4 = f(X[:, k] + dt * k3, U[:, k])
            #     x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # # beta = casadi.atan(self.lr / (self.lf + self.lr) * casadi.tan(self.delta_dv[i]))
            beta = 0.0
            # x_[k+1]
            self.mpc.subject_to(
                    self.x_dv[i + 1] == self.x_dv[i] + self.Ts * (self.v_dv[i] * casadi.cos(self.psi_dv[i] + beta)))
            # y_[k+1]
            self.mpc.subject_to(
                    self.y_dv[i + 1] == self.y_dv[i] + self.Ts * (self.v_dv[i] * casadi.sin(self.psi_dv[i] + beta)))
            # vel_[k+1]
            self.mpc.subject_to(self.v_dv[i + 1] == self.v_dv[i] + self.Ts * (self.acc_dv[i]))
            # psi_[k+1]
            # self.mpc.subject_to(
            #     self.psi_dv[i + 1] == self.psi_dv[i] + self.Ts * ((self.v_dv[i] / self.lr) * casadi.sin(beta)))
            self.mpc.subject_to(
                    self.psi_dv[i + 1] == self.psi_dv[i] +
                    self.Ts * ((self.v_dv[i] / self.wheelbase) * casadi.tan(self.delta_dv[i])))

            # temp_x = casadi.vertcat(self.x_dv[i], self.y_dv[i], self.v_dv[i], self.psi_dv[i])
            # temp_u = casadi.vertcat(self.acc_dv[i], self.delta_dv[i])
            # integration_function[i+1] = self.kin_bic_model.setup_discretizer(temp_x, temp_u, ode=model_to_discretize,
            #                                                                  dt=self.Ts)
            # # # todo: remove
            # # discrete_solution = self.kin_bic_model.discretize(integration_function, temp_x, temp_u)

        # state constraints
        self.mpc.subject_to(self.mpc.bounded(vel_bound[0], self.v_dv, vel_bound[1]))
        self.mpc.subject_to(self.mpc.bounded(-2 * casadi.pi, self.psi_dv, 2 * casadi.pi))

        # Input Bound Constraints
        self.mpc.subject_to(self.mpc.bounded(acc_bound[0], self.acc_dv, acc_bound[1]))
        self.mpc.subject_to(
            self.mpc.bounded(np.radians(delta_bound[0]),
                             self.delta_dv,
                             np.radians(delta_bound[1])))

        # Input Rate Bound Constraints. todo: refactor without for loop and test, i.e use a for-loop or map to assign the rates and set the bound
        slack_flag = int(not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))))
        self.jerk_dv[0] = self.acc_dv[0] - self.previous_input[0]
        self.delta_rate_dv[0] = self.delta_dv[0] - self.previous_input[1]
        self.mpc.subject_to(self.mpc.bounded(
                (jerk_bound[0] * self.Ts) - (slack_flag * self.V_scale_u_rate[0, 0] * self.sl_acc_dv[0]),
                self.jerk_dv[0],
                (jerk_bound[1] * self.Ts) + (slack_flag * self.V_scale_u_rate[0, 0] * self.sl_acc_dv[0])))

        self.mpc.subject_to(self.mpc.bounded(
                (np.radians(delta_rate_bound[0]) * self.Ts) - (slack_flag * self.V_scale_u_rate[1, 1] * self.sl_delta_dv[0]),
                self.delta_rate_dv[0],
                (np.radians(delta_rate_bound[1]) * self.Ts) + (slack_flag * self.V_scale_u_rate[1, 1] * self.sl_delta_dv[0])))

        for i in range(self.horizon - 1):
            self.jerk_dv[i + 1] = self.acc_dv[i + 1] - self.acc_dv[i]
            self.delta_rate_dv[i + 1] = self.delta_dv[i + 1] - self.delta_dv[i]
            self.mpc.subject_to(self.mpc.bounded(
                    (jerk_bound[0] * self.Ts) - (slack_flag * self.V_scale_u_rate[0, 0] * self.sl_acc_dv[i + 1]),
                    self.jerk_dv[i + 1],
                    (jerk_bound[1] * self.Ts) + (slack_flag * self.V_scale_u_rate[0, 0] * self.sl_acc_dv[i + 1])))
            self.mpc.subject_to(self.mpc.bounded(
                    (np.radians(delta_rate_bound[0]) * self.Ts) - (slack_flag * self.V_scale_u_rate[1, 1] * self.sl_delta_dv[i + 1]),
                    self.delta_rate_dv[i + 1],
                    (np.radians(delta_rate_bound[1]) * self.Ts) + (slack_flag * self.V_scale_u_rate[1, 1] * self.sl_delta_dv[i + 1])
            ))
        # Slack Bound Constraints.
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            if self.slack_upper_bound_u_rate is None:
                self.mpc.subject_to(
                    0 <= self.sl_acc_dv)  # test removing the conditional by using (0 <= self.sl_delta_dv) <= casadi.inf and declaring slack bound u rate as inf or 1E15
                self.mpc.subject_to(0 <= self.sl_delta_dv)  # test removing the conditional by using (0 <= self.sl_delta_dv) <= casadi.inf and declaring slack bound u rate as inf or 1E15
            else:
                # self.mpc.subject_to((0 <= self.sl_acc_dv) <= self.slack_upper_bound_u_rate[0])
                # self.mpc.subject_to((0 <= self.sl_delta_dv) <= self.slack_upper_bound_u_rate[1])
                self.mpc.subject_to(self.mpc.bounded(0, self.sl_acc_dv, self.slack_upper_bound_u_rate[0]))
                self.mpc.subject_to(self.mpc.bounded(0, self.sl_delta_dv, self.slack_upper_bound_u_rate[1]))

        # Other Constraints
        # e.g. things like collision avoidance or lateral acceleration bounds could go here. for static obstacles, the position will be repeated horizon times, for dynamic obstacles, the k-th position should be added in the loop

    def _quad_form(self, z, Q):
        return casadi.mtimes(z, casadi.mtimes(Q, z.T))  # z.T @ Q @ z

    def objective_function_setup(self):
        """
        path following
        """
        cost = 0
        # tracking cost
        for i in range(self.horizon):
            if self.normalize_yaw_error:
                cost += self._quad_form(self.z_dv[i + 1, 0:3] - self.z_ref[i, 0:3], self.Q[0:3, 0:3])
                cost += self._quad_form(casadi.fmod(self.z_dv[i + 1, 3] - self.z_ref[i, 3] + np.pi, 2 * np.pi) - np.pi,
                                        self.Q[3, 3])  # normalizes the angle in the range [-pi, pi)
            else:
                cost += self._quad_form(self.z_dv[i + 1, :] - self.z_ref[i, :], self.Q)

        # input cost.
        for i in range(self.horizon - 1):
            cost += self._quad_form(self.u_dv[i, :], self.R)

        # input derivative cost.
        self.u_rate_dv[0, :] = (self.u_dv[0, :] - self.previous_input.T)  # / self.Ts
        for i in range(self.horizon - 1):
            self.u_rate_dv[i + 1, :] = (self.u_dv[i + 1, :] - self.u_dv[i, :])  # / self.Ts
            cost += self._quad_form(self.u_rate_dv[i + 1, :], self.Rd)

        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            if self.slack_objective_is_quadratic:
                cost += self.P_u_rate[0, 0] * casadi.sum1(self.sl_acc_dv ** 2) + self.P_u_rate[1, 1] * casadi.sum1(self.sl_delta_dv ** 2)  # slack cost
            else:
                slack_weights = casadi.diag(self.P_u_rate)  # to convert to a column vector
                cost += slack_weights[0] * casadi.sum1(self.sl_acc_dv) + slack_weights[1] * casadi.sum1(self.sl_delta_dv)  # slack cost

        cost += self._quad_form(self.z_dv[self.horizon, :] - self.z_ref[self.horizon - 1, :], self.Qf)  # terminal state

        cost *= 0.5
        return cost

    def setup_solver(self, cost, solver_options=None, solver='ipopt', suppress_output=False):
        # to setup warmstarting and codegen (https://github.com/casadi/casadi/discussions/3434#discussioncomment-7404877)

        self.mpc.minimize(cost)
        # Ipopt with custom options: https://web.casadi.org/docs/ -> see sec 9.1 on Opti stack.
        if solver_options is None:
            # self.mpc.solver('ipopt', p_opts, s_opts)
            solver_options = {
                # 'warmstart': True,
            }
            if solver == 'ipopt':
                ipopt_options = {
                    'ipopt.print_level': not suppress_output,
                    'ipopt.sb': 'yes',
                    'ipopt.max_iter': 2000,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6,
                }
                solver_options.update(ipopt_options)

            # todo: test
            elif solver == 'osqp':
                osqp_options = {
                    'verbose': not suppress_output,
                    'max_iter': 2000,
                    'eps_pr': 1e-8,
                    'eps_r': 1e-8,
                    'warm_start_dual': True,
                    'warm_start_primal': True
                }
                solver_options.update(osqp_options)
        solver_options.update(
                {
                    'print_time': not suppress_output,
                    'verbose': not suppress_output,
                    'expand': True,
                    # 'max_cpu_time': 0.1,
                    # 'max_iter': 1000,
                    'record_time': True
                }
        )
        self.mpc.solver(solver, solver_options)
        return self.mpc.solve

    def update_initial_condition(self, x0, y0, vel0, psi0):
        self.mpc.set_value(self.current_state, [x0, y0, vel0, psi0])

    def update_reference(self, x_ref, y_ref, v_ref, psi_ref):
        self.mpc.set_value(self.x_ref, x_ref)
        self.mpc.set_value(self.y_ref, y_ref)
        self.mpc.set_value(self.v_ref, v_ref)
        self.mpc.set_value(self.psi_ref, psi_ref)

    def update_previous_input(self, acc_prev, delta_prev):
        self.mpc.set_value(self.previous_input, [acc_prev, delta_prev])

    def update_u_rate(self, u_prev_value, u_dv_value):
        # not required since the Opti stack keeps track of the variable.
        return

    def update(self, state=None, ref_traj=None, previous_input=None, warmstart_variables=None):
        self.update_initial_condition(*state)
        self.update_reference(*ref_traj)
        self.update_previous_input(*previous_input)

        if self.warmstart and len(warmstart_variables) > 0:
            # Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
            # Format: self.mpc.set_initial(self.solution.value_variables())
            self.mpc.set_initial(self.z_dv, warmstart_variables['z_ws'])
            self.mpc.set_initial(self.u_dv, warmstart_variables['u_ws'])
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                self.mpc.set_initial(self.sl_dv, warmstart_variables['sl_ws'])

            self.mpc.set_initial(self.mpc.lam_g, warmstart_variables['lam_g'])

    def solve(self, debug=True):
        """
        Call update method before this.
        :return:
        """
        st = time.process_time()

        sl_mpc = np.zeros((self.horizon, self.nu)) # or return None
        try:
            sol = self.solver()  # todo: make this an attribute

            # warmstarting
            self.mpc.set_value(sol.value_parameters())

            # Optimal solution.
            u_mpc = sol.value(self.u_dv)
            z_mpc = sol.value(self.z_dv)
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                sl_mpc = sol.value(self.sl_dv)
            z_ref = sol.value(self.z_ref)
            u_prev = sol.value(self.previous_input)
            iteration_count = sol.stats()["iter_count"]

            cost = sol.value(self.mpc.f)
            g = sol.value(self.mpc.g)
            lam_g = sol.value(self.mpc.lam_g)
            lam_x = None  # sol.value(self.mpc.lam_x)
            lam_p = None  # sol.value(self.mpc.lam_p)

            self.lbg = sol.value(self.mpc.lbg)
            self.ubg = sol.value(self.mpc.ubg)
            u_rate = sol.value(self.u_rate_dv)
            # jerk = sol.value(self.jerk_dv)
            # delta_rate = sol.value(self.delta_rate_dv)
            # u_rate_alt = np.array([jerk, delta_rate]).T  # same as u_rate above

            # for index in range(g.size):
            #     print(f"{self.lbg[index]} <= {g[index]} <= {self.ubg[index]}")

            return_state = sol.stats()["return_status"]
            success = sol.stats()["success"]

            t_proc_callback_fun = sol.stats()["t_proc_callback_fun"]
            t_proc_nlp_f = sol.stats()["t_proc_nlp_f"]
            t_proc_nlp_g = sol.stats()["t_proc_nlp_g"]
            t_proc_nlp_grad = sol.stats()["t_proc_nlp_grad"]
            t_proc_nlp_grad_f = sol.stats()["t_proc_nlp_grad_f"]
            t_proc_nlp_hess_l = sol.stats()["t_proc_nlp_hess_l"]
            t_proc_nlp_jac_g = sol.stats()["t_proc_nlp_jac_g"]
            t_proc_total = sol.stats()["t_proc_total"]

            t_wall_callback_fun = sol.stats()["t_wall_callback_fun"]
            t_wall_nlp_f = sol.stats()["t_wall_nlp_f"]
            t_wall_nlp_g = sol.stats()["t_wall_nlp_g"]
            t_wall_nlp_grad = sol.stats()["t_wall_nlp_grad"]
            t_wall_nlp_grad_f = sol.stats()["t_wall_nlp_grad_f"]
            t_wall_nlp_hess_l = sol.stats()["t_wall_nlp_hess_l"]
            t_wall_nlp_jac_g = sol.stats()["t_wall_nlp_jac_g"]
            t_wall_total = sol.stats()["t_wall_total"]
            # print(sol.stats())  # todo:
            is_opt = True

            # # get functions used by the solver.
            # # Could be useful when replacing default functions.
            # # See (https://github.com/casadi/casadi/wiki/FAQ:-How-to-specify-a-custom-Hessian-approximation%3F | https://groups.google.com/g/casadi-users/c/XnDBUWrPTlQ)
            # hess_l = self.mpc.debug.casadi_solver.get_function('nlp_hess_l')
        except Exception as e:
            # Suboptimal solution (e.g. timed out).
            u_mpc = self.mpc.debug.value(self.u_dv)
            z_mpc = self.mpc.debug.value(self.z_dv)
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                sl_mpc = self.mpc.debug.value(self.sl_dv)
            z_ref = self.mpc.debug.value(self.z_ref)
            u_rate = self.mpc.debug.value(self.u_rate_dv)
            u_prev = self.mpc.debug.value(self.previous_input)
            iteration_count = None
            is_opt = False
            success = False
            lam_g = self.mpc.debug.value(self.mpc.lam_g)
            lam_x = None  # sol.value(self.mpc.lam_x)
            lam_p = None  # sol.value(self.mpc.lam_p)

        solve_time = time.process_time() - st

        sol_dict = {'u_control': u_mpc[0, :],
                    'u_mpc': u_mpc,
                    'z_mpc': z_mpc,
                    'u_rate': u_rate,
                    'u_prev': u_prev,
                    'sl_mpc': sl_mpc,
                    'z_ref': z_ref,
                    'optimal': is_opt,
                    'solve_time': solve_time,
                    'iter_count': iteration_count,
                    'solver_status': success,
                    'lam_x': lam_x,
                    'lam_g': lam_g,
                    'lam_p': lam_p,
                    'solver_stats': sol.stats(),
                    }
        return sol_dict

    def set_weights(self, Q, R, Qf=None, Rd=None):
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        if Qf is not None:
            self.Qf = casadi.diag(Qf)
        if Rd is not None:
            self.Rd = casadi.diag(Rd)

    def reset(self, z0=None):
        if z0 is None:
            self.z_dv_value = np.zeros((self.nx, self.horizon + 1))
        self.u_dv_value = np.zeros((self.nu, self.horizon))
        self.u_rate_dv_value = np.zeros((self.nu, self.horizon))
        self.solve()

    def run(self, z0, u0, zref, zk, u_prev):
        # update
        # solve
        pass


if __name__ == '__main__':
    kmco_instance = KinematicMPCCasadiOpti(x0=[0., 0., 1., 0.])
    solution_dictionary = kmco_instance.solve()
    print(solution_dictionary)
