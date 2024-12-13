"""
See:
    To parallelize casadi:
        * https://web.casadi.org/docs/#for-loop-equivalents
        * https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py#L157

Todo:
    * set solver [done]
    * simulate [done]
    * setup cost and constraints using vertcat [done]
    * set constraints as lbx, lbg, ubx, ubg instead [done]
    * add terminal state cost [done]
    * add rate change cost [done]
    * add method to set weights [done]
    * add method to reset, i.e set all back to zero, e.g u_prev, z_k, z_dv, z_ref, [Done]
    * add solver options [Done]
    * add model generation/loading
    * create run method to first update, (optionally) set weights, (optionally) reset, solve
    * setup rate constraints
    * setup slack on rate constraints
    * rename variables
    * replace for loops with map
    * implement my discretizer

"""
import types
import time
import numpy as np
from scipy import linalg
import casadi

from trajectory_following_ros2.casadi.kinematic_bicycle_model import KinematicBicycleModel


class KinematicMPCCasadi(object):
    """docstring for ClassName"""

    def __init__(self, vehicle=None, horizon=15, sample_time=0.02, wheelbase=0.256,
                 nx=4, nu=2, x0=None, u0=None,
                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                 symbol_type='SX', warmstart=True,
                 solver_options=None, solver_type='nlp', solver='ipopt', suppress_ipopt_output=True):
        """Constructor for KinematicMPCCasadi"""
        self.horizon = horizon
        self.nx = nx
        self.nu = nu
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.Qf = casadi.diag(Qf)
        self.Rd = casadi.diag(Rd)
        self.warmstart = warmstart
        self.solver_options = solver_options

        self.Ts = sample_time
        self.T_predict = self.Ts * self.horizon  # prediction horizon, i.e T_predict / horizon = dt
        self.wheelbase = wheelbase
        self.lf = wheelbase / 2
        self.lr = self.wheelbase - self.lf
        self.width = 0.192

        self.vehicle = vehicle
        self.symbol_type = symbol_type
        if self.symbol_type.lower() == 'SX'.lower():
            self.symbol_type = casadi.SX
        else:
            self.symbol_type = casadi.MX

        if vehicle is not None:
            self.vehicle = vehicle
            self.model = self.vehicle.model
        else:
            self.vehicle = KinematicBicycleModel(nx=nx, nu=nu, x0=x0,
                                               vel_bound=vel_bound, delta_bound=vel_bound, acc_bound=acc_bound,
                                               jerk_bound=jerk_bound,
                                               delta_rate_bound=delta_rate_bound,
                                               vehicle_parameters={"wheelbase": self.wheelbase}, sample_time=self.Ts,
                                               model_type='kinematic', model_name='vehicle_kinematic_model',
                                               symbol_type='SX',
                                               discretization_method='cvodes')
            self.model = self.vehicle.model

        self.ode = self.model.f_expl_expr
        self.ode = casadi.substitute(self.ode, self.model.params.wheelbase, self.wheelbase)
        self.ode_linear = self.vehicle.linear_model  # todo: remove ss variables
        self.ode_linear = casadi.substitute(self.ode_linear, casadi.vertcat(self.model.xss, self.model.u0),
                                            casadi.vertcat(self.model.x, self.model.u))
        self.ode_function = self.vehicle.create_ode_function(self.ode,
                                                           self.model.x, self.model.u,
                                                           function_name='nonlinear_ode')
        self.ode_linear_function = self.vehicle.create_ode_function(self.ode_linear,
                                                                  self.model.x, self.model.u,
                                                                  function_name='linear_ode')
        self.u_prev = None  # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_k = x0  # current state:  [x_0, y_0, psi_0, v_0]
        self.u_prev_value = u0  # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_k_value = x0  # current state:  [x_0, y_0, psi_0, v_0]

        self.x_ref = None
        self.y_ref = None
        self.v_ref = None
        self.psi_ref = None
        self.z_ref = np.zeros((self.nx, self.horizon + 1))  # reference trajectory ([x, y, vel, psi], horizon)
        self.u_ref = np.zeros((self.nu, self.horizon))  # reference input ([acc, delta], horizon)
        self.z_ref_value = np.zeros((self.nx, self.horizon + 1))  # reference trajectory ([x, y, vel, psi], horizon)
        self.u_ref_value = np.zeros((self.nu, self.horizon))  # reference input ([acc, delta], horizon)

        self.x_dv = None
        self.y_dv = None
        self.v_dv = None
        self.psi_dv = None
        self.z_dv = np.zeros((self.nx, self.horizon + 1))  # actual/predicted/openloop trajectory
        self.z_dv_value = np.zeros((self.nx, self.horizon + 1))

        self.acc_dv = None
        self.delta_dv = None
        self.u_dv = np.zeros((self.nu, self.horizon))  # actual/predicted/openloop inputs
        self.u_dv_value = np.zeros((self.nu, self.horizon))

        self.sl_acc_dv = None
        self.sl_delta_dv = None
        self.sl_dv = np.zeros((self.nu, self.horizon))  # slack variables for input rates

        self.mpc, self.objective, self.constraints = self.initialize_mpc(horizon=self.horizon, nx=self.nx, nu=self.nu)

        # define the objective function and constraints. todo: setup objective first
        self.constraints, self.lbg, self.ubg, self.lbx, self.ubx = self.constraints_setup(vel_bound=vel_bound,
                                                                                          delta_bound=delta_bound,
                                                                                          acc_bound=acc_bound,
                                                                                          jerk_bound=jerk_bound,
                                                                                          delta_rate_bound=delta_rate_bound,
                                                                                          reset=False)
        self.objective = self.objective_function_setup()

        if x0 is not None:
            self.update_initial_condition(x0[0], x0[1], x0[2], x0[3])

            # provide time-varing parameters: setpoints/references
            self.update_reference([self.Ts * (k + 1) for k in range(self.horizon + 1)],
                                  (self.horizon + 1) * [float(x0[1])],
                                  (self.horizon + 1) * [float(x0[2])],
                                  (self.horizon + 1) * [float(x0[3])])

        if u0 is not None:
            self.update_previous_input(u0[0, 0], u0[1, 0])

        self.solver, self.opt_variables, \
        self.opt_params, self.opt_constraints = self.setup_solver(cost=self.objective, constraints=self.constraints,
                                                                  suppress_output=suppress_ipopt_output,
                                                                  solver_options=self.solver_options,
                                                                  solver=solver,
                                                                  solver_type=solver_type)

        num_tries = 5  # 1, 5, 10. todo: expose as a parameter and separate function
        for _ in range(num_tries):
            # a kind of hacky way to fix not finding a solution the first time the solver is called
            self.solution = self.solve()

    def initialize_mpc(self, horizon, nx, nu):
        mpc = types.SimpleNamespace()
        # Parameters
        self.u_prev = self.symbol_type.sym('u_prev', nu)
        self.z_k = self.symbol_type.sym('z_k', nx)

        # Reference trajectory
        self.z_ref = self.symbol_type.sym('z_ref', nx, horizon + 1)
        self.u_ref = self.symbol_type.sym('u_ref', nu, horizon)

        # Actual/Predicted/Open-Loop Trajectory
        self.z_dv = self.symbol_type.sym('z_dv', nx, horizon + 1)

        # Control inputs
        self.u_dv = self.symbol_type.sym('u_dv', nu, horizon)

        # Slack variables
        self.sl_dv = self.symbol_type.sym('sl_dv', nu, horizon)

        objective = 0.0
        constraints = []
        # todo: remove objective and constraints since they're not being used
        return mpc, objective, constraints

    def constraints_setup(
            self, vel_bound=None, delta_bound=None, acc_bound=None,
            jerk_bound=None, delta_rate_bound=None, reset=False
    ):
        """
        G constraints (lbg, ubg): dynamic constraints in the form of an equation/function that change over time
            Examples -> zdot = f(x, u)
                        z_[k+1] = z_k + f(x, u) * dt
                        z_[k+1] = Az_k + Bu_k
        X (box) constraints (lbx, ubx): static constraints on [x, u]:
            Examples -> min <= variable <= max

        To add input rate constraints, augment the dynamics.
            See: https://forces.embotech.com/Documentation/examples/high_level_rate_constraints/index.html
        :param vel_bound:
        :param delta_bound:
        :param acc_bound:
        :param jerk_bound:
        :param delta_rate_bound:
        :param reset:
        :return:
        """
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

        u_min = np.array([[acc_bound[0]], [delta_bound[0]]])
        u_max = np.array([[acc_bound[1]], [delta_bound[1]]])

        u_dot_min = np.array([[jerk_bound[0]], [delta_rate_bound[0]]])
        u_dot_max = np.array([[jerk_bound[1]], [delta_rate_bound[1]]])

        constraints = []
        '''
        Set lower bound of inequality constraint to zero to force: 
            1. n*N state dynamics
            2. n terminal contraints and
            3. CVX hull constraint
            
            [0] * (self.nx * (self.horizon + 1)) + [0] * self.nx 
        '''
        # lbg = [0] * (self.nx * (self.horizon + 1))  # + [0] * self.nx + [0]  # 0.0
        # ubg = [0] * (self.nx * (self.horizon + 1))  # + [0] * self.nx + [0]  # 0.0

        lbg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)  # + [0] * self.nx + [0]  # 0.0
        ubg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)  # + [0] * self.nx + [0]  # 0.0

        # initial state constraints
        constraints.append(self.z_dv[:, 0] - self.z_k)

        for k in range(self.horizon):
            # todo: create a new symbolic state for x_next and update it at each timestep using my integrator function
            # state model/dynamics constraints.  a=b or a-b
            ode_symbolic = self.ode_function(self.z_dv[:, k], self.u_dv[:, k])
            # ode_symbolic = self.ode_linear_function(self.z_dv[:, k], self.u_dv[:, k])
            z_dt = ode_symbolic * self.Ts  # todo: replace with my integration function
            # z_dt = self.vehicle.get_next_state()

            z_next = self.z_dv[:, k] + z_dt

            # A = self.symbol_type.eye(self.nx) + self.Ts * self.symbol_type.jacobian(ode_symbolic, self.z_dv[:, k])
            # B = self.Ts * self.symbol_type.jacobian(ode_symbolic, self.u_dv[:, k])
            # z_next = A @ self.z_dv[:, k] + B @ self.u_dv[:, k]

            dynamics = self.z_dv[:, k + 1] - z_next  # (self.z_dv[:, k + 1] == z_next)
            constraints.append(dynamics)  # todo: use casadi map or fold

        constraints = casadi.vertcat(*constraints)

        # Input Rate Bound Constraints (lbx)
        u_prev = self.u_prev
        # u_dot = self.u_dv[:, 0] - self.u_prev
        # constraints.append(u_dot >= (u_dot_min * self.Ts - self.sl_dv[:, i + 1]))
        # constraints.append(u_dot <= (u_dot_max * self.Ts + self.sl_dv[:, i + 1]))
        for k in range(self.horizon):
            u_dot = self.u_dv[:, k] - u_prev  # self.u_dv[:, k + 1] - self.u_dv[:, k + 1]
            constraints = casadi.vertcat(constraints, u_dot)
            lbg = casadi.vertcat(lbg, u_dot_min)  # (u_dot_min * self.Ts - self.sl_dv[:, i + 1])
            ubg = casadi.vertcat(ubg, u_dot_max)  # (u_dot_max * self.Ts + self.sl_dv[:, i + 1])
            u_prev = self.u_dv[:, k]

            # # Other Constraints.
            # E.g slack: see https://forces.embotech.com/Documentation/examples/high_level_soft_constraints/index.html
            # constraints = casadi.vertcat(constraints, self.sl_dv[:, k])  # 0 <= self.sl_dv
            # lbg = casadi.vertcat(lbg, 0)
            # ubg = casadi.vertcat(ubg, casadi.inf)
            # # e.g. things like collision avoidance or lateral acceleration bounds could go here.

        # lbx = [-casadi.inf, -casadi.inf, vel_bound[0], -casadi.inf] * (self.horizon + 1)  # state constraints
        # ubx = [casadi.inf, casadi.inf, vel_bound[1], casadi.inf] * (self.horizon + 1)  # state constraints
        #
        # lbx.extend([acc_bound[0], np.radians(delta_bound[0])] * self.horizon)  # input constraints
        # ubx.extend([acc_bound[1], np.radians(delta_bound[1])] * self.horizon)  # input constraints

        lbx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))
        ubx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))

        # state constraints (self.v_dv, vel_bound[1])
        lbx[0: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # X lower bound
        lbx[1: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # Y lower bound
        lbx[2: self.nx * (self.horizon + 1): self.nx] = vel_bound[0]  # vel lower bound
        lbx[3: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # psi lower bound

        ubx[0: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # X upper bound
        ubx[1: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # Y upper bound
        ubx[2: self.nx * (self.horizon + 1): self.nx] = vel_bound[1]  # vel upper bound
        ubx[3: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # psi upper bound

        # controls
        lbx[self.nx * (self.horizon + 1)::self.nu] = acc_bound[0]  # acc lower bound
        lbx[self.nx * (self.horizon + 1) + 1::self.nu] = np.radians(delta_bound[0])  # delta lower bound

        ubx[self.nx * (self.horizon + 1)::self.nu] = acc_bound[1]  # acc lower bound
        ubx[self.nx * (self.horizon + 1) + 1::self.nu] = np.radians(delta_bound[1])  # delta lower bound
        return constraints, lbg, ubg, lbx, ubx

    def _quad_form(self, z, Q):
        return casadi.mtimes(z.T, casadi.mtimes(Q, z))  # z.T @ Q @ z

    def objective_function_setup(self):
        cost = 0
        ''' Lagrange/stage cost '''
        # tracking cost
        u_prev = self.u_prev
        for k in range(self.horizon):
            state_error = self.z_dv[:, k] - self.z_ref[:, k]  # self.z_dv[:, k + 1] - self.z_ref[:, k]
            cost += self._quad_form(state_error, self.Q)

            # input cost.
            if k < (self.horizon - 1):
                control_error = self.u_dv[:, k]  # - self.u_ref[:, k]
                cost += self._quad_form(control_error, self.R)

                # input derivative cost
                input_rate = self.u_dv[:, k] - u_prev
                cost += self._quad_form(input_rate, self.Rd)  # self.u_dv[:, k + 1] - self.u_dv[:, k]
                u_prev = self.u_dv[:, k]

        # # slack cost on input derivatives
        # cost += casadi.sum1(self.sl_dv)

        ''' Mayer/terminal cost '''
        cost += self._quad_form(self.z_dv[:, self.horizon] - self.z_ref[:, self.horizon],
                                self.Qf)  # terminal state

        # cost *= 0.5
        return cost

    def setup_solver(self, cost, constraints, solver_type='nlp',
                     solver_options=None, solver='ipopt', suppress_output=True):
        flat_z = casadi.reshape(self.z_dv, self.nx * (self.horizon + 1), 1)  # flatten (self.nx * (self.horizon + 1), 1)
        flat_u = casadi.reshape(self.u_dv, self.nu * self.horizon, 1)  # flatten (self.nu * self.horizon, 1)
        flat_z_ref = casadi.reshape(self.z_ref, self.nx * (self.horizon + 1),
                                    1)  # flatten (self.nx * (self.horizon + 1), 1)
        flat_z_k = casadi.reshape(self.z_k, self.nx, 1)
        flat_u_ref = casadi.reshape(self.u_ref, -1, 1)  # flatten     (self.nu * self.horizon, 1)
        flat_u_prev = casadi.reshape(self.u_prev, self.nu, 1)

        # flatten and append inputs and states
        opt_variables = casadi.vertcat(
                flat_z,
                flat_u
        )
        opt_params = casadi.vertcat(
                flat_z_ref,
                flat_z_k,
                flat_u_prev
        )  # casadi.vertcat(flat_z_ref, flat_u_ref)

        # lbg, ubg constraints (i.e dynamic constraints in the form of an equation)
        g = casadi.vertcat(constraints)

        optimization_prob = {'f': cost, 'x': opt_variables, 'p': opt_params, 'g': g}

        if solver_options is None:
            solver_options = {
                'record_time': True,
            }

            if solver == 'ipopt':
                ipopt_options = {
                    # 'ipotp.max_iter': 2000,
                    'ipopt.print_level': not suppress_output,
                    'ipopt.sb': 'yes'
                }
                solver_options.update(ipopt_options)

        solver_options.update(
                {
                    'print_time': not suppress_output,
                    'verbose': not suppress_output,
                    'expand': True,
                }
        )

        if solver_type == 'quad':
            '''
            f must be convex and g must be linear
            solvers: qpoases, osqp
            '''
            quad_solver = solver
            if quad_solver not in ["osqp", "qpoases"]:
                quad_solver = 'osqp'  # osqp, qpoases
                print(f"Solver: {solver} not valid. Defaulting to {quad_solver}")

            if quad_solver == "osqp":
                quad_settings = {
                    'warm_start_dual': True,
                    'warm_start_primal': True
                }
                solver_options.update(quad_settings)
            solver = casadi.qpsol('solver', quad_solver, optimization_prob, solver_options)
        else:
            nlp_solver = solver
            nlp_settings = {
                'ipopt.max_iter': 2000,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6,
                # "linear_solver": "ma27",  # Comment this line if you don't have MA27
            }
            solver_options.update(nlp_settings)
            solver = casadi.nlpsol('solver', nlp_solver, optimization_prob, solver_options)
        return solver, opt_variables, opt_params, g

    def update_initial_condition(self, x0, y0, vel0, psi0):
        self.z_k_value = [x0, y0, vel0, psi0]

    def update_reference(self, x_ref, y_ref, v_ref, psi_ref, acc_ref=0.0, delta_ref=0.0):
        """
        Todo: pass z_ref instead
        :param x_ref:
        :param y_ref:
        :param v_ref:
        :param psi_ref:
        :return:
        """
        self.z_ref_value[0, :] = x_ref
        self.z_ref_value[1, :] = y_ref
        self.z_ref_value[2, :] = v_ref
        self.z_ref_value[3, :] = psi_ref

        self.u_ref_value[0, :] = acc_ref
        self.u_ref_value[1, :] = delta_ref

    def update_previous_input(self, acc_prev, delta_prev):
        self.u_prev_value[0, 0] = acc_prev
        self.u_prev_value[1, 0] = delta_prev

    def update(self, state=None, ref_traj=None, previous_input=None, warmstart_variables=None):
        self.update_initial_condition(*state)
        self.update_reference(*ref_traj)
        self.update_previous_input(*previous_input)

        if self.warmstart and len(warmstart_variables) > 0:
            # Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
            self.z_dv_value = warmstart_variables['z_ws']
            self.u_dv_value = warmstart_variables['u_ws']
            # self.mpc.set_initial(self.sl_dv, warmstart_variables['sl_ws'])

    def solve(self):
        """
        Call update method before this.
        :return:
        """
        st = time.process_time()

        try:
            """
            init_control, x0, opt_variables: [u_dv, z_dv], [(nu, horizon), (nx, horizon+1)]
            control_parameter, c_p, opt_parameters: [u_ref, x_ref]
            
            Can just substitue values using casadi.substitute, 
                e.g self.u_dv_value[:, 0] = casadi.substitute(self.u_dv[:, 0], self.u_dv[:, 0], [acc_prev, delta_prev])
            """
            opt_variables = casadi.vertcat(
                    casadi.reshape(self.z_dv_value, self.nx * (self.horizon + 1), 1),
                    casadi.reshape(self.u_dv_value, self.nu * self.horizon, 1)
            )
            opt_parameters = casadi.vertcat(
                    casadi.reshape(self.z_ref_value, self.nx * (self.horizon + 1), 1),
                    casadi.reshape(self.z_k_value, self.nx, 1),
                    casadi.reshape(self.u_prev_value, self.nu, 1)
            )

            result_dict = self.solver(x0=opt_variables, p=opt_parameters,
                                      lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
            sol = result_dict['x']

            # Optimal solution.
            u_mpc = casadi.reshape(sol[self.nx * (self.horizon + 1):], self.nu,
                                   self.horizon).full()  # sol.value(self.u_dv)
            z_mpc = casadi.reshape(sol[:self.nx * (self.horizon + 1)], self.nx,
                                   self.horizon + 1).full()  # sol.value(self.z_dv)
            # sl_mpc = # sol.value(self.sl_dv)
            z_ref = casadi.reshape(opt_parameters[:self.nx * (self.horizon + 1)], self.nx, self.horizon + 1).full()
            iteration_count = self.solver.stats()['iter_count']  # sol.stats()["iter_count"]
            is_opt = True  # self.solver.stats()['success']
            success = self.solver.stats()['success']
        except Exception as e:
            # Suboptimal solution (e.g. timed out). Todo: decide what to do
            u_mpc = np.zeros((self.nu, self.horizon))  # self.mpc.debug.value(self.u_dv)
            z_mpc = np.zeros((self.nx, self.horizon + 1))  # self.mpc.debug.value(self.z_dv)
            sl_mpc = None  # self.mpc.debug.value(self.sl_dv)
            z_ref = casadi.reshape(opt_parameters[:self.nx * (self.horizon + 1)], self.nx,
                                   self.horizon + 1).full()  # self.mpc.debug.value(self.z_ref)
            iteration_count = None
            is_opt = False  # self.solver.stats()['success']
            success = False

        solve_time = time.process_time() - st  # self.solver.stats()['t_wall_nlp_f']  # t_wall_total

        sol_dict = {'u_control': u_mpc[:, 0],
                    'u_mpc': u_mpc,
                    'z_mpc': z_mpc,
                    # 'sl_mpc': sl_mpc,
                    'z_ref': z_ref.T,
                    'optimal': is_opt,
                    'solve_time': solve_time,  # todo: get from solver time
                    'iter_count': iteration_count,
                    'solver_status': success
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
        self.z_k_value = self.symbol_type.sym('z_k', self.nx)
        self.solve()

    def run(self, z0, u0, zref, zk, u_prev):
        # update
        # solve
        pass


if __name__ == '__main__':
    dt = 0.0001
    x0 = np.array([[1., 0.3, 3, 0.1]]).T
    u0 = np.array([[1., 0.3]]).T
    kin_mpc = KinematicMPCCasadi(vehicle=None, horizon=25, sample_time=dt, wheelbase=0.256,
                                 nx=4, nu=2, x0=x0, u0=u0,
                                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                                 symbol_type='SX', suppress_ipopt_output=True)
    solution_dictionary = kin_mpc.solve()
    print(solution_dictionary)
