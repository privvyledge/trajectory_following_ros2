"""
See:
    To parallelize casadi:
        * https://web.casadi.org/docs/#for-loop-equivalents
        * https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py#L157
"""
import time
import numpy as np
import casadi

from trajectory_following_ros2.casadi.kinematic_bicycle_model import KinematicBicycleModel
from trajectory_following_ros2.casadi._casadi_base import KinematicMPCBase


class KinematicMPCCasadi(KinematicMPCBase):
    """Continuous-time kinematic MPC using a function-based NLP formulation."""

    def __init__(self, vehicle=None, horizon=15, sample_time=0.02, wheelbase=0.256,
                 nx=4, nu=2, x0=None, u0=None,
                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                 vel_bound=(-5.0, 5.0), delta_bound=(-np.radians(23.0), np.radians(23.0)), acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-np.radians(352.9411764706), np.radians(352.9411764706)),
                 symbol_type='MX', warmstart=True,
                 solver_options=None, solver_type='nlp', solver='ipopt', suppress_ipopt_output=True,
                 normalize_yaw_error=True,
                 slack_weights_u_rate=(1e-6, 1e-6),
                 slack_scale_u_rate=(1.0, 1.0),
                 slack_upper_bound_u_rate=None,
                 slack_objective_is_quadratic=False,
                 code_gen_mode='jit'):
        super().__init__(
            vehicle=vehicle, horizon=horizon, sample_time=sample_time, wheelbase=wheelbase,
            nx=nx, nu=nu, x0=x0, u0=u0,
            Q=Q, R=R, Qf=Qf, Rd=Rd,
            vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
            jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound,
            symbol_type=symbol_type, warmstart=warmstart,
            solver_options=solver_options, solver_type=solver_type, solver=solver,
            suppress_ipopt_output=suppress_ipopt_output,
            normalize_yaw_error=normalize_yaw_error,
            slack_weights_u_rate=slack_weights_u_rate,
            slack_scale_u_rate=slack_scale_u_rate,
            slack_upper_bound_u_rate=slack_upper_bound_u_rate,
            slack_objective_is_quadratic=slack_objective_is_quadratic,
            code_gen_mode=code_gen_mode,
            num_obstacles=0)

    def _build_vehicle_model(self, vehicle, nx, nu, x0,
                             vel_bound, delta_bound, acc_bound,
                             jerk_bound, delta_rate_bound):
        discrete_model = False
        if vehicle is not None:
            self.vehicle = vehicle
            self.model = self.vehicle.model
        else:
            self.vehicle = KinematicBicycleModel(
                nx=nx, nu=nu, x0=x0,
                vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
                jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound,
                vehicle_parameters={"wheelbase": self.wheelbase}, sample_time=self.Ts,
                model_type='kinematic', model_name='vehicle_kinematic_model',
                symbol_type=self.symbol_type,
                discretization_method='cvodes', discrete=discrete_model)
            self.model = self.vehicle.model

        self.ode = self.model.f_expl_expr
        self.ode = casadi.substitute(self.ode, self.model.params.wheelbase, self.wheelbase)
        self.ode_linear = self.vehicle.linear_model
        if not discrete_model:
            self.ode_linear = casadi.substitute(
                self.ode_linear,
                casadi.vertcat(self.model.xss, self.model.u0),
                casadi.vertcat(self.model.x, self.model.u))
        else:
            self.ode = casadi.substitute(self.ode, self.model.params.dt, self.Ts)
            self.ode_linear = casadi.substitute(self.ode, self.model.params.dt, self.Ts)
        self.ode_function = self.vehicle.create_ode_function(
            self.ode, self.model.x, self.model.u, function_name='nonlinear_ode')
        self.ode_linear_function = self.vehicle.create_ode_function(
            self.ode_linear, self.model.x, self.model.u, function_name='linear_ode')

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
        """
        if delta_bound is None:
            delta_bound = [-np.radians(23.0), np.radians(23.0)]

        if vel_bound is None:
            vel_bound = [-10.0, 10.0]

        if acc_bound is None:
            acc_bound = [-3.0, 3.0]

        if jerk_bound is None:
            jerk_bound = [-1.5, 1.5]

        if delta_rate_bound is None:
            delta_rate_bound = [-np.radians(352.9411764706), np.radians(352.9411764706)]

        u_min = np.array([[acc_bound[0]], [delta_bound[0]]])
        u_max = np.array([[acc_bound[1]], [delta_bound[1]]])

        u_dot_min = np.array([[jerk_bound[0]], [delta_rate_bound[0]]])
        u_dot_max = np.array([[jerk_bound[1]], [delta_rate_bound[1]]])

        constraints = []

        lbg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)
        ubg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)

        # initial state constraints
        constraints.append(self.z_dv[:, 0] - self.z_k)

        for k in range(self.horizon):
            ode_symbolic = self.ode_function(self.z_dv[:, k], self.u_dv[:, k])
            z_dt = ode_symbolic * self.Ts
            z_next = self.z_dv[:, k] + z_dt
            dynamics = self.z_dv[:, k + 1] - z_next
            constraints.append(dynamics)

        constraints = casadi.vertcat(*constraints)

        # Input Rate Bound Constraints
        u_prev = self.u_prev
        u_dot_list = []
        slack_flag = int(not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))))
        for k in range(self.horizon):
            u_dot = self.u_dv[:, k] - u_prev
            """
            The formulation below, i.e LBG - slack <= constraints <= UBG + slack is valid.
            However, casadi requires LBG and UBG to only include numbers.
            Therefore, we reformulate with an equivalent expression:
               i. LBG <= u_dot + slack <= np.inf (i.e, UBG), and
               ii. -UBG <= -u_dot + slack <= np.inf (or -np.inf (i.e, LBG) <= u_dot - slack <= UBG)
            """
            # i.
            constraints = casadi.vertcat(
                constraints,
                u_dot + (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))
            lbg = casadi.vertcat(lbg, u_dot_min * self.Ts)
            ubg = casadi.vertcat(ubg, casadi.inf * casadi.DM.ones((self.nu, 1)) * self.Ts)

            # ii.
            constraints = casadi.vertcat(
                constraints,
                -u_dot + (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))
            lbg = casadi.vertcat(lbg, -u_dot_max * self.Ts)
            ubg = casadi.vertcat(ubg, casadi.inf * casadi.DM.ones((self.nu, 1)) * self.Ts)

            u_prev = self.u_dv[:, k]
            u_dot_list.append(u_dot)

        u_dot_list = casadi.horzcat(*u_dot_list)
        self.u_rate_dv = u_dot_list

        lbx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))
        ubx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))

        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            lbx = casadi.vertcat(lbx, casadi.DM.zeros((self.nu * self.horizon, 1)))
            ubx = casadi.vertcat(ubx, casadi.DM.zeros((self.nu * self.horizon, 1)))

        # state constraints (self.v_dv, vel_bound[1]).
        lbx[0: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # X lower bound
        lbx[1: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # Y lower bound
        lbx[2: self.nx * (self.horizon + 1): self.nx] = vel_bound[0]  # vel lower bound
        lbx[3: self.nx * (self.horizon + 1): self.nx] = -2 * casadi.pi  # psi lower bound

        ubx[0: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # X upper bound
        ubx[1: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # Y upper bound
        ubx[2: self.nx * (self.horizon + 1): self.nx] = vel_bound[1]  # vel upper bound
        ubx[3: self.nx * (self.horizon + 1): self.nx] = 2 * casadi.pi  # psi upper bound

        # controls
        lbx[self.nx * (self.horizon + 1):(self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = acc_bound[0]  # acc lower bound
        lbx[self.nx * (self.horizon + 1) + 1:(self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = delta_bound[0]  # delta lower bound

        ubx[self.nx * (self.horizon + 1):(self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = acc_bound[1]  # acc lower bound
        ubx[self.nx * (self.horizon + 1) + 1:(self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = delta_bound[1]  # delta lower bound

        # slack constraints
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            lbx[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = 0.0
            lbx[self.nx * (self.horizon + 1) + (self.nu * self.horizon) + 1:(self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = 0.0

            ubx[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = self.slack_upper_bound_u_rate[0]
            ubx[self.nx * (self.horizon + 1) + (self.nu * self.horizon) + 1:(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = self.slack_upper_bound_u_rate[1]
        return constraints, lbg, ubg, lbx, ubx

    def setup_solver(self, cost, constraints, solver_type='nlp',
                     solver_options=None, solver='ipopt', suppress_output=True):
        flat_z = casadi.reshape(self.z_dv, self.nx * (self.horizon + 1), 1)
        flat_u = casadi.reshape(self.u_dv, self.nu * self.horizon, 1)
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            flat_sl = casadi.reshape(self.sl_dv, self.nu * self.horizon, 1)
        flat_z_ref = casadi.reshape(self.z_ref, self.nx * (self.horizon + 1), 1)
        flat_z_k = casadi.reshape(self.z_k, self.nx, 1)
        flat_u_ref = casadi.reshape(self.u_ref, -1, 1)
        flat_u_prev = casadi.reshape(self.u_prev, self.nu, 1)

        opt_variables = casadi.vertcat(flat_z, flat_u)
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            opt_variables = casadi.vertcat(opt_variables, flat_sl)

        opt_params = casadi.vertcat(flat_z_ref, flat_z_k, flat_u_prev)

        g = casadi.vertcat(constraints)

        optimization_prob = {'f': cost, 'x': opt_variables, 'p': opt_params, 'g': g}

        if solver_options is None:
            solver_options = {
                'record_time': True,
                'print_time': not suppress_output,
                'verbose': not suppress_output,
                'expand': True,
            }

            if solver == 'ipopt':
                ipopt_options = {
                    'ipopt.print_level': not suppress_output,
                    'ipopt.sb': 'yes',
                    'ipopt.max_iter': 2000,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6,
                    'error_on_fail': 0,  # to raise an exception if the solver fails to find a solution
                    # "ipopt.linear_solver": "ma27",  # Comment this line if you don't have MA27
                }
                solver_options.update(ipopt_options)

                if self.warmstart:
                    warmstart_options = {
                        'ipopt.warm_start_init_point': 'yes',
                        'ipopt.warm_start_bound_push': 1e-8,
                        'ipopt.warm_start_mult_bound_push': 1e-8,
                        'ipopt.mu_init': 1e-5,
                        'ipopt.bound_relax_factor': 1e-9,
                    }
                    solver_options.update(warmstart_options)

        compiler = "gcc"
        flags = ["-O3"]
        if compiler == "gcc":
            flags = [
                "-Ofast",  # enables even more optimization than -O3 but could be less precise/safe but is okay in most situations
                "-march=native",  # optimizes for the specific hardware (CPU) but isn't transferable
                "-fPIC",  # for shared_libraries
            ]  # for performance boost using gcc

        if self.code_gen_mode == 'jit':
            # JIT compilation automatically compiles helper functions as well as the solver.
            # Yields about a 2.5x increase in performance
            # By default, the compiler will be gcc or cl.exe
            jit_options = {
                "jit": True, "compiler": "shell", "jit_temp_suffix": False,
                "jit_options": {
                    "flags": flags, "compiler": compiler, "verbose": True,
                    "compiler_flags": flags
                }, # use ("compiler": "ccache gcc") if ccache is installed for a performance boost
                'jit_cleanup': False  # True: delete fails on shutdown
                       }
            solver_options.update(jit_options)

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
                    # 'verbose': not suppress_output,
                    # 'max_iter': 2000,
                    # 'eps_pr': 1e-8,
                    # 'eps_r': 1e-8,
                    'warm_start_dual': True,
                    'warm_start_primal': True,
                    # 'sparse': True,  # test with osqp and qpoases
                }
                solver_options.update(quad_settings)
            solver = casadi.qpsol('solver', quad_solver, optimization_prob, solver_options)
        else:
            nlp_solver = solver
            # nlp_settings = {
            #
            # }
            # solver_options.update(nlp_settings)
            solver = casadi.nlpsol('solver', nlp_solver, optimization_prob, solver_options)
            if self.code_gen_mode == 'external':
                # Generate C code for the NLP functions. todo: rename the files
                solver.generate_dependencies("nlp.c")
                import subprocess
                # On Windows, use other flags
                cmd_args = [compiler, "-fPIC", "-shared"] + flags + ["nlp.c", "-o", "nlp.so"]
                subprocess.run(cmd_args)

                # Create a new NLP solver instance from the compiled code
                solver = casadi.nlpsol("solver", nlp_solver, "./nlp.so")
        return solver, opt_variables, opt_params, g

    def solve(self):
        """Call update method before this."""
        st = time.process_time()

        sl_mpc = np.zeros((self.horizon, self.nu))
        try:
            opt_variables = casadi.vertcat(
                casadi.reshape(self.z_dv_value, self.nx * (self.horizon + 1), 1),
                casadi.reshape(self.u_dv_value, self.nu * self.horizon, 1)
            )
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                opt_variables = casadi.vertcat(
                    opt_variables,
                    casadi.reshape(self.sl_dv_value, self.nu * self.horizon, 1))

            opt_parameters = casadi.vertcat(
                casadi.reshape(self.z_ref_value, self.nx * (self.horizon + 1), 1),
                casadi.reshape(self.z_k_value, self.nx, 1),
                casadi.reshape(self.u_prev_value, self.nu, 1),
            )

            dual_variables = {}
            if self.warmstart:
                if self.lam_x_value is not None:
                    dual_variables['lam_x0'] = self.lam_x_value
                if self.lam_g_value is not None:
                    dual_variables['lam_g0'] = self.lam_g_value

            result_dict = self.solver(x0=opt_variables, p=opt_parameters,
                                      lbg=self.lbg, lbx=self.lbx,
                                      ubg=self.ubg, ubx=self.ubx, **dual_variables)

            sol = result_dict['x']
            cost = result_dict['f']
            g = result_dict['g']
            lam_g = result_dict['lam_g']  # lagrange multipliers (constraints)
            lam_x = result_dict['lam_x']  # lagrange multipliers (optimization variables)
            lam_p = result_dict['lam_p']  # lagrange multipliers (parameters)

            # todo: add hessian calculation here for convexity check
            #  (https://www.mathworks.com/help/optim/ug/hessian.html#bsapedt | https://github.com/casadi/casadi/issues/3297#issuecomment-1662226255 | https://github.com/meco-group/rockit/blob/master/rockit/ocp.py | https://groups.google.com/g/casadi-users/c/h05oCge2vkk)

            z_mpc = casadi.reshape(sol[:self.nx * (self.horizon + 1)],
                                   self.nx, self.horizon + 1).full()
            u_mpc = casadi.reshape(
                sol[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nu * self.horizon],
                self.nu, self.horizon).full()
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                sl_mpc = casadi.reshape(
                    sol[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon))],
                    self.nu, self.horizon).full()

            self.update_u_rate(self.u_prev_value, u_mpc)
            u_rate = self.u_rate_dv_value

            z_ref = casadi.reshape(
                opt_parameters[:self.nx * (self.horizon + 1)],
                self.nx, self.horizon + 1).full()
            z_k_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nx],
                self.nx, 1).full()
            u_prev_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1) + self.nx:],
                self.nu, 1).full()
            iteration_count = self.solver.stats()['iter_count']
            is_opt = True
            success = self.solver.stats()['success']

            # Useful for debugging; keep commented unless actively debugging.
            initial_condition_constraint = casadi.reshape(g[:self.nx], self.nx, 1)
            model_dynamics = casadi.reshape(g[self.nx:self.nx + (self.nx * self.horizon)], self.nx, self.horizon)
            u_rate_i = casadi.reshape(
                g[self.nx + (self.nx * self.horizon):self.nx + (self.nx * self.horizon) + self.nu * self.horizon],
                self.nu, -1).full()
            u_rate_ii = casadi.reshape(
                g[self.nx + (self.nx * self.horizon) + self.nu * self.horizon:self.nx + (self.nx *
                    self.horizon) + (self.nu * self.horizon) + (self.nu * self.horizon)],
                self.nu, -1).full()

            # # get functions used by the solver.
            # # Could be useful when replacing default functions.
            # # See (https://github.com/casadi/casadi/wiki/FAQ:-How-to-specify-a-custom-Hessian-approximation%3F | https://groups.google.com/g/casadi-users/c/XnDBUWrPTlQ)
            # hess_l = self.solver.get_function('nlp_hess_l')

            # # To verify for debugging purposes (should be commented out)
            # opt_to_state_input_dict = {
            #     'x_0': 'x', 'x_1': 'y', 'x_2': 'vel', 'x_3': 'psi',
            #     'u_0': 'acc', 'u_1': 'delta'
            # }
            # for index in range(g.size1()):
            #     print(f"{self.lbg[index]} <= {g[index]} <= {self.ubg[index]}")
            # print("\n")
            #
            # # print state (z) constraints.
            # for index in range(0, self.nx * (self.horizon + 1), self.nx):
            #     expression = ""
            #     for state in range(self.nx):
            #         key = f'x_{state}'
            #         expression += f"{self.lbx[index + state]} <= {sol[index + state]} <= {self.ubx[index + state]}\n"
            #     print(expression)
            # print("\n")
            #
            # # print input (u) constraints.
            # for index in range(self.nx * (self.horizon + 1), self.nx * (self.horizon + 1) + self.nu * self.horizon,
            #                    self.nu):
            #     expression = ""
            #     for input_ in range(self.nu):
            #         expression += f"{self.lbx[index + input_]} <= {sol[index + input_]} <= {self.ubx[index + input_]}\n"
            #     print(expression)
            # print("\n")
            #
            # # print input (slack) constraints.
            # if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            #     for index in range(self.nx * (self.horizon + 1) + (self.nu * self.horizon), (
            #             self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)), self.nu):
            #         expression = ""
            #         for input_ in range(self.nu):
            #             # key = f'u_{input_}'
            #             expression += f"{self.lbx[index + input_]} <= {sol[index + input_]} <= {self.ubx[index + input_]}\n"
            #         print(expression)
            #     print("\n")
            #
            # # # print input rate constraints. todo
            # # for index in range(self.nx * (self.horizon + 1), self.nx * (self.horizon + 1) + self.nu * self.horizon,
            # #                    self.nu):
            # #     expression = ""
            # #     for input_ in range(self.nu):
            # #         expression += f"{self.lbx[index + input_]} <= {sol[index + input_]} <= {self.ubx[index + input_]}\n"
            # #     print(expression)
            # # print("\n")
        except Exception as e:
            u_mpc = np.zeros((self.nu, self.horizon))
            z_mpc = np.zeros((self.nx, self.horizon + 1))
            u_rate = np.zeros((self.nu, self.horizon))
            z_ref = casadi.reshape(
                opt_parameters[:self.nx * (self.horizon + 1)],
                self.nx, self.horizon + 1).full()
            z_k_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nx],
                self.nx, 1).full()
            u_prev_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1) + self.nx:],
                self.nu, 1).full()
            iteration_count = None
            is_opt = False
            success = False
            lam_g = np.zeros(self.lbg.shape)
            lam_x = np.zeros_like(opt_parameters)
            lam_p = np.zeros_like(opt_parameters)

        solve_time = time.process_time() - st

        sol_dict = {'u_control': u_mpc[:, 0],
                    'u_mpc': u_mpc,
                    'z_mpc': z_mpc,
                    'u_rate': u_rate,
                    'u_prev': u_prev_value,
                    'sl_mpc': sl_mpc,
                    'z_ref': z_ref.T,
                    'optimal': is_opt,
                    'solve_time': solve_time,
                    'iter_count': iteration_count,
                    'solver_status': success,
                    'lam_x': lam_x,
                    'lam_g': lam_g,
                    'lam_p': lam_p,
                    'solver_stats': self.solver.stats()
                    }

        self.solution_dict = sol_dict
        return sol_dict


if __name__ == '__main__':
    dt = 0.0001
    x0 = np.array([[1., 0.3, 3, 0.1]]).T
    u0 = np.array([[1., 0.3]]).T
    kin_mpc = KinematicMPCCasadi(vehicle=None, horizon=25, sample_time=dt, wheelbase=0.256,
                                 nx=4, nu=2, x0=x0, u0=u0,
                                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                                 vel_bound=(-5.0, 5.0),
                                 delta_bound=(-np.radians(23.0), np.radians(23.0)),
                                 acc_bound=(-3.0, 3.0),
                                 jerk_bound=(-1.5, 1.5),
                                 delta_rate_bound=(-np.radians(352.9411764706), np.radians(352.9411764706)),
                                 symbol_type='SX', suppress_ipopt_output=True)
    solution_dictionary = kin_mpc.solve()
    print(solution_dictionary)