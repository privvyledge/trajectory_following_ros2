"""
Todo:
    * cleanup and make a simple extensible class/interface
    * move helper functions to a different file
    * add frenet model (https://github.com/bohlinz/genesis_path_follower/blob/master/scripts/controllers/kinematic_frenet_mpc.py | https://github.com/acados/acados/blob/master/examples/acados_python/race_cars/bicycle_model.py#L97)
"""
import types

import numpy as np
import casadi


class KinematicBicycleModel(object):
    """docstring for ClassName"""

    def __init__(self, nx=4, nu=2, x0=None,
                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                 vehicle_parameters=None, sample_time=0.001, symbol_type='MX',
                 model_type='kinematic', model_name='vehicle_kinematic_model', discretization_method='cvodes',
                 discrete=False):
        """Constructor for KinematicBicycleModel"""
        # model info
        self.model_name = model_name
        self.model_type = model_type
        self.symbol_type = symbol_type
        self.discrete = discrete

        # vehicle parameters
        if vehicle_parameters is None:
            vehicle_parameters = {'wheelbase': 0.256}

        self.wheelbase = vehicle_parameters['wheelbase']

        self.nx = nx
        self.nu = nu

        self.z0 = x0

        self.Ts = sample_time
        self.discretization_method = discretization_method

        # constraints
        self.vel_bound = vel_bound
        self.delta_bound = delta_bound
        self.acc_bound = acc_bound
        self.jerk_bound = jerk_bound
        self.delta_rate_bound = acc_bound

        if self.discrete:
            self.model, self.constraints = self.setup_discrete_ltv_model(symbol_type=self.symbol_type)
            self.linear_model = self.model
        else:
            self.model, self.constraints = self.setup_model(symbol_type=self.symbol_type)
            [self.linear_model, self.ode_linear_function,
             self.system_matrices, self.system_matrices_function_handles] = self.setup_linearizer(self.model.f_expl_expr,
                                                                                                  y_meas=None,
                                                                                                  x=self.model.x,
                                                                                                  u=self.model.u,
                                                                                                  xss=self.model.xss,
                                                                                                  uss=self.model.u0,
                                                                                                  wheelbase=self.model.params.wheelbase,
                                                                                                  method='jac')
            self.linear_model, x_next = self.linearize(self.linear_model, wheelbase=self.model.params.wheelbase)

    def setup_model(self, symbol_type='MX',
                    z0=None, u0=(0., 0.),
                    vel_bound=None, delta_bound=None, acc_bound=None, jerk_bound=None, delta_rate_bound=None):
        # todo: remove constraints from model since that isn't from the kinematics
        # todo: create new models class
        # initialize structs
        constraint = types.SimpleNamespace()  # todo: switch to a class instead
        model = types.SimpleNamespace()  # todo: switch to a class instead

        if symbol_type.lower() == 'MX'.lower():
            symbol_type = casadi.MX

        elif symbol_type.lower() == 'SX'.lower():
            symbol_type = casadi.SX

        else:
            raise Exception('Invalid Symbol Type')

        # state
        x = symbol_type.sym("x")
        y = symbol_type.sym("y")
        vel = symbol_type.sym("vel")
        psi = symbol_type.sym("psi")
        z = casadi.vertcat(x, y, vel, psi)
        num_states = z.shape[0]

        # initial_state/steady state/linearization point
        xss = symbol_type.sym("xss")
        yss = symbol_type.sym("yss")
        velss = symbol_type.sym("velss")
        psiss = symbol_type.sym("psiss")
        zss = casadi.vertcat(xss, yss, velss, psiss)

        # input
        acc = symbol_type.sym("acc")
        delta = symbol_type.sym("delta")
        u = casadi.vertcat(acc, delta)
        num_inputs = u.shape[0]

        # initial input
        acc_prev = symbol_type.sym("acc_prev")
        delta_prev = symbol_type.sym("delta_prev")
        u_prev = casadi.vertcat(acc_prev, delta_prev)
        if u0 is None:
            u0 = u_prev

        # input rate symbols.
        jerk = symbol_type.sym("jerk")
        jerk = acc - acc_prev

        delta_rate = symbol_type.sym("delta_rate")
        delta_rate = delta - delta_prev

        u_dot1 = casadi.vertcat(jerk, delta_rate)
        u_dot2 = u - u0  #
        u_dot = u - u_prev

        # state derivatives
        x_dot = symbol_type.sym("xdot")  # vel * casadi.cos(psi)
        y_dot = symbol_type.sym("ydot")  # vel * casadi.sin(psi)
        vel_dot = symbol_type.sym("vel_dot")  # acc
        psi_dot = symbol_type.sym("psi_dot")  # (vel / wheelbase) * casadi.tan(delta)
        z_dot = casadi.vertcat(x_dot, y_dot, vel_dot, psi_dot)  # ode-nonlinear

        # algebraic variables
        z_algebraic = casadi.vertcat([])

        # parameters
        '''Race car parameters'''
        wheelbase = symbol_type.sym('wheelbase')  # m
        dt = symbol_type.sym('dt')
        z_ref = symbol_type.sym('z_ref', z.shape)
        z_k = symbol_type.sym('z_k', z.shape)
        u_ref = symbol_type.sym('u_ref', u.shape)
        u_prev = symbol_type.sym('u_prev', u.shape)
        parameters = casadi.vertcat(
            wheelbase,
            z_ref,
            u_ref,
            z_k,
            u_prev
    )

        # dynamics (continuous non-linear model)
        f_expl = casadi.vertcat(
                vel * casadi.cos(psi),
                vel * casadi.sin(psi),
                acc,
                (vel / wheelbase) * casadi.tan(delta)
        )  # continuous non-linear model

        '''Constraints/bounds'''
        if delta_bound is None:
            delta_bound = self.delta_bound

        if vel_bound is None:
            vel_bound = self.vel_bound

        if acc_bound is None:
            acc_bound = self.acc_bound

        if jerk_bound is None:
            jerk_bound = self.jerk_bound

        if delta_rate_bound is None:
            delta_rate_bound = self.delta_rate_bound

        # state bounds
        # model.x_min = -np.inf
        # model.x_max = np.inf
        #
        # model.y_min = -np.inf
        # model.y_max = np.inf

        model.v_min = vel_bound[0]
        model.v_max = vel_bound[1]

        # model.psi_min = -2 * np.pi
        # model.psi_max = -2 * np.pi

        # input bounds
        model.delta_min = np.radians(delta_bound[0])  # minimum steering angle [rad]
        model.delta_max = np.radians(delta_bound[1])  # maximum steering angle [rad]

        model.acc_min = acc_bound[0]
        model.acc_max = acc_bound[1]

        # input rate bounds. Todo: setup as non-linear constraints
        model.delta_rate_min = np.radians(delta_rate_bound[0])
        model.delta_rate_max = np.radians(delta_rate_bound[1])

        model.jerk_min = jerk_bound[0]
        model.jerk_max = jerk_bound[1]

        # (optional) Define initial conditions
        if z0 is None:
            if self.z0 is not None:
                model.x0 = self.z0
            else:
                model.x0 = np.zeros(z.shape[0])
        else:
            model.x0 = z0

        constraint.expr = casadi.SX.sym('', 0)
        # Define constraints struct. todo: test with rate contraints
        # constraint.expr = casadi.vertcat(vel, delta, acc)  # casadi.vertcat(vel, delta, acc, (delta_rate, jerk))
        # constraint.u_dot_min = np.array([[jerk_bound[0]], [delta_rate_bound[0]]])
        # constraint.u_dot_max = np.array([[jerk_bound[0]], [delta_rate_bound[1]]])

        # todo: add slack to rate constraints or use Acados slacks instead
        # constraint.delta_rate_min = delta_rate_bound[0]
        # constraint.jerk_min = jerk_bound[0] = None

        # constraint.delta_rate_max = delta_rate_bound[1]
        # constraint.jerk_max = jerk_bound[1] = None

        # constraint.delta_rate = Function("delta_rate", [delta_rate], [delta_rate])
        # constraint.jerk = Function("jerk", [jerk], [jerk])
        # constraint.u_dot = Function("u_dot", [u, u0], [u_dot])

        # Define model struct
        params = types.SimpleNamespace()
        params.wheelbase = wheelbase
        # params.mass = 0.2

        model.f_impl_expr = z_dot - f_expl
        model.f_expl_expr = f_expl
        # model.f_disk = self.setup_discretizer(z, u, parameters)
        model.x = z
        model.xss = zss
        model.xdot = z_dot
        model.u = u
        model.u0 = u_prev
        model.z = z_algebraic
        model.p = parameters
        model.name = self.model_name
        model.params = params

        return model, constraint

    def setup_discrete_ltv_model(self, symbol_type='MX',
                    z0=None, u0=(0., 0.),
                    vel_bound=None, delta_bound=None, acc_bound=None, jerk_bound=None, delta_rate_bound=None):
        # todo: remove constraints from model since that isn't from the kinematics
        # todo: create new models class
        # initialize structs
        constraint = types.SimpleNamespace()  # todo: switch to a class instead
        model = types.SimpleNamespace()  # todo: switch to a class instead

        if symbol_type.lower() == 'MX'.lower():
            symbol_type = casadi.MX

        elif symbol_type.lower() == 'SX'.lower():
            symbol_type = casadi.SX

        else:
            raise Exception('Invalid Symbol Type')

        # state
        x = symbol_type.sym("x")
        y = symbol_type.sym("y")
        vel = symbol_type.sym("vel")
        psi = symbol_type.sym("psi")
        z = casadi.vertcat(x, y, vel, psi)
        num_states = z.shape[0]

        # input
        acc = symbol_type.sym("acc")
        delta = symbol_type.sym("delta")
        u = casadi.vertcat(acc, delta)
        num_inputs = u.shape[0]

        # initial input
        acc_prev = symbol_type.sym("acc_prev")
        delta_prev = symbol_type.sym("delta_prev")
        u_prev = casadi.vertcat(acc_prev, delta_prev)
        if u0 is None:
            u0 = u_prev

        # input rate symbols.
        jerk = symbol_type.sym("jerk")
        jerk = acc - acc_prev

        delta_rate = symbol_type.sym("delta_rate")
        delta_rate = delta - delta_prev

        u_dot1 = casadi.vertcat(jerk, delta_rate)
        u_dot2 = u - u0  #
        u_dot = u - u_prev

        # state derivatives
        x_dot = symbol_type.sym("xdot")  # vel * casadi.cos(psi)
        y_dot = symbol_type.sym("ydot")  # vel * casadi.sin(psi)
        vel_dot = symbol_type.sym("vel_dot")  # acc
        psi_dot = symbol_type.sym("psi_dot")  # (vel / wheelbase) * casadi.tan(delta)
        z_dot = casadi.vertcat(x_dot, y_dot, vel_dot, psi_dot)  # ode-nonlinear

        # algebraic variables
        z_algebraic = casadi.vertcat([])

        # parameters
        '''Race car parameters'''
        wheelbase = symbol_type.sym('wheelbase')  # m
        dt = symbol_type.sym('dt')
        z_ref = symbol_type.sym('z_ref', z.shape)
        z_k = symbol_type.sym('z_k', z.shape)
        u_ref = symbol_type.sym('u_ref', u.shape)
        u_prev = symbol_type.sym('u_prev', u.shape)
        parameters = casadi.vertcat(
            wheelbase,
            z_ref,
            u_ref,
            z_k,
            u_prev,
            dt
    )

        A = casadi.blockcat([
            [1, 0, casadi.cos(psi) * dt, -vel * casadi.sin(psi) * dt],
            [0, 1, casadi.sin(psi) * dt, vel * casadi.cos(psi) * dt],
            [0, 0, 1, 0],
            [0, 0, dt * casadi.tan(delta) / self.wheelbase, 1],
        ])
        B = casadi.blockcat([
            [0, 0],
            [0, 0],
            [dt, 0],  #
            [0, dt * vel / (self.wheelbase * casadi.cos(delta) ** 2)],
        ])
        G = casadi.vertcat(vel * casadi.sin(psi) * psi * dt,
                           -vel * casadi.cos(psi) * psi * dt,
                           0,
                           -(vel * delta) * dt / (self.wheelbase * casadi.cos(delta) ** 2))

        # dynamics (discrete linear model)
        f_expl = A @ z + B @ u + G  # discrete linear model

        '''Constraints/bounds'''
        if delta_bound is None:
            delta_bound = self.delta_bound

        if vel_bound is None:
            vel_bound = self.vel_bound

        if acc_bound is None:
            acc_bound = self.acc_bound

        if jerk_bound is None:
            jerk_bound = self.jerk_bound

        if delta_rate_bound is None:
            delta_rate_bound = self.delta_rate_bound

        # state bounds
        # model.x_min = -np.inf
        # model.x_max = np.inf
        #
        # model.y_min = -np.inf
        # model.y_max = np.inf

        model.v_min = vel_bound[0]
        model.v_max = vel_bound[1]

        # model.psi_min = -2 * np.pi
        # model.psi_max = -2 * np.pi

        # input bounds
        model.delta_min = np.radians(delta_bound[0])  # minimum steering angle [rad]
        model.delta_max = np.radians(delta_bound[1])  # maximum steering angle [rad]

        model.acc_min = acc_bound[0]
        model.acc_max = acc_bound[1]

        # input rate bounds. Todo: setup as non-linear constraints
        model.delta_rate_min = np.radians(delta_rate_bound[0])
        model.delta_rate_max = np.radians(delta_rate_bound[1])

        model.jerk_min = jerk_bound[0]
        model.jerk_max = jerk_bound[1]

        # (optional) Define initial conditions
        if z0 is None:
            if self.z0 is not None:
                model.x0 = self.z0
            else:
                model.x0 = np.zeros(z.shape[0])
        else:
            model.x0 = z0


        # Define model struct
        params = types.SimpleNamespace()
        params.wheelbase = wheelbase
        # params.mass = 0.2
        params.dt = dt

        model.f_impl_expr = z_dot - f_expl
        model.f_expl_expr = f_expl
        model.x = z
        model.xdot = z_dot
        model.u = u
        model.u0 = u_prev
        model.z = z_algebraic
        model.p = parameters
        model.name = self.model_name
        model.params = params

        return model, constraint

    def substitute_symbols(self, ode, symbols, values):
        ode = casadi.substitute(ode, casadi.vertcat(*symbols), casadi.vertcat(*values))
        return ode

    def create_ode_function(self, ode, x, u, function_name='ode'):
        ode_function = casadi.Function(function_name, [x, u], [ode],
                                       ['x', 'u'], ['ode'])
        return ode_function

    def setup_linearizer(self, ode, y_meas=None, x=None, u=None, xss=None, uss=None, wheelbase=None, method='jac'):
        """
        Creates/initializes the linearizer
        :param ode:
        :param y_meas:
        :param x:
        :param u:
        :param xss: this can be the same as x or a different point
        :param uss:
        :param wheelbase:
        :param method: solver (recommended), jacobian
        :return:
        """
        num_states = ode.shape[0]
        num_inputs = u.shape[0]

        C_linear = np.eye(num_states, num_states)
        D_linear = np.zeros((C_linear.shape[0], num_inputs))

        if method == 'jac':
            # to obtain state space matrices
            A_linear = casadi.jacobian(ode, x)
            B_linear = casadi.jacobian(ode, u)

            if y_meas is not None:
                C_linear = casadi.jacobian(y_meas, x)
                D_linear = casadi.jacobian(y_meas, u)

            # substitute x, u symbols with xss, uss respectively
            A_linear = casadi.substitute(A_linear, casadi.vertcat(x, u), casadi.vertcat(xss, uss))
            B_linear = casadi.substitute(B_linear, casadi.vertcat(x, u), casadi.vertcat(xss, uss))

            # Function handles
            A_linear_function = casadi.Function('A_linear_function', [xss, uss, wheelbase], [A_linear],
                                                ['xss', 'uss', 'wheelbase'], ['A_linear'])
            B_linear_function = casadi.Function('B_linear_function', [xss, uss, wheelbase], [B_linear],
                                                ['xss', 'uss', 'wheelbase'], ['A_linear'])
            system_matrices_function_handles = [A_linear_function, B_linear_function]

            # Linear ODE
            ode_linear = A_linear @ x + B_linear @ u
            ode_linear_function = casadi.Function('ode_linear_function', [x, u, xss, uss, wheelbase], [ode_linear],
                                                  ['x', 'u', 'xss', 'uss', 'wheelbase'], ['ode_linear'])

            # LTV State-space model
            system_matrices = [A_linear, B_linear, C_linear, D_linear]

            return ode_linear, ode_linear_function, system_matrices, system_matrices_function_handles

        else:
            # default case
            ode_linear = casadi.linearize(ode, casadi.vertcat(x, u), casadi.vertcat(xss, uss))
            ode_linear_function = casadi.Function('ode_linear_function', [x, u, xss, uss, wheelbase], [ode_linear],
                                                  ['x', 'u', 'xss', 'uss', 'wheelbase'], ['ode_linear'])

            A_linear = np.eye(num_states)
            B_linear = np.zeros((num_states, num_inputs))
            system_matrices = [A_linear, B_linear, C_linear, D_linear]
            return ode_linear, ode_linear_function, system_matrices, None

    def get_linear_solution(self, ode_linear_function, x, u, xss, uss, wheelbase=0.256):
        x_next = ode_linear_function(x, u, xss, uss, wheelbase)
        return x_next

    def linearize(self, ode_linear, x=None, u=None, xss=None, uss=None, wheelbase=None,
                  get_linear_solution=False, ode_linear_function=None):
        # todo: remove or find a better way to use

        # (optional) substitue wheelbase symbol with value to prepare. Might remove as this just needs to be done once
        # ode_linear = casadi.substitute(ode_linear, wheelbase, self.wheelbase)
        ode_linear = self.substitute_symbols(ode_linear, [wheelbase], [self.wheelbase])

        x_next = None
        if get_linear_solution:
            x_next = self.get_linear_solution(ode_linear_function, x, u, xss, uss, wheelbase)

        return ode_linear, x_next

    def setup_discretizer(self, state, input, parameters=None, integration_method='cvodes', dt=0.01, ode=None):
        """
        Todo: directly right the integrator to output SX or MX variables
        :param state: state symbol
        :param input: input symbol
        :param parameters:
        :param integration_method: cvodes, collocation, rk (RK4)
        :param ode: should be passed unless passing to Acados model
        :return:
        """
        # todo: check casadi version because p=parameters and u=input in newer vesions as opposed to p=inputs
        dae = {'x': state, 'p': input, 'ode': ode}
        integration_options = {
            't0': 0,
            'tf': dt}
        integrator_function = casadi.integrator('integrator_function', integration_method, dae, integration_options)
        return integrator_function

    def discretize(self, integrator_function, state, input):
        x_next = integrator_function(x0=state, p=input)['xf']
        return x_next.full()

    def get_next_state(self, ode, x, u, linearize=True, discretize=True):
        x_next = 0
        # x_next = self.get_linear_solution(ode, x, u, xss, uss, wheelbase)


if __name__ == '__main__':
    kin_bic_model = KinematicBicycleModel(nx=4, nu=2, x0=None,
                                          vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                                          jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                                          vehicle_parameters=None, sample_time=0.001,
                                          model_type='kinematic', model_name='vehicle_kinematic_model',
                                          discretization_method='cvodes')

    model_to_discretize = kin_bic_model.substitute_symbols(kin_bic_model.model.f_expl_expr,
                                                           [kin_bic_model.model.params.wheelbase],
                                                           [kin_bic_model.wheelbase])
    model_to_discretize2 = kin_bic_model.linear_model

    # linearization point
    model_to_discretize2 = casadi.substitute(
            model_to_discretize2, casadi.vertcat(kin_bic_model.model.xss, kin_bic_model.model.u0, kin_bic_model.model.params.wheelbase),
            casadi.vertcat(kin_bic_model.model.x, kin_bic_model.model.u, kin_bic_model.wheelbase))
    integration_function = kin_bic_model.setup_discretizer(kin_bic_model.model.x, kin_bic_model.model.u,
                                                           ode=model_to_discretize, dt=0.0001)
    integration_function2 = kin_bic_model.setup_discretizer(kin_bic_model.model.x, kin_bic_model.model.u,
                                                            ode=model_to_discretize2, dt=0.0001)
    discrete_solution = kin_bic_model.discretize(integration_function, [0., 0., 0., 0.], [1., 0.2])

    print("Done")
