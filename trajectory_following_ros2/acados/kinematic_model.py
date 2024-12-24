"""
Similar to bicycle_model.py

See:
    * https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py
    * https://github.com/commaai/openpilot/blob/master/selfdrive/controls/tests/test_lateral_mpc.py
    * https://github.com/mlab-upenn/mpcc/blob/master/scripts/acados/acados_settings.py#L126

Todo: get actuation limit values from arguments instead of hardcoding here
"""

import types

import numpy as np
from casadi import *


def kinematic_model(symbol_type='SX'):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "vehicle_kinematic_model"

    '''
    (optional) Todo: Load track parameters
    '''

    '''Casadi Model'''
    if symbol_type.lower() == 'MX'.lower():
        symbol = MX
        
    elif symbol_type.lower() == 'SX'.lower():
        symbol = SX
    
    else:
        raise Exception('Invalid Symbol Type')

    # set up states & controls
    x = symbol.sym("x")
    y = symbol.sym("y")
    vel = symbol.sym("vel")
    psi = symbol.sym("psi")

    # state
    z = vertcat(
            x,
            y,
            vel,
            psi
    )

    # inputs
    acc = symbol.sym("acc")
    delta = symbol.sym("delta")
    u = vertcat(
            acc,
            delta
    )

    # state derivatives
    xdot = symbol.sym("xdot")
    ydot = symbol.sym("ydot")
    veldot = symbol.sym("veldot")
    psidot = symbol.sym("psidot")
    zdot = vertcat(
            xdot,
            ydot,
            veldot,
            psidot
    )

    # algebraic variables
    algebraic_variables = vertcat([])

    # parameters
    '''Race car parameters'''
    wheelbase = symbol.sym('wheelbase')  # m
    z_ref = symbol.sym('z_ref', z.shape)
    u_ref = symbol.sym('u_ref', u.shape)
    z_k = symbol.sym('z_k', z.shape)
    u_prev = symbol.sym('u_prev', u.shape)
    parameters = vertcat(
            wheelbase,
            z_ref,
            u_ref,
            z_k,
            u_prev
    )

    # dynamics (continuous non-linear model)
    f_expl = vertcat(
            vel * cos(psi),
            vel * sin(psi),
            acc,
            (vel / wheelbase) * tan(delta)
    )  # continuous non-linear model

    '''Constraints/bounds. 
    Todo: add rate constraints'''
    # state bounds
    # model.x_min = -inf
    # model.x_max = inf
    #
    # model.y_min = -inf
    # model.y_max = inf

    model.vel_min = -1.5
    model.vel_max = 1.5

    model.psi_min = -np.pi
    model.psi_max = np.pi

    # input bounds
    model.acc_min = -3.0
    model.acc_max = 3.0

    model.delta_min = -np.radians(27.0)  # minimum steering angle [rad]
    model.delta_max = np.radians(27.0)  # maximum steering angle [rad]

    # Define initial conditions
    model.x0 = np.array([0., 0., 0., 0.])

    # # Define constraints struct
    # constraint.expr = vertcat(
    #         psi,
    #         # acc,  # if the dynamics are updated with u as states
    #         # delta  # if the dynamics are updated with u as states
    # )

    # Define model struct
    params = types.SimpleNamespace()
    params.wheelbase = wheelbase
    model.f_impl_expr = zdot - f_expl
    model.f_expl_expr = f_expl
    # model.f_disk = discretizer(x_k, u_k, p_k)  # provide function
    model.x = z
    model.xdot = zdot
    model.u = u
    model.z = algebraic_variables
    model.p = parameters
    model.name = model_name
    model.params = params

    return model, constraint


