"""
Solver-agnostic discrete-time dynamics for the kinematic backends.

Given a continuous-time explicit ODE ``xdot = f(x, u, p)``, build a symbolic
one-step map ``x_{k+1} = F(x_k, u_k, p)`` using a selectable integration
scheme. Intended to be shared by the CasADi discrete MPC model
(``casadi/kinematic_bicycle_model.py``) and acados (``acados/kinematic_model.py``)
so both backends discretize through a single path.

Methods:
    'euler'        Forward Euler (1st order). Explicit symbolic expression.
    'rk4'          Classic 4-stage Runge-Kutta (4th order). Explicit symbolic
                   expression. Recommended for the MPC prediction model.
    'cvodes'       SUNDIALS CVODES (variable-step BDF), 'rk' (fixed-step RK),
    'rk'           and 'collocation' (Gauss-Legendre) wrap ``casadi.integrator``
    'collocation'  and return a black-box Function call. They are heavier to
                   differentiate inside an NLP; prefer 'rk4' for the MPC model
                   and reserve these for simulation / reference.

For 'euler' and 'rk4', ``dt`` may be a CasADi symbol, so the caller can keep
``dt`` parametric and substitute a numeric value later. For the
integrator-backed methods ``dt`` must be a concrete number, because
``casadi.integrator`` bakes the step (``tf``) into the generated integrator at
construction time.
"""
import casadi

EXPLICIT_METHODS = ('euler', 'rk4')
INTEGRATOR_METHODS = ('cvodes', 'idas', 'rk', 'collocation')


def _build_integrator(name, method, dae, dt):
    """Construct a casadi.integrator across CasADi versions.

    CasADi >= 3.6 uses the positional ``integrator(name, solver, dae, t0, tf)``
    signature; older versions take the step via an options dict
    ``{'t0': ..., 'tf': ...}``. Try the modern form first and fall back to the
    legacy form on a signature mismatch.
    """
    t0, tf = 0.0, float(dt)
    try:
        return casadi.integrator(name, method, dae, t0, tf)
    except (TypeError, NotImplementedError):
        return casadi.integrator(name, method, dae, {'t0': t0, 'tf': tf})


def make_discrete_dynamics(f_expl, x, u, dt, p=None, method='rk4',
                           function_name='disc_dyn'):
    """Discretize a continuous explicit ODE into a one-step map.

    :param f_expl: continuous-time ODE expression ``xdot = f(x, u, p)``.
    :param x: state symbol (column vector).
    :param u: input symbol (column vector).
    :param dt: integration step. Symbolic (MX/SX) is allowed for 'euler'/'rk4';
        integrator-backed methods require a numeric ``dt``.
    :param p: optional parameter symbol(s) held constant over the step
        (e.g. wheelbase). ``None`` if the ODE has no extra parameters.
    :param method: integration scheme (see module docstring).
    :param function_name: name used for the generated CasADi objects.
    :return: CasADi expression for ``x_{k+1}`` in terms of ``x, u`` (and ``p``).
    """
    method = method.lower()

    ode_args = [x, u] if p is None else [x, u, p]
    f = casadi.Function(f'{function_name}_ode', ode_args, [f_expl])

    def _f(xk):
        return f(xk, u) if p is None else f(xk, u, p)

    if method == 'euler':
        x_next = x + dt * _f(x)
    elif method == 'rk4':
        k1 = _f(x)
        k2 = _f(x + 0.5 * dt * k1)
        k3 = _f(x + 0.5 * dt * k2)
        k4 = _f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    elif method in INTEGRATOR_METHODS:
        if isinstance(dt, (casadi.MX, casadi.SX)):
            raise ValueError(
                f"method='{method}' requires a numeric dt because "
                f"casadi.integrator bakes 'tf' at construction time; "
                f"got a symbolic dt of type {type(dt).__name__}.")
        params = u if p is None else casadi.vertcat(u, p)
        dae = {'x': x, 'p': params, 'ode': f_expl}
        # SUNDIALS solvers ('cvodes'/'idas') are black boxes that only support
        # MX graphs (no eval_sx) and are heavy to differentiate inside an NLP --
        # prefer 'rk4' for the MPC model and reserve these for simulation.
        integrator = _build_integrator(function_name, method, dae, dt)
        x_next = integrator(x0=x, p=params)['xf']
    else:
        raise ValueError(
            f"Unknown integration method '{method}'. Expected one of "
            f"{EXPLICIT_METHODS + INTEGRATOR_METHODS}.")

    return x_next
