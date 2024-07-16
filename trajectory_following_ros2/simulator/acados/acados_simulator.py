import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

from acados_template import AcadosModel
from acados_template import AcadosSim, AcadosSimSolver
from trajectory_following_ros2.acados.kinematic_model import kinematic_model


class Simulator(object):
    """docstring for ClassName"""

    def __init__(self, vehicle=None, sample_time=0.02, generate=True, build=True, with_cython=True,
                 integrator_config_file="kinematic_bicycle_acados_integrator.json",
                 code_export_directory="c_generated_code_integrator"):
        """Constructor for Simulator"""

        # model = vehicle.model
        model = kinematic_model()
        self.acados_integrator = self.acados_integrator_settings(model=model, dt=sample_time,
                                                                 generate=generate, build=build,
                                                                 with_cython=with_cython,
                                                                 integrator_config_file=integrator_config_file,
                                                                 code_export_directory=code_export_directory)

    def acados_integrator_settings(self, model, dt, generate=True, build=True, with_cython=True,
                                   integrator_config_file="kinematic_bicycle_acados_integrator.json",
                                   code_export_directory="c_generated_code_integrator"):
        # generate = True  # generates the OCP and stores in the json file
        # build = True  # builds/compiles the model and stores in code_export_directory
        # the cython version is faster than bare C because there is no call overhead as opposed to the C code call overhead
        # linear_mpc=True, use discrete model, else nonlinear continuous model

        # export model
        model, constraint = model

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

        # dimensions
        nx = model.x.size()[0]  # number of states
        nu = model.u.size()[0]  # number of inputs
        nparams = model.p.size()[0]  # number of parameter

        # optionally discretize model if ode_type="linear" or "discrete"
        sim = AcadosSim()
        sim.model = model_ac
        sim.parameter_values = np.array([
            0.256,  # wheelbase
            *np.zeros(nx),  # zref
            *np.zeros(nu),  # uref
            *np.zeros(nx),  # zk
            *np.zeros(nu),  # u_prev
        ])
        sim.solver_options.integrator_type = 'IRK'  # IRK, ERK, GNSF
        # for implicit integrator
        # sim.solver_options.newton_iter = 10  # for implicit integrator
        # sim.solver_options.newton_tol = 1e-8
        # sim.solver_options.collocation_type = "GAUSS_LEGENDRE"
        sim.solver_options.num_stages = 4  # 8, 4, 2
        sim.solver_options.num_steps = 3  # 100, 3, 2
        # sim.solver_options.nlp_solver_tol_eq = 1e-9
        sim.solver_options.T = dt
        # sensitivity_propagation
        # sim.solver_options.sens_adj = False
        # sim.solver_options.sens_forw = False
        # sim.solver_options.sens_hess = False
        sim.code_export_directory = code_export_directory
        if with_cython:
            sim.code_export_directory = f"{sim.code_export_directory}_cython"

            if generate:
                AcadosSimSolver.generate(sim, json_file=integrator_config_file)
            if build:
                AcadosSimSolver.build(sim.code_export_directory, with_cython=with_cython)

            acados_integrator = AcadosSimSolver.create_cython_solver(integrator_config_file)

        else:
            acados_integrator = AcadosSimSolver(sim, json_file='kinematic_bicycle_acados_integrator.json', build=build,
                                                generate=generate, verbose=True)

        return acados_integrator
