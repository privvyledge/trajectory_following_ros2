"""
Kinematic bicycle simulator using an acados IRK integrator.

Publishes odometry, acceleration feedback, and a TF transform at `update_rate` Hz,
acting as a physics plant for hardware-free testing paired with a controller node.
"""
import os
import tempfile

import numpy as np

from trajectory_following_ros2.simulator.base_simulator import BaseSimulator, main_spin
from trajectory_following_ros2.simulator.acados.acados_simulator import Simulator

# Acados Cython build embeds the CWD into Makefile -I flags without quoting.
# Paths with spaces (e.g. OneDrive) break cython. Write generated code to /tmp
# where the path is guaranteed to have no spaces.
_ACADOS_TMP = os.path.join(tempfile.gettempdir(), 'trajectory_following_ros2', 'acados_sim')
os.makedirs(_ACADOS_TMP, exist_ok=True)


class AcadosSimulator(BaseSimulator):

    def __init__(self):
        super().__init__('kinematic_acados_simulator')

    def _build_integrator(self):
        code_dir = os.path.join(_ACADOS_TMP, 'c_generated_code_integrator_cython')
        already_built = os.path.exists(os.path.join(code_dir, 'acados_sim_solver_pyx.so'))
        sim = Simulator(
            vehicle=None, sample_time=self.sample_time,
            generate=not already_built, build=not already_built, with_cython=True,
            integrator_config_file=os.path.join(_ACADOS_TMP, 'kinematic_bicycle_acados_integrator.json'),
            code_export_directory=os.path.join(_ACADOS_TMP, 'c_generated_code_integrator'))
        self._integrator = sim.acados_integrator
        self._integrator.set('u', self.uk.flatten())
        self._integrator.set('x', self.zk.flatten())
        self._integrator.solve()

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        self._integrator.set('x', x)
        self._integrator.set('u', u)
        self._integrator.set('p', np.concatenate([
            [self.WHEELBASE],
            np.zeros(self.NX),       # zref (unused by integrator dynamics)
            np.zeros(self.NU),       # uref (unused by integrator dynamics)
            x,                       # zk
            self.u_prev.flatten(),
        ]))
        self._integrator.solve()
        return self._integrator.get('x')


def main(args=None):
    main_spin(AcadosSimulator, args)


if __name__ == '__main__':
    main()
