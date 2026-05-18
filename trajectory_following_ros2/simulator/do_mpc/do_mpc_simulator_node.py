"""
Kinematic bicycle simulator using do-mpc's SUNDIALS CVODES integrator.

Publishes odometry, acceleration feedback, and a TF transform at `update_rate` Hz,
acting as a physics plant for hardware-free testing paired with a controller node.
"""
import numpy as np

from trajectory_following_ros2.simulator.base_simulator import BaseSimulator, main_spin
from trajectory_following_ros2.do_mpc.model import BicycleKinematicModel
from trajectory_following_ros2.simulator.do_mpc.do_mpc_simulator import Simulator


class DoMpcSimulator(BaseSimulator):

    def __init__(self):
        super().__init__('kinematic_do_mpc_simulator')

    def _build_integrator(self):
        vehicle_model = BicycleKinematicModel(
            length=self.WHEELBASE, width=0.192, sample_time=self.sample_time)
        # TVP placeholders required by the do-mpc model structure; not used by simulator.
        vehicle_model.wp_id = 0
        vehicle_model.reference_path = [[0., 0., 0., 0.]]
        sim = Simulator(vehicle_model, sample_time=self.sample_time)
        self._integrator = sim.simulator
        self._integrator.x0 = self.zk

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # do-mpc CVODES integrator tracks state internally; x argument is unused.
        return self._integrator.make_step(u.reshape(-1, 1)).flatten()


def main(args=None):
    main_spin(DoMpcSimulator, args)


if __name__ == '__main__':
    main()