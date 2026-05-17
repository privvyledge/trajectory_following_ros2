from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SolverResult:
    """Normalized output from any MPC solver backend."""
    accel_cmd: float = 0.0          # [m/s²]  first optimal input
    steering_cmd: float = 0.0       # [rad]   first optimal input
    velocity_cmd: float = 0.0       # [m/s]   k+1 predicted velocity (sent to VESC as speed)
    jerk_cmd: Optional[float] = None
    steering_rate_cmd: Optional[float] = None
    u_sequence: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))   # (nu, N)
    x_sequence: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))   # (nx, N+1)
    u_prev: np.ndarray = field(default_factory=lambda: np.zeros(2))            # (nu,) for warm-start
    is_optimal: bool = False
    solve_time: float = 0.0
    status: str = 'uninitialized'


class BaseSolver(ABC):
    """Abstract interface that every MPC backend adapter must implement."""

    def initialize(self, x0: np.ndarray) -> None:
        """One-time warm-start after the first odometry message. Default: no-op."""

    @abstractmethod
    def solve(self, x0: np.ndarray, xref: np.ndarray,
              u_prev: np.ndarray) -> SolverResult:
        """
        Run one MPC step.

        Parameters
        ----------
        x0   : shape (nx,)   — current state [x, y, vel, psi]
        xref : shape (nx, N+1) — reference trajectory
        u_prev : shape (nu,) — previous applied input [acc, delta] (rad for steering)

        Returns
        -------
        SolverResult
        """

    @property
    @abstractmethod
    def nx(self) -> int:
        """Number of states (4: x, y, vel, psi)."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Number of inputs (2: acc, delta)."""
