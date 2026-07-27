"""Abstract base class for visualization backends."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional  # noqa: F401


class BaseVizBackend(ABC):

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    @abstractmethod
    def log_vehicle_pose(self, x: float, y: float, yaw: float, speed: float,
                         stamp=None) -> None:
        """Vehicle position + heading.  Called every odom callback (~20 Hz)."""

    @abstractmethod
    def log_full_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        """Complete reference trajectory (published once, transient-local)."""

    @abstractmethod
    def log_predicted_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        """MPC predicted horizon."""

    @abstractmethod
    def log_ref_window(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        """MPC reference window (current lookahead)."""

    @abstractmethod
    def log_goal(self, x: float, y: float, stamp=None) -> None:
        """Current goal point."""

    # ------------------------------------------------------------------
    # Time-series — commanded actions
    # ------------------------------------------------------------------

    @abstractmethod
    def log_commands(self, accel: float, steer_deg: float, speed: float,
                     stamp=None) -> None:
        """AckermannDriveStamped outputs from the controller."""

    @abstractmethod
    def log_yaw_rate(self, yaw_rate: float) -> None:
        """Desired yaw rate (rad/s).  No message stamp — backend uses own clock."""

    # ------------------------------------------------------------------
    # Time-series — state feedback
    # ------------------------------------------------------------------

    @abstractmethod
    def log_accel_actual(self, accel: float, stamp=None) -> None:
        """Measured longitudinal acceleration (m/s²)."""

    @abstractmethod
    def log_solve_time(self, solve_time_s: float) -> None:
        """Solver wall time in seconds.  No message stamp — backend uses own clock."""

    # ------------------------------------------------------------------
    # Time-series — errors (computed in the node, passed here)
    # ------------------------------------------------------------------

    @abstractmethod
    def log_errors(self, cte: float, heading_err_deg: float) -> None:
        """Cross-track error (m) and heading error (deg)."""

    # ------------------------------------------------------------------
    # Optional — only called when the corresponding topic is subscribed
    # ------------------------------------------------------------------

    def log_actuator_feedback(self, accel: float, steer_deg: float, speed: float,
                              stamp=None) -> None:
        """Actual values echoed from the hardware actuator.  Default no-op."""

    def log_reference_cmd(self, accel: float, steer_deg: float, speed: float,
                          stamp=None) -> None:
        """Reference commands from a driver-recorded dataset.  Default no-op."""

    def log_obstacles(self, obstacles: list, margin_offset: float = 0.0,
                      stamp=None) -> None:
        """Obstacles with shape type, position, orientation, dimensions, and margin offset. Default no-op."""

    def log_footprint_polygon(self, pts: list, stamp=None) -> None:
        """Polygon coordinates (x, y) of the ego footprint. Default no-op."""

    def set_keepout(self, ego_radius: float, safe_distance: float) -> None:
        """Update the drawn ego/keep-out radii after startup.  Default no-op.

        Called when the visualizer adopts the controller's live values, so the drawn
        keep-out matches the one the solver actually enforces.
        """

    # ------------------------------------------------------------------
    # Lifecycle

    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Called when the node is destroyed.  Default no-op."""
