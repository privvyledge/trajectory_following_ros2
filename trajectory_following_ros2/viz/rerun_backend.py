"""Rerun visualization backend."""
import math
from typing import List, Tuple, Callable

import rerun as rr

from trajectory_following_ros2.viz.rerun_helpers import ENTITY, COLORS, ros_stamp_to_nanos
from trajectory_following_ros2.viz.base_viz_backend import BaseVizBackend


class RerunBackend(BaseVizBackend):
    """Wraps all rr.log calls extracted verbatim from the original visualizer_node."""

    def __init__(self, app_name: str, spawn_viewer: bool,
                 connect_addr: str, recording_path: str,
                 stamp_fn: Callable):
        """
        stamp_fn — zero-argument callable returning the current ROS stamp.
                   Signature: () -> builtin_interfaces.msg.Time
                   Used for log_yaw_rate and log_solve_time which have no msg stamp.
        """
        rr.init(app_name)
        if spawn_viewer:
            rr.spawn()
        if connect_addr:
            rr.connect_tcp(connect_addr)
        if recording_path:
            rr.save(recording_path)
        self._stamp_fn = stamp_fn

    # ------------------------------------------------------------------
    # Internal time helpers
    # ------------------------------------------------------------------

    def _set_time_from_stamp(self, stamp) -> None:
        rr.set_time_nanos('ros_time', ros_stamp_to_nanos(stamp))

    def _set_time_now(self) -> None:
        rr.set_time_nanos('ros_time', ros_stamp_to_nanos(self._stamp_fn()))

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    def log_vehicle_pose(self, x: float, y: float, yaw: float, speed: float,
                         stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['vehicle_pos'],
               rr.Points2D([[x, y]], colors=[COLORS['vehicle']], radii=0.1))
        arrow = 0.4
        rr.log(ENTITY['vehicle_heading'],
               rr.Arrows2D(origins=[[x, y]],
                           vectors=[[math.cos(yaw) * arrow, math.sin(yaw) * arrow]],
                           colors=[COLORS['vehicle']]))
        rr.log(ENTITY['speed_actual'], rr.Scalar(speed))
        rr.log(ENTITY['heading_deg'],  rr.Scalar(math.degrees(yaw)))

    def log_full_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['full_path'],
               rr.LineStrips2D([pts], colors=[COLORS['full_path']], radii=0.02))

    def log_predicted_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['predicted'],
               rr.LineStrips2D([pts], colors=[COLORS['predicted']], radii=0.03))

    def log_ref_window(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['ref_window'],
               rr.LineStrips2D([pts], colors=[COLORS['ref_window']], radii=0.03))

    def log_goal(self, x: float, y: float, stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['goal'],
               rr.Points2D([[x, y]], colors=[COLORS['goal']], radii=0.12))

    # ------------------------------------------------------------------
    # Time-series — commanded actions
    # ------------------------------------------------------------------

    def log_commands(self, accel: float, steer_deg: float, speed: float,
                     stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['accel_commanded'],     rr.Scalar(accel))
        rr.log(ENTITY['steer_commanded_deg'], rr.Scalar(steer_deg))
        rr.log(ENTITY['speed_commanded'],     rr.Scalar(speed))

    def log_yaw_rate(self, yaw_rate: float) -> None:
        self._set_time_now()
        rr.log(ENTITY['yaw_rate_desired'], rr.Scalar(yaw_rate))

    # ------------------------------------------------------------------
    # Time-series — state feedback
    # ------------------------------------------------------------------

    def log_accel_actual(self, accel: float, stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['accel_actual'], rr.Scalar(accel))

    def log_solve_time(self, solve_time_s: float) -> None:
        self._set_time_now()
        rr.log(ENTITY['solve_time_ms'], rr.Scalar(solve_time_s * 1e3))

    # ------------------------------------------------------------------
    # Time-series — errors
    # ------------------------------------------------------------------

    def log_errors(self, cte: float, heading_err_deg: float) -> None:
        rr.log(ENTITY['cross_track_error'], rr.Scalar(cte))
        rr.log(ENTITY['heading_error_deg'], rr.Scalar(heading_err_deg))

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def log_actuator_feedback(self, accel: float, steer_deg: float, speed: float,
                              stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['steer_actual_deg'], rr.Scalar(steer_deg))
        rr.log(ENTITY['speed_actual_fb'],  rr.Scalar(speed))
        rr.log(ENTITY['accel_actual_fb'],  rr.Scalar(accel))

    def log_reference_cmd(self, accel: float, steer_deg: float, speed: float,
                          stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['accel_reference'],     rr.Scalar(accel))
        rr.log(ENTITY['steer_reference_deg'], rr.Scalar(steer_deg))
        rr.log(ENTITY['speed_reference'],     rr.Scalar(speed))