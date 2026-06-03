"""Rerun visualization backend."""
import math
import warnings
from typing import List, Tuple, Callable

import rerun as rr

from trajectory_following_ros2.viz.rerun_helpers import (
    ENTITY, COLORS, ros_stamp_to_nanos, build_blueprint)
from trajectory_following_ros2.viz.base_viz_backend import BaseVizBackend


# Multi-sink ("tee") — streaming to a live viewer AND a .rrd file at the same
# time — was added in rerun 0.23 (rr.set_sinks + rr.GrpcSink / rr.FileSink).
# On older rerun a recording stream has a single sink and the last one set
# wins, so a live viewer and file recording are mutually exclusive there.
_MULTISINK_MIN_VERSION = (0, 23, 0)


def _rerun_version() -> Tuple[int, int, int]:
    """Installed rerun version as a (major, minor, patch) int tuple."""
    parts = []
    for token in rr.__version__.split('+')[0].split('.')[:3]:
        digits = ''.join(c for c in token if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


class RerunBackend(BaseVizBackend):
    """Wraps all rr.log calls extracted verbatim from the original visualizer_node."""

    def __init__(self, app_name: str, spawn_viewer: bool,
                 connect_addr: str, recording_path: str,
                 stamp_fn: Callable,
                 serve_web: bool = False, web_port=None,
                 open_browser: bool = True):
        """
        stamp_fn — zero-argument callable returning the current ROS stamp.
                   Signature: () -> builtin_interfaces.msg.Time
                   Used for log_yaw_rate and log_solve_time which have no msg stamp.
        serve_web — serve the Rerun web viewer over HTTP and stream data over a
                   WebSocket instead of spawning the native viewer. The browser
                   renders with WebGPU/WebGL, bypassing the native Vulkan path
                   (broken on WSL2 — no VK_ICD_FILENAMES workaround needed). Takes
                   precedence over spawn_viewer/connect_addr when set.
        web_port — HTTP port for the web viewer (None = rerun default, 9090).
        open_browser — auto-open the system browser at the viewer URL.
        """
        rr.init(app_name)
        self._configure_sinks(spawn_viewer, connect_addr, recording_path,
                              serve_web, web_port, open_browser)
        # Pin an explicit layout. Without this, rerun's auto-layout dumps every
        # scalar onto one time-series view and (notably in the web viewer) splits
        # the heading arrow into its own spatial view, detached from the pose.
        blueprint = build_blueprint()
        if blueprint is not None:
            rr.send_blueprint(blueprint)
        self._stamp_fn = stamp_fn

    # ------------------------------------------------------------------
    # Sink configuration
    # ------------------------------------------------------------------

    def _configure_sinks(
            self, spawn_viewer: bool, connect_addr: str, recording_path: str,
            serve_web: bool = False, web_port=None,
            open_browser: bool = True) -> None:
        """Wire up rerun output sinks.

        A live viewer (spawn or connect) and a .rrd recording can run at the
        same time only on rerun >= 0.23 (multi-sink tee). On older rerun the
        single-sink model makes them mutually exclusive; in that case the
        persistent .rrd recording is kept and the live viewer is skipped.
        """
        # Browser web viewer takes precedence as the live sink. It renders in the
        # browser via WebGPU/WebGL, bypassing the native Vulkan path that is broken
        # on WSL2 — no VK_ICD_FILENAMES / lavapipe workaround needed in this mode.
        # serve_web is single-sink on rerun < 0.23, so it is mutually exclusive
        # with a .rrd recording; keep the web viewer and warn if both were asked.
        if serve_web:
            if recording_path:
                warnings.warn(
                    'serve_web and a .rrd recording_path are mutually exclusive on '
                    f'rerun {rr.__version__} (single-sink). Keeping the web viewer and '
                    'skipping the recording. Record in a separate run, or upgrade to '
                    'rerun >= 0.23 for simultaneous sinks.', stacklevel=2)
            rr.serve_web(open_browser=open_browser, web_port=web_port)
            return

        live_requested = bool(spawn_viewer) or bool(connect_addr)
        file_requested = bool(recording_path)

        # One output (or none): legacy single-sink calls — valid on every version.
        if not (live_requested and file_requested):
            if spawn_viewer:
                rr.spawn()
            if connect_addr:
                rr.connect_tcp(connect_addr)
            if recording_path:
                rr.save(recording_path)
            return

        # Both a live viewer and a file recording were requested.
        multisink_supported = (
            _rerun_version() >= _MULTISINK_MIN_VERSION
            and hasattr(rr, 'set_sinks')
            and hasattr(rr, 'GrpcSink')
            and hasattr(rr, 'FileSink')
        )
        if multisink_supported:
            try:
                self._configure_multisink(spawn_viewer, connect_addr, recording_path)
                return
            except Exception as exc:  # API drift across rerun versions — degrade.
                warnings.warn(
                    f'Rerun multi-sink (tee) setup failed ({exc}); falling back to '
                    f'recording only ({recording_path}).', stacklevel=2)
        else:
            warnings.warn(
                'Simultaneous live viewer + .rrd recording requires rerun >= 0.23 '
                f'(installed: {rr.__version__}). On this version a recording stream has '
                'a single sink, so they are mutually exclusive. Keeping the .rrd '
                'recording and skipping the live viewer. To view live, open the file in '
                f'a separate viewer (it tails as it grows): rerun {recording_path}',
                stacklevel=2)

        # <0.23 (or a failed tee): prefer the persistent recording over live view.
        rr.save(recording_path)

    def _configure_multisink(
            self, spawn_viewer: bool, connect_addr: str, recording_path: str) -> None:
        """Tee log data to a live viewer and a .rrd file (rerun >= 0.23)."""
        sinks = []
        if spawn_viewer:
            # Launch the viewer process without claiming the default sink, then
            # route to it explicitly with a GrpcSink alongside the FileSink.
            rr.spawn(connect=False)
            sinks.append(rr.GrpcSink())
        if connect_addr:
            sinks.append(rr.GrpcSink(url=f'rerun+http://{connect_addr}/proxy'))
        sinks.append(rr.FileSink(recording_path))
        rr.set_sinks(*sinks)

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
