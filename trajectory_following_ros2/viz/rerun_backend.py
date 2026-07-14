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
                 open_browser: bool = True,
                 ego_radius: float = 0.0, safe_distance: float = 0.0,
                 vehicle_length: float = 0.58, vehicle_width: float = 0.31,
                 vehicle_height: float = 0.12, footprint_rear_axle_offset: float = 0.19):
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
        self._log_world_axes()

        self._ego_radius = ego_radius
        self._safe_distance = safe_distance
        self._vehicle_length = vehicle_length
        self._vehicle_width = vehicle_width
        self._vehicle_height = vehicle_height
        self._footprint_rear_axle_offset = footprint_rear_axle_offset
        self._footprint_poly = []

    # ------------------------------------------------------------------
    # World frame
    # ------------------------------------------------------------------
    # Rerun's 2-D view uses the image convention (+X right, +Y *down*), while
    # ROS REP-103 is +X forward, +Y left, +Z up. We reconcile the two by
    # negating Y on every spatial log: a ROS point at +Y (left) is logged at
    # -Y, which Rerun then renders *up*. The result is a ROS-aligned view
    # (X right, Y up, yaw CCW), matching RViz. All spatial logs below go
    # through _xy() / _strip() so the convention is applied in exactly one place.

    @staticmethod
    def _xy(x: float, y: float) -> List[float]:
        """ROS (x, y) → Rerun 2-D screen coordinates (negate Y)."""
        return [x, -y]

    @staticmethod
    def _strip(pts: List[Tuple[float, float]]) -> List[List[float]]:
        """ROS polyline → Rerun 2-D screen coordinates (negate Y)."""
        return [[px, -py] for px, py in pts]

    def _log_world_axes(self) -> None:
        """Static origin marker + X/Y axis arrows in the ROS frame.

        Logged once as static data so the frame is visible at every point on
        the timeline. Arrows are expressed directly in screen coordinates:
        ROS +X (forward) points right, ROS +Y (left) points up.
        """
        axis_len = 1.0  # metres
        rr.log(ENTITY['world_origin'],
               rr.Points2D([[0.0, 0.0]], colors=[COLORS['origin']], radii=0.06),
               static=True)
        rr.log(ENTITY['world_axes'],
               rr.Arrows2D(
                   origins=[[0.0, 0.0], [0.0, 0.0]],
                   # +X right, +Y up (screen). Y arrow is negated like all data.
                   vectors=[[axis_len, 0.0], [0.0, -axis_len]],
                   colors=[COLORS['axis_x'], COLORS['axis_y']],
                   labels=['x', 'y']),
               static=True)

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
            # ws_port defaults to 9877 in rerun 0.22; the served web viewer at
            # http://localhost:<web_port>/ shows the generic start page unless the
            # data-source URL is appended. Bare http://localhost:9090 will NOT connect.
            wp = web_port if web_port is not None else 9090
            ws_port = 9877
            rr.serve_web(open_browser=open_browser, web_port=wp, ws_port=ws_port)
            print(
                f'[RerunBackend] Web viewer ready. Open this FULL URL in your browser '
                f'(bare http://localhost:{wp} shows the empty start page):\n'
                f'    http://localhost:{wp}/?url=ws://localhost:{ws_port}\n'
                f'    (WSL2: localhost forwards to Windows automatically.)',
                flush=True)
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
               rr.Points2D([self._xy(x, y)], colors=[COLORS['vehicle']], radii=0.1))
        arrow = 0.4
        rr.log(ENTITY['vehicle_heading'],
               rr.Arrows2D(origins=[self._xy(x, y)],
                           vectors=[self._xy(math.cos(yaw) * arrow,
                                             math.sin(yaw) * arrow)],
                           colors=[COLORS['vehicle']]))
        rr.log(ENTITY['speed_actual'], rr.Scalar(speed))
        rr.log(ENTITY['heading_deg'],  rr.Scalar(math.degrees(yaw)))

        # 1. Log Ego Radius Circle (translucent Points2D)
        if self._ego_radius > 0.0:
            rr.log(ENTITY['ego_radius'],
                   rr.Points2D([self._xy(x, y)], colors=[COLORS['ego_radius']], radii=self._ego_radius))

        # 2. Log Safe Distance Circle (radius = ego_radius + safe_distance)
        if self._safe_distance > 0.0 or self._ego_radius > 0.0:
            rr.log(ENTITY['safe_distance'],
                   rr.Points2D([self._xy(x, y)], colors=[COLORS['safe_distance']], radii=self._ego_radius + self._safe_distance))

        # 3. Log Footprint (Nav2 polygon prism or default 3D box)
        if self._footprint_poly:
            self._log_prism(ENTITY['vehicle_footprint'], self._footprint_poly, self._vehicle_height, COLORS['ego_footprint'])
        else:
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            cx = x + self._footprint_rear_axle_offset * cos_y
            cy = y + self._footprint_rear_axle_offset * sin_y
            cz = self._vehicle_height / 2.0
            q_neg = [0.0, 0.0, math.sin(-yaw / 2), math.cos(-yaw / 2)]

            rr.log(ENTITY['vehicle_footprint'],
                   rr.Boxes3D(
                       centers=[[cx, -cy, cz]],
                       half_sizes=[[self._vehicle_length / 2.0, self._vehicle_width / 2.0, self._vehicle_height / 2.0]],
                       rotations=[q_neg],
                       colors=[COLORS['ego_footprint']]
                   ))

    def log_full_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['full_path'],
               rr.LineStrips2D([self._strip(pts)], colors=[COLORS['full_path']],
                               radii=0.02))

    def log_predicted_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['predicted'],
               rr.LineStrips2D([self._strip(pts)], colors=[COLORS['predicted']],
                               radii=0.03))

    def log_ref_window(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['ref_window'],
               rr.LineStrips2D([self._strip(pts)], colors=[COLORS['ref_window']],
                               radii=0.03))

    def log_goal(self, x: float, y: float, stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        rr.log(ENTITY['goal'],
               rr.Points2D([self._xy(x, y)], colors=[COLORS['goal']], radii=0.12))

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

    def _log_prism(self, entity_path: str, pts_2d: List[Tuple[float, float]], height: float, color: List[int]) -> None:
        bottom_pts = [[p[0], -p[1], 0.0] for p in pts_2d]
        top_pts = [[p[0], -p[1], height] for p in pts_2d]

        if bottom_pts:
            bottom_pts.append(bottom_pts[0])
            top_pts.append(top_pts[0])

        verticals = []
        for b, t in zip(bottom_pts[:-1], top_pts[:-1]):
            verticals.append([b, t])

        strips = [bottom_pts, top_pts] + verticals
        rr.log(entity_path, rr.LineStrips3D(strips, colors=[color], radii=0.015))

    def log_footprint_polygon(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)
        self._footprint_poly = list(pts)

    def log_obstacles(self, obstacles: List[dict], margin_offset: float = 0.0, stamp=None) -> None:
        if stamp is not None:
            self._set_time_from_stamp(stamp)

        if not obstacles:
            rr.log(ENTITY['obstacles_circles'], rr.Points2D([]))
            rr.log(ENTITY['obstacles_margin_circles'], rr.Points2D([]))
            rr.log(ENTITY['obstacles_boxes'], rr.LineStrips2D([]))
            rr.log(ENTITY['obstacles_margin_boxes'], rr.LineStrips2D([]))
            return

        circle_positions = []
        circle_radii = []
        box_strips = []
        box_margin_strips = []

        for obs in obstacles:
            o_type = obs['type']
            x = obs['x']
            y = obs['y']
            yaw = obs['yaw']
            dims = obs['dimensions']

            if o_type in ('SPHERE', 'CYLINDER') and dims:
                r = dims[0] if o_type == 'SPHERE' else (dims[1] if len(dims) >= 2 else dims[0])
                circle_positions.append(self._xy(x, y))
                circle_radii.append(r)
            elif o_type == 'BOX' and len(dims) >= 2:
                length, width = dims[0], dims[1]

                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                dx = length / 2.0
                dy = width / 2.0

                local_corners = [
                    (dx, dy),
                    (dx, -dy),
                    (-dx, -dy),
                    (-dx, dy),
                    (dx, dy)
                ]
                strip = [[x + lx * cos_y - ly * sin_y, -(y + lx * sin_y + ly * cos_y)] for lx, ly in local_corners]
                box_strips.append(strip)

                if margin_offset > 0.0:
                    mx = dx + margin_offset
                    my = dy + margin_offset
                    local_margin_corners = [
                        (mx, my),
                        (mx, -my),
                        (-mx, -my),
                        (-mx, my),
                        (mx, my)
                    ]
                    margin_strip = [[x + lx * cos_y - ly * sin_y, -(y + lx * sin_y + ly * cos_y)] for lx, ly in local_margin_corners]
                    box_margin_strips.append(margin_strip)

        if circle_positions:
            rr.log(ENTITY['obstacles_circles'],
                   rr.Points2D(circle_positions, radii=circle_radii, colors=[COLORS['obstacle']]))
            if margin_offset > 0.0:
                margin_radii = [r + margin_offset for r in circle_radii]
                rr.log(ENTITY['obstacles_margin_circles'],
                       rr.Points2D(circle_positions, radii=margin_radii, colors=[COLORS['obstacle_margin']]))
            else:
                rr.log(ENTITY['obstacles_margin_circles'], rr.Points2D([]))
        else:
            rr.log(ENTITY['obstacles_circles'], rr.Points2D([]))
            rr.log(ENTITY['obstacles_margin_circles'], rr.Points2D([]))

        if box_strips:
            rr.log(ENTITY['obstacles_boxes'],
                   rr.LineStrips2D(box_strips, colors=[COLORS['obstacle']], radii=0.03))
            if box_margin_strips:
                rr.log(ENTITY['obstacles_margin_boxes'],
                       rr.LineStrips2D(box_margin_strips, colors=[COLORS['obstacle_margin']], radii=0.015))
            else:
                rr.log(ENTITY['obstacles_margin_boxes'], rr.LineStrips2D([]))
        else:
            rr.log(ENTITY['obstacles_boxes'], rr.LineStrips2D([]))
            rr.log(ENTITY['obstacles_margin_boxes'], rr.LineStrips2D([]))
