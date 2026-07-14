"""Native matplotlib visualization backend.

Runs a daemon thread that owns all matplotlib state and refreshes at 10 Hz.
ROS callbacks write into thread-safe deque ring buffers; the animation
function snapshots them under a lock.

matplotlib.use('TkAgg') must be called before importing pyplot — this module
handles that at import time, and raises ImportError on headless systems so
the node's outer except-block can skip this backend gracefully.
"""
import collections
import math
import threading
from typing import List, Tuple

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms

except Exception as e:
    raise ImportError(f'matplotlib TkAgg unavailable: {e}') from e


from trajectory_following_ros2.viz.base_viz_backend import BaseVizBackend


class MatplotlibBackend(BaseVizBackend):
    """Daemon-thread matplotlib figure with 2-D map + 4 time-series panels."""

    def __init__(self, buffer_size: int = 300,
                 video_path: str = '', video_fps: int = 10,
                 ego_radius: float = 0.0, safe_distance: float = 0.0,
                 vehicle_length: float = 0.58, vehicle_width: float = 0.31,
                 vehicle_height: float = 0.12, footprint_rear_axle_offset: float = 0.19):
        self._lock = threading.Lock()
        # Video recording (optional). When video_path is non-empty the daemon
        # thread sets up a MovieWriter and grabs every animation frame; the
        # writer is finalized in shutdown(). Guarded by its own lock because
        # grab_frame() (daemon thread) and finish() (main thread) both write
        # to the same ffmpeg subprocess pipe.
        self._video_path = video_path
        self._video_fps = max(1, int(video_fps))
        self._writer = None
        self._writer_lock = threading.Lock()
        self._buf = {
            'speed_actual':    collections.deque(maxlen=buffer_size),
            'speed_cmd':       collections.deque(maxlen=buffer_size),
            'steer_cmd_deg':   collections.deque(maxlen=buffer_size),
            'accel_cmd':       collections.deque(maxlen=buffer_size),
            'accel_actual':    collections.deque(maxlen=buffer_size),
            'cte':             collections.deque(maxlen=buffer_size),
            'heading_err_deg': collections.deque(maxlen=buffer_size),
            'solve_time_ms':   collections.deque(maxlen=buffer_size),
            'yaw_rate':        collections.deque(maxlen=buffer_size),
            # optional
            'steer_fb_deg':    collections.deque(maxlen=buffer_size),
            'speed_fb':        collections.deque(maxlen=buffer_size),
            'accel_fb':        collections.deque(maxlen=buffer_size),
            'steer_ref_deg':   collections.deque(maxlen=buffer_size),
            'speed_ref':       collections.deque(maxlen=buffer_size),
            'accel_ref':       collections.deque(maxlen=buffer_size),
        }
        # Spatial state
        self._vehicle_xy = (0.0, 0.0)
        self._vehicle_yaw = 0.0
        self._vehicle_trail = collections.deque(maxlen=buffer_size)
        self._full_path = []
        self._pred_path = []
        self._ref_window = []
        self._goal_xy = None
        self._full_path_dirty = False

        self._obstacles = []
        self._margin_offset = 0.0
        self._obstacles_dirty = False
        self._obstacle_patches = []

        self._ego_radius = ego_radius
        self._safe_distance = safe_distance
        self._vehicle_length = vehicle_length
        self._vehicle_width = vehicle_width
        self._vehicle_height = vehicle_height
        self._footprint_rear_axle_offset = footprint_rear_axle_offset
        self._footprint_poly = []
        self._ego_circles = []
        self._footprint_patches = []

        self._display_thread = threading.Thread(
            target=self._run_display, daemon=True, name='mpl_viz')
        self._display_thread.start()

    # ------------------------------------------------------------------
    # Daemon thread — all matplotlib state lives here
    # ------------------------------------------------------------------

    def _run_display(self):
        fig = plt.figure(figsize=(14, 8), num='Trajectory Visualizer')
        gs = fig.add_gridspec(4, 2, wspace=0.35, hspace=0.55)

        ax_map = fig.add_subplot(gs[:, 0])
        ax_speed = fig.add_subplot(gs[0, 1])
        ax_steer = fig.add_subplot(gs[1, 1])
        ax_errors = fig.add_subplot(gs[2, 1])
        ax_diag = fig.add_subplot(gs[3, 1])

        # --- ax_map artists (created once, updated via set_data) ---
        ax_map.set_title('2-D Map')
        ax_map.set_aspect('equal', adjustable='datalim')
        ax_map.set_xlabel('x (m)')
        ax_map.set_ylabel('y (m)')

        (line_full_path,) = ax_map.plot([], [], color='grey', lw=1.0, label='full path')
        (line_ref_window,) = ax_map.plot([], [], color='orange', lw=1.5, label='ref window')
        (line_pred_path,) = ax_map.plot([], [], color='lime', lw=1.5, label='predicted')
        (line_trail,) = ax_map.plot([], [], color='cyan', lw=0.5, alpha=0.5, label='trail')
        (dot_vehicle,) = ax_map.plot([], [], 'o', color='cyan', ms=7, label='vehicle')
        (dot_goal,) = ax_map.plot([], [], 'o', color='red', ms=8, label='goal')
        # heading arrow via annotation — store as list so we can remove/re-add
        self._hdg_arrow = [None]

        ax_map.legend(loc='upper right', fontsize=7)

        # --- ax_speed ---
        ax_speed.set_title('Speed (m/s)')
        ax_speed.set_ylabel('m/s')
        (line_spd_act,) = ax_speed.plot([], [], color='cyan',  lw=1.2, label='actual')
        (line_spd_cmd,) = ax_speed.plot([], [], color='orange', lw=1.0, ls='--', label='cmd')
        ax_speed.legend(loc='upper right', fontsize=7)

        # --- ax_steer ---
        ax_steer.set_title('Steering cmd (deg)')
        ax_steer.set_ylabel('deg')
        (line_steer,) = ax_steer.plot([], [], color='magenta', lw=1.2)

        # --- ax_errors (twin axes: CTE left, heading right) ---
        ax_errors.set_title('Errors')
        ax_errors.set_ylabel('CTE (m)', color='tab:blue')
        ax_errors.tick_params(axis='y', labelcolor='tab:blue')
        ax_err2 = ax_errors.twinx()
        ax_err2.set_ylabel('Heading err (deg)', color='tab:orange')
        ax_err2.tick_params(axis='y', labelcolor='tab:orange')
        (line_cte,) = ax_errors.plot([], [], color='tab:blue', lw=1.2, label='CTE')
        (line_hdg_err,) = ax_err2.plot([], [], color='tab:orange', lw=1.2, label='Hdg err')

        # --- ax_diag ---
        ax_diag.set_title('Diagnostics')
        ax_diag.set_ylabel('accel cmd (m/s²)', color='tab:green')
        ax_diag.tick_params(axis='y', labelcolor='tab:green')
        ax_diag2 = ax_diag.twinx()
        ax_diag2.set_ylabel('solve time (ms)', color='tab:red')
        ax_diag2.tick_params(axis='y', labelcolor='tab:red')
        (line_accel_cmd,) = ax_diag.plot([], [], color='tab:green', lw=1.2, label='accel cmd')
        (line_solve_time,) = ax_diag2.plot([], [], color='tab:red', lw=1.2, label='solve ms')

        def _xs(buf):
            return list(range(len(buf)))

        def _animate(_frame):
            with self._lock:
                # snapshot scalars
                spd_act = list(self._buf['speed_actual'])
                spd_cmd = list(self._buf['speed_cmd'])
                steer = list(self._buf['steer_cmd_deg'])
                accel = list(self._buf['accel_cmd'])
                cte = list(self._buf['cte'])
                hdg_err = list(self._buf['heading_err_deg'])
                solve = list(self._buf['solve_time_ms'])
                # spatial
                vx, vy = self._vehicle_xy
                vyaw = self._vehicle_yaw
                trail = list(self._vehicle_trail)
                full = list(self._full_path)
                pred = list(self._pred_path)
                ref_win = list(self._ref_window)
                goal = self._goal_xy
                fp_dirty = self._full_path_dirty
                if fp_dirty:
                    self._full_path_dirty = False
                obs_dirty = self._obstacles_dirty
                obstacles = list(self._obstacles)
                margin_offset = self._margin_offset
                if obs_dirty:
                    self._obstacles_dirty = False

                viz_ego_radius = self._ego_radius
                viz_safe_distance = self._safe_distance
                length = self._vehicle_length
                width = self._vehicle_width
                axle_offset = self._footprint_rear_axle_offset
                footprint_poly = list(self._footprint_poly)

            # --- map ---
            if fp_dirty and full:
                fx, fy = zip(*full)
                line_full_path.set_data(fx, fy)

            if pred:
                px, py = zip(*pred)
                line_pred_path.set_data(px, py)
            else:
                line_pred_path.set_data([], [])

            if ref_win:
                rx, ry = zip(*ref_win)
                line_ref_window.set_data(rx, ry)
            else:
                line_ref_window.set_data([], [])

            if trail:
                tx, ty = zip(*trail)
                line_trail.set_data(tx, ty)
            else:
                line_trail.set_data([], [])

            dot_vehicle.set_data([vx], [vy])

            if goal is not None:
                dot_goal.set_data([goal[0]], [goal[1]])

            # obstacles — remove previous, draw new
            if obs_dirty:
                while self._obstacle_patches:
                    patch = self._obstacle_patches.pop()
                    patch.remove()

                for obs in obstacles:
                    o_type = obs['type']
                    x = obs['x']
                    y = obs['y']
                    yaw = obs['yaw']
                    dims = obs['dimensions']

                    if o_type in ('SPHERE', 'CYLINDER') and dims:
                        r = dims[0] if o_type == 'SPHERE' else (dims[1] if len(dims) >= 2 else dims[0])

                        c_obs = patches.Circle((x, y), r, color='red', alpha=0.5, label='_obstacle')
                        ax_map.add_patch(c_obs)
                        self._obstacle_patches.append(c_obs)

                        if margin_offset > 0.0:
                            c_margin = patches.Circle((x, y), r + margin_offset, color='red', fill=False, ls='--', lw=1.0, label='_obstacle_margin')
                            ax_map.add_patch(c_margin)
                            self._obstacle_patches.append(c_margin)

                    elif o_type == 'BOX' and len(dims) >= 2:
                        length_obs, width_obs = dims[0], dims[1]

                        rect = patches.Rectangle((-length_obs/2, -width_obs/2), length_obs, width_obs, color='red', alpha=0.5, label='_obstacle')
                        trans = transforms.Affine2D().rotate(yaw).translate(x, y) + ax_map.transData
                        rect.set_transform(trans)
                        ax_map.add_patch(rect)
                        self._obstacle_patches.append(rect)

                        if margin_offset > 0.0:
                            rect_margin = patches.Rectangle(
                                (-(length_obs/2 + margin_offset), -(width_obs/2 + margin_offset)),
                                length_obs + 2 * margin_offset,
                                width_obs + 2 * margin_offset,
                                color='red', fill=False, ls='--', lw=1.0, label='_obstacle_margin'
                            )
                            rect_margin.set_transform(trans)
                            ax_map.add_patch(rect_margin)
                            self._obstacle_patches.append(rect_margin)

            # Draw ego circles & footprint
            if self._ego_circles:
                for patch in self._ego_circles:
                    patch.remove()
                self._ego_circles = []

            if viz_ego_radius > 0.0:
                c_ego = patches.Circle((vx, vy), viz_ego_radius, color='purple', fill=False, ls='-', lw=1.0, label='_ego_radius')
                ax_map.add_patch(c_ego)
                self._ego_circles.append(c_ego)

            if viz_ego_radius > 0.0 or viz_safe_distance > 0.0:
                c_safe = patches.Circle((vx, vy), viz_ego_radius + viz_safe_distance, color='orange', fill=False, ls='-.', lw=1.0, label='_safe_distance')
                ax_map.add_patch(c_safe)
                self._ego_circles.append(c_safe)

            if self._footprint_patches:
                for patch in self._footprint_patches:
                    patch.remove()
                self._footprint_patches = []

            if footprint_poly:
                poly = patches.Polygon(footprint_poly, closed=True, color='cyan', fill=False, lw=1.5, label='_footprint')
                ax_map.add_patch(poly)
                self._footprint_patches.append(poly)
            else:
                rect_ego = patches.Rectangle((axle_offset - length/2, -width/2), length, width, color='cyan', fill=False, lw=1.5, label='_footprint')
                rect_ego.set_transform(transforms.Affine2D().rotate(vyaw).translate(vx, vy) + ax_map.transData)
                ax_map.add_patch(rect_ego)
                self._footprint_patches.append(rect_ego)

            # heading arrow — remove previous, draw new
            if self._hdg_arrow[0] is not None:
                self._hdg_arrow[0].remove()
            arrow_len = 0.4
            self._hdg_arrow[0] = ax_map.annotate(
                '', xy=(vx + math.cos(vyaw) * arrow_len,
                         vy + math.sin(vyaw) * arrow_len),
                xytext=(vx, vy),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))

            ax_map.relim()
            ax_map.autoscale_view()

            # --- speed ---
            line_spd_act.set_data(_xs(spd_act), spd_act)
            line_spd_cmd.set_data(_xs(spd_cmd), spd_cmd)
            ax_speed.relim()
            ax_speed.autoscale_view()

            # --- steer ---
            line_steer.set_data(_xs(steer), steer)
            ax_steer.relim()
            ax_steer.autoscale_view()

            # --- errors ---
            line_cte.set_data(_xs(cte), cte)
            line_hdg_err.set_data(_xs(hdg_err), hdg_err)
            ax_errors.relim()
            ax_errors.autoscale_view()
            ax_err2.relim()
            ax_err2.autoscale_view()

            # --- diag ---
            line_accel_cmd.set_data(_xs(accel), accel)
            line_solve_time.set_data(_xs(solve), solve)
            ax_diag.relim()
            ax_diag.autoscale_view()
            ax_diag2.relim()
            ax_diag2.autoscale_view()

            # --- video frame grab (last, after the canvas is updated) ---
            if self._writer is not None:
                with self._writer_lock:
                    if self._writer is not None:
                        try:
                            self._writer.grab_frame()
                        except Exception:
                            # writer torn down (shutdown) or pipe closed; stop
                            self._writer = None

        self._setup_writer(fig)

        self._ani = animation.FuncAnimation(
            fig, _animate,
            interval=100,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()  # blocks the daemon thread — intentional

    def _setup_writer(self, fig):
        """Open a MovieWriter for video_path. .gif → Pillow, else ffmpeg."""
        if not self._video_path:
            return
        try:
            is_gif = self._video_path.lower().endswith('.gif')
            if is_gif:
                writer = animation.PillowWriter(fps=self._video_fps)
            else:
                if not animation.FFMpegWriter.isAvailable():
                    print('[MatplotlibBackend] ffmpeg not found on PATH; '
                          'video recording disabled. Install ffmpeg or use a '
                          '.gif video_path.')
                    return
                writer = animation.FFMpegWriter(fps=self._video_fps, bitrate=2400)
            writer.setup(fig, self._video_path, dpi=100)
            with self._writer_lock:
                self._writer = writer
            print(f'[MatplotlibBackend] recording video to {self._video_path} '
                  f'@ {self._video_fps} fps')
        except Exception as e:
            print(f'[MatplotlibBackend] video recording disabled ({e}).')
            self._writer = None

    # ------------------------------------------------------------------
    # BaseVizBackend implementation
    # ------------------------------------------------------------------

    def log_vehicle_pose(self, x: float, y: float, yaw: float, speed: float,
                         stamp=None) -> None:
        with self._lock:
            self._vehicle_xy = (x, y)
            self._vehicle_yaw = yaw
            self._vehicle_trail.append((x, y))
            self._buf['speed_actual'].append(speed)

    def log_full_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        with self._lock:
            self._full_path = list(pts)
            self._full_path_dirty = True

    def log_predicted_path(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        with self._lock:
            self._pred_path = list(pts)

    def log_ref_window(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        with self._lock:
            self._ref_window = list(pts)

    def log_goal(self, x: float, y: float, stamp=None) -> None:
        with self._lock:
            self._goal_xy = (x, y)

    def log_commands(self, accel: float, steer_deg: float, speed: float,
                     stamp=None) -> None:
        with self._lock:
            self._buf['accel_cmd'].append(accel)
            self._buf['steer_cmd_deg'].append(steer_deg)
            self._buf['speed_cmd'].append(speed)

    def log_yaw_rate(self, yaw_rate: float) -> None:
        with self._lock:
            self._buf['yaw_rate'].append(yaw_rate)

    def log_accel_actual(self, accel: float, stamp=None) -> None:
        with self._lock:
            self._buf['accel_actual'].append(accel)

    def log_solve_time(self, solve_time_s: float) -> None:
        with self._lock:
            self._buf['solve_time_ms'].append(solve_time_s * 1e3)

    def log_errors(self, cte: float, heading_err_deg: float) -> None:
        with self._lock:
            self._buf['cte'].append(cte)
            self._buf['heading_err_deg'].append(heading_err_deg)

    def log_actuator_feedback(self, accel: float, steer_deg: float, speed: float,
                              stamp=None) -> None:
        with self._lock:
            self._buf['accel_fb'].append(accel)
            self._buf['steer_fb_deg'].append(steer_deg)
            self._buf['speed_fb'].append(speed)

    def log_reference_cmd(self, accel: float, steer_deg: float, speed: float,
                          stamp=None) -> None:
        with self._lock:
            self._buf['accel_ref'].append(accel)
            self._buf['steer_ref_deg'].append(steer_deg)
            self._buf['speed_ref'].append(speed)

    def log_obstacles(self, obstacles: List[dict], margin_offset: float = 0.0, stamp=None) -> None:
        with self._lock:
            if self._obstacles != obstacles or self._margin_offset != margin_offset:
                self._obstacles = list(obstacles)
                self._margin_offset = margin_offset
                self._obstacles_dirty = True

    def log_footprint_polygon(self, pts: List[Tuple[float, float]], stamp=None) -> None:
        with self._lock:
            self._footprint_poly = list(pts)

    def shutdown(self) -> None:
        with self._writer_lock:
            if self._writer is not None:
                try:
                    self._writer.finish()
                    print(f'[MatplotlibBackend] video saved to {self._video_path}')
                except Exception as e:
                    print(f'[MatplotlibBackend] failed to finalize video ({e}).')
                self._writer = None
        try:
            plt.close('all')
        except Exception:
            pass
