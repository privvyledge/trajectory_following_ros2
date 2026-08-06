#!/usr/bin/env python3
"""Check that an extracted drive is a usable reference, and plot it.

Takes the two CSVs ``bag_to_waypoints.py`` writes and answers the questions that
decide whether a recorded run can be handed to the controller as a route:

* Is the geometry followable? Curvature is measured against **arc length**, not
  sample index -- dense or unevenly spaced waypoints fake enormous curvature
  under a plain index gradient. The implied steering ``atan(L * kappa)`` is then
  compared against the steering limit.
* Does the recorded command chain agree with itself? Commanded steering is
  compared against the servo position converted back through the VESC gains, and
  commanded speed against the motor ERPM command and the measured VESC ERPM.
  A disagreement means the CSV's speed column does not describe what the car did.

Writes PNGs and prints a numeric summary. With ``--rerun-serve`` it also streams
the path and the time series to a Rerun web viewer.
"""

import argparse
import math
import os
import socket

import numpy as np
import pandas as pd

# VESC calibration, from f1tenth_launch/config/vehicle/vesc.yaml.
SERVO_GAIN = -1.4
SERVO_OFFSET = 0.56
SPEED_TO_ERPM_GAIN = 3750.0


def resample_by_arclength(x, y, spacing):
    """Interpolate a path onto uniform arc-length spacing.

    Curvature from unevenly spaced samples is dominated by the spacing, not the
    geometry, so every curvature number below is computed on this grid.
    """
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    s, x, y = s[keep], x[keep], y[keep]
    grid = np.arange(0.0, s[-1], spacing)
    return grid, np.interp(grid, s, x), np.interp(grid, s, y)


def curvature(s, x, y):
    """Signed curvature of a uniformly arc-length-sampled path."""
    dx, dy = np.gradient(x, s), np.gradient(y, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    denom = np.power(dx * dx + dy * dy, 1.5)
    return np.where(denom > 1e-12, (dx * ddy - dy * ddx) / np.maximum(denom, 1e-12), 0.0)


def apply_loader_smoothing(way):
    """Smooth x/y exactly the way waypoint_loader does at runtime.

    The controller never sees the raw CSV -- waypoint_loader B-splines it first.
    Judging followability on the raw points therefore measures a path that is
    never driven; this reproduces the runtime geometry instead of approximating
    it, so the two cannot drift apart.
    """
    from trajectory_following_ros2.utils import filters

    smoothed = filters.smooth_and_interpolate_coordinates(
        coordinates=way[['x', 'y']].to_numpy(), method='bspline',
        polynomial_order=3, weight_smooth=0.3)
    out = way.iloc[:len(smoothed)].copy()
    out[['x', 'y']] = smoothed
    return out


def geometry_report(way, wheelbase, max_steer, spacing, smooth_window):
    x, y = way['x'].to_numpy(), way['y'].to_numpy()
    s, xr, yr = resample_by_arclength(x, y, spacing)
    kappa = curvature(s, xr, yr)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        kappa = np.convolve(kappa, kernel, mode='same')

    steer = np.arctan(wheelbase * kappa)
    over = np.abs(steer) > max_steer
    gap = math.hypot(x[-1] - x[0], y[-1] - y[0])

    stats = {
        'waypoints': len(x),
        'path_length_m': float(s[-1]),
        'closure_gap_m': gap,
        'closed_loop': gap < 0.5,
        'mean_spacing_m': float(np.mean(np.hypot(np.diff(x), np.diff(y)))),
        'max_abs_curvature_1pm': float(np.abs(kappa).max()),
        'min_turn_radius_m': float(1.0 / max(np.abs(kappa).max(), 1e-9)),
        'max_implied_steer_deg': float(np.degrees(np.abs(steer).max())),
        'p99_implied_steer_deg': float(np.degrees(np.percentile(np.abs(steer), 99))),
        'frac_over_steer_limit': float(over.mean()),
    }
    return stats, s, xr, yr, kappa, steer


def actuator_report(act):
    """Cross-check the recorded command chain against itself."""
    stats = {}
    have = set(act.columns)

    if {'cmd_steer', 'servo_pos'} <= have:
        # Invert servo = gain * steer + offset.
        servo_steer = (act['servo_pos'] - SERVO_OFFSET) / SERVO_GAIN
        err = servo_steer - act['cmd_steer']
        stats['steer_cmd_vs_servo_rms_deg'] = float(np.degrees(np.sqrt((err ** 2).mean())))
        stats['steer_cmd_vs_servo_max_deg'] = float(np.degrees(np.abs(err).max()))
        act['servo_steer'] = servo_steer

    if {'cmd_speed', 'motor_erpm'} <= have:
        erpm_speed = act['motor_erpm'] / SPEED_TO_ERPM_GAIN
        err = erpm_speed - act['cmd_speed']
        stats['speed_cmd_vs_erpm_rms_mps'] = float(np.sqrt((err ** 2).mean()))
        act['erpm_speed'] = erpm_speed

    if {'vesc_erpm', 'odom_vx'} <= have:
        meas = act['vesc_erpm'] / SPEED_TO_ERPM_GAIN
        act['vesc_speed'] = meas
        stats['meas_speed_vs_odom_rms_mps'] = float(
            np.sqrt(((meas - act['odom_vx']) ** 2).mean()))

    for col, label in (('cmd_speed', 'cmd_speed'), ('odom_vx', 'odom_vx'),
                       ('cmd_steer', 'cmd_steer')):
        if col in have:
            stats['max_abs_%s' % label] = float(act[col].abs().max())

    if 'vesc_voltage' in have:
        stats['vesc_voltage_min_v'] = float(act['vesc_voltage'].min())
    return stats


def make_plots(name, outdir, way, stats, s, steer, max_steer, act):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    written = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(way['x'], way['y'], c=way['vx'], s=6, cmap='viridis')
    axes[0].plot(way['x'].iloc[0], way['y'].iloc[0], 'go', ms=10, label='start')
    axes[0].plot(way['x'].iloc[-1], way['y'].iloc[-1], 'rx', ms=10, label='end')
    axes[0].set_aspect('equal')
    axes[0].set_title('%s path (%.1f m, gap %.2f m)'
                      % (name, stats['path_length_m'], stats['closure_gap_m']))
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    fig.colorbar(sc, ax=axes[0], label='vx [m/s]')

    axes[1].plot(s, np.degrees(steer), lw=1)
    axes[1].axhline(np.degrees(max_steer), color='r', ls='--', label='steer limit')
    axes[1].axhline(-np.degrees(max_steer), color='r', ls='--')
    axes[1].set_title('implied steering atan(L*kappa) vs arc length')
    axes[1].set_xlabel('arc length [m]')
    axes[1].set_ylabel('steering [deg]')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, '%s_path.png' % name)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)

    if act is not None and len(act):
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for col, label in (('cmd_speed', 'commanded'), ('erpm_speed', 'motor ERPM cmd'),
                           ('vesc_speed', 'VESC measured'), ('odom_vx', 'odom vx')):
            if col in act:
                axes[0].plot(act['t'], act[col], lw=1, label=label)
        axes[0].set_ylabel('speed [m/s]')
        axes[0].legend(ncol=4)
        axes[0].grid(alpha=0.3)
        axes[0].set_title('%s actuator chain' % name)

        for col, label in (('cmd_steer', 'commanded'), ('servo_steer', 'servo position'),
                           ('teleop_steer', 'teleop')):
            if col in act:
                axes[1].plot(act['t'], np.degrees(act[col]), lw=1, label=label)
        axes[1].axhline(np.degrees(max_steer), color='r', ls='--', lw=0.8)
        axes[1].axhline(-np.degrees(max_steer), color='r', ls='--', lw=0.8)
        axes[1].set_ylabel('steering [deg]')
        axes[1].legend(ncol=3)
        axes[1].grid(alpha=0.3)

        for col, label in (('vesc_current', 'motor current [A]'),
                           ('vesc_voltage', 'input voltage [V]')):
            if col in act:
                axes[2].plot(act['t'], act[col], lw=1, label=label)
        axes[2].set_ylabel('VESC')
        axes[2].set_xlabel('time [s]')
        axes[2].legend(ncol=2)
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, '%s_actuators.png' % name)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def host_ip():
    """Best-effort primary IPv4 of this host, for printing the viewer URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def circle(cx, cy, radius, segments=48):
    """Closed ground-plane circle as a 3-D line strip at z = 0."""
    a = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    return np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a),
                            np.zeros_like(a)])


def footprint(x, y, yaw, length, width, rear_overhang):
    """Vehicle footprint rectangle at a pose, about the rear-axle reference point."""
    back, front = -rear_overhang, length - rear_overhang
    half = width / 2.0
    local = np.array([[back, -half], [front, -half], [front, half], [back, half],
                      [back, -half]])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = local @ rot.T + np.array([x, y])
    return np.column_stack([world, np.zeros(len(world))])


def build_blueprint(name):
    """Pin the layout so the world view and the time series are both visible.

    Without an explicit blueprint the auto-layout buries the spatial view among
    one plot per scalar entity, which reads as "the viewer only shows curves".
    """
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='/world', name='%s route' % name),
            rrb.Vertical(
                rrb.TimeSeriesView(origin='/actuators/speed', name='speed'),
                rrb.TimeSeriesView(origin='/actuators/steer', name='steering'),
                rrb.TimeSeriesView(origin='/geometry', name='path geometry'),
            ),
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state='collapsed'),
        collapse_panels=True,
    )


def log_to_rerun(name, way, act, arclength, kappa, steer, web_port, ws_port, ego):
    """Stream the route, the ego geometry and the actuator series to Rerun.

    The world view is 3-D, and a Rerun 3-D view silently drops 2-D archetypes
    (no warning, just an empty world), so every ground-plane entity is logged at
    z = 0 as a 3-D archetype.
    """
    import rerun as rr

    rr.init('bag_run_%s' % name)
    rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port,
                 default_blueprint=build_blueprint(name))

    xyz = np.column_stack([way['x'], way['y'], np.zeros(len(way))])
    rr.log('world/path', rr.LineStrips3D([xyz], colors=[0, 160, 255], radii=0.015),
           static=True)
    rr.log('world/waypoints', rr.Points3D(xyz, colors=[220, 220, 220], radii=0.008),
           static=True)
    rr.log('world/start', rr.Points3D(xyz[:1], colors=[0, 255, 0], radii=0.06), static=True)
    rr.log('world/end', rr.Points3D(xyz[-1:], colors=[255, 0, 0], radii=0.06), static=True)

    # Reverse segments are what make a recorded drive unusable as a route, so
    # give them their own entity rather than burying them in the path colour.
    reverse = way['vx'].to_numpy() < -0.05
    if reverse.any():
        rr.log('world/reverse', rr.Points3D(xyz[reverse], colors=[255, 80, 255], radii=0.02),
               static=True)

    # --- ego geometry, animated along the route -----------------------------
    # The keep-out drawn here must equal the one the controller enforces:
    # ego_radius + safe_distance per collision disc, at the disc offsets, which
    # is why these come from the platform config rather than being invented.
    t_route = way['total_time_elapsed'].to_numpy()
    yaws = way['yaw'].to_numpy()
    for i in range(len(way)):
        rr.set_time_seconds('run_time', float(t_route[i]))
        x, y, yaw = float(way['x'].iloc[i]), float(way['y'].iloc[i]), float(yaws[i])
        rr.log('world/ego/footprint',
               rr.LineStrips3D([footprint(x, y, yaw, ego['length'], ego['width'],
                                          ego['rear_overhang'])],
                               colors=[255, 210, 0], radii=0.008))
        discs, keepouts = [], []
        for offset in ego['disc_offsets']:
            cx = x + offset * math.cos(yaw)
            cy = y + offset * math.sin(yaw)
            discs.append(circle(cx, cy, ego['radius']))
            keepouts.append(circle(cx, cy, ego['radius'] + ego['safe_distance']))
        rr.log('world/ego/discs', rr.LineStrips3D(discs, colors=[0, 220, 120], radii=0.005))
        rr.log('world/ego/keepout', rr.LineStrips3D(keepouts, colors=[255, 90, 90],
                                                    radii=0.004))

    # Columnar sends: a per-sample log loop over ~7k rows x 9 channels is slow
    # enough to dominate the run.
    rr.send_columns(
        'geometry/implied_steer_deg',
        indexes=[rr.TimeSecondsColumn('arclength_m', arclength)],
        columns=rr.Scalar.columns(scalar=np.degrees(steer)),
    )
    rr.send_columns(
        'geometry/curvature_1pm',
        indexes=[rr.TimeSecondsColumn('arclength_m', arclength)],
        columns=rr.Scalar.columns(scalar=kappa),
    )

    if act is not None and len(act):
        times = act['t'].to_numpy()
        # Grouped so the blueprint can show speed and steering as two plots
        # rather than one plot per scalar.
        groups = {
            'speed': ('cmd_speed', 'erpm_speed', 'vesc_speed', 'odom_vx'),
            'steer': ('cmd_steer', 'servo_steer', 'teleop_steer'),
            'vesc': ('vesc_current', 'vesc_voltage'),
        }
        for group, cols in groups.items():
            for col in cols:
                if col in act:
                    values = act[col].to_numpy()
                    if group == 'steer':
                        values = np.degrees(values)
                    rr.send_columns(
                        'actuators/%s/%s' % (group, col),
                        indexes=[rr.TimeSecondsColumn('run_time', times)],
                        columns=rr.Scalar.columns(scalar=values),
                    )

    ip = host_ip()
    print('\nRerun web viewer:')
    print('  http://%s:%d/?url=ws://%s:%d' % (ip, web_port, ip, ws_port))
    print('  (the localhost URL rerun prints itself is not reachable from another machine)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('waypoints', help='waypoint CSV from bag_to_waypoints.py')
    parser.add_argument('--actuator-csv', default=None)
    parser.add_argument('--name', default=None, help='label for outputs (default: CSV stem)')
    parser.add_argument('--outdir', default='out/bag_report')
    parser.add_argument('--wheelbase', type=float, default=0.256)
    parser.add_argument('--max-steer-deg', type=float, default=27.0)
    parser.add_argument('--spacing', type=float, default=0.05,
                        help='arc-length resample spacing for curvature, m (default: 0.05)')
    parser.add_argument('--curvature-smooth', type=int, default=9,
                        help='boxcar width over the arc-length grid (default: 9)')
    parser.add_argument('--loader-smoothing', action='store_true',
                        help="apply waypoint_loader's B-spline smoothing before measuring, "
                             'i.e. report the geometry the controller actually tracks')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--rerun-serve', action='store_true')
    # Ego collision geometry for the viewer. Defaults are the f1tenth platform
    # values (config/platforms/f1tenth.yaml) so drawn == enforced; --platform-yaml
    # reads them from the file instead, which is what to use once the vehicle is
    # recalibrated rather than editing numbers here.
    parser.add_argument('--platform-yaml', default=None,
                        help='read ego_radius / ego_disc_offsets / safe_distance from this '
                             'platform YAML instead of the built-in f1tenth defaults')
    parser.add_argument('--ego-length', type=float, default=0.58)
    parser.add_argument('--ego-width', type=float, default=0.31)
    parser.add_argument('--ego-rear-overhang', type=float, default=0.19,
                        help='rear-axle reference point to the tail, m (default: 0.19)')
    parser.add_argument('--web-port', type=int, default=9090)
    parser.add_argument('--ws-port', type=int, default=9877)
    args = parser.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.waypoints))[0]
    way = pd.read_csv(args.waypoints, skipinitialspace=True)
    act = None
    if args.actuator_csv and os.path.exists(args.actuator_csv):
        act = pd.read_csv(args.actuator_csv)

    if args.loader_smoothing:
        way = apply_loader_smoothing(way)

    max_steer = math.radians(args.max_steer_deg)
    stats, s, xr, yr, kappa, steer = geometry_report(
        way, args.wheelbase, max_steer, args.spacing, args.curvature_smooth)
    if act is not None:
        stats.update(actuator_report(act))

    print('=== %s ===' % name)
    for key, value in stats.items():
        print('  %-30s %s' % (key, ('%.4f' % value) if isinstance(value, float) else value))

    if not args.no_plots:
        for path in make_plots(name, args.outdir, way, stats, s, steer, max_steer, act):
            print('  plot: %s' % path)

    if args.rerun_serve:
        ego = {'radius': 0.212, 'disc_offsets': [0.045, 0.335], 'safe_distance': 0.15,
               'length': args.ego_length, 'width': args.ego_width,
               'rear_overhang': args.ego_rear_overhang}
        if args.platform_yaml:
            import yaml
            params = yaml.safe_load(open(args.platform_yaml))['/**']['ros__parameters']
            ego['radius'] = float(params.get('ego_radius', ego['radius']))
            ego['disc_offsets'] = list(params.get('ego_disc_offsets', ego['disc_offsets']))
            ego['safe_distance'] = float(params.get('safe_distance', ego['safe_distance']))
        print('  ego: radius %.3f m, discs %s, safe_distance %.3f m'
              % (ego['radius'], ego['disc_offsets'], ego['safe_distance']))
        log_to_rerun(name, way, act, s, kappa, steer, args.web_port, args.ws_port, ego)
        print('\nserving; Ctrl-C to stop')
        try:
            while True:
                import time
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
