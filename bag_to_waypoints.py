#!/usr/bin/env python3
"""Turn a recorded rosbag2 drive into a waypoint CSV the loader can replay.

Reads the fused odometry out of an mcap bag and writes the same 15-column CSV
``waypoint_recorder`` produces, so ``waypoint_loader`` consumes it unchanged.
Optionally writes a second, wide CSV of the actuator chain (commands, servo
position, motor ERPM, measured VESC state) resampled onto a uniform grid, for
checking that the recorded path is actually drivable.

The bag is read in a single pass with a topic filter, so the 20+ GB of camera
payload alongside the odometry is never deserialized.

A localizer's map-frame correction usually arrives as a separate, sensor-free bag
meant to be played alongside the original, because the original carries the
actuator chain and the correction carries only pose. ``--correction-bag`` merges
the two. Both routes it offers are exact rather than interpolated: the correction
is produced by re-stamping nothing, so every pose shares a header stamp
field-for-field with the odometry message it came from.

Example
-------
    python3 bag_to_waypoints.py /mnt/shared_dir/bags/20260805/loop_laps_173558 \
        --namespace gosling1 -o data/gosling1_loop_laps.csv --actuator-csv out/loop_act.csv

    # same run, drift-corrected against a map-frame bag
    python3 bag_to_waypoints.py /mnt/shared_dir/bags/20260805/loop_laps_173558 \
        --correction-bag /mnt/shared_dir/deliverables/20260805/mapframe_loop_laps_173558 \
        --namespace gosling1 -o data/gosling1_loop_laps.csv
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

# Command/feedback channels pulled for the actuator report. Each entry is
# (topic suffix under the namespace, column prefix). Missing topics are skipped.
ACTUATOR_TOPICS = [
    ('vehicle/ackermann_cmd', 'cmd'),
    ('ackermann_drive', 'drive'),
    ('teleop', 'teleop'),
    ('safety', 'safety'),
    ('vehicle/commands/motor/speed', 'motor_erpm'),
    ('vehicle/commands/servo/position', 'servo_pos'),
    ('vehicle/sensors/core', 'vesc'),
]


def yaw_from_quaternion(x, y, z, w):
    """Extract the yaw (Z) Euler angle from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def collect_map_to_odom(messages, map_frame, odom_frame):
    """Build a time-ordered map->odom transform table from /tf messages.

    Raw pre-map recordings have no map frame at all; a bag produced by a mapper
    (e.g. RTAB-Map) publishes map->odom as the drift correction. Composing it
    onto the fused odometry is what turns a drifting multi-lap recording into a
    route whose laps overlay.
    """
    times, tx, ty, yaw = [], [], [], []
    for stamp, msg in messages:
        for tf in msg.transforms:
            parent = tf.header.frame_id.lstrip('/')
            child = tf.child_frame_id.lstrip('/')
            if parent != map_frame or child != odom_frame:
                continue
            t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            q = tf.transform.rotation
            times.append(t if t > 0.0 else stamp)
            tx.append(tf.transform.translation.x)
            ty.append(tf.transform.translation.y)
            yaw.append(yaw_from_quaternion(q.x, q.y, q.z, q.w))
    if not times:
        return None
    order = np.argsort(times)
    return (np.array(times)[order], np.array(tx)[order],
            np.array(ty)[order], np.unwrap(np.array(yaw)[order]))


def apply_map_to_odom(odom, table, map_frame):
    """Compose the latest map->odom onto each odometry sample (zero-order hold).

    Only pose is transformed. The twist stays untouched: it is expressed in
    base_link, which the correction does not move.
    """
    times, tx, ty, yaw = table
    out = []
    for row in odom:
        stamp = row[0]
        i = int(np.clip(np.searchsorted(times, stamp, side='right') - 1, 0, len(times) - 1))
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        x, y = row[2], row[3]
        mx = tx[i] + c * x - s * y
        my = ty[i] + s * x + c * y
        # Rotate the orientation about Z by the correction yaw.
        base = yaw_from_quaternion(row[5], row[6], row[7], row[8]) + yaw[i]
        out.append((row[0], map_frame, mx, my, row[4],
                    0.0, 0.0, math.sin(base / 2.0), math.cos(base / 2.0),
                    row[9], row[10], row[11]))
    return out


def open_bag(uri, storage_id, topics):
    """Open a rosbag2 for sequential reading, filtered to ``topics``.

    Returns the reader plus a topic -> message-type-name map taken from the
    bag's own metadata, so no assumption about types is baked in here.
    """
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions('', ''),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in topics if t in type_map]
    missing = [t for t in topics if t not in type_map]
    if missing:
        print('  (not in bag, skipped): %s' % ', '.join(missing), file=sys.stderr)
    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    return reader, type_map, present


def read_bag(uri, storage_id, odom_topic, actuator_topics, tf_topic=None):
    """Single filtered pass over the bag. Returns (odom_rows, actuator_series, tf)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    wanted = [t for t, _ in actuator_topics]
    if odom_topic:
        wanted.append(odom_topic)
    if tf_topic:
        wanted.append(tf_topic)
    reader, type_map, present = open_bag(uri, storage_id, wanted)
    msg_classes = {t: get_message(type_map[t]) for t in present}

    odom = []
    tf = []
    series = {t: [] for t in present if t not in (odom_topic, tf_topic)}

    while reader.has_next():
        topic, data, bag_ts = reader.read_next()
        msg = deserialize_message(data, msg_classes[topic])
        # Prefer the message's own stamp; Float64 has no header, and a zero
        # stamp means the publisher never filled one in.
        stamp = bag_ts * 1e-9
        header = getattr(msg, 'header', None)
        if header is not None:
            hs = header.stamp.sec + header.stamp.nanosec * 1e-9
            if hs > 0.0:
                stamp = hs

        if topic == odom_topic:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            tw = msg.twist.twist
            odom.append((
                stamp, msg.header.frame_id,
                p.x, p.y, p.z,
                q.x, q.y, q.z, q.w,
                tw.linear.x, tw.linear.y, tw.angular.z,
            ))
        elif topic == tf_topic:
            tf.append((stamp, msg))
        else:
            series[topic].append((stamp, msg))

    return odom, series, tf


def trim_reverse_tail(odom, min_speed):
    """Drop a trailing reverse manoeuvre.

    Off by default, and it should stay off for any drive whose reverse was
    driven deliberately: reversing is a supported feature of this controller
    (signed ``vx`` reference, ``allow_reversing``, the loader's reverse-aware
    yaw recomputation), so a route that ends by backing up is a route that
    exercises it, not a recording artifact. Use this flag only when the tail is
    genuinely incidental -- e.g. the driver repositioning the car after the run
    was already over -- and you want the route to end on the last forward
    motion.

    Only the tail is trimmed -- a mid-route reverse is part of the driven line
    and is left alone.
    """
    last_forward = None
    for i in range(len(odom) - 1, -1, -1):
        if odom[i][9] > min_speed:
            last_forward = i
            break
    if last_forward is None or last_forward == len(odom) - 1:
        return odom, 0
    return odom[:last_forward + 1], len(odom) - 1 - last_forward


def build_waypoints(odom, min_distance, min_speed, trim_stationary):
    """Decimate raw odometry into waypoints.

    Leading and trailing stationary samples are dropped: a pile of near-identical
    points at the start doubles as the goal on a closed route and makes the
    controller latch "final goal reached" on lap one.
    """
    if not odom:
        raise SystemExit('no odometry messages found on the requested topic')

    arr = np.array([r[0:1] + r[2:] for r in odom], dtype=float)
    frame_id = odom[0][1]
    t, x, y, z = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    qx, qy, qz, qw = arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7]
    vx, vy, omega = arr[:, 8], arr[:, 9], arr[:, 10]

    lo, hi = 0, len(t) - 1
    if trim_stationary:
        moving = np.abs(vx) >= min_speed
        if not moving.any():
            raise SystemExit(
                'vehicle never exceeded --min-speed %.3f m/s; nothing to extract' % min_speed)
        lo, hi = int(np.argmax(moving)), int(len(moving) - 1 - np.argmax(moving[::-1]))

    keep = [lo]
    for i in range(lo + 1, hi + 1):
        j = keep[-1]
        if math.hypot(x[i] - x[j], y[i] - y[j]) >= min_distance:
            keep.append(i)
    if keep[-1] != hi:
        keep.append(hi)
    keep = np.array(keep)

    t0 = t[keep[0]]
    rows = []
    prev_t = None
    for i in keep:
        elapsed = t[i] - t0
        dt = 0.0 if prev_t is None else elapsed - prev_t
        prev_t = elapsed
        rows.append({
            'frame_id': frame_id,
            'total_time_elapsed': '%.6f' % elapsed,
            'dt': '%.6f' % dt,
            'x': repr(x[i]), 'y': repr(y[i]), 'z': repr(z[i]),
            'yaw': repr(yaw_from_quaternion(qx[i], qy[i], qz[i], qw[i])),
            'qx': repr(qx[i]), 'qy': repr(qy[i]), 'qz': repr(qz[i]), 'qw': repr(qw[i]),
            'vx': repr(vx[i]), 'vy': repr(vy[i]),
            'speed': repr(float(math.hypot(vx[i], vy[i]))),
            'omega': repr(omega[i]),
        })
    # t0 is returned so the actuator CSV can share this exact time origin: two
    # files on different origins cannot be scrubbed together in a viewer.
    return rows, (t[hi] - t[lo]), len(t), float(t0)


WAYPOINT_COLUMNS = ['frame_id', 'total_time_elapsed', 'dt', 'x', 'y', 'z', 'yaw',
                    'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'speed', 'omega']


def write_waypoints(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=WAYPOINT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def extract_channel(prefix, messages):
    """Flatten one topic's messages into {column: (times, values)}."""
    if not messages:
        return {}
    sample = messages[0][1]
    times = np.array([m[0] for m in messages], dtype=float)
    out = {}

    if hasattr(sample, 'drive'):                       # AckermannDriveStamped
        out['%s_speed' % prefix] = (times, np.array([m[1].drive.speed for m in messages]))
        out['%s_steer' % prefix] = (times, np.array([m[1].drive.steering_angle for m in messages]))
    elif hasattr(sample, 'state'):                     # VescStateStamped
        for field, column in (('speed', 'erpm'), ('voltage_input', 'voltage'),
                              ('current_motor', 'current'), ('duty_cycle', 'duty')):
            if hasattr(sample.state, field):
                out['%s_%s' % (prefix, column)] = (
                    times, np.array([getattr(m[1].state, field) for m in messages]))
    elif hasattr(sample, 'data'):                      # Float64
        out[prefix] = (times, np.array([float(m[1].data) for m in messages]))
    return out


def write_actuator_csv(path, series, prefixes, odom, rate, t_origin):
    """Zero-order-hold every channel onto one uniform grid and write it wide."""
    channels = {}
    for topic, messages in series.items():
        channels.update(extract_channel(prefixes[topic], messages))

    if odom:
        ot = np.array([r[0] for r in odom], dtype=float)
        channels['odom_x'] = (ot, np.array([r[2] for r in odom]))
        channels['odom_y'] = (ot, np.array([r[3] for r in odom]))
        channels['odom_yaw'] = (ot, np.array(
            [yaw_from_quaternion(r[5], r[6], r[7], r[8]) for r in odom]))
        channels['odom_vx'] = (ot, np.array([r[9] for r in odom]))
        channels['odom_omega'] = (ot, np.array([r[11] for r in odom]))

    if not channels:
        return 0

    t_start = max(v[0][0] for v in channels.values())
    t_end = min(v[0][-1] for v in channels.values())
    if t_end <= t_start:
        raise SystemExit('actuator channels do not overlap in time')
    grid = np.arange(t_start, t_end, 1.0 / rate)

    columns = sorted(channels)
    table = np.empty((len(grid), len(columns) + 1))
    # Shared origin with the waypoint CSV, so t < 0 is the pre-drive dwell.
    table[:, 0] = grid - t_origin
    for k, name in enumerate(columns, start=1):
        times, values = channels[name]
        # Zero-order hold: index of the newest sample at or before each grid point.
        idx = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1)
        table[:, k] = values[idx]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    np.savetxt(path, table, delimiter=',', fmt='%.6f',
               header=','.join(['t'] + columns), comments='')
    return len(grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='rosbag2 directory')
    parser.add_argument('-o', '--output', required=True, help='waypoint CSV to write')
    parser.add_argument('--namespace', default='gosling1',
                        help='topic namespace (default: gosling1; "" for none)')
    parser.add_argument('--odom-topic', default=None,
                        help='override the fused-odometry topic (default: <ns>/odometry/local)')
    parser.add_argument('--storage', default='mcap', help='rosbag2 storage id (default: mcap)')
    parser.add_argument('--map-frame', default=None, metavar='FRAME',
                        help='compose the <FRAME>-><odom-frame> transform from the bag TF onto '
                             'every pose, e.g. --map-frame map for a mapper-corrected bag. '
                             'Raw pre-map recordings have no such transform and must omit this.')
    parser.add_argument('--odom-frame', default='odom',
                        help='child frame of the map correction (default: odom)')
    parser.add_argument('--tf-topic', default=None,
                        help='TF topic carrying the correction (default: <ns>/tf)')
    parser.add_argument('--correction-bag', default=None, metavar='BAG',
                        help='second, sensor-free bag carrying a localizer map-frame '
                             'correction for this same run. Alone, its already-composed '
                             'pose topic (--pose-topic) is used directly; add --map-frame '
                             'to instead take only its <FRAME>-><odom-frame> transform and '
                             'compose that onto the primary bag odometry. Either way the '
                             'actuator chain still comes from the primary bag.')
    parser.add_argument('--pose-topic', default=None,
                        help='composed map-frame pose topic in the correction bag '
                             '(default: <ns>/pose_map)')
    parser.add_argument('--min-distance', type=float, default=0.03,
                        help='minimum spacing between kept waypoints, metres (default: 0.03)')
    parser.add_argument('--min-speed', type=float, default=0.05,
                        help='speed below which a sample counts as stationary (default: 0.05)')
    parser.add_argument('--keep-stationary', action='store_true',
                        help='do not trim the stationary head/tail of the run')
    parser.add_argument('--trim-reverse-tail', action='store_true',
                        help='drop a trailing reverse manoeuvre, so the route ends on the '
                             'last forward motion instead of backing up at the goal')
    parser.add_argument('--actuator-csv', default=None,
                        help='also write a wide actuator CSV here')
    parser.add_argument('--actuator-rate', type=float, default=50.0,
                        help='resample rate for the actuator CSV, Hz (default: 50)')
    args = parser.parse_args()

    ns = args.namespace.strip('/')
    prefix = ('/%s/' % ns) if ns else '/'
    odom_topic = args.odom_topic or (prefix + 'odometry/local')
    actuator = [(prefix + suffix, name) for suffix, name in ACTUATOR_TOPICS]
    prefixes = dict(actuator)

    tf_topic = None
    if args.map_frame:
        tf_topic = args.tf_topic or (prefix + 'tf')

    print('reading %s' % args.bag)
    if args.correction_bag:
        # The correction bag holds no actuator data and the primary bag holds no
        # map frame, so each is read for only what it owns.
        if args.map_frame:
            # Take the transform from the correction bag and compose it onto the
            # primary bag's odometry.
            odom, series, _ = read_bag(args.bag, args.storage, odom_topic, actuator, None)
            _, _, tf = read_bag(args.correction_bag, args.storage, None, [], tf_topic)
        else:
            # Use the correction bag's already-composed pose. Its twist is the
            # source odometry's, untouched -- body-frame, so the map correction
            # does not apply to it -- which is what vx/speed need.
            pose_topic = args.pose_topic or (prefix + 'pose_map')
            _, series, _ = read_bag(args.bag, args.storage, None, actuator, None)
            print('reading %s' % args.correction_bag)
            odom, _, tf = read_bag(args.correction_bag, args.storage, pose_topic, [], None)
            if not odom:
                raise SystemExit(
                    'no messages on %s in %s; pass --map-frame to use its TF instead, '
                    'or --pose-topic to name the composed pose topic'
                    % (pose_topic, args.correction_bag))
            print('  map-frame pose   : %d samples from %s' % (len(odom), pose_topic))
            # A localizer that publishes pose only leaves twist at zero, which
            # yields a route of zero reference speeds -- structurally valid and
            # silently undrivable. AMCL's pose_map copied twist from
            # odometry/local; do not assume a later localizer does the same.
            if max(abs(r[9]) for r in odom) < 1e-9:
                raise SystemExit(
                    '%s in %s carries no twist (vx identically zero); the '
                    'waypoint vx/speed columns would all be zero. Re-run with '
                    "--map-frame map to compose that bag's map->odom onto the "
                    "primary bag's odometry instead, which keeps its twist."
                    % (pose_topic, args.correction_bag))
    else:
        odom, series, tf = read_bag(args.bag, args.storage, odom_topic, actuator, tf_topic)

    if args.map_frame:
        table = collect_map_to_odom(tf, args.map_frame, args.odom_frame)
        if table is None:
            raise SystemExit(
                'no %s -> %s transform on %s; this bag is not mapper-corrected'
                % (args.map_frame, args.odom_frame,
                   args.correction_bag or args.bag))
        odom = apply_map_to_odom(odom, table, args.map_frame)
        drift = float(np.hypot(table[1], table[2]).max())
        print('  map correction   : %d transforms, max |translation| %.3f m'
              % (len(table[0]), drift))

    if args.trim_reverse_tail:
        odom, dropped = trim_reverse_tail(odom, args.min_speed)
        print('  reverse tail     : dropped %d odometry samples' % dropped)

    rows, duration, raw_count, t_origin = build_waypoints(
        odom, args.min_distance, args.min_speed, not args.keep_stationary)
    write_waypoints(args.output, rows)

    xs = np.array([float(r['x']) for r in rows])
    ys = np.array([float(r['y']) for r in rows])
    length = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
    gap = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    speeds = np.array([float(r['vx']) for r in rows])

    print('  odometry samples : %d -> %d waypoints' % (raw_count, len(rows)))
    print('  moving duration  : %.1f s' % duration)
    print('  path length      : %.2f m' % length)
    print('  start/end gap    : %.3f m  (%s)'
          % (gap, 'closed' if gap < 0.5 else 'open'))
    print('  vx range         : %.2f .. %.2f m/s (mean |vx| %.2f)'
          % (speeds.min(), speeds.max(), np.abs(speeds).mean()))
    print('  wrote %s' % args.output)

    if args.actuator_csv:
        n = write_actuator_csv(args.actuator_csv, series, prefixes, odom,
                               args.actuator_rate, t_origin)
        print('  wrote %s (%d rows @ %.0f Hz)' % (args.actuator_csv, n, args.actuator_rate))


if __name__ == '__main__':
    main()
