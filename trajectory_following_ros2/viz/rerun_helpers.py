"""Shared constants for the Rerun trajectory visualizer."""

# ---------------------------------------------------------------------------
# Entity paths
# Hierarchy: world/ for spatial, signals/ for time-series.
# ---------------------------------------------------------------------------
ENTITY = {
    # Spatial
    'world_origin':      'world/frame/origin',
    'world_axes':        'world/frame/axes',
    'vehicle_pos':       'world/vehicle/position',
    'vehicle_heading':   'world/vehicle/heading',
    'vehicle_axes':      'world/vehicle/axes',
    'full_path':         'world/paths/full_reference',
    'ref_window':        'world/paths/mpc_reference_window',
    'predicted':         'world/paths/mpc_predicted',
    'goal':              'world/paths/goal_point',
    'obstacles_circles':         'world/obstacles/circles/actual',
    'obstacles_margin_circles':  'world/obstacles/circles/margin',
    'obstacles_boxes':           'world/obstacles/boxes/actual',
    'obstacles_margin_boxes':    'world/obstacles/boxes/margin',
    'vehicle_footprint':         'world/vehicle/footprint',
    'ego_radius':                'world/vehicle/ego_radius',
    'safe_distance':             'world/vehicle/safe_distance',
    # State
    'speed_actual':      'signals/state/speed_mps',
    'heading_deg':       'signals/state/heading_deg',
    'accel_actual':      'signals/state/accel_measured_mps2',
    # Errors
    'cross_track_error': 'signals/errors/cross_track_m',
    'heading_error_deg': 'signals/errors/heading_deg',
    # Commanded actions (from AckermannDriveStamped)
    'accel_commanded':       'signals/actions/commanded/accel_mps2',
    'steer_commanded_deg':   'signals/actions/commanded/steer_deg',
    'speed_commanded':       'signals/actions/commanded/speed_mps',
    # MPC derived outputs
    'yaw_rate_desired':      'signals/actions/commanded/yaw_rate_rads',
    # Actuator feedback — actual values echoed by the hardware (optional)
    'steer_actual_deg':      'signals/actions/actuator_feedback/steer_deg',
    'speed_actual_fb':       'signals/actions/actuator_feedback/speed_mps',
    'accel_actual_fb':       'signals/actions/actuator_feedback/accel_mps2',
    # Reference / dataset commands (optional — e.g. driver recording)
    'accel_reference':       'signals/actions/reference/accel_mps2',
    'steer_reference_deg':   'signals/actions/reference/steer_deg',
    'speed_reference':       'signals/actions/reference/speed_mps',
    # Solver diagnostics
    'solve_time_ms':         'signals/solver/solve_time_ms',
}

# ---------------------------------------------------------------------------
# Colors — RGBA uint8
# ---------------------------------------------------------------------------
COLORS = {
    'origin':     [230, 230, 230, 255],   # white-ish
    'axis_x':     [230,  40,  40, 255],   # red   — ROS +X (forward)
    'axis_y':     [40,  220,  40, 255],   # green — ROS +Y (left)
    'axis_z':     [60,  120, 255, 255],   # blue  — ROS +Z (up)
    'vehicle':    [0,   200, 255, 255],   # cyan
    'full_path':  [120, 120, 120, 160],   # dim grey
    'ref_window': [255, 200,   0, 255],   # amber
    'predicted':  [50,  230,  50, 255],   # green
    'goal':       [255,  60,  60, 255],   # red
    'commanded':  [255, 140,   0, 255],   # orange
    'reference':  [100, 120, 255, 255],   # blue
    'feedback':   [200, 100, 255, 255],   # purple
    'obstacle':          [255,   0,   0, 255],   # solid red — the body itself
    # Keep-out outlines. Each must be a distinct *hue* from the shape it
    # surrounds: the margin used to be translucent red like the obstacle, so
    # only the (larger) margin was visible and the two read as one object.
    'obstacle_margin':   [255, 200,  40, 255],   # yellow  — obstacle + keep-out
    'ego_footprint':     [0,   255, 150, 255],   # bright mint green
    'ego_radius':        [190,  90, 255, 255],   # purple  — ego inflation
    'safe_distance':     [255, 140,   0, 255],   # orange  — ego + safe distance
}


def ros_stamp_to_nanos(stamp) -> int:
    """Convert a ROS builtin_interfaces/Time stamp to integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def build_blueprint():
    """Explicit Rerun blueprint for the trajectory visualizer.

    Without a blueprint, Rerun auto-generates the layout: it dumps every scalar
    onto a single time-series view and (especially in the web viewer) splits the
    heading-arrow entity into its own spatial view, detached from the
    vehicle position and path. This pins:

    - one 3-D spatial view rooted at ``world`` (pose + heading + paths + goal +
      footprint prism share the same view and coordinate frame). It is a *3-D*
      view, so every spatial entity must be logged with a 3-D archetype —
      ``rerun_backend`` logs ground-plane data at z = 0 in the ROS frame, with
      ViewCoordinates RIGHT_HAND_Z_UP on ``world`` to orient the camera; and
    - time-series views grouped by *physical quantity* so commanded / actual /
      reference signals of the same unit overlay on one axis.

    Returns the blueprint, or ``None`` if the installed rerun has no blueprint
    API (caller then falls back to the auto-layout).
    """
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return None

    spatial = rrb.Spatial3DView(name='World', origin='world', contents='world/**')

    speed = rrb.TimeSeriesView(name='Speed (m/s)', contents=[
        ENTITY['speed_actual'],
        ENTITY['speed_commanded'],
        ENTITY['speed_actual_fb'],
        ENTITY['speed_reference'],
    ])
    accel = rrb.TimeSeriesView(name='Acceleration (m/s²)', contents=[
        ENTITY['accel_actual'],
        ENTITY['accel_commanded'],
        ENTITY['accel_actual_fb'],
        ENTITY['accel_reference'],
    ])
    steer = rrb.TimeSeriesView(name='Steering (deg)', contents=[
        ENTITY['steer_commanded_deg'],
        ENTITY['steer_actual_deg'],
        ENTITY['steer_reference_deg'],
    ])
    heading = rrb.TimeSeriesView(name='Heading & yaw rate', contents=[
        ENTITY['heading_deg'],
        ENTITY['yaw_rate_desired'],
    ])
    errors = rrb.TimeSeriesView(name='Tracking errors', contents=[
        ENTITY['cross_track_error'],
        ENTITY['heading_error_deg'],
    ])
    solver = rrb.TimeSeriesView(name='Solver', contents=[
        ENTITY['solve_time_ms'],
    ])

    signals = rrb.Grid(speed, accel, steer, heading, errors, solver)

    return rrb.Blueprint(
        rrb.Horizontal(spatial, signals, column_shares=[1.0, 1.0]),
        # 'collapsed' hides the entity/attribute toggles outright in the web
        # viewer; 'hidden' would be worse. Left expandable so the entity tree
        # and per-entity visibility toggles stay reachable.
        rrb.BlueprintPanel(state='expanded'),
        rrb.SelectionPanel(state='collapsed'),
        rrb.TimePanel(state='collapsed'),
    )
