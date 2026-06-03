"""Shared constants for the Rerun trajectory visualizer."""

# ---------------------------------------------------------------------------
# Entity paths
# Hierarchy: world/ for spatial, signals/ for time-series.
# ---------------------------------------------------------------------------
ENTITY = {
    # Spatial
    'vehicle_pos':       'world/vehicle/position',
    'vehicle_heading':   'world/vehicle/heading',
    'full_path':         'world/paths/full_reference',
    'ref_window':        'world/paths/mpc_reference_window',
    'predicted':         'world/paths/mpc_predicted',
    'goal':              'world/paths/goal_point',
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
    'vehicle':    [0,   200, 255, 255],   # cyan
    'full_path':  [120, 120, 120, 160],   # dim grey
    'ref_window': [255, 200,   0, 255],   # amber
    'predicted':  [50,  230,  50, 255],   # green
    'goal':       [255,  60,  60, 255],   # red
    'commanded':  [255, 140,   0, 255],   # orange
    'reference':  [100, 120, 255, 255],   # blue
    'feedback':   [200, 100, 255, 255],   # purple
}


def ros_stamp_to_nanos(stamp) -> int:
    """Convert a ROS builtin_interfaces/Time stamp to integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def build_blueprint():
    """Explicit Rerun blueprint for the trajectory visualizer.

    Without a blueprint, Rerun auto-generates the layout: it dumps every scalar
    onto a single time-series view and (especially in the web viewer) splits the
    ``Arrows2D`` heading entity into its own spatial view, detached from the
    vehicle position and path. This pins:

    - one 2-D spatial view rooted at ``world`` (pose + heading + paths + goal
      share the same view and coordinate frame), and
    - time-series views grouped by *physical quantity* so commanded / actual /
      reference signals of the same unit overlay on one axis.

    Returns the blueprint, or ``None`` if the installed rerun has no blueprint
    API (caller then falls back to the auto-layout).
    """
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return None

    spatial = rrb.Spatial2DView(name='World', origin='world', contents='world/**')

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
        rrb.BlueprintPanel(state='collapsed'),
        rrb.SelectionPanel(state='collapsed'),
        rrb.TimePanel(state='collapsed'),
    )
