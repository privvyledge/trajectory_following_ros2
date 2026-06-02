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
