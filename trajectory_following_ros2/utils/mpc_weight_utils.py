"""Bryson's rule weight computation for MPC cost matrices."""
import numpy as np


def bryson_weights(
    max_state_errors: dict,
    max_inputs: dict,
    max_input_rates: dict = None,
) -> tuple:
    """Diagonal Q, R (and optionally Rd) via Bryson's rule: weight_ii = 1 / e_max_i².

    Args:
        max_state_errors: {'x', 'y', 'v', 'psi'} — max acceptable state error (m, m, m/s, rad).
        max_inputs: {'a', 'delta'} — max acceptable input magnitude (m/s², rad).
        max_input_rates: optional {'jerk', 'steer_rate'} — max acceptable input rate (m/s³, rad/s).

    Returns:
        (Q, R, Rd) where Q is (4×4), R is (2×2), Rd is (2×2) or None.
    """
    Q = np.diag([
        1.0 / max_state_errors['x'] ** 2,
        1.0 / max_state_errors['y'] ** 2,
        1.0 / max_state_errors['v'] ** 2,
        1.0 / max_state_errors['psi'] ** 2,
    ])
    R = np.diag([
        1.0 / max_inputs['a'] ** 2,
        1.0 / max_inputs['delta'] ** 2,
    ])
    Rd = None
    if max_input_rates is not None:
        Rd = np.diag([
            1.0 / max_input_rates['jerk'] ** 2,
            1.0 / max_input_rates['steer_rate'] ** 2,
        ])
    return Q, R, Rd