"""Regression coverage for the offline obstacle-208 restart gate."""
import csv

import phase2_gate


FIELDNAMES = [
    'ref_idx', 'ego_x', 'ego_y', 'ego_yaw', 'applied_steering', 'applied_speed',
    'avoidance_stop', 'avoidance_required_offset', 'avoidance_bound', 'sel_id',
    'physical_clearance', 'status',
]


def _write_trace(path, forward_progress=0.0, avoidance_stop=1):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for tick in range(240):
            progress = forward_progress * tick / 239.0
            writer.writerow({
                'ref_idx': 139,
                'ego_x': phase2_gate.RESTART_POSE[0],
                'ego_y': phase2_gate.RESTART_POSE[1] - progress,
                'ego_yaw': phase2_gate.RESTART_FORWARD_YAW,
                'applied_steering': phase2_gate.MAX_STEER,
                'applied_speed': -0.1,
                'avoidance_stop': avoidance_stop,
                'avoidance_required_offset': 2.4,
                'avoidance_bound': 2.0,
                'sel_id': 208,
                'physical_clearance': 0.3,
                'status': 0,
            })


def test_restart_gate_reproduces_directional_stall(tmp_path):
    trace = tmp_path / 'stalled.csv'
    _write_trace(trace)

    result = phase2_gate.classify(trace)

    assert result['restart_setup_valid'] is True
    assert result['restart_pass'] is False
    assert result['restart_tick'] is None


def test_restart_gate_accepts_forward_progress(tmp_path):
    trace = tmp_path / 'restarted.csv'
    _write_trace(trace, forward_progress=1.0)

    result = phase2_gate.classify(trace)

    assert result['restart_setup_valid'] is True
    assert result['restart_pass'] is True
    assert result['restart_tick'] is not None


def test_restart_gate_allows_a_candidate_to_avoid_the_latch(tmp_path):
    trace = tmp_path / 'wrong_scene.csv'
    _write_trace(trace, avoidance_stop=0)

    result = phase2_gate.classify(trace)

    assert result['restart_scene_valid'] is True
    assert result['restart_setup_valid'] is True
    assert result['restart_pass'] is False


def test_restart_gate_rejects_an_incomplete_window(tmp_path):
    trace = tmp_path / 'short.csv'
    _write_trace(trace)

    result = phase2_gate.classify(trace, restart_window_s=20.0)

    assert result['restart_scene_valid'] is True
    assert result['restart_window_complete'] is False
    assert result['restart_setup_valid'] is False
