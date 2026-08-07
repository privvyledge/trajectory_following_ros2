"""Regression coverage for the offline obstacle-208 restart gate."""
import csv

import phase2_gate
import pytest


FIELDNAMES = [
    'ref_idx', 'ego_x', 'ego_y', 'ego_yaw', 'applied_steering', 'applied_speed',
    'avoidance_stop', 'avoidance_required_offset', 'avoidance_bound', 'sel_id',
    'n_selected', 'physical_clearance', 'forward_escape_active', 'status',
]


def _write_trace(path, forward_progress=0.0, avoidance_stop=1,
                 selected_share=1.0, bound_share=1.0, dropout_tick=None,
                 escape_active_until=0):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for tick in range(240):
            progress = forward_progress * tick / 239.0
            selected = tick < 240 * selected_share and tick != dropout_tick
            writer.writerow({
                'ref_idx': 139,
                'ego_x': phase2_gate.RESTART_POSE[0],
                'ego_y': phase2_gate.RESTART_POSE[1] - progress,
                'ego_yaw': phase2_gate.RESTART_FORWARD_YAW,
                'applied_steering': phase2_gate.MAX_STEER,
                'applied_speed': -0.1,
                'avoidance_stop': avoidance_stop,
                'avoidance_required_offset': 2.4 if tick < 240 * bound_share else 1.0,
                'avoidance_bound': 2.0,
                'sel_id': 208 if selected else -1,
                'n_selected': 6 if selected else 3,
                'physical_clearance': 0.3,
                'forward_escape_active': int(tick < escape_active_until),
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


@pytest.mark.parametrize('share, expected', [(0.49, False), (0.5, True)])
def test_restart_gate_requires_sustained_bound_evidence(tmp_path, share, expected):
    trace = tmp_path / 'scene.csv'
    _write_trace(trace, bound_share=share)

    result = phase2_gate.classify(trace)

    assert result['restart_scene_valid'] is expected


def test_restart_gate_marks_one_tick_selection_dropout_invalid(tmp_path):
    trace = tmp_path / 'dropout.csv'
    _write_trace(trace, forward_progress=1.0, dropout_tick=80,
                 escape_active_until=240)

    result = phase2_gate.classify(trace, escape_arm='on')

    assert result['restart_selection_dropout_ticks'] == 1
    assert result['restart_scene_integrity'] is False
    assert result['restart_setup_valid'] is False
    assert result['restart_pass'] is False


def test_restart_gate_rejects_progress_reached_only_after_escape_aborts(tmp_path):
    trace = tmp_path / 'post_abort.csv'
    _write_trace(trace, forward_progress=1.0, escape_active_until=100)

    result = phase2_gate.classify(trace, escape_arm='on')

    assert result['restart_tick'] > 100
    assert result['restart_escape_attributed'] is False
    assert result['restart_setup_valid'] is True
    assert result['restart_pass'] is False


def test_gate_glob_skips_waypoint_csv(tmp_path, capsys):
    trace = tmp_path / 'solver.csv'
    _write_trace(trace, forward_progress=1.0)
    route = tmp_path / 'route_from_120.csv'
    route.write_text('x,y,yaw,vx\n0,0,0,0\n')

    rc = phase2_gate.main([
        str(tmp_path / '*.csv'), '--target', 'restart-208',
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert 'Skipping' in captured.err
    assert 'route_from_120.csv' in captured.err
