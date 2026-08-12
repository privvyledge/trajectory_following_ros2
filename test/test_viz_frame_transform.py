"""The visualizer must draw the vehicle in the same frame as the route.

Odometry arrives in the localizer's frame while the route arrives in the
controller's global_frame. Without a transform the drawn vehicle is offset by the
whole map->odom correction — measured on hardware as (0.588, -0.710) m and +91.0
degrees — which silently invalidates the drawn cross-track error, keep-out rings
and goal point on every map-frame route.

These call the production method unbound against a stub, so no ROS graph is needed.
"""
import math
import threading

import pytest

from trajectory_following_ros2.viz.visualizer_node import VisualizerNode


class _StubLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, msg, **kwargs):
        self.warnings.append(msg)

    def info(self, msg, **kwargs):
        pass


class _StubBuffer:
    """Returns a fixed transform, or raises to model TF not being up yet."""

    def __init__(self, transform=None):
        self._transform = transform
        self.calls = []

    def lookup_transform(self, target, source, when):
        self.calls.append((target, source))
        if self._transform is None:
            raise RuntimeError('no such transform')
        return self._transform


def _make_transform(tx, ty, yaw):
    class _T:
        pass

    t = _T()
    t.transform = _T()
    t.transform.translation = _T()
    t.transform.translation.x = tx
    t.transform.translation.y = ty
    t.transform.translation.z = 0.0
    t.transform.rotation = _T()
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = math.sin(yaw / 2.0)
    t.transform.rotation.w = math.cos(yaw / 2.0)
    return t


class _StubNode:
    def __init__(self, global_frame='map', transform=None):
        self._mutex = threading.Lock()
        self.global_frame = global_frame
        self.tf_buffer = _StubBuffer(transform)
        self._warned_pose_frame = False
        self._logger = _StubLogger()

    def get_logger(self):
        return self._logger


def _to_global(node, x, y, yaw, frame):
    return VisualizerNode._to_global_frame(node, x, y, yaw, frame)


def test_applies_the_map_to_odom_correction():
    """The gosling1 case: +91 deg of rotation between the two frames."""
    yaw_corr = math.radians(91.0)
    node = _StubNode(transform=_make_transform(0.588, -0.710, yaw_corr))

    x, y, yaw = _to_global(node, 2.0, 1.0, 0.0, 'odom')

    c, s = math.cos(yaw_corr), math.sin(yaw_corr)
    assert x == pytest.approx(c * 2.0 - s * 1.0 + 0.588)
    assert y == pytest.approx(s * 2.0 + c * 1.0 - 0.710)
    assert yaw == pytest.approx(yaw_corr)
    assert node.tf_buffer.calls == [('map', 'odom')]


def test_identical_frames_skip_the_lookup():
    """global_frame:=odom is the historical case and must stay a no-op."""
    node = _StubNode(global_frame='odom')

    assert _to_global(node, 2.0, 1.0, 0.5, 'odom') == (2.0, 1.0, 0.5)
    assert node.tf_buffer.calls == []


def test_unset_global_frame_is_a_no_op():
    """Before any route arrives there is nothing to be consistent with."""
    node = _StubNode(global_frame='')

    assert _to_global(node, 2.0, 1.0, 0.5, 'odom') == (2.0, 1.0, 0.5)
    assert node.tf_buffer.calls == []


def test_missing_transform_falls_back_to_raw_pose_and_warns_once():
    """Mirrors the controller: warn once, then keep drawing rather than go blank."""
    node = _StubNode(transform=None)

    for _ in range(3):
        assert _to_global(node, 2.0, 1.0, 0.5, 'odom') == (2.0, 1.0, 0.5)

    assert len(node._logger.warnings) == 1
    assert 'global_frame' in node._logger.warnings[0]
