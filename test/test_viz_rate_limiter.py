"""Rate limiting for best-effort visualization streams."""
from trajectory_following_ros2.viz.base_viz_backend import StreamRateLimiter


def test_rate_limiter_caps_each_stream_independently():
    now = [10.0]
    limiter = StreamRateLimiter(5.0, time_fn=lambda: now[0])

    assert limiter.allow('vehicle')
    assert not limiter.allow('vehicle')
    assert limiter.allow('obstacles')

    now[0] += 0.199
    assert not limiter.allow('vehicle')
    now[0] += 0.002
    assert limiter.allow('vehicle')


def test_non_positive_frequency_disables_limiting():
    limiter = StreamRateLimiter(0.0, time_fn=lambda: 10.0)

    assert limiter.allow('vehicle')
    assert limiter.allow('vehicle')
