"""Pure-Python tests for the obstacle aggregator's merge/gate/evict state machine."""
import pytest

from trajectory_following_ros2.obstacle_aggregator import ObstacleAggregatorState


def _items(*specs):
    """Build ``(id, x, y, payload)`` items; the payload is the id, for readability."""
    return [(obj_id, x, y, obj_id) for obj_id, x, y in specs]


def _ids(records):
    return [record[0] for record in records]


def test_ids_are_namespaced_per_source():
    state = ObstacleAggregatorState(num_sources=2, id_namespace_stride=10000)
    state.ingest(0, _items((3, 0.0, 0.0)), now=0.0)
    state.ingest(1, _items((3, 1.0, 0.0)), now=0.0)

    records, _ = state.collect(now=0.0)

    # Both sources emitted id 3; without namespacing the controller's side hysteresis
    # would apply one obstacle's committed go-around side to the other.
    assert sorted(_ids(records)) == [3, 10003]


def test_namespaced_ids_are_stable_across_republishes():
    state = ObstacleAggregatorState(num_sources=1)
    state.ingest(0, _items((7, 0.0, 0.0)), now=0.0)

    first, _ = state.collect(now=0.1)
    second, _ = state.collect(now=0.2)

    assert _ids(first) == _ids(second) == [7]


def test_sources_merge_instead_of_replacing_each_other():
    state = ObstacleAggregatorState(num_sources=2)
    state.ingest(0, _items((1, 0.0, 0.0)), now=0.0)
    state.ingest(1, _items((1, 5.0, 0.0), (2, 6.0, 0.0)), now=0.0)
    # A second message on source 0 replaces only source 0's own content.
    state.ingest(0, _items((4, 1.0, 0.0)), now=0.1)

    records, _ = state.collect(now=0.1)

    assert sorted(_ids(records)) == [4, 10001, 10002]


def test_gate_is_applied_at_collect_so_a_latched_source_follows_the_ego():
    # The static source publishes once, at t=0, while the ego is 100 m away from the
    # object. Gating at ingest would drop it permanently; gating at collect lets it
    # reappear as the vehicle approaches.
    state = ObstacleAggregatorState(num_sources=1, gate_radius=50.0,
                                    expiring=[False])
    state.set_ego(0.0, 0.0)
    state.ingest(0, _items((1, 100.0, 0.0)), now=0.0)

    assert _ids(state.collect(now=0.0)[0]) == []

    state.set_ego(80.0, 0.0)
    assert _ids(state.collect(now=100.0)[0]) == [1]


def test_gate_passes_everything_before_the_first_ego_fix():
    state = ObstacleAggregatorState(num_sources=1, gate_radius=1.0)
    state.ingest(0, _items((1, 500.0, 0.0)), now=0.0)

    # No odometry yet: dropping obstacles here would silently hide a real one.
    assert _ids(state.collect(now=0.0)[0]) == [1]


def test_gate_disabled_keeps_every_object():
    state = ObstacleAggregatorState(num_sources=1, gate_radius=0.0)
    state.set_ego(0.0, 0.0)
    state.ingest(0, _items((1, 1e6, 0.0)), now=0.0)

    assert _ids(state.collect(now=0.0)[0]) == [1]


def test_stale_source_is_evicted_once():
    state = ObstacleAggregatorState(num_sources=1, source_timeout=1.0)
    state.ingest(0, _items((1, 0.0, 0.0)), now=0.0)

    records, evicted = state.collect(now=0.5)
    assert _ids(records) == [1] and evicted == []

    records, evicted = state.collect(now=1.5)
    assert _ids(records) == [] and evicted == [0]

    # The warning fires once, not on every publish tick.
    records, evicted = state.collect(now=2.5)
    assert _ids(records) == [] and evicted == []


def test_latched_source_is_exempt_from_eviction():
    state = ObstacleAggregatorState(num_sources=2, source_timeout=1.0,
                                    expiring=[False, True])
    state.ingest(0, _items((1, 0.0, 0.0)), now=0.0)
    state.ingest(1, _items((1, 0.0, 0.0)), now=0.0)

    records, evicted = state.collect(now=10.0)

    assert _ids(records) == [1]
    assert evicted == [1]


def test_recovered_source_can_be_evicted_again():
    state = ObstacleAggregatorState(num_sources=1, source_timeout=1.0)
    state.ingest(0, _items((1, 0.0, 0.0)), now=0.0)
    assert state.collect(now=2.0)[1] == [0]

    state.ingest(0, _items((1, 0.0, 0.0)), now=2.0)
    assert _ids(state.collect(now=2.1)[0]) == [1]
    assert state.collect(now=4.0)[1] == [0]


def test_zero_timeout_disables_eviction():
    state = ObstacleAggregatorState(num_sources=1, source_timeout=0.0)
    state.ingest(0, _items((1, 0.0, 0.0)), now=0.0)

    records, evicted = state.collect(now=1e6)

    assert _ids(records) == [1] and evicted == []


def test_bad_construction_is_rejected():
    with pytest.raises(ValueError):
        ObstacleAggregatorState(num_sources=0)
    with pytest.raises(ValueError):
        ObstacleAggregatorState(num_sources=2, expiring=[True])
    with pytest.raises(IndexError):
        ObstacleAggregatorState(num_sources=1).ingest(1, [], now=0.0)
