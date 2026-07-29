# Changelog

Only changes that are visible to *downstream packages* are recorded here. Internal
refactors of the controller nodes, solver backends, launch files, and configs are not —
see `git log` for those.

## Downstream API policy

`trajectory_following_ros2.utils` is imported by at least one external package, so the
members listed under [Published surface](#published-surface) are treated as a published
API. When one of them is renamed, removed, or has its signature changed:

1. Record it under an `### Unreleased` heading below, naming the old and new spelling.
2. Say so in the commit subject too (e.g. `api(utils): rename Trajectory.check_goal -> is_goal_reached`),
   so a `git log --oneline` grep on the consumer side finds it.

A deprecated alias is optional, not required. A rename that is only discoverable by
crashing the consumer at runtime is the failure mode this policy exists to prevent —
it has happened once (`Trajectory.check_goal` -> `Trajectory.is_goal_reached`, which
crashed a downstream node on its first control callback and went unnoticed for two
releases).

Anything in `utils` *not* on the list below is internal and may change freely.

## Published surface

`utils.Trajectory.Trajectory` — constructed as
`Trajectory(search_index_number=..., goal_tolerance=..., stop_speed=...)`.

| Member | Kind |
|---|---|
| `calc_ref_trajectory` | method |
| `is_goal_reached` | method |
| `arclength_index_advance` | attribute (bool flag; also settable) |
| `current_index` | attribute |
| `projection_status` | attribute |
| `state` | attribute |
| `state_key_to_column` | attribute |
| `trajectory` | attribute |
| `trajectory_keys` | attribute |
| `trajectory_key_to_column` | attribute |

`utils.trajectory_utils` — `normalize_angle`, `cumulative_distance_along_path`,
`calculate_curvature_single`, `calculate_curvature_all`, `calculate_current_arc_length`,
`calc_path_relative_time`.

## Unreleased

- `utils.Trajectory` no longer imports `matplotlib` at module scope. The import was
  unused (nothing in the module plots), so it was dropped rather than deferred. Importing
  `trajectory_following_ros2.utils.Trajectory` no longer pulls `matplotlib` — relevant to
  consumers running a venv whose numpy differs from the system matplotlib's, where the
  transitive import failed with `numpy.core.multiarray failed to import`.
- `utils.trajectory_utils` no longer imports `pandas` at module scope. It is used only by
  the module's `__main__` demo block, where the import now lives.

No changes to the published surface above.
