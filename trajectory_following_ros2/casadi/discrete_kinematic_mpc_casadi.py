"""
Backward-compatibility shim.

The discrete-time formulation was merged into the single
``KinematicMPCCasadi`` class in ``kinematic_mpc_casadi.py`` (the discrete path was
the superset: obstacle avoidance, sqpmethod/qrqp solver infrastructure, and
wall-clock timing). The old continuous formulation is mathematically identical to
``discrete_model_type='nonlinear', discrete_integration_method='euler'``, so a
separate discrete class is no longer needed.

``DiscreteKinematicMPCCasadi`` remains here as an alias so existing imports keep
working. Prefer importing ``KinematicMPCCasadi`` directly. New default for the
merged class is ``discrete_integration_method='rk4'``; the historical discrete
default was the same. Note the merged class defaults ``num_obstacles=0`` /
``collision_avoidance_scheme='euclidean'`` (the old discrete defaults of
``num_obstacles=1`` / ``'cbf'`` were never relied on by the controller node, which
always passes these explicitly).
"""
from trajectory_following_ros2.casadi.kinematic_mpc_casadi import KinematicMPCCasadi

# Alias for backward compatibility.
DiscreteKinematicMPCCasadi = KinematicMPCCasadi

__all__ = ['DiscreteKinematicMPCCasadi', 'KinematicMPCCasadi']
