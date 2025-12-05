"""Control module for OSC action generation.

Computes normalized actions [-1, 1] for robosuite's OSC controller.
OSC handles impedance control internally (kp=150, damping=1).
"""

from .cartesian_pd import (
    CartesianPDController,
    compute_pregrasp_pose,
    pd_control,
    quat_error,
    quat_multiply,
    quat_conjugate,
)

__all__ = [
    "CartesianPDController",
    "compute_pregrasp_pose",
    "pd_control",
    "quat_error",
    "quat_multiply",
    "quat_conjugate",
]
