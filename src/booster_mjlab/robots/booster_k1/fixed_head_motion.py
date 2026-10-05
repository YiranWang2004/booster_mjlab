"""Project serial K1 reference clips onto the body-only joint layout."""

from dataclasses import replace

import numpy as np

from booster_mjlab.motion import MotionFile

from .k1_20dof_constants import K1_20DOF_JOINT_ORDER
from .k1_constants import K1_JOINT_ORDER


def to_20dof(motion: MotionFile) -> MotionFile:
    """Drop head columns after dataset augmentation, before preparing AMP/reset data."""
    dof_pos = np.asarray(motion.dof_pos)
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected 2D dof_pos, got shape {dof_pos.shape}")
    if dof_pos.shape[1] == len(K1_20DOF_JOINT_ORDER):
        return motion
    if dof_pos.shape[1] != len(K1_JOINT_ORDER):
        raise ValueError(
            f"Expected 22 serial or 20 body-only dof columns, got {dof_pos.shape[1]}"
        )
    return replace(motion, dof_pos=dof_pos[:, 2:].copy())
