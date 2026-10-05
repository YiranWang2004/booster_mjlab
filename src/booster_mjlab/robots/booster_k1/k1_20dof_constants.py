"""Serial K1 with 20 body joints and a rigid head at yaw=0, pitch=0 rad."""

from dataclasses import replace
from math import cos, sin

import mujoco
from mjlab.entity import EntityCfg

from .k1_constants import (
    K1_ACTION_SCALE,
    K1_ACTUATOR_HT4438,
    K1_JOINT_ORDER,
    get_k1_robot_cfg,
    get_spec,
)

K1_HEAD_JOINTS = K1_JOINT_ORDER[:2]
K1_20DOF_JOINT_ORDER = K1_JOINT_ORDER[2:]
K1_FIXED_HEAD_PITCH = 0.0
K1_20DOF_ACTION_SCALE = {
    pattern: scale for pattern, scale in K1_ACTION_SCALE.items() if pattern != "Head_.*"
}


def get_k1_20dof_spec() -> mujoco.MjSpec:
    """Remove the head hinges while retaining both bodies and their inertia/geoms."""
    spec = get_spec()
    # Both head hinge anchors are at their body origins, and the XML's head
    # body quaternions are identity. Bake the chosen pitch into Head_2's frame.
    spec.body("Head_2").quat = (
        cos(K1_FIXED_HEAD_PITCH / 2),
        0.0,
        sin(K1_FIXED_HEAD_PITCH / 2),
        0.0,
    )
    for name in K1_HEAD_JOINTS:
        spec.delete(spec.joint(name))
    return spec


def get_k1_20dof_robot_cfg() -> EntityCfg:
    """Build a fresh K1 config with only the 20 body actuators."""
    cfg = get_k1_robot_cfg()
    cfg.spec_fn = get_k1_20dof_spec
    assert cfg.articulation is not None
    cfg.articulation = replace(
        cfg.articulation,
        actuators=tuple(
            actuator
            for actuator in cfg.articulation.actuators
            if actuator is not K1_ACTUATOR_HT4438
        ),
    )
    assert cfg.init_state.joint_pos is not None
    cfg.init_state = replace(
        cfg.init_state,
        joint_pos={
            name: value
            for name, value in cfg.init_state.joint_pos.items()
            if name not in K1_HEAD_JOINTS
        },
    )
    return cfg
