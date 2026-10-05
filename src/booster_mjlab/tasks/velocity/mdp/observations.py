from __future__ import annotations

from functools import partial

from rsl_rl.env import VecEnv
from tensordict import TensorDict

import torch

K1_JOINT_DIM = 22
K1_ACTION_DIM = 22
K1_BODY_JOINT_DIM = 20
SUPPORTED_JOINT_DIMS = (K1_BODY_JOINT_DIM, K1_JOINT_DIM)

# Joint slots whose axes change sign under a left/right mirror.
K1_INVERTED_JOINT_INDICES: tuple[int, ...] = (
    0,
    3,
    5,
    7,
    9,
    11,
    12,
    15,
    17,
    18,
    21,
)

# The parallel ankle's A/B crank axes both point along +y. Unlike serial ankle
# roll, neither crank changes sign under a left/right mirror.
K1_PARALLEL_INVERTED_JOINT_INDICES: tuple[int, ...] = tuple(
    index for index in K1_INVERTED_JOINT_INDICES if index not in (15, 21)
)
POLICY_DIM_NO_BASE_LIN_VEL = 75
POLICY_DIM_20DOF = 69
CRITIC_EXTRA_DIM = 15


def _augment_symmetries(
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    inverted_indices: tuple[int, ...],
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply symmetry augmentations to observations and actions."""

    if obs is not None:
        actor_obs = obs["actor"]
        critic_obs = obs["critic"]
        actor_dim = actor_obs.shape[1]
        critic_dim = critic_obs.shape[1]
        if not _is_supported_policy_dim(actor_dim):
            raise ValueError(
                f"Unsupported actor observation dim: {actor_dim}. "
                f"Expected one of {SUPPORTED_POLICY_DIMS}."
            )
        expected_critic_dim = actor_dim + CRITIC_EXTRA_DIM
        if critic_dim != expected_critic_dim:
            raise ValueError(
                f"Critic observation dim mismatch: got {critic_dim}, "
                f"expected {expected_critic_dim}."
            )

        n_envs = actor_obs.shape[0]

        # augment the batch size to 2 different symmetries
        actor_aug = torch.zeros(n_envs * 2, actor_obs.shape[1], device=actor_obs.device)
        actor_aug[:n_envs] = actor_obs
        actor_aug[n_envs : 2 * n_envs] = flip_k1_policy_obs_left_right(
            actor_obs, inverted_indices
        )

        critic_aug = torch.zeros(
            n_envs * 2, critic_obs.shape[1], device=critic_obs.device
        )
        critic_aug[:n_envs] = critic_obs
        critic_aug[n_envs : 2 * n_envs] = flip_k1_critic_obs_left_right(
            critic_obs, inverted_indices
        )

        obs = TensorDict(
            {"actor": actor_aug, "critic": critic_aug}, batch_size=(n_envs * 2,)
        )

    if actions is not None:
        if actions.shape[1] not in SUPPORTED_JOINT_DIMS:
            raise ValueError(
                f"Unsupported action dim: {actions.shape[1]}. "
                f"Expected one of {SUPPORTED_JOINT_DIMS}."
            )

        n_envs = actions.shape[0]

        # augment the batch size to 2 different symmetries
        actions_aug = torch.zeros(n_envs * 2, actions.shape[1], device=actions.device)
        actions_aug[:n_envs] = actions[:]
        actions_aug[n_envs : 2 * n_envs] = flip_k1_action_left_right(
            actions, inverted_indices
        )

        actions = actions_aug

    return obs, actions


def augment_symmetries(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the serial-ankle K1."""
    del env
    return _augment_symmetries(obs, actions, K1_INVERTED_JOINT_INDICES)


def augment_symmetries_parallel(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the parallel-ankle K1."""
    del env
    return _augment_symmetries(obs, actions, K1_PARALLEL_INVERTED_JOINT_INDICES)


def flip_k1_action_left_right(
    action: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    action = action.clone()
    # switch left and right joints
    action = _switch_k1_joints_left_right(action, inverted_indices)
    return action


def flip_k1_policy_obs_left_right(
    obs: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    obs = obs.clone()
    policy_dim = obs.shape[1]
    num_linear_obs_blocks = _get_num_linear_obs_blocks(policy_dim)
    if num_linear_obs_blocks is None:
        raise ValueError(
            f"Unsupported policy observation dim: {policy_dim}. "
            f"Expected one of {SUPPORTED_POLICY_DIMS}."
        )

    # No leading base linear terms in the current velocity policy layout.
    # Each 3D vector mirrors with [x, y, z] -> [x, -y, z].
    for i in range(num_linear_obs_blocks):
        start = i * 3
        obs[:, start : start + 3] = obs[:, start : start + 3] * obs.new_tensor(
            [1.0, -1.0, 1.0]
        )

    base_offset = num_linear_obs_blocks * 3

    # base ang vel
    obs[:, base_offset : base_offset + 3] = obs[
        :, base_offset : base_offset + 3
    ] * obs.new_tensor([-1.0, 1.0, -1.0])
    # projected gravity
    obs[:, base_offset + 3 : base_offset + 6] = obs[
        :, base_offset + 3 : base_offset + 6
    ] * obs.new_tensor([1.0, -1.0, 1.0])

    joint_pos_start = base_offset + 6
    joint_dim = (policy_dim - 9) // 3
    joint_vel_start = joint_pos_start + joint_dim
    last_actions_start = joint_vel_start + joint_dim
    command_start = last_actions_start + joint_dim

    # joint pos
    obs[:, joint_pos_start:joint_vel_start] = _switch_k1_joints_left_right(
        obs[:, joint_pos_start:joint_vel_start], inverted_indices
    )
    # joint vel
    obs[:, joint_vel_start:last_actions_start] = _switch_k1_joints_left_right(
        obs[:, joint_vel_start:last_actions_start], inverted_indices
    )
    # last actions
    obs[:, last_actions_start:command_start] = _switch_k1_joints_left_right(
        obs[:, last_actions_start:command_start], inverted_indices
    )
    # velocity command
    obs[:, command_start : command_start + 3] = obs[
        :, command_start : command_start + 3
    ] * obs.new_tensor([1.0, -1.0, -1.0])

    return obs


def flip_k1_critic_obs_left_right(
    obs: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    obs = obs.clone()
    policy_dim = obs.shape[1] - CRITIC_EXTRA_DIM
    if not _is_supported_policy_dim(policy_dim):
        raise ValueError(
            f"Unsupported critic observation dim: {obs.shape[1]}. "
            f"Expected policy part of one of {SUPPORTED_POLICY_DIMS} "
            f"plus {CRITIC_EXTRA_DIM} extra dims."
        )

    # policy obs
    obs[:, :policy_dim] = flip_k1_policy_obs_left_right(
        obs[:, :policy_dim], inverted_indices
    )

    # base lin vel: [x, y, z] -> [x, -y, z]
    base_lin_vel_start = policy_dim
    obs[:, base_lin_vel_start : base_lin_vel_start + 3] = obs[
        :, base_lin_vel_start : base_lin_vel_start + 3
    ] * obs.new_tensor([1.0, -1.0, 1.0])

    foot_height_start = policy_dim + 3
    foot_air_time_start = foot_height_start + 2
    foot_contact_start = foot_air_time_start + 2
    foot_contact_forces_start = foot_contact_start + 2

    # foot height
    obs[:, [foot_height_start, foot_height_start + 1]] = obs[
        :, [foot_height_start + 1, foot_height_start]
    ]
    # foot air time
    obs[:, [foot_air_time_start, foot_air_time_start + 1]] = obs[
        :, [foot_air_time_start + 1, foot_air_time_start]
    ]
    # foot contact
    obs[:, [foot_contact_start, foot_contact_start + 1]] = obs[
        :, [foot_contact_start + 1, foot_contact_start]
    ]
    # foot contact forces
    obs[
        :,
        [
            foot_contact_forces_start,
            foot_contact_forces_start + 1,
            foot_contact_forces_start + 2,
            foot_contact_forces_start + 3,
            foot_contact_forces_start + 4,
            foot_contact_forces_start + 5,
        ],
    ] = obs[
        :,
        [
            foot_contact_forces_start + 3,
            foot_contact_forces_start + 4,
            foot_contact_forces_start + 5,
            foot_contact_forces_start,
            foot_contact_forces_start + 1,
            foot_contact_forces_start + 2,
        ],
    ] * obs.new_tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

    return obs


SUPPORTED_POLICY_DIMS = (POLICY_DIM_20DOF, POLICY_DIM_NO_BASE_LIN_VEL)


def _is_supported_policy_dim(policy_dim: int) -> bool:
    return _get_num_linear_obs_blocks(policy_dim) is not None


def _get_num_linear_obs_blocks(policy_dim: int) -> int | None:
    """Infer number of leading 3D base linear vectors from policy dim.

    Layout is:
    [base_ang_vel(3), projected_gravity(3), joint_pos(J), joint_vel(J),
     last_action(J), command(3)], with J=22 or J=20 (fixed head).
    """
    if policy_dim not in SUPPORTED_POLICY_DIMS:
        return None
    return 0


def _switch_k1_joints_left_right(
    joints: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    """Switch left and right joints in the K1 joint tensor.

    Joint order:

    'Head_Yaw' (Inverted), 'Head_Pitch',
    'Left_Shoulder_Pitch', 'Left_Shoulder_Roll' (Inverted), 'Left_Elbow_Pitch', 'Left_Elbow_Yaw' (Inverted),
    'Right_Shoulder_Pitch', 'Right_Shoulder_Roll' (Inverted), 'Right_Elbow_Pitch', 'Right_Elbow_Yaw' (Inverted),
    'Left_Hip_Pitch', 'Left_Hip_Roll' (Inverted), 'Left_Hip_Yaw' (Inverted), 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll' (Inverted),
    'Right_Hip_Pitch', 'Right_Hip_Roll' (Inverted), 'Right_Hip_Yaw' (Inverted), 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll' (Inverted)

    Joints marked as "Inverted" need to have their sign flipped when calculating the left-right symmetry.

    Args:
        joints: Shape (batch, 22), or (batch, 20) with the head slots removed.
    """
    joint_dim = joints.shape[1]
    if joint_dim not in SUPPORTED_JOINT_DIMS:
        raise ValueError(
            f"Unsupported joint tensor dim: {joint_dim}. "
            f"Expected one of {SUPPORTED_JOINT_DIMS}."
        )
    joints_flipped = torch.zeros_like(joints)

    head_dim = joint_dim - K1_BODY_JOINT_DIM
    # Head joints (absent in the 20DoF variants).
    joints_flipped[:, :head_dim] = joints[:, :head_dim]
    # Shoulders and Elbows
    joints_flipped[:, head_dim : head_dim + 4] = joints[:, head_dim + 4 : head_dim + 8]
    joints_flipped[:, head_dim + 4 : head_dim + 8] = joints[:, head_dim : head_dim + 4]
    # Hips, Knees, Ankles
    joints_flipped[:, head_dim + 8 : head_dim + 14] = joints[
        :, head_dim + 14 : head_dim + 20
    ]
    joints_flipped[:, head_dim + 14 : head_dim + 20] = joints[
        :, head_dim + 8 : head_dim + 14
    ]

    # Sign indices are expressed in the original 22-joint layout.
    if head_dim == 0:
        inverted_indices = tuple(index - 2 for index in inverted_indices if index >= 2)

    joints_flipped[:, list(inverted_indices)] = -joints_flipped[
        :, list(inverted_indices)
    ]

    return joints_flipped


flip_k1_parallel_action_left_right = partial(
    flip_k1_action_left_right,
    inverted_indices=K1_PARALLEL_INVERTED_JOINT_INDICES,
)
flip_k1_parallel_policy_obs_left_right = partial(
    flip_k1_policy_obs_left_right,
    inverted_indices=K1_PARALLEL_INVERTED_JOINT_INDICES,
)
flip_k1_parallel_critic_obs_left_right = partial(
    flip_k1_critic_obs_left_right,
    inverted_indices=K1_PARALLEL_INVERTED_JOINT_INDICES,
)
