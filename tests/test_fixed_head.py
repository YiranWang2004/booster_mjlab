"""Run with: uv run python -m unittest discover -s tests -p 'test_fixed_head.py'."""

from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import mjlab  # noqa: F401  # Initialize mjlab before its task extensions.
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
import mujoco
import numpy as np
from tensordict import TensorDict
import torch

from booster_mjlab.motion import MotionFile, MotionLoader
from booster_mjlab.robots.booster_k1.fixed_head_motion import to_20dof
from booster_mjlab.robots.booster_k1.k1_20dof_constants import (
    K1_20DOF_JOINT_ORDER,
    K1_FIXED_HEAD_PITCH,
    get_k1_20dof_robot_cfg,
)
from booster_mjlab.robots.booster_k1.k1_constants import (
    K1_JOINT_ORDER,
    get_k1_robot_cfg,
)
from booster_mjlab.tasks.velocity.mdp.observations import (
    augment_symmetries,
    flip_k1_action_left_right,
    flip_k1_critic_obs_left_right,
    flip_k1_parallel_action_left_right,
    flip_k1_policy_obs_left_right,
)

MOTION_TRANSFORM = "booster_mjlab.robots.booster_k1.fixed_head_motion:to_20dof"


def _drop_head_obs(obs: torch.Tensor) -> torch.Tensor:
    """Select body joint slots in each of the three original 22-value blocks."""
    return torch.cat(
        (obs[:, :6], obs[:, 8:28], obs[:, 30:50], obs[:, 52:72], obs[:, 72:]),
        dim=-1,
    )


class FixedHeadTests(unittest.TestCase):
    def test_model_has_20_actuators_and_preserves_head_kinematics(self):
        original = Entity(get_k1_robot_cfg()).spec.compile()
        fixed = Entity(get_k1_20dof_robot_cfg()).spec.compile()
        self.assertEqual((original.nq, original.nv, original.nu), (29, 28, 22))
        self.assertEqual((fixed.nq, fixed.nv, fixed.nu), (27, 26, 20))
        self.assertEqual(
            tuple(fixed.joint(i).name for i in range(1, fixed.njnt)),
            K1_20DOF_JOINT_ORDER,
        )
        np.testing.assert_allclose(fixed.body_mass, original.body_mass)
        self.assertEqual(fixed.ngeom, original.ngeom)

        original_data = mujoco.MjData(original)
        original_data.qpos[original.joint("Head_Pitch").qposadr[0]] = (
            K1_FIXED_HEAD_PITCH
        )
        fixed_data = mujoco.MjData(fixed)
        mujoco.mj_forward(original, original_data)
        mujoco.mj_forward(fixed, fixed_data)
        for name in ("Head_1", "Head_2"):
            np.testing.assert_allclose(
                fixed_data.body(name).xpos, original_data.body(name).xpos, atol=1e-12
            )
            np.testing.assert_allclose(
                fixed_data.body(name).xmat, original_data.body(name).xmat, atol=1e-12
            )

    def test_all_serial_variants_and_amp_reset_transforms(self):
        variants = [task for task in list_tasks() if task.endswith("-K1-20DoF")]
        self.assertEqual(len(variants), 14)
        for task in variants:
            with self.subTest(task=task):
                original_task = task.removesuffix("-20DoF")
                agent = load_rl_cfg(task)
                original_agent = load_rl_cfg(original_task)
                self.assertEqual(
                    agent.experiment_name, original_agent.experiment_name + "_20dof"
                )
                for play in (False, True):
                    cfg = load_env_cfg(task, play=play)
                    robot = Entity(cfg.scene.entities["robot"])
                    self.assertEqual(tuple(robot.joint_names), K1_20DOF_JOINT_ORDER)
                    self.assertNotIn("Head_.*", cfg.actions["joint_pos"].scale)
                    posture = cfg.rewards["upper_body_posture"].params
                    self.assertNotIn("Head_.*", posture["asset_cfg"].joint_names)
                    for regime in ("std_standing", "std_walking", "std_running"):
                        self.assertNotIn("Head_.*", posture[regime])
                    if "-Amp" in task:
                        self.assertEqual(agent.dataset_transform, MOTION_TRANSFORM)
                        self.assertEqual(
                            cfg.events["reset_robot_from_motion"].params[
                                "dataset_transform"
                            ],
                            MOTION_TRANSFORM,
                        )
                original_robot = Entity(
                    load_env_cfg(original_task).scene.entities["robot"]
                )
                self.assertEqual(tuple(original_robot.joint_names), K1_JOINT_ORDER)

    def test_body_symmetry_matches_22_joint_projection(self):
        for dim, flip in (
            (75, flip_k1_policy_obs_left_right),
            (90, flip_k1_critic_obs_left_right),
        ):
            original = torch.arange(2 * dim, dtype=torch.float32).reshape(2, dim)
            body = _drop_head_obs(original)
            mirrored = flip(body)
            torch.testing.assert_close(mirrored, _drop_head_obs(flip(original)))
            torch.testing.assert_close(flip(mirrored), body)

        original_actions = torch.arange(44, dtype=torch.float32).reshape(2, 22)
        for flip in (flip_k1_action_left_right, flip_k1_parallel_action_left_right):
            torch.testing.assert_close(
                flip(original_actions[:, 2:]), flip(original_actions)[:, 2:]
            )
        obs = TensorDict(
            {"actor": torch.randn(2, 69), "critic": torch.randn(2, 84)},
            batch_size=(2,),
        )
        actions = torch.randn(2, 20)
        augmented, augmented_actions = augment_symmetries(None, obs, actions)
        self.assertEqual(augmented["actor"].shape, (4, 69))
        self.assertEqual(augmented["critic"].shape, (4, 84))
        self.assertEqual(augmented_actions.shape, (4, 20))
        torch.testing.assert_close(augmented["actor"][:2], obs["actor"])
        torch.testing.assert_close(augmented_actions[:2], actions)
        self.assertEqual(augment_symmetries(None, None, actions)[1].shape, (4, 20))
        self.assertIsNone(augment_symmetries(None, obs, None)[1])

    def test_motion_projection_preserves_leg_features_and_reset_order(self):
        motion = MotionFile(
            fps=30,
            root_pos=np.zeros((12, 3), dtype=np.float32),
            root_rot=np.tile([0.0, 0.0, 0.0, 1.0], (12, 1)),
            dof_pos=np.arange(12 * 22, dtype=np.float32).reshape(12, 22) * 0.001,
        )
        projected = to_20dof(motion)
        np.testing.assert_array_equal(projected.dof_pos, motion.dof_pos[:, 2:])
        self.assertIs(to_20dof(projected), projected)
        self.assertEqual(motion.dof_pos.shape, (12, 22))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            with path.open("wb") as file:
                pickle.dump(vars(motion), file)
            kwargs = dict(
                dataset_root=path,
                simulation_dt=0.02,
                speed_factor=1.0,
                num_amp_obs_steps=3,
                augmentations=[{"name": "mirror"}],
                include_base_lin_vel=True,
                include_base_ang_vel=False,
                include_projected_gravity=True,
            )
            original = MotionLoader(**kwargs, joint_indices=list(range(10, 22)))
            body = MotionLoader(
                **kwargs,
                joint_indices=list(range(8, 20)),
                dataset_transform=MOTION_TRANSFORM,
            )
            self.assertEqual(body.all_obs.shape[1], 3 * 30)
            self.assertEqual(body.all_states.shape[1], 4 + 2 * 20 + 6)
            torch.testing.assert_close(body.all_obs, original.all_obs)
            torch.testing.assert_close(
                body.all_states[:, :4], original.all_states[:, :4]
            )
            torch.testing.assert_close(
                body.all_states[:, 4:24], original.all_states[:, 6:26]
            )
            torch.testing.assert_close(
                body.all_states[:, 24:44], original.all_states[:, 28:48]
            )
            torch.testing.assert_close(
                body.all_states[:, 44:], original.all_states[:, 48:]
            )


if __name__ == "__main__":
    unittest.main()
