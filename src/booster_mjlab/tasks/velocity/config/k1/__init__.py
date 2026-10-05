from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
    booster_k1_flat_env_cfg,
    booster_k1_rough_env_cfg,
    with_fixed_head,
)
from .rl_cfg import (
    booster_k1_amp_ppo_runner_cfg,
    booster_k1_amp_ppo_symmetric_runner_cfg,
    booster_k1_ppo_runner_cfg,
    booster_k1_symmetric_ppo_runner_cfg,
)

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner
from booster_mjlab.tasks.velocity.config import with_amp_obs_group


def _with_amp_reset_cfg(env_cfg, runner_cfg):
    return with_amp_obs_group(
        env_cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        dataset_transform=runner_cfg.dataset_transform,
        include_base_lin_vel=True,
    )


# Task name -> (env cfg factory, runner cfg factory). Every entry is registered
# twice: once with Adam and once with Muon (``-Muon`` inserted before ``-Booster``).
_PPO_TASKS = {
    "Rough": (booster_k1_rough_env_cfg, booster_k1_ppo_runner_cfg),
    "Flat": (booster_k1_flat_env_cfg, booster_k1_ppo_runner_cfg),
    "Rough-DA": (booster_k1_rough_env_cfg, booster_k1_symmetric_ppo_runner_cfg),
    "Flat-DA": (booster_k1_flat_env_cfg, booster_k1_symmetric_ppo_runner_cfg),
}
_AMP_TASKS = {
    "Rough-Amp": (booster_k1_rough_env_cfg, booster_k1_amp_ppo_runner_cfg),
    "Flat-Amp": (booster_k1_flat_env_cfg, booster_k1_amp_ppo_runner_cfg),
    "Flat-Amp-DA": (booster_k1_flat_env_cfg, booster_k1_amp_ppo_symmetric_runner_cfg),
}

for fixed_head in (False, True):
    suffix = "-20DoF" if fixed_head else ""
    for use_muon in (False, True):
        optimizer = "-Muon" if use_muon else ""
        for tasks, is_amp in ((_PPO_TASKS, False), (_AMP_TASKS, True)):
            for name, (env_cfg_fn, rl_cfg_fn) in tasks.items():
                env_cfg = env_cfg_fn()
                play_env_cfg = env_cfg_fn(play=True)
                rl_cfg = rl_cfg_fn(use_muon=use_muon)
                if fixed_head:
                    env_cfg = with_fixed_head(env_cfg)
                    play_env_cfg = with_fixed_head(play_env_cfg)
                    rl_cfg.experiment_name += "_20dof"
                    if is_amp:
                        rl_cfg.dataset_transform = (
                            "booster_mjlab.robots.booster_k1.fixed_head_motion:to_20dof"
                        )
                if is_amp:
                    env_cfg = _with_amp_reset_cfg(env_cfg, rl_cfg)
                    play_env_cfg = _with_amp_reset_cfg(play_env_cfg, rl_cfg)
                register_mjlab_task(
                    task_id=f"Mjlab-Velocity-{name}{optimizer}-Booster-K1{suffix}",
                    env_cfg=env_cfg,
                    play_env_cfg=play_env_cfg,
                    rl_cfg=rl_cfg,
                    runner_cls=(
                        VelocityAmpOnPolicyRunner if is_amp else BoosterOnPolicyRunner
                    ),
                )
