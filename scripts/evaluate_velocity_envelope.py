"""Headless repeated fixed body-frame command evaluation and closed voxel surface."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--size', type=int, default=9)
p.add_argument('--limits', type=float, nargs=3, default=[3, 2.5, 5])
p.add_argument('--device', default='cuda:1')
p.add_argument('--axis-only', action='store_true')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
configure_torch_backends()
torch.manual_seed(20261002)
axes = [np.linspace(-v, v, a.size) for v in a.limits]
commands = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
if a.axis_only:
    commands = np.unique(np.concatenate([np.eye(3)[j][None]*axes[j][:,None] for j in range(3)]),axis=0)
task = 'Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1'
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = len(commands)*5
cfg.seed = 20261002
cfg.events.pop('push_robot', None)
cfg.curriculum = {}
# Keep auto-reset, but permanently flag any reset as failure.
cmdcfg = cfg.commands['twist']
cmdcfg.heading_command = False; cmdcfg.ranges.heading = None
cmdcfg.rel_standing_envs = cmdcfg.rel_heading_envs = cmdcfg.rel_world_envs = cmdcfg.rel_forward_envs = 0
cmdcfg.init_velocity_prob = 0
cmdcfg.resampling_time_range = (1e9, 1e9)
rlcfg = load_rl_cfg(task)
# AMP replay memory is irrelevant to inference.
rlcfg.amp_replay_buffer_size = 1
raw = ManagerBasedRlEnv(cfg, device=a.device)
env = RslRlVecEnvWrapper(raw, clip_actions=rlcfg.clip_actions)
runner = load_runner_cls(task)(env, asdict(rlcfg), device=a.device)
runner.load(a.checkpoint, load_cfg={'actor': True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)
term = raw.command_manager.get_term('twist')
fixed = torch.tensor(np.repeat(commands, 5, axis=0), dtype=torch.float32, device=a.device)
term.vel_command_b[:] = fixed
term.is_standing_env[:] = False; term.is_heading_env[:] = False; term.is_world_env[:] = False
obs = env.get_observations()
robot = raw.scene['robot']
failed = torch.zeros(env.num_envs, dtype=torch.bool, device=a.device)
window = []
windows = []
steps = round(15/raw.step_dt); warmup = round(5/raw.step_dt); width = round(1/raw.step_dt)
t0 = time.time()
with torch.inference_mode():
    for s in range(steps):
        obs, _, dones, _ = env.step(policy(obs))
        vel = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], -1)
        failed |= dones.bool() | (robot.data.projected_gravity_b[:, 2] > -np.cos(np.deg2rad(63))) | (robot.data.root_link_pos_w[:, 2] < .35) | ~torch.isfinite(vel).all(-1)
        term.vel_command_b[:] = fixed
        term.is_standing_env[:] = False; term.is_heading_env[:] = False; term.is_world_env[:] = False
        # Command only changes on reset; repair the observation after that change.
        if dones.any():
            obs = env.get_observations()
        if s >= warmup:
            window.append(vel.clone())
            if len(window) == width:
                windows.append(torch.stack(window).mean(0).cpu().numpy()); window=[]
        if (s+1) % 50 == 0:
            print(f'PROGRESS {s+1}/{steps}, elapsed={time.time()-t0:.1f}s, failed={failed.sum().item()}', flush=True)
w = np.stack(windows).reshape(-1, len(commands), 5, 3)
f = failed.cpu().numpy().reshape(len(commands), 5)
# Signed lower bound, zero-axis drift constraint, plus explicit upper tracking bound.
avg = w.mean(2)
nonzero = np.abs(commands) > 1e-6
ratio = avg*np.sign(commands)[None] / np.maximum(np.abs(commands)[None], 1e-6)
axis_ok = np.where(nonzero[None], ratio >= .9, np.abs(avg) <= np.array([.1, .1, .15]))
accurate_ok = np.where(nonzero[None], (ratio >= .9) & (ratio <= 1.1), np.abs(avg) <= np.array([.1, .1, .15])).all(-1)
accurate_feasible = (accurate_ok.mean(0)>=.9) & ~f.any(1)
window_ok = axis_ok.all(-1)
feasible = (window_ok.mean(0) >= .9) & ~f.any(1)
np.savez_compressed(out/'measurements.npz', commands=commands, window_velocities=w, failed=f, feasible=feasible, accurate_feasible=accurate_feasible, window_ok=window_ok, axes=np.stack(axes))
np.savetxt(out/'summary.csv', np.column_stack([commands, w.mean((0,2)), w.mean(0).std(1), f.sum(1), window_ok.mean(0), feasible]), delimiter=',', header='vx,vy,vyaw,mean_vx,mean_vy,mean_vyaw,std_vx,std_vy,std_vyaw,failed_repeats,stable_window_fraction,feasible', comments='')
meta = dict(task=task, checkpoint=str(Path(a.checkpoint).resolve()), repetitions=5, seed=20261002, warmup_seconds=5, measurement_seconds=10, window_seconds=1, limits=a.limits, grid_size=a.size, axis_only=a.axis_only, resolution=[float(x[1]-x[0]) for x in axes], feasible_points=int(feasible.sum()), total_points=len(commands), elapsed_seconds=time.time()-t0, criterion='All five survive; >=90% of 1s windows of five-repeat mean: each nonzero component >=90% signed command; zero drift <=[0.1,0.1,0.15].', random_push=False, startup_randomization=True)
meta['feasible_extents'] = [commands[feasible].min(0).tolist(), commands[feasible].max(0).tolist()] if feasible.any() else None
meta['boundary_feasible'] = bool((feasible & np.any(np.isclose(np.abs(commands), a.limits), axis=1)).any())
(out/'metadata.json').write_text(json.dumps(meta, indent=2))
env.close()
print(json.dumps(meta, indent=2), flush=True)
