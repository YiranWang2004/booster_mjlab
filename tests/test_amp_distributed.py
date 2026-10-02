"""Run with: uv run python -m unittest discover -s tests -p 'test_amp_distributed.py'."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import mjlab  # noqa: F401  # Initialize mjlab before importing its task extensions.
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from booster_mjlab.amp.algorithms.amp_ppo import AmpPPO
from booster_mjlab.amp.modules.discriminator import Discriminator


def _algorithm(rank: int, use_muon: bool) -> AmpPPO:
    torch.manual_seed(100 + rank)
    discriminator = Discriminator(
        input_dim=6,
        hidden_layer_sizes=[8, 4],
        reward_scale=1.0,
        loss_type="bce",
        empirical_normalization=True,
    )
    # Synchronization only needs the modules, not a simulator or motion dataset.
    return AmpPPO(
        actor=nn.Linear(6, 2),
        critic=nn.Linear(6, 1),
        storage=None,
        amp=SimpleNamespace(discriminator=discriminator),
        use_muon=use_muon,
        multi_gpu_cfg={"global_rank": rank, "world_size": 2},
    )


def _assert_replicated(module: nn.Module) -> None:
    for tensor in module.state_dict().values():
        reference = tensor.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(tensor, reference, rtol=0, atol=0)


def _worker(rank: int, store: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for use_muon in (False, True):
            alg = _algorithm(rank, use_muon)
            discriminator = alg.amp.discriminator
            # Make buffer divergence observable too, even before training.
            discriminator.amp_normalizer._mean.fill_(rank)
            alg.broadcast_parameters()
            for module in (alg.actor, alg.critic, discriminator):
                _assert_replicated(module)

            # Unequal local batch sizes must produce pooled normalization stats.
            discriminator.update_normalization(torch.full((rank + 2, 6), rank + 1.0))
            normalizer = discriminator.amp_normalizer
            torch.testing.assert_close(normalizer.mean, torch.full((6,), 1.6))
            assert normalizer.count.item() == 5
            _assert_replicated(discriminator)

            for _ in range(3):
                policy = torch.randn(8, 6) + rank
                expert = torch.randn(8, 6) - rank
                scores = discriminator(torch.cat((policy, expert)))
                amp_loss, penalty = discriminator.compute_loss(
                    scores[:8],
                    scores[8:],
                    expert,
                    policy,
                )
                loss = (
                    alg.actor(policy).square().mean()
                    + alg.critic(policy).square().mean()
                )
                alg.optimizer.zero_grad()
                (loss + amp_loss + penalty).backward()

                # Independent oracle: gather both local gradients before reduction.
                params = list(discriminator.parameters())
                local = torch.cat([p.grad.flatten() for p in params])
                gathered = [torch.empty_like(local) for _ in range(2)]
                dist.all_gather(gathered, local)
                expected = torch.stack(gathered).mean(0)
                alg.reduce_parameters()
                actual = torch.cat([p.grad.flatten() for p in params])
                torch.testing.assert_close(actual, expected)
                alg.optimizer.step()
                for module in (alg.actor, alg.critic, discriminator):
                    _assert_replicated(module)

            restored = _algorithm(rank, use_muon)
            restored.load(alg.save(), load_cfg=None, strict=True)
            restored.broadcast_parameters()
            for name, value in discriminator.state_dict().items():
                torch.testing.assert_close(
                    restored.amp.discriminator.state_dict()[name],
                    value,
                )
            _assert_replicated(restored.amp.discriminator)
    finally:
        dist.destroy_process_group()


class AmpDistributedTest(unittest.TestCase):
    def test_two_rank_adam_and_muon(self) -> None:
        with TemporaryDirectory() as directory:
            mp.spawn(
                _worker, args=(str(Path(directory) / "store"),), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
