from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from awm.training.objective import FlowObjective, set_trainable
from awm.training.recipe import load_recipe
from test_training import batch, tiny_model


def worker(rank, address):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(42)
        recipe = load_recipe('joint')
        model = tiny_model()
        model.enable_gradient_checkpointing()
        parameters = set_trainable(model, recipe['phase'])
        objective = torch.nn.parallel.DistributedDataParallel(
            FlowObjective(model, recipe), find_unused_parameters=True)
        optimizer = torch.optim.AdamW(parameters, lr=.001)
        for update in range(2):
            for micro in range(2):
                data = batch((bool(rank),), True)
                context = [torch.randn(3, 16)]
                loss, _ = objective(data, context)
                (loss / 2).backward()
            optimizer.step()
            optimizer.zero_grad()
        for parameter in parameters:
            peer = parameter.detach().clone()
            dist.broadcast(peer, 0)
            torch.testing.assert_close(peer, parameter, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_action_free_and_action_ranks_train_together(tmp_path):
    address = (tmp_path / 'rendezvous').resolve().as_uri()
    mp.spawn(worker, args=(address,), nprocs=2, join=True)
