import torch
import os 
from torch import nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, Schedule1F1B
from model import Transformer, LLAMA_8B, LLAMA_DEBUG
from typing import Any, Dict
from config import parse_args

def train(rank, world_size, device, model_args, batch_size, seq_len, num_iters, num_batches, num_microbatches):

    # load model
    model = Transformer(model_args)
    print("loaded model")

    # create pipeline
    layers_per_rank = model_args.n_layers // world_size
    mb_size = batch_size // num_microbatches
    mb_ex = torch.randint(
        0,
        model_args.vocab_size,
        (mb_size, seq_len),
        device=device,
    )
    pipe = pipeline(
        module=model,
        mb_args=(mb_ex,),
        split_spec={
            f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING
            for i in range(1, world_size)
        }
    )

    # pipeline schedule
    criterion = torch.nn.CrossEntropyLoss()
    # stage = pipe.get_stage_module(rank)
    stage = pipe.build_stage(rank, device)
    schedule = Schedule1F1B(stage, num_microbatches, loss_fn=criterion)

    # generate data
    if rank == 0:
        input = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, seq_len),
            device=device,
        )
    else:
        target = torch.randn(
            batch_size,
            seq_len,
            model_args.vocab_size,
            requires_grad=True,
            device=device,
        )

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    for i in range(num_iters):
        for step in range(num_batches):
            if rank == 0:
                x = input
                schedule.step(x)
            else:
                losses = []
                y = target
                schedule.step(target=y, losses=losses)

            optimizer.step()
            optimizer.zero_grad()

    dist.barrier()
    dist.destroy_process_group()
    


def main(rank, world_size, device, args: Dict[str, Any]) -> None:
    train(
        rank,
        world_size,
        device,
        LLAMA_DEBUG,
        args["batch_size"],
        args["seq_len"],
        args["num_iters"],
        args["num_batches"],
        args["num_microbatches"],
    )

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    device = None
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    args = parse_args()
    main(rank, world_size, device, args)