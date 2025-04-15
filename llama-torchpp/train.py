import torch
import os 
import csv
import numpy as np
from torch import nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, SplitPoint, Schedule1F1B
from torch.profiler import profile, record_function, ProfilerActivity
from model import Transformer, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_DEBUG
from typing import Any, Dict
from config import parse_args

def log_to_csv(output_path, timestamp, rank, elapses, warmup: float=0.2):
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}/{timestamp}_rank{rank}.csv"
    
    with open(output_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "mean", "std", "percent"])

        for key, vals in elapses.items():
            elapses[key] = vals[int(warmup * len(vals)):]
        total_mean = np.mean(elapses["total"])

        for key, vals in elapses.items():
            mean = np.mean(vals)
            std = np.std(vals)
            pct = (mean / total_mean * 100)
            w.writerow([key, round(mean), round(std), round(pct)])

def train(rank, world_size, device, model_args, output_path, timestamp, batch_size=100, seq_len=10, num_iters=2, num_batches=10, num_microbatches=4):
    assert world_size == 2

    # load model

    layers_per_rank = model_args.n_layers // world_size
    if rank == 0: int(model_args.n_layers * 5 / 8)
    if rank == 1: model_args.n_layers - int(model_args.n_layers * 5 / 8)

    model = Transformer(rank, layers_per_rank, device, model_args)
    print(f"[Rank {rank}] Loaded model")

    # create pipeline
    if rank == 0:
        model.norm = None
        model.output = None
    elif rank == world_size - 1:
        model.tok_embeddings = None
    else:
        model.norm = None
        model.output = None
        model.tok_embeddings = None

    stage = PipelineStage(
        model,
        rank,
        world_size,
        device,
        # group=...,
    )
    model.to(device)
    model.train()

    # pipeline schedule
    criterion = torch.nn.CrossEntropyLoss()
    schedule = Schedule1F1B(stage, num_microbatches, loss_fn=criterion)

    # generate data
    if rank == 0:
        input = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, seq_len),
            device=device,
        )
    target = torch.randn(
        batch_size,
        seq_len,
        model_args.vocab_size,
        requires_grad=True,
        device=device,
    )

    dist.barrier()

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    for iter in range(num_iters):
        model.init_tracing()
        model.update_tracing("start")
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        #     with record_function(f"Rank {rank} iter {iter}"):
        losses = []
        y = target
        if rank == 0:
            x = input
            schedule.step(x, target=y, losses=losses)
        else:
            schedule.step(target=y, losses=losses)
        loss = (
            torch.mean(torch.stack(losses)).to(device)
            if rank == world_size - 1
            else torch.tensor([-1.0], device=device)
        )
        optimizer.step()
        optimizer.zero_grad()
        # model.num_batches_updated += 1
        # if model.num_batches_updated == num_microbatches:
        model.update_tracing("end")
        model.finish_tracing()
        # log_to_txt(output_path, timestamp, rank, prof.key_averages().table(sort_by="cuda_time_total"))
        # prof.export_chrome_trace(output_path + f"trace_rank{rank}_iter{iter}.json")

    elapses = model.fetch_traces()
    print(f"Rank {rank}", elapses)
    log_to_csv(output_path, timestamp, rank, elapses)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    print(f"[Rank {rank}] Device: {device}")
    args = parse_args()
    if args["model"] == "LLAMA_1B":
        model = LLAMA_1B
    elif args["model"] == "LLAMA_3B":
        model = LLAMA_3B
    elif args["model"] == "LLAMA_8B":
        model = LLAMA_8B
    elif args["model"] == "LLAMA_DEBUG":
        model = LLAMA_DEBUG
    else:
        assert False and "'model' must be LLAMA_[DEBUG,1B,3B,8B]"

    train(
        rank,
        world_size,
        device,
        model,
        args["output_path"],
        args["timestamp"],
        args["batch_size"],
        args["seq_len"],
        args["num_iters"],
        args["num_batches"],
        args["num_microbatches"],
    )
