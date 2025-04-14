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

# load model

model_args = LLAMA_DEBUG
device = "cuda:0"
model = Transformer(device, model_args)
model.to(device)

# generate data

batch_size, seq_len = 128, 32
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

# train

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

for iter in range(10):
    model.init_tracing()
    model.update_tracing("start")
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    #     with record_function(f"Rank {rank} iter {iter}"):
    pred = model(input)
    loss = criterion(pred, target)

    optimizer.step()
    optimizer.zero_grad()

    model.update_tracing("end")
    model.finish_tracing()
    # log_to_txt(output_path, timestamp, rank, prof.key_averages().table(sort_by="cuda_time_total"))
    # prof.export_chrome_trace(output_path + f"trace_rank{rank}_iter{iter}.json")

elapses = model.fetch_traces()

warmup = 0.2
for key, vals in elapses.items():
    elapses[key] = vals[int(warmup * len(vals)):]
total_mean = np.mean(elapses["total"])

for key, vals in elapses.items():
    mean = np.mean(vals)
    std = np.std(vals)
    pct = (mean / total_mean * 100)
    print(key, round(mean), round(std), round(pct))