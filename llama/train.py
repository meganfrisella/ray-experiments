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
batch_size = 64
seq_len = 32
num_iters = 25

device = "cuda:0"
stg1_layers = int(model_args.n_layers * 5 / 8)
print(f"stg 1: {stg1_layers}/{model_args.n_layers} layers")
model = Transformer(device, model_args, stg1_layers)
model.to(device)
print("loaded model")

# generate data

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

for iter in range(num_iters):
    if iter % 5 == 0: print(f"iter {iter}/{num_iters}")
    model.init_tracing()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True, 
            profile_memory=True,
            with_stack=True) as prof:
        with record_function("single"):
            model.update_tracing("start")
            pred = model(input)
            loss = criterion(pred, target)
            optimizer.step()
            optimizer.zero_grad()
            model.update_tracing("end")
    prof.export_chrome_trace("single.json")

    model.finish_tracing()
    # log_to_txt(output_path, timestamp, rank, prof.key_averages().table(sort_by="cuda_time_total"))
    # prof.export_chrome_trace(output_path + f"trace_rank{rank}_iter{iter}.json")

elapses = model.fetch_traces()

warmup = 0.2
for key, vals in elapses.items():
    elapses[key] = vals[int(warmup * len(vals)):]
total_mean = np.mean(elapses["total"])

for key, vals in elapses.items():
    print(f"{key} last elapse: {vals[-1]}")
    mean = np.mean(vals)
    std = np.std(vals)
    pct = (mean / total_mean * 100)
    print(key, round(mean), round(std), round(pct))
