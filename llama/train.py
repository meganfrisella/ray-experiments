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
import time

TEST_RANK = None
model_args = LLAMA_3B
batch_size = 256
seq_len = 32
num_iters = 10

layers_per_rank = model_args.n_layers // 2
device = "cuda:0"

# load model

assert TEST_RANK == 0 or TEST_RANK == 1 or TEST_RANK == None

if TEST_RANK == None:
    model = Transformer(0, model_args.n_layers, device, model_args)
else:
    model = Transformer(TEST_RANK, layers_per_rank, device, model_args)
    if TEST_RANK == 0:
        model.norm = None
        model.output = None
    else:
        model.tok_embeddings = None
model.to(device)
# model.train()

print("loaded model")

# generate data

if TEST_RANK == None:
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
        device=device,
    )
elif TEST_RANK == 0:
    input = torch.randint(
        0,
        model_args.vocab_size,
        (batch_size, seq_len),
        device=device,
    )
    grad = torch.randn(
        batch_size,
        seq_len,
        model_args.dim,
        device=device,
    )
else:
    input = torch.randn(
        batch_size,
        seq_len,
        model_args.dim,
        device=device,
    )
    target = torch.randn(
        batch_size,
        seq_len,
        model_args.vocab_size,
        device=device,
    )

# train

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

fwd_times = []
bwd_times = []
for iter in range(num_iters):
    # if iter % 5 == 0: print(f"iter {iter}/{num_iters}")
    # model.init_tracing()
    # model.update_tracing("start")
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #         record_shapes=True, 
    #         profile_memory=True,
    #         with_stack=True) as prof:
    #     with record_function(f"stage{TEST_RANK}"):

    fwd_start = time.perf_counter()
    pred = model(input)
    fwd_end = time.perf_counter()
    fwd_times.append((fwd_end-fwd_start)*1000)

    # prof.export_chrome_trace(f"forward_single_torch.json")

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #         record_shapes=True, 
    #         profile_memory=True,
    #         with_stack=True) as prof:
    #     with record_function(f"stage{TEST_RANK}"):

    bwd_start = time.perf_counter()
    if TEST_RANK == 0:
        pred.backward(grad)
        optimizer.step()
        optimizer.zero_grad()
    else:
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    bwd_end = time.perf_counter()
    bwd_times.append((bwd_end-bwd_start)*1000)

    # prof.export_chrome_trace(f"backward_single_torch.json")

print(
    f"fwd time: {np.mean(fwd_times)} ms"
)
print(
    f"bwd time: {np.mean(bwd_times)} ms"
)

# elapses = model.fetch_traces()

# def print_output(elapses, warmup=0.2):
#     for key, vals in elapses.items():
#         elapses[key] = vals[int(warmup * len(vals)):]
#     total_mean = np.mean(elapses["total"])

#     for key, vals in elapses.items():
#         # print(f"Rank {TEST_RANK} {key} last elapse: {vals[-1]}")
#         mean = np.mean(vals)
#         std = np.std(vals)
#         pct = (mean / total_mean * 100)
#         print(f"Rank {TEST_RANK}", key, round(mean), round(std), round(pct))

# print_output(elapses)
