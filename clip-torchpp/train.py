import torch
import os 
import csv
import numpy as np
from torch import nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, SplitPoint, Schedule1F1B, ScheduleGPipe
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.pipelining.microbatch import TensorChunkSpec

from model import CLIP
from typing import Any, Dict
from argparse import ArgumentParser

class CLIPSchedule(ScheduleGPipe):
    def _compute_loss(self, output, target):
        """
        image_feats: (B, D) float tensor, already L2-normalized
        text_feats:  (B, D) float tensor, already L2-normalized
        logit_scale: scalar tensor, typically model.logit_scale.exp()
        """
        # 1) compute similarity matrix
        #    S[i,j] = cosine(image_i, text_j) * exp(logit_scale)
        # logits = logit_scale * image_feats @ text_feats.t()        # (B, B)

        # 2) labels are just [0,1,2,…,B-1]
        logits = output
        logits *= 14.0
        # B = logits.shape[0]
        # labels = torch.arange(B, device=logits.device)
        if target is None:
            target = torch.arange(B, device=logits.device, dtype=torch.long)
        else:
            target = target.squeeze(0)
            B = logits.size(0)
            assert logits.ndim == 2 and logits.size(1) == B, \
                f"Expected square logits (BxB), got {tuple(logits.shape)}"
            assert target.ndim == 1 and target.size(0) == B, \
                f"Expected target shape ({B},), got {tuple(target.shape)}"
            assert target.min() >= 0 and target.max() < B, \
                f"Target values must be in [0,{B-1}], saw [{int(target.min())},{int(target.max())}]"
        # 3) image→text loss and text→image loss
        # print(logits.shape, target.shape)
        loss_i2t = F.cross_entropy(logits,   target)  # treat rows as preds over text
        loss_t2i = F.cross_entropy(logits.t(), target)  # treat cols as preds over image
        # print(int(os.environ["RANK"]), "loss", loss_i2t, loss_t2i)
        # 4) average the two
        return (loss_i2t + loss_t2i) / 2     

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
            # print(f"Rank {rank} {key} last elapse: {vals[-1]}")
            mean = np.mean(vals)
            std = np.std(vals)
            pct = (mean / total_mean * 100)
            w.writerow([key, round(mean), round(std), round(pct)])

def step(rank, schedule, target, input, optimizer):
    losses = []
    y = target
    if rank == 0:
        x = input
        schedule.step(*input, target=y, losses=losses)
    else:
        # USE ONLY FOR DEBUGGING
        with torch.autograd.detect_anomaly():
            schedule.step(target=target, losses=losses)
    #print(losses)
    #torch.cuda.synchronize()
    optimizer.step()
    optimizer.zero_grad()

import torch.nn.functional as F

def clip_contrastive_loss(logits, target=None):
    """
    image_feats: (B, D) float tensor, already L2-normalized
    text_feats:  (B, D) float tensor, already L2-normalized
    logit_scale: scalar tensor, typically model.logit_scale.exp()
    """
    # 1) compute similarity matrix
    #    S[i,j] = cosine(image_i, text_j) * exp(logit_scale)
    # logits = logit_scale * image_feats @ text_feats.t()        # (B, B)

    # 2) labels are just [0,1,2,…,B-1]
    logits *= 14.0
    # B = logits.shape[0]
    # labels = torch.arange(B, device=logits.device)
    if target is None:
        target = torch.arange(B, device=logits.device, dtype=torch.long)
    else:
        target = target.squeeze(0)
        B = logits.size(0)
        assert logits.ndim == 2 and logits.size(1) == B, \
            f"Expected square logits (B×B), got {tuple(logits.shape)}"
        assert target.ndim == 1 and target.size(0) == B, \
            f"Expected target shape ({B},), got {tuple(target.shape)}"
        assert target.min() >= 0 and target.max() < B, \
            f"Target values must be in [0,{B-1}], saw [{int(target.min())},{int(target.max())}]"
    # 3) image→text loss and text→image loss
    # print(logits.shape, target.shape)
    loss_i2t = F.cross_entropy(logits,   target)  # treat rows as preds over text
    loss_t2i = F.cross_entropy(logits.t(), target)  # treat cols as preds over image
    # print(int(os.environ["RANK"]), "loss", loss_i2t, loss_t2i)
    # 4) average the two
    return (loss_i2t + loss_t2i) / 2

def train(rank, world_size, device, model_args, batch_size=100, seq_len=10, num_iters=2, num_batches=10, num_microbatches=4):
    assert world_size == 3

    # load model

    model = CLIP(**model_args)

    # create pipeline 
    if rank == 0:
        # text
        model.visual = None
    elif rank == 1:
        # vision
        model.transformer = None
        model.ln_final = None
        model.token_embedding = None
        model.positional_embedding = None
    else:
        # combine
        model.visual = None
        model.transformer = None
        model.ln_final = None
        model.token_embedding = None
        model.positional_embedding = None
    model.stage = rank

    stage = PipelineStage(
        model,
        rank,
        world_size,
        device,
        # group=...,
    )
    model.to(device)
    print(f"[Rank {rank}] Loaded")
    # model.train()

    # pipeline schedule
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = clip_contrastive_loss
    # schedule = Schedule1F1B(stage, num_microbatches, loss_fn=criterion)
    schedule = ScheduleGPipe(stage, num_microbatches, loss_fn=criterion, args_chunk_spec=(TensorChunkSpec(0), TensorChunkSpec(0)))
    # schedule = CLIPSchedule(stage, num_microbatches, loss_fn=criterion, args_chunk_spec=(TensorChunkSpec(0), TensorChunkSpec(0)))


    # generate data
    input = None
    if rank == 0:
        width = model_args["image_resolution"]
        input = (
            # image input
            torch.randn(batch_size, 3, width, width,
                dtype=torch.float32,
                device=device,
            ),
            # text input
            torch.randint(
                0,
                model_args['vocab_size'],
                (batch_size, seq_len),
                device=device,
            )
        )

        # pipe = Pipe(
        #     model, 
        #     chunks=num_microbatches, 
        #     input_chunk_spec=(TensorChunkSpec(0), TensorChunkSpec(0)),
        #     output_chunk_spec=TensorChunkSpec(0),
        #     checkpoint='never')


    # target = torch.randn(
    #     batch_size,
    #     seq_len,
    #     model_args['vocab_size'],
    #     requires_grad=True,
    #     device=device,
    # )
    mb_size = batch_size // num_microbatches
    labels = torch.arange(mb_size, device=device)
    target = labels.unsqueeze(0).repeat(num_microbatches, 1)

    dist.barrier()

    # train
    import time

    optimizer = torch.optim.AdamW(model.parameters())

    # step(rank, schedule, target, input, optimizer)
    # exit()

    warmup = 3
    for _ in range(warmup):
        step(rank, schedule, target, input, optimizer)

    start = time.perf_counter()
    for iter in range(num_iters):
        # model.init_tracing()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        #         record_shapes=True, 
        #         profile_memory=True,
        #         with_stack=True) as prof:
            # model.update_tracing("start")
        step(rank, schedule, target, input, optimizer)
            # model.update_tracing("end")
            # prof.export_chrome_trace(f"rank{rank}_distrib.json")
            # model.num_batches_updated += 1
            # if model.num_batches_updated == num_microbatches:
            # model.finish_tracing()
            # log_to_txt(output_path, timestamp, rank, prof.key_averages().table(sort_by="cuda_time_total"))
        # prof.export_chrome_trace(f"rank{rank}_step.json")
    end = time.perf_counter()

    # elapses = model.fetch_traces()
    # log_to_csv(output_path, timestamp, rank, elapses)
    print(
        f"gpipe throughput: {(num_iters * batch_size * seq_len)/(end - start):.0f} tokens/sec"
    )
    print(
        f"gpipe time: {(end - start)*1000/num_iters:.0f} ms"
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(0)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print("device count", torch.cuda.device_count())
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    print(f"[Rank {rank}] Device: {device}")


    parser = ArgumentParser()
    
    parser.add_argument('--num-iters', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-microbatches', type=int)
    parser.add_argument('--model', type=str)

    args = parser.parse_args() 
    clip_config = {
        "embed_dim": 512,
        "image_resolution": 224,
        "vision_layers": 2,
        "vision_width": 256,
        "vision_patch_size": 32,
        "context_length": 32,
        "vocab_size": 49408,
        "transformer_width": 256,
        "transformer_heads": 8,
        "transformer_layers": 2,
    }

    train(
        rank,
        world_size,
        device,
        model_args=clip_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iters=args.num_iters,
        num_microbatches=args.num_microbatches,
    )
