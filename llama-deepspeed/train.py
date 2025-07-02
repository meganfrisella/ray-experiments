
import torch
import torch.nn.functional as F
from model import Transformer, LLAMA_DEBUG, LLAMA_3B
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import argparse
import os
import csv
import time
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import Dataset, DataLoader
import numpy as np


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
            print(f"Rank {rank} {key} last elapse: {vals[-1]}")
            mean = np.mean(vals)
            std = np.std(vals)
            pct = (mean / total_mean * 100)
            w.writerow([key, round(mean), round(std), round(pct)])



def to_layers(model, device):
    for layer in model.layers:
        # layer.mod.freqs_cis = layer.mod.freqs_cis.to(device)
        layer.freqs_cis = layer.freqs_cis.to(device)
        # if layer.mod.mask is not None:
        #     layer.mod.mask = layer.mod.mask.to(device)
        if layer.mask is not None:
            layer.mask = layer.mask.to(device)

    layers = torch.nn.Sequential(
        model.tok_embeddings.to(device),
        *[layer.to(device) for layer in model.layers],
        model.norm.to(device),
        model.output.to(device),
    )
    return layers


class DummyLlamaDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size,(self.seq_len,)).long()
        labels = torch.randn(self.seq_len,self.vocab_size)
        return input_ids, labels

    
def train(args, device):
    params = LLAMA_3B
    model = Transformer(args.local_rank, args.seq_len, params)
    pipe = PipelineModule(
        layers=to_layers(model, device),
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
    )
    dataset = DummyLlamaDataset(args.steps * args.batch_size, args.seq_len, params.vocab_size)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
    )

    # events = {"start": [], "end": []}
    warmup = 3
    for _ in range(warmup):
        engine.train_batch()

    start = time.perf_counter()
    for i in range(args.steps):
        # model.init_tracing()
        # model.update_tracing("start")
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        #         record_shapes=True, 
        #         profile_memory=True,
        #         with_stack=True) as prof:
        #     with record_function("distrib"):
        loss = engine.train_batch()
        # prof.export_chrome_trace(f"rank{args.local_rank}_ds.json")

        # model.update_tracing("end")
        # model.finish_tracing()
    end = time.perf_counter()

    # elapses = model.fetch_traces()
    # log_to_csv(args.output_path, args.timestamp, args.local_rank, elapses)
    print(
        f"1f1b throughput: {(args.steps * args.batch_size * args.seq_len)/(end - start):.0f} tokens/sec"
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=2,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int)
    parser.add_argument('--seq-len',
                        type=int)
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser.add_argument('-o',
                        '--output-path',
                            type=str)
    parser.add_argument('-t',
                        '--timestamp',
                            type=str)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    args.local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{args.local_rank % torch.cuda.device_count()}")

    deepspeed.init_distributed(dist_backend=args.backend)
    torch.cuda.set_device(device)

    print(f"Rank {args.local_rank} device {device}")

    train(args, device)