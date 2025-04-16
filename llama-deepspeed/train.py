
import torch
import torch.nn.functional as F
from model import Transformer, LLAMA_DEBUG
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

def to_layers(model, device):
    for layer in model.layers:
        layer.freqs_cis = layer.freqs_cis.to(device)
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
    params = LLAMA_DEBUG
    model = Transformer(args.local_rank, args.seq_len, params)
    model = PipelineModule(
        layers=to_layers(model, device),
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
    )
    dataset = DummyLlamaDataset(args.steps * args.batch_size, args.seq_len, params.vocab_size)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
    )

    elapses = []
    for i in range(args.steps):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        loss = engine.train_batch()

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        time = start_event.elapsed_time(end_event)
        elapses.append(time)
        print(f"rank {args.local_rank} iter {i} loss {loss} time {time}")

    print(f"Rank {args.local_rank} total time: {np.sum(elapses)/args.steps}")

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