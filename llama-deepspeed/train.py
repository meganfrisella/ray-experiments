
import torch
import torch.nn.functional as F
from model import EmbeddingBlock, FreqsCis, TransformerBlock, RMSNorm, OutputLayer, LLAMA_DEBUG
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

class EmbeddingPipe(EmbeddingBlock):
    def forward(self, tokens):
        seqlen = tokens.shape[1]
        h = super().forward(tokens)
        return h, seqlen

class FreqsCisPipe(FreqsCis):
    def forward(self, args):
        h, seqlen = args
        return super().forward(h, seqlen)

class TransformerBlockPipe(TransformerBlock):
    def forward(self, args):
        h, freqs_cis, mask = args
        start_pos = 0
        return super().forward(h, start_pos, freqs_cis, mask)

class RMSNormPipe(RMSNorm):
    def forward(self, args):
        h, _, _ = args
        return super().forward(h)

class OutputLayerPipe(OutputLayer):
    def forward(self, h):
        out = super().forward(h)
        print("HERE", out, out.requires_grad)
        return out

class CrossEntropyLoss_(torch.nn.Module):
  def forward(self, args):
    logits, labels = args
    logits = logits.requires_grad_(True)
    return torch.nn.CrossEntropyLoss(logits, labels)

def to_layers(params, device):
    # layers = [
    #     LayerSpec(EmbeddingPipe, params.vocab_size, params.dim),
    #     LayerSpec(FreqsCisPipe, params.dim // params.n_heads, params.max_seq_len * 2, params.rope_theta, device),
    #     *[LayerSpec(TransformerBlockPipe, idx, params)
    #       for idx in range(params.n_layers)],
    #     LayerSpec(RMSNormPipe, params.dim, params.norm_eps),
    #     LayerSpec(OutputLayerPipe, params.dim, params.vocab_size, bias=False),
    #     LayerSpec(CrossEntropyLoss_),
    # ]
    layers = torch.nn.Sequential(
        EmbeddingPipe(params.vocab_size, params.dim),
        FreqsCisPipe(params.dim // params.n_heads, params.max_seq_len * 2, params.rope_theta, device),
        *[TransformerBlockPipe(idx, params) for idx in range(params.n_layers)],
        RMSNormPipe(params.dim, params.norm_eps),
        OutputLayerPipe(params.dim, params.vocab_size, bias=False),
        # CrossEntropyLoss_(),
    )
    return layers

def get_model(params, args, device, activation_checkpointing_config=None):
    class LlamaModelPipe(PipelineModule):
        def __init__(self, params, device, **kwargs):
            super().__init__(
                layers=to_layers(params, device),
                **kwargs
            )

    return LlamaModelPipe(params,
                        device,
                        loss_fn=torch.nn.CrossEntropyLoss(),
                        num_stages=args.pipeline_parallel_size,
                        partition_method='parameters')

def build_dataset(rank, device, steps, batch_size, seq_len, vocab_size):
    class MyDataset(Dataset):
        def __init__(self, input, target):
            self.input = input
            self.target = target
        def __len__(self):
            return len(input)
        def __getitem__(self, idx):
            return self.input[idx], self.target[idx]

    # dist.barrier()
    # if rank != 0:
    #     dist.barrier()

    input = torch.randint(
        0,
        vocab_size,
        (steps * batch_size, seq_len),
        # device=device,
    )
    target = torch.randn(
        steps * batch_size,
        seq_len,
        vocab_size,
        # requires_grad=True,
        # device=device,
    )
    dataset = MyDataset(input, target)

    # if rank == 0:
    #     dist.barrier()
    
    return dataset
    # return DataLoader(dataset, batch_size=batch_size)

def train(args, device):
    params = LLAMA_DEBUG
    model = get_model(params, args, device)
    dataset = build_dataset(args.local_rank, device, args.steps, args.batch_size, args.seq_len, params.vocab_size)
    # loader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
    )

    elapses = []
    for i in range(args.steps):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # train_iter = iter(loader)
        loss = engine.train_batch() # (data_iter=train_iter)

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        time = start_event.elapsed_time(end_event)
        elapses.append(time)
        print(f"rank {args.local_rank} iter {i} loss {loss} time {time}")

    print(f"Rank {args.local_rank} total time: {np.sum(elapses)/args.steps}")
    dist.barrier()
    dist.destroy_process_group()

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
    # dist.init_process_group(backend="nccl", rank=args.local_rank, world_size=args.pipeline_parallel_size)

    deepspeed.init_distributed(dist_backend=args.backend)
    torch.cuda.set_device(device)

    print(f"Rank {args.local_rank} device {device}")

    train(args, device)