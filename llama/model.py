# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector

from torch.profiler import profile, record_function, ProfilerActivity

# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )


logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


LLAMA_DEBUG = ModelArgs(
    dim=512,  # 1/2
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=256,
    ffn_dim_multiplier=1.5,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)


LLAMA_1B = ModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=256,
    ffn_dim_multiplier=1.5,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)


LLAMA_3B = ModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)

LLAMA_8B = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # self.wq = ColumnParallelLinear(
        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.wk = ColumnParallelLinear(
        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.wv = ColumnParallelLinear(
        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.wo = RowParallelLinear(
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            # input_is_parallel=True,
            # init_method=lambda x: x,
        )

        # [NOTE] Disable KV cache during training.
        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # [NOTE] Disable KV cache during training.

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # self.w1 = ColumnParallelLinear(
        self.w1 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.w2 = RowParallelLinear(
        self.w2 = torch.nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            # input_is_parallel=True,
            # init_method=lambda x: x,
        )
        # self.w3 = ColumnParallelLinear(
        self.w3 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, rank, layers_per_rank, device, params: ModelArgs):
        super().__init__()
        self.rank = rank
        self.device = device
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.fwd_time = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

        # buckets = [
        #     "VocabParallelEmbedding",
        #     [
        #         "Attention",
        #         "FeedForward",
        #         "RMSNorm * 2",
        #     ],
        #     "ColumnParallelLinear",
        # ]

        def log_size(layer, indent=0):
            num_params = sum(p.numel() for p in layer.parameters())
            size_mib = num_params * 4 / (1024 * 1024)
            indent_str = "  " * indent
            logger.info(
                f"{indent_str}{layer.__class__.__name__}: {round(size_mib)} MiB"
            )
            if size_mib < 25:
                return
            for _, child in layer.named_children():
                log_size(child, indent + 1)

        # self.tok_embeddings = VocabParallelEmbedding(
        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size,
            params.dim,
            # init_method=lambda x: x,
        )
        log_size(self.tok_embeddings)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(layers_per_rank):
            self.layers.append(TransformerBlock(layer_id, params))
        log_size(self.layers[0])

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        log_size(self.norm)
        # self.output = ColumnParallelLinear(
        self.output = torch.nn.Linear(
            params.dim,
            params.vocab_size,
            bias=False,
            # init_method=lambda x: x,
        )
        log_size(self.output)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        ).to(device)

    def init_tracing(self):
    #   self.num_batches_forwarded = 0
    #   self.num_batches_updated = 0
      self.events: Dict[str, Any] = {
          "start": [],
          "end": [],
          "fwd.starts": [],
          "fwd.ends": [],
          "bwd.starts": [],
          "bwd.ends": [],
          "upd.starts": [],
          "upd.ends": [],
      }
      self.fwd_total = 0
      self.fwd_cnt = 0
      torch.cuda.synchronize()

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def update_tracing(self, key: str):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        assert key in self.events
        self.events[key].append(event)

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()

        assert len(self.events["start"]) == 1
        assert len(self.events["end"]) == 1

        def millis_to_micros(millis: float) -> int:
            return round(millis * 1e3)

        total = self.events["start"][0].elapsed_time(self.events["end"][0])
        fw_total = sum(
            [
                fw_start.elapsed_time(fw_end)
                for fw_start, fw_end in zip(
                    self.events["fwd.starts"],
                    self.events["fwd.ends"],
                )
            ]
        )
        bw_total = sum(
            [
                bw_start.elapsed_time(bw_end)
                for bw_start, bw_end in zip(
                    self.events["bwd.starts"],
                    self.events["bwd.ends"],
                )
            ]
        )
        up_total = sum(
            [
                up_start.elapsed_time(up_end)
                for up_start, up_end in zip(
                    self.events["upd.starts"],
                    self.events["upd.ends"],
                )
            ]
        )
        self.elapses["total"].append(millis_to_micros(total))
        self.elapses["fwd_total"].append(millis_to_micros(fw_total))
        self.elapses["bwd_total"].append(millis_to_micros(bw_total))
        self.elapses["upd_total"].append(millis_to_micros(up_total))

    def forward(self, tokens: torch.TensorType):

        # self.update_tracing("fwd.starts")

        seqlen = tokens.shape[1]
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # self.freqs_cis = self.freqs_cis.to(h.device)
        start_pos = 0
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h

        # self.update_tracing("fwd.ends")

        return output
