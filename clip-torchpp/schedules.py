from torch.distributed.pipelining.schedules import PipelineScheduleSingle

# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

from torch.distributed.pipelining._utils import generate_stage_to_rank_mapping
from torch.distributed.pipelining.microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from torch.distributed.pipelining.stage import _PipelineStageBase, _sorted_batch_p2p


class ClipGPipeSchedule(PipelineScheduleSingle):
    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs, target_mbs, losses = (
            self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        )

        # initialize each stage’s weights on the very first chunk
        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        fwd_sends_to_wait = []

        # Forward
        for i in range(self._n_microbatches):
            with record_function(f"Forward chunk {i}"):
                # 1) recv activations from previous stage (no‐ops on stage 0)
                ops = self._stage.get_fwd_recv_ops(i)
                for work in _sorted_batch_p2p(ops, desc="fwd_recv").values():
                    work.wait()

                # 2) run this stage’s forward
                output = self._stage.forward_one_chunk(i,
                                                      arg_mbs[i],
                                                      kwarg_mbs[i])

                # 3) only the last stage (idx==2) computes & records the contrastive loss
                if self._stage.stage_index == 2:
                    self._maybe_compute_loss(self._stage,
                                             output,
                                             target_mbs,
                                             i)

                # 4) send activations to next stage (no‐ops on stage 2)
                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

        # wait for all the sends to drain before starting backward
        for work in fwd_sends_to_wait:
            work.wait()

        # if nobody asked for gradients, we’re done
        if not self._has_backward:
            return

        bwd_sends_to_wait = []

        # ─── Backward ──────────────────────────────────────────────────────────────
        for i in range(self._n_microbatches):
            with record_function(f"Backward chunk {i}"):
                # 1) recv gradients from next stage (no‐ops on stage 2)
                ops = self._stage.get_bwd_recv_ops(i)
                for work in _sorted_batch_p2p(ops, desc="bwd_recv").values():
                    work.wait()

                # 2) pull out the loss for this chunk (only non‐None on stage 2)
                loss = self._maybe_get_loss(self._stage, i)

                # 3) run backward on this stage
                self._stage.backward_one_chunk(
                    i,
                    loss=loss,
                    last_backward=(i == self._n_microbatches - 1),
                )

                # 4) send gradients to previous stage (no‐ops on stage 0)
                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

        # scale grads if requested
        self._stage.scale_grads(
            grad_scale_factor=self._n_microbatches
            if self.scale_grads
            else 1
        )

        # stash losses into the user‐provided container
        self._update_losses(self._stage, losses)

        # finally wait for all backward‐sends to drain
        for work in bwd_sends_to_wait:
            work.wait() 
