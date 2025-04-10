
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig, LlamaForCausalLM
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import argparse
import os
from torch.utils.data import Dataset, DataLoader

class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, attention_mask, position_ids = args
        inputs_embeds = super().forward(input_ids)
        return inputs_embeds, attention_mask, position_ids


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, idx: int, activation_checkpointing: bool = False):
        super().__init__(config, idx)
        self.activation_checkpointing = activation_checkpointing
        # for name, param in self.named_parameters():
        #     if "norm" in name:
        #         continue
        #     param.data = param.data.to(dtype)

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, attention_mask, position_ids = args
        outputs = LlamaDecoderLayer.forward(self,
                                            hidden_states,
                                            attention_mask,
                                            position_ids,
                                            )
        return outputs[0], attention_mask, position_ids

    def _ckpt_forward(self, args):
        hidden_states, attention_mask, position_ids = args

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return LlamaDecoderLayer.forward(module, *inputs)

            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
            None,
        )
        # layer_outputs = torch.utils.checkpoint.checkpoint(
        #     create_custom_forward(self),
        #     hidden_states,
        #     attention_mask,
        #     position_ids,
        #     None,
        # )

        return outputs, attention_mask, position_ids


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, attention_mask, position_ids = args
        last_hidden_states = super().forward(hidden_states)
        return last_hidden_states


class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        hidden_states = args
        logits = super().forward(hidden_states)
        return logits


class LossLayer(torch.nn.Module):
    def forward(self, args):
        logits, labels = args
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss


def to_layers(model_config, activation_checkpointing=False):
    layers = [
        LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
        # TiedLayerSpec("weight", EmbeddingPipe, model_config.vocab_size, model_config.hidden_size, tied_weight_attr="weight"),
        *[LayerSpec(ParallelTransformerLayerPipe, model_config, idx, activation_checkpointing)
          for idx in range(model_config.num_hidden_layers)],
        LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
        LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False),
        # TiedLayerSpec("weight", LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False,
        #               tied_weight_attr="weight"),
        # LayerSpec(LossLayer),
    ]
    return layers

def get_model(model_config, args, activation_checkpointing_config=None):
    class LlamaModelPipe(PipelineModule):
        def __init__(self, model_config, **kwargs):
            super().__init__(
                layers=to_layers(model_config),
                **kwargs
            )

    return LlamaModelPipe(model_config,
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method='parameters')

def build_dataloader(steps, batch_size, seq_len, vocab_size):
    class MyDataset(Dataset):
        def __init__(self, input, target):
            self.input = input
            self.target = target
        def __len__(self):
            return len(input)
        def __getitem__(self, idx):
            return self.input[idx], self.target[idx]

    input = torch.randint(
        0,
        vocab_size,
        (steps, batch_size, seq_len),
        device=device,
    )
    target = torch.randn(
        steps,
        batch_size,
        seq_len,
        vocab_size,
        requires_grad=True,
        device=device,
    )

    dataset = MyDataset(input, target)
    return DataLoader(dataset, batch_size=batch_size)

def train(args, device):
    model_args = LlamaConfig(
        dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        vocab_size=128,
        rms_norm_eps=1e-5,
        rope_theta=500000,
    )
    model = get_model(model_args, args)
    loader = build_dataloader(args.steps, args.batch_size, args.seq_len, model_args.vocab_size)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # training_data=(input,target),
    )

    elapses = []
    for i in range(args.steps):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        train_iter = iter(loader)
        loss = engine.train_batch(data_iter=train_iter)

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

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{args.local_rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    print(f"Rank {args.local_rank} device {device}")

    train(args, device)