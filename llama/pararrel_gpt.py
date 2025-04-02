# $ torchrun --nproc-per-node 4 pippy_gpt2.py

import argparse
import os

import torch
import torch.distributed as dist
import time
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint, Schedule1F1B

from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Model

# from hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):
    # Model configs
    config = GPT2Config()
    config.n_embd = args.n_embd or config.n_embd
    config.n_layer = 2 or config.n_layer
    config.n_head = 8 or config.n_head
    config.num_labels = 4
    # config.pad_token_id = config.eos_token_id
    print("[Rank {}] Using device: {}".format(args.rank, args.device))

    # Create model
    model_class = GPT2ForSequenceClassification
    model_name = "GPT2ForSequenceClassification"
    gpt2 = model_class(config)
    gpt2.to(args.device)
    # gpt2.eval()
    if args.rank == 0:
        print(gpt2.config)
        # print(f"GPT-2 total number of params = {get_number_of_params(gpt2) // 10 ** 6}M")
        print(gpt2)

    optimizer = torch.optim.AdamW(gpt2.parameters(), lr = 3e-5)
    # print(gpt2.parameters())
    loss_fn=torch.nn.CrossEntropyLoss()

    # Example microbatch inputs
    # mb_inputs = generate_inputs_for_model(
    #     model_class, gpt2, model_name, args.batch_size // args.chunks, args.device)

    example_input_microbatch = torch.randint(0, gpt2.config.vocab_size, device = args.device,  dtype=torch.int64, requires_grad = False, size = (args.batch_size // args.chunks, 1024))
    y = torch.randint(0, gpt2.config.num_labels -1 , size = (args.batch_size,),device = args.device,  dtype=torch.int64, requires_grad = False)
    attention_mask = torch.ones_like(example_input_microbatch)

    # example_input_microbatch = x.chunk(args.chunks)[0]
    # Pipeline split spec
    decoders_per_rank = (gpt2.config.n_layer + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {decoders_per_rank}")
    split_spec = {
        f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    # Create pipeline representation
    pipe = pipeline(
        gpt2,
        mb_args=(example_input_microbatch,),
        mb_kwargs={ 'use_cache': False, 'output_attentions': False},
        split_spec=split_spec,
    )

    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)
    # print(f"Pipeline stage {args.rank} {get_number_of_params(smod) // 10 ** 6}M params")

    # Create schedule runtime
    stage = pipe.build_stage(
        args.rank,
        device=args.device,
    )

    # def tokenwise_loss_fn(outputs, targets):
    #     loss_fn = torch.nn.CrossEntropyLoss()
    #     print(f'size3131: {outputs}')
    #     print(f'size3132: {targets}')
    #     return loss_fn(outputs, targets)



    # Attach to a schedule
    schedule = ScheduleGPipe(stage, n_microbatches = args.chunks, loss_fn = loss_fn)

    # Full batch inputs as in single-worker case
    # inputs = generate_inputs_for_model(
    #     model_class, gpt2, model_name, args.batch_size, args.device, True)

    # data_without_labels = {key: value for key, value in inputs.items() if key != 'labels'}
    # labels = inputs['labels']

    # inputs = generate_inputs_for_model(
    #     model_class, gpt2, model_name, args.batch_size, args.device, True)
    # abc = inputs['input_ids']
    # print(f'inputsabc size: {inputs}')

  # 10 inoputs, 
    x1 = torch.randint(0, gpt2.config.vocab_size, device = args.device,  dtype=torch.int64, requires_grad = False, size = (args.batch_size, 1024))
    x2 = torch.randint(0, gpt2.config.vocab_size, device = args.device,  dtype=torch.int64, requires_grad = False, size = (args.batch_size, 1024))
    batches = {0 : x1, 1: x2}
    # print(f'x : {x}')

    gpt2.train()
    # print(f'sizedef: {x.shape}')
    # Run
    num_training_steps = 100

    start_time = time.perf_counter()
    for i in range(num_training_steps):
        for step in range(args.batches):
            x = batches[step]
            # attention_mask = torch.ones_like(x)
        # Optionally, regenerate or shuffle your batch data here.
        # For this demo we re-use the same x and y every step.

            if args.rank == 0:
                # Rank 0 feeds the input into the pipeline
                # print(f'x: {x}')
                schedule.step(x)
            else:
                # Other ranks provide targets and capture losses
                losses = []
                schedule.step(target=y, losses=losses)
                if losses:
                    # Print loss from the first loss tensor (if available)
                    print(f"[Rank {args.rank}] Step {i + 1}/{num_training_steps}, Loss: {losses[0].item()}")

            # After each microbatch pass, update the model parameters
            optimizer.step()
            optimizer.zero_grad()

        if args.rank == 0:
            print(f"[Rank {args.rank}] Completed training step {i+1}/{num_training_steps}")

  
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f'total run time: {runtime}')
    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {args.rank} completes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default= 128)
    # Note: this specific example requires: 1) a batch size that is divisible by
    # the number of chunks; 2) the division result (i.e. chunk size) must be 1,
    # otherwise padding token must be provided too (see GPT-2's forward function)
    parser.add_argument('--batch_size', type=int, default= 128)
    parser.add_argument('--batches', type=int, default= 1)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    # 36 batch size, 

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)
