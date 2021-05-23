import os
import numpy as np
import torch
import torch.nn as nn
from data import DatasetSeqLMIter
from model import ProLM
from trainer import SeqLMTrainer
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter


parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)

if args.multi_node:
    distributed.init_process_group(
        backend=args.backend,
        init_method='env://'
        # world_size=args.world_size,
        # rank=args.rank,
    )

# GPU mode: 0 - 1 gpu, 1 -- 1 Node, 2 -- multi Nodes
gpu_mode = 1
cuda_device_id = 0
if distributed.is_available():
    if distributed.is_initialized():
        gpu_mode = 2
        cuda_device_id = args.local_rank
elif not torch.cuda.is_available():
    gpu_mode = 0

# Setup cuda device
print('only multi_node training supported')
assert(gpu_mode == 2)

writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

torch.cuda.set_device(cuda_device_id)
device = torch.device(f"cuda:{cuda_device_id}")

global_rank = distributed.get_rank()
# print(f"Loading Train Dataset: {global_rank}")

train_data_path = f'data/multinode/uniref50_train_shuffle_{global_rank}_iter.csv'
print(train_data_path)

train_dataset = DatasetSeqLMIter(train_data_path)

train_data_loader = DataLoader(train_dataset, num_workers=args.num_workers)
test_data_loader = None

# build model
print("Building BERT model")
model = ProLM(len(train_dataset.vocab),
              hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

if args.restart:
    print("reload pretrained BERT model")
    model.load_state_dict(torch.load(args.restart_file, map_location=torch.device('cpu')))

model.to(device)

# Distributed GPU training if CUDA can detect more than 1 GPU
print(f"Using GPU {cuda_device_id}")
model = nn.parallel.DistributedDataParallel(model,
                                            device_ids=[cuda_device_id],
                                            output_device=cuda_device_id)

trainer = SeqLMTrainer(writer, model,
                       train_dataloader=train_data_loader,
                       test_dataloader=test_data_loader,
                       lr=args.lr,
                       betas=(args.adam_beta1, args.adam_beta2),
                       weight_decay=args.adam_weight_decay,
                       warmup_steps=args.warmup_steps,
                       lr_scheduler=args.lr_scheduler,
                       device=device,
                       log_freq=args.log_freq)


print("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)

    # Saving the current model on file_path
    if distributed.get_rank() == 0:
        output_path = args.output_path + f"{args.save_prefix}_ep{epoch}"
        torch.save(model.module.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)

    if test_data_loader is not None:
        trainer.test(epoch)

