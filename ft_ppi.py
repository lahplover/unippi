import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import BERTLM
from trainer import BERTTrainer
from data import DatasetInterDomain
import options


parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)

# GPU mode: 0 - 1 gpu, 1 -- 1 Node, 2 -- multi Nodes
gpu_mode = 1
cuda_device_id = 0

# Setup cuda device
if (gpu_mode == 1) & torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


final_test_dataset = DatasetInterDomain(args.test_dataset, seq_len=args.seq_len, seq_mode=args.seq_mode,
                                        relative_3d_size=10, relative_3d_step=2,
                                        relative_3d=args.relative_3d, on_memory=args.on_memory)

final_test_data_loader = DataLoader(final_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

print("Building BERT model")
# Initialize the BERT Language Model, with BERT model
model = BERTLM(len(final_test_dataset.vocab),
               hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads,
               seq_mode=args.seq_mode,
               abs_position_embed=args.abs_position_embed,
               relative_attn=args.relative_attn,
               relative_1d=args.relative_1d,
               max_relative_1d_positions=10,
               relative_3d=args.relative_3d,
               relative_3d_vocab_size=len(final_test_dataset.vocab_3d))

print("reload pretrained BERT model")
model.load_state_dict(torch.load(args.restart_file,  map_location=torch.device('cpu')))

model.to(device)


if (gpu_mode == 1) and (torch.cuda.device_count() > 1):
    print("Using %d GPUS for BERT" % torch.cuda.device_count())
    model = nn.DataParallel(model)


writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

trainer = BERTTrainer(writer, model,
                      seq_mode=args.seq_mode,
                      train_dataloader=final_test_data_loader,
                      test_dataloader=final_test_data_loader,
                      lr=args.lr,
                      betas=(args.adam_beta1, args.adam_beta2),
                      weight_decay=args.adam_weight_decay,
                      warmup_steps=args.warmup_steps,
                      lr_scheduler=args.lr_scheduler,
                      device=device,
                      log_freq=args.log_freq)

print('Testing')
trainer.test(1)


