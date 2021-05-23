import os
import numpy as np
import torch
import torch.nn as nn
from data import DatasetBlock, DatasetBlocksp
from model import Transformer
from trainer import Seq2SeqTrainer
import options
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as pl
from tqdm import tqdm


def load_dataset(args):
    if args.task == 'block':
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = DatasetBlock(args.test_dataset, seq_len=args.seq_len)
    elif args.task == 'blocksp':
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = DatasetBlocksp(args.test_dataset, seq_len=args.seq_len)
    else:
        raise ValueError('unknown task name')

    return test_dataset


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
if gpu_mode == 2:
    torch.cuda.set_device(cuda_device_id)
    device = torch.device(f"cuda:{cuda_device_id}")
elif (gpu_mode == 1) & torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

test_dataset = load_dataset(args)

test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# build model
print("Building BERT model")
# Initialize the BERT Language Model, with BERT model
model = Transformer(len(test_dataset.vocab),
                    hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

if args.restart:
    print("reload pretrained BERT model")
    model.load_state_dict(torch.load(args.restart_file, map_location=torch.device('cpu')))

model.to(device)

# Distributed GPU training if CUDA can detect more than 1 GPU
if gpu_mode == 2:
    print(f"Using GPU {cuda_device_id}")
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cuda_device_id],
                                                output_device=cuda_device_id,
                                                find_unused_parameters=True)
elif (gpu_mode == 1) and (torch.cuda.device_count() > 1):
    print("Using %d GPUS for BERT" % torch.cuda.device_count())
    model = nn.DataParallel(model)

model.eval()
torch.set_grad_enabled(False)

# print(f'exp_{args.exp_i}, learning_rate: {args.lr}, weight_decay: {args.weight_decay}')
writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

trainer = Seq2SeqTrainer(writer, model,
                      train_dataloader=test_data_loader,
                      test_dataloader=test_data_loader,
                      lr=args.lr,
                      betas=(args.adam_beta1, args.adam_beta2),
                      weight_decay=args.adam_weight_decay,
                      warmup_steps=args.warmup_steps,
                      lr_scheduler=args.lr_scheduler,
                      device=device,
                      log_freq=args.log_freq,
                      label_smoothing=args.label_smoothing)

print('Testing')
trainer.test(0)

# avg_loss = 0.0
# total_correct = 0
# total_element = 0
# len_data_loader = len(test_data_loader)
#
# loss_fn = nn.NLLLoss(ignore_index=0)
#
# for i, data in tqdm(enumerate(test_data_loader)):
#     data = {key: value.to(device) for key, value in data.items()}
#
#     seq_output = model.forward(data["src"], data["tgt_x"])
#
#     # NLLLoss of predicting masked token word
#     # (N, T, E) --> (N, E, T)
#     print(seq_output.transpose(1, 2).argmax(dim=1))
#
#     loss = loss_fn(seq_output.transpose(1, 2), data["tgt_y"])
#
#     # masked token prediction accuracy
#     idx = (data["tgt_y"] > 0)
#     # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx])
#     # print(mask_lm_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["bert_label"][idx]))
#     correct = seq_output.transpose(1, 2).argmax(dim=1)[idx].eq(data["tgt_y"][idx]).sum().item()
#     batch_n_element = data["tgt_y"][idx].nelement()
#     total_correct += correct
#     total_element += batch_n_element
#
#     avg_loss += loss.item()
#
#     print(f'loss: {loss}  Accuracy/test: {100.0 * correct / batch_n_element}')
#
#     total_correct += correct
#     total_element += batch_n_element
#
# print(f"avg_loss= {avg_loss / len_data_loader}, total_acc= {total_correct * 100.0 / total_element}")
#
#


