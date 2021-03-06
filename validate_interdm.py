import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data import DatasetPDBInter, DatasetSeqInter, DatasetPDBInterDomain
from model import InterDM
from trainer import DMTrainer
import options
from tqdm import tqdm


def load_dataset(args):
    print("Loading Test Dataset", args.test_dataset)
    if args.task == 'pdb':
        test_dataset = DatasetPDBInter(args.test_dataset, seq_len=args.seq_len,
                                       relative_3d=args.relative_3d,
                                       relative_3d_size=10, relative_3d_step=2,
                                       target_intra_dm=args.target_intra_dm)
    elif args.task == 'pfam':
        test_dataset = DatasetSeqInter(args.test_dataset, seq_len=args.seq_len,
                                       relative_3d=args.relative_3d,
                                       relative_3d_size=10, relative_3d_step=2,
                                       target_intra_dm=args.target_intra_dm)
    elif args.task == 'interfam':
        test_dataset = DatasetPDBInterDomain(args.test_dataset, seq_len=args.seq_len,
                                             relative_3d=args.relative_3d,
                                             relative_3d_size=10, relative_3d_step=2,
                                             target_intra_dm=args.target_intra_dm)
    else:
        raise ValueError('unknown task name')
    return test_dataset


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


test_dataset = load_dataset(args)

test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

print("Building model")
# Initialize the BERT Language Model, with BERT model
model = InterDM(len(test_dataset.vocab),
               hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads,
               seq_mode=args.seq_mode,
               abs_position_embed=args.abs_position_embed,
               relative_attn=args.relative_attn,
               relative_1d=args.relative_1d,
               max_relative_1d_positions=10,
               relative_3d=args.relative_3d,
               relative_3d_vocab_size=len(test_dataset.vocab_3d))

print("reload pretrained model")
model.load_state_dict(torch.load(args.restart_file,  map_location=torch.device('cpu')))

model.to(device)


if (gpu_mode == 1) and (torch.cuda.device_count() > 1):
    print("Using %d GPUS" % torch.cuda.device_count())
    model = nn.DataParallel(model)


writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

trainer = DMTrainer(writer, model,
                    no_msa_index=test_dataset.vocab_3d['no_msa'],
                    train_dataloader=test_data_loader,
                    test_dataloader=test_data_loader,
                    lr=args.lr,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    warmup_steps=args.warmup_steps,
                    lr_scheduler=args.lr_scheduler,
                    device=device,
                    log_freq=args.log_freq)


if not args.visual:
    print('Testing')
    trainer.test(1)
else:
    # visualization
    import matplotlib.pyplot as pl
    # no_msa_index = test_dataset.vocab_3d['no_msa']
    # dm_weight = (no_msa_index - torch.arange(no_msa_index + 1, device=device,
    #                                          requires_grad=False, dtype=torch.float)) ** 2
    dm_weight = torch.tensor([100] * 5 + [50] * 5 + [1, 0],
                             device=device, requires_grad=False, dtype=torch.float)
    loss_dm = nn.NLLLoss(weight=dm_weight)

    for i, data in tqdm(enumerate(test_data_loader)):
        data = {key: value.to(device) for key, value in data.items()}

        dist_mat_output = model(data["seq_input"], data["segment_label"],
                                     distance_matrix=data["dist_mat_input"])
        loss = loss_dm(dist_mat_output, data["dist_mat_target"])
        print(loss)
        for j in [0, 8]:
            fig = pl.figure()
            ax = pl.subplot(131)
            pl.imshow(dist_mat_output[j].argmax(dim=0))
            ax = pl.subplot(132)
            pl.imshow(data['dist_mat_target'][j])
            if args.relative_3d:
                ax = pl.subplot(133)
                pl.imshow(data['dist_mat_input'][j])
            pl.savefig(f'fig_dm/r3d-interpfam-{i}-{j}.pdf')
            pl.close(fig)

# loss = loss_dm(dist_mat_output[0:1], data["dist_mat_target"][0:1])



