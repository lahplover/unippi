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


def load_dataset(args):
    if args.task == 'block':
        train_data_path = args.train_dataset
        print("Loading Train Dataset", train_data_path)
        train_dataset = DatasetBlock(train_data_path, seq_len=args.seq_len)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = DatasetBlock(args.test_dataset, seq_len=args.seq_len) \
            if args.test_dataset is not None else None
    elif args.task == 'blocksp':
        train_data_path = args.train_dataset
        print("Loading Train Dataset", train_data_path)
        train_dataset = DatasetBlocksp(train_data_path, seq_len=args.seq_len)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = DatasetBlocksp(args.test_dataset, seq_len=args.seq_len) \
            if args.test_dataset is not None else None
    else:
        raise ValueError('unknown task name')

    return train_dataset, test_dataset


def main():
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

    # print(f"world size: {distributed.get_world_size()}")
    # print(f"rank: {distributed.get_rank()}")
    # for large dataset, prepare a subset of dataset for each GPU, load the dataset with id=get_rank()
    # print("Loading Train Dataset", args.train_dataset)
    # train_data_path = args.train_dataset

    train_dataset, test_dataset = load_dataset(args)

    print("Creating Dataloader")
    if gpu_mode == 2:
        datasampler = DistributedSampler(train_dataset, shuffle=True)
        # datasampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, sampler=datasampler)
        # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        #                                num_workers=args.num_workers)
    else:
        datasampler = RandomSampler(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, sampler=datasampler)
        # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        #                                num_workers=args.num_workers)

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    # build model
    print("Building BERT model")
    # Initialize the BERT Language Model, with BERT model
    model = Transformer(len(train_dataset.vocab),
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

    # print(f'exp_{args.exp_i}, learning_rate: {args.lr}, weight_decay: {args.weight_decay}')
    writer = SummaryWriter(log_dir=args.log_dir + f'/exp{args.exp_i}')

    trainer = Seq2SeqTrainer(writer, model,
                          train_dataloader=train_data_loader,
                          test_dataloader=test_data_loader,
                          lr=args.lr,
                          betas=(args.adam_beta1, args.adam_beta2),
                          weight_decay=args.adam_weight_decay,
                          warmup_steps=args.warmup_steps,
                          lr_scheduler=args.lr_scheduler,
                          device=device,
                          log_freq=args.log_freq)

    if args.epochs > 0:
        print("Training Start")
        for epoch in range(args.epochs):
            output_path = args.output_path + f"{args.save_prefix}_ep{epoch}"
            # trainer.train(epoch, output_path=output_path)
            trainer.train(epoch)
            if epoch % args.save_freq == 0:
                # Saving the current BERT model on file_path
                if gpu_mode == 0:
                    torch.save(model.state_dict(), output_path)
                elif gpu_mode == 1:
                    torch.save(model.module.state_dict(), output_path)
                elif gpu_mode == 2:
                    if distributed.get_rank() == 0:
                        torch.save(model.module.state_dict(), output_path)
                # model.to(device)
                print("EP:%d Model Saved on:" % epoch, output_path)

            if test_data_loader is not None:
                trainer.test(epoch)
    else:
        print("run validation only")
        trainer.test(0)


if __name__ == '__main__':
    main()
