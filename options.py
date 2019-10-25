import argparse


def get_training_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", type=str, default=None, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("--dmpeak_dataset", type=str, default=None, help="dmpeak data path")

    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", type=str, default='./', help="ex)output/bert.model")
    parser.add_argument("--restart", action='store_true', default=False, help="restart with trained model")
    parser.add_argument("--restart_file", type=str, default='.ep0', help="saved model")

    parser.add_argument("--multi_node", action='store_true', default=False, help="multi Nodes training.")
    parser.add_argument('--backend', type=str, default='nccl', help='Name of the backend to use.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank of the current process within a node.')
    # parser.add_argument("--gpu_mode", type=int, default=0, help="GPU mode: 0 - 1 gpu, 1 -- 1 Node, 2 -- multi Nodes")
    # parser.add_argument("--cuda_devices", type=int, default=None, help="CUDA device ids")

    parser.add_argument("--task", type=str, default='pdb', help="task = pdb / pfam / interfam")
    parser.add_argument("--visual", action='store_true', default=False, help="visualization of the model")
    parser.add_argument("--seq_mode", type=str, default='one', help="seq mode = one / two / 2domain")

    parser.add_argument("--abs_position_embed", action='store_true', default=False, help="absolute position embedding")
    parser.add_argument("--relative_attn", action='store_true', default=False, help="use relative attention")
    parser.add_argument("--relative_1d", action='store_true', default=False, help="relative 1d position")
    parser.add_argument("--relative_3d", action='store_true', default=False, help="relative 3d attention")
    parser.add_argument("--target_intra_dm", action='store_true', default=False, help="use intra_dm as training target")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=1, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=12, help="maximum sequence len")

    # parser.add_argument("-cl", "--corpus_lines", type=int, default=1000, help="total number of lines in corpus")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    # parser.add_argument("--on_memory", action='store_true', default=False, help="Loading on memory: true or false")

    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--log_dir", type=str, default='./runs', help="directory for tensorboard log")
    parser.add_argument("--save_freq", type=int, default=10, help="save model every n epochs")
    parser.add_argument("--save_prefix", type=str, default='', help="prefix for saving a model")

    parser.add_argument("--exp_i", type=int, default=0, help="experiment id")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--lr_scheduler", type=str, default='decay', help="learning rate scheduler")

    parser.add_argument("--warmup_steps", type=int, default=300, help="warm up steps")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser


def parse_args_and_arch(parser):
    args = parser.parse_args()
    return args
