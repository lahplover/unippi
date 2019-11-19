''' Translate input text with trained model. '''

import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from dataset import TranslationDataset
from model import Translator


parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('--model', required=True,
                    help='Path to model .pt file')
parser.add_argument('--src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('--output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('--beam_size', type=int, default=3,
                    help='Beam size')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch size')
parser.add_argument('--n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('--no_cuda', action='store_true')

args = parser.parse_args()
args.cuda = not args.no_cuda

args.hidden = 256
args.layers = 4
args.attn_heads = 8
args.seq_len = 12

# Prepare DataLoader
test_dataset = TranslationDataset(args.src, seq_len=args.seq_len)
args.vocab_len = len(test_dataset.vocab)


test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

translator = Translator(args)

with open(args.output, 'wt') as f:
    for src in tqdm(test_data_loader, mininterval=2, desc='  - (Test)', leave=False):
        all_hyp, all_scores = translator.translate_batch(src)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred_line = ' '.join([test_dataset.idx2vocab[idx] for idx in idx_seq])
                f.write(pred_line + '\n')
print('[Info] Finished.')

