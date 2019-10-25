import torch.nn as nn
from .bert import BERT


class FTPPI(nn.Module):
    """
    finetune the Next Sentence Prediction Model
    """

    def __init__(self, vocab_size, hidden=128, n_layers=1, attn_heads=8,
                 seq_mode='one',
                 abs_position_embed=True,
                 relative_attn=False,
                 relative_1d=False,
                 max_relative_1d_positions=10,
                 relative_3d=False,
                 relative_3d_vocab_size=10):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.seq_mode = seq_mode

        self.bert = BERT(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads,
                         seq_mode=seq_mode,
                         abs_position_embed=abs_position_embed,
                         relative_attn=relative_attn,
                         relative_1d=relative_1d,
                         max_relative_1d_positions=max_relative_1d_positions,
                         relative_3d=relative_3d,
                         relative_3d_vocab_size=relative_3d_vocab_size)

        self.next_sentence = NextSentencePrediction(hidden)

    def forward(self, x, segment_label=None, distance_matrix=None):
        x = self.bert(x, segment_label, distance_matrix)
        return self.next_sentence(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    x: (N, S, E) -> (N, E) -> (N, 2)
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = x[:, 0]
        return self.softmax(self.linear(x))


