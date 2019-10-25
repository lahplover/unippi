import torch.nn as nn
from .bert import BERT


class BlockLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, vocab_size, hidden=128, n_layers=1, attn_heads=8,
                 seq_mode='two',
                 abs_position_embed=True,
                 relative_attn=False,
                 relative_1d=False,
                 max_relative_1d_positions=10,
                 relative_3d=False,
                 relative_3d_vocab_size=10):

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

        self.masked_lm = MaskedLanguageModel(hidden, vocab_size)

    def forward(self, x, segment_label=None, distance_matrix=None):
        x = self.bert(x, segment_label, distance_matrix, language_model=True)
        x = self.masked_lm(x)
        return x


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # (N, S, E)
        x = self.softmax(self.linear(x))
        return x

