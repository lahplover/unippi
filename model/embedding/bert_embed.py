import torch.nn as nn
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1, seq_mode='one', abs_position_embed=True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        self.token = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.abs_position_embed = abs_position_embed
        if abs_position_embed:
            self.position = PositionalEmbedding(d_model=embed_size)
        self.seq_mode = seq_mode
        if seq_mode == 'two':
            self.segment = nn.Embedding(3, embedding_dim=embed_size)

    def forward(self, sequence, segment_label=None):
        x = self.token(sequence)
        if self.abs_position_embed:
            x = x + self.position(sequence)
        if self.seq_mode == 'two':
            x = x + self.segment(segment_label)
        # x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
