import torch.nn as nn
import torch
from .embedding import BERTEmbedding


class SeqLM(nn.Module):
    def __init__(self, vocab_size, hidden=128, n_layers=1, attn_heads=8, dropout=0):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: Transformer model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden,
                                       dropout=dropout, seq_mode='one', abs_position_embed=True)

        self.encoder = Encoder(hidden, n_layers, attn_heads, dropout)
        # self.decoder = Decoder(self.embedding, hidden, n_layers, attn_heads, dropout)
        self.seq_lm = SeqLanguageModel(hidden, vocab_size)

    def forward(self, x):
        """
        mask1: src_key_padding_mask, (N, S)
        mask2: src_mask, (S, S)
        :param x:  (N, S)
        :return:
        """
        # masking for padded token
        # torch.ByteTensor([batch_size, seq_len)
        padding_mask = (x == 0)

        x = self.embedding(x)
        x = self.encoder(x, padding_mask)

        # print('in model output size ', x.size())
        # return self.next_sentence(x), self.mask_lm(x)
        x = self.seq_lm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, hidden=128, n_layers=1, attn_heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=attn_heads,
                                                   dim_feedforward=hidden*4,
                                                   dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden)
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=encoder_norm)

        # encoder decoder share embedding??

    def forward(self, x, padding_mask):

        seq_len = x.size(1)
        # have to put mask into device, requires_grad=False, otherwise the memory will not be released?
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, requires_grad=False))
        mask_tgt = mask.masked_fill((mask == 0), float('-inf')).masked_fill(mask == 1, float(0.0))

        # print(mask1)
        x = x.transpose(0, 1)  # (N, S, E) -> (S, N, E)
        x = self._encoder(x, src_key_padding_mask=padding_mask, mask=mask_tgt)
        return x


class SeqLanguageModel(nn.Module):
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
        x = self.softmax(self.linear(x))
        # (T, N, E) --> (N, T, E)
        x = x.transpose(0, 1)
        return x
