import torch.nn as nn
import torch
from .embedding import BERTEmbedding


class Transformer(nn.Module):
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

        self.encoder = Encoder(self.embedding, hidden, n_layers, attn_heads, dropout)
        self.decoder = Decoder(self.embedding, hidden, n_layers, attn_heads, dropout)
        self.mask_lm = MaskedLanguageModel(hidden, vocab_size)

    def forward(self, x1, x2):
        """
        mask1: src_key_padding_mask, (N, S)
        mask2: tgt_mask, (T, T)
        :param x1:  (N, S)
        :param x2:  (N, T)
        :return:
        """
        memory, memory_mask = self.encoder(x1)
        x = self.decoder(x2, memory, memory_mask)

        # print('in model output size ', x.size())
        # return self.next_sentence(x), self.mask_lm(x)
        x = self.mask_lm(x)

        return x


class Encoder(nn.Module):
    def __init__(self, embedding, hidden=128, n_layers=1, attn_heads=8, dropout=0.1):
        super().__init__()
        self._embedding = embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=attn_heads,
                                                   dim_feedforward=hidden*4,
                                                   dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden)
        self._encoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        # encoder decoder share embedding??

    def forward(self, x1):
        # masking for padded token
        # torch.ByteTensor([batch_size, seq_len)
        mask1 = (x1 == 0)
        # print(mask1)
        x1 = self._embedding(x1)
        x1 = x1.transpose(0, 1)  # (N, S, E) -> (S, N, E)
        memory = self._encoder(x1, src_key_padding_mask=mask1)
        return memory, mask1

class Decoder(nn.Module):
    def __init__(self, embedding, hidden=128, n_layers=1, attn_heads=8, dropout=0.1):
        super().__init__()
        self._embedding = embedding

        # Language model masking
        # TODO: how to verify tril is the correct language model mask?
        # https://github.com/pytorch/pytorch/blob/43a2fd0e2451177bd4e9c7990a7126a576a1cfe4/torch/nn/functional.py#L3109
        # assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
        # attn_mask = attn_mask.unsqueeze(0)
        # attn_output_weights += attn_mask
        # attn_output = torch.bmm(attn_output_weights, v)
        # the attention multiplication for one sequence and one head: (tgt_len, src_len) x (src_len, hidden)
        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # mask = torch.tril(torch.ones(seq_length, seq_length))
        # mask_tgt = mask.masked_fill((mask == 0), float('-inf')).masked_fill(mask == 1, float(0.0))
        # self.register_buffer("mask_tgt", mask_tgt)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden, nhead=attn_heads,
                                                   dim_feedforward=hidden*4,
                                                   dropout=dropout)
        decoder_norm = nn.LayerNorm(hidden)
        self._decoder = nn.TransformerDecoder(decoder_layer, n_layers, decoder_norm)

    def forward(self, x2, memory, memory_key_padding_mask, tgt_key_padding_mask=None):
        seq_len = x2.size(1)
        # have to put mask into device, requires_grad=False, otherwise the memory will not be released?
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x2.device, requires_grad=False))
        mask_tgt = mask.masked_fill((mask == 0), float('-inf')).masked_fill(mask == 1, float(0.0))

        x2 = self._embedding(x2)
        x2 = x2.transpose(0, 1)
        x = self._decoder(x2, memory, tgt_mask=mask_tgt,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        return x

# class BERTLM(nn.Module):
#     """
#     BERT Language Model
#     Next Sentence Prediction Model + Masked Language Model
#     """
#
#     def __init__(self, vocab_size, hidden=128, n_layers=1, attn_heads=8, seq_length=12):
#         """
#         :param bert: BERT model which should be trained
#         :param vocab_size: total vocab size for masked_lm
#         """
#
#         super().__init__()
#         self.bert = BERT(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, seq_length=seq_length)
#         self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
#
#     def forward(self, x1, x2):
#         x = self.bert(x1, x2)
#         # print('in model output size ', x.size())
#         # return self.next_sentence(x), self.mask_lm(x)
#         x = self.mask_lm(x)
#         return x


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
        x = self.softmax(self.linear(x))
        # (T, N, E) --> (N, T, E)
        x = x.transpose(0, 1)
        return x
