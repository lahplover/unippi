import torch.nn as nn
import torch
from .embedding import BERTEmbedding, RelationEmbedding
# import torch.nn.functional as F
from .attention import MultiHeadedAttention, RelativeAttention


class TransformerEncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout,
                 relative_attention=False, relation_embed_size=None):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.relative_attention = relative_attention
        if not relative_attention:
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        else:
            self.attention = RelativeAttention(num_heads=attn_heads, d_model=hidden, d_relation=relation_embed_size)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.linear1 = nn.Linear(hidden, feed_forward_hidden)
        self.linear2 = nn.Linear(feed_forward_hidden, hidden)
        # self.activation = F.gelu()
        self.activation = nn.ReLU()

    def forward(self, x, mask=None, relation_bias_matrix=None):
        if self.relative_attention:
            x_att = self.attention(x, x, x, relation_bias_matrix, mask)
        else:
            x_att = self.attention(x, x, x, mask)
        # sublayer residue connection and the LayerNorm, follows the Pytorch transformer implementation
        x = self.norm1(x + self.dropout1(x_att))

        x_ff = self.linear2(self.dropout2(self.activation(self.linear1(x))))

        x = self.norm2(x + self.dropout3(x_ff))
        return x


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1,
                 seq_mode='one', abs_position_embed=True,
                 relative_attn=True,
                 relation_embed_size=16,
                 relative_1d=True, max_relative_1d_positions=10,
                 relative_3d=False, relative_3d_vocab_size=10,
                 relative_attention_constrained=True):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.relative_attention_constrained = relative_attention_constrained

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        self.seq_mode = seq_mode

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden,
                                       seq_mode=seq_mode, abs_position_embed=abs_position_embed)

        if relative_attn:
            self.relation_embedding = RelationEmbedding(relation_embed_size,
                                                        relative_1d, max_relative_1d_positions,
                                                        relative_3d, relative_3d_vocab_size,
                                                        dropout=dropout)
        else:
            self.relation_embedding = None

        # multi-layers transformer blocks, deep network
        if not self.relative_attention_constrained:
            self.transformer_blocks = nn.ModuleList(
                [TransformerEncoderLayer(hidden, attn_heads, hidden * 4, dropout, relative_attn, relation_embed_size)
                 for _ in range(n_layers)])
        else:
            module_list = [TransformerEncoderLayer(hidden, attn_heads, hidden * 4, dropout, relative_attention=False),
                           TransformerEncoderLayer(hidden, attn_heads, hidden * 4, dropout, relative_attention=False)]
            for _ in range(n_layers-2):
                module_list.append(TransformerEncoderLayer(hidden, attn_heads, hidden * 4, dropout,
                                                           relative_attn, relation_embed_size))
            self.transformer_blocks = nn.ModuleList(module_list)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=attn_heads,
        #                                            dim_feedforward=hidden*4, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    def forward(self, x, segment_info=None, distance_matrix=None, language_model=False):
        """
        :param x: [N, S]
        :param segment_info: [N, S]
        :param distance_matrix: [N, S, S]
        :return:
        """
        # # masking for padded token
        # mask = (x == 0)
        #
        # # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)
        #
        # x = x.transpose(0, 1)  # (N, S, E) -> (S, N, E)
        #
        # # running over multiple transformer blocks
        # x = self.transformer_encoder.forward(x, src_key_padding_mask=mask)
        #
        # x = x.transpose(0, 1)  # (S, N, E) -> (N, S, E)

        batch_size, seq_len = x.size()

        if language_model:
            # attention masking for language model, keep the lower-half of the matrix
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, requires_grad=False))
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            # attention masking for padded token
            # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
            mask = (x > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)  # [N, S, E]

        if self.relation_embedding:
            relation_bias_matrix = self.relation_embedding(x.device, batch_size, seq_len, distance_matrix)  # [N, S, S, E]
        else:
            relation_bias_matrix = None

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask, relation_bias_matrix)

        # i = 0
        # for transformer in self.transformer_blocks:
        #     if i < 2:
        #         x = transformer.forward(x, mask, relation_bias_matrix)
        #     else:
        #         x = transformer.forward(x, mask)
        #     i += 1

        return x
