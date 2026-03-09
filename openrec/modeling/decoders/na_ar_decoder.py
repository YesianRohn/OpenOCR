import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openrec.modeling.common import Mlp


class NaARDecoder(nn.Module):
    """A transformer model. User is able to modify the attributes as needed.
    The architechture is based on the paper "Attention Is All You Need". Ashish
    Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
    Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you
    need. In Advances in Neural Information Processing Systems, pages
    6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        nhead=None,
        beam_size=0,
        num_decoder_layers=6,
        max_len=25,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        scale_embedding=True,
    ):
        super(NaARDecoder, self).__init__()
        self.out_channels = out_channels
        self.ignore_index = out_channels - 1
        self.bos = out_channels - 2
        self.eos = 0
        self.max_len = max_len
        d_model = in_channels
        dim_feedforward = d_model * 4
        nhead = nhead if nhead is not None else d_model // 32
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=attention_dropout_rate,
                batch_first=True
            ),
            num_layers=num_decoder_layers
)

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model,
                                      self.out_channels - 2,
                                      bias=False)
        w0 = np.random.normal(0.0, d_model**-0.5,
                              (d_model, self.out_channels - 2)).astype(
                                  np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        src, memory_key_padding_mask = src
        tgt = tgt[:, :-1]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[1], device=src.get_device())

        memory = src  # B N C
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, data=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(B, sN, C)`.
            - tgt: :math:`(B, tN, C)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """

        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :2 + max_len]
            res = self.forward_train(src, tgt)
        else:
            res = self.forward_test(src)
        return res

    def forward_test(self, src):
        src, memory_key_padding_mask = src
        bs = src.shape[0]
        memory = src
        dec_seq = torch.full((bs, self.max_len + 1),
                             self.ignore_index,
                             dtype=torch.int64,
                             device=src.get_device())
        dec_seq[:, 0] = self.bos
        logits = []

        for len_dec_seq in range(0, self.max_len):
            dec_seq_embed = self.embedding(
                dec_seq[:, :len_dec_seq + 1])  # N dim 26+10 # </s>  012 a
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(
                dec_seq_embed.shape[1], src.get_device())
            tgt = dec_seq_embed  # bs, 3, dim #bos, a, b, c, ... eos
            tgt = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
            dec_output = tgt
            dec_output = dec_output[:, -1:, :]

            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            logits.append(word_prob)
            if len_dec_seq < self.max_len:
                # greedy decode. add the next token index to the target input
                dec_seq[:, len_dec_seq + 1] = word_prob.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if (dec_seq == self.eos).any(dim=-1).all():
                    break
        logits = torch.cat(logits, dim=1)
        return logits

    def generate_square_subsequent_mask(self, sz, device):
        """Generate a square mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions
        are filled with float(0.0).
        """
        mask = torch.zeros([sz, sz], dtype=torch.float32)
        mask_inf = torch.triu(
            torch.full((sz, sz), dtype=torch.float32, fill_value=-torch.inf),
            diagonal=1,
        )
        mask = mask + mask_inf
        return mask.to(device)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        # pe = torch.permute(pe, [1, 0, 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # x = x.permute([1, 0, 2])
        # x = x + self.pe[:x.shape[0], :]
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)  # .permute([1, 0, 2])


class PositionalEncoding_2d(nn.Module):
    """Inject some information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding_2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(torch.unsqueeze(pe, 0), [1, 0, 2])
        self.register_buffer('pe', pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.0)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.0)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        w_pe = self.pe[:x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = torch.permute(w_pe, [1, 2, 0])
        w_pe = torch.unsqueeze(w_pe, 2)

        h_pe = self.pe[:x.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = torch.permute(h_pe, [1, 2, 0])
        h_pe = torch.unsqueeze(h_pe, 3)

        x = x + w_pe + h_pe
        x = torch.permute(
            torch.reshape(x,
                          [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]),
            [2, 0, 1],
        )

        return self.dropout(x)


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)
