##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-03-28 11:49:16 pm
# @copyright MIT License
#

import torch

import config as c
from model.enc_dec_transformers import TransformerEncoderDecoder


class TestTransformers:
    def setup_class(self):
        self.batch_size = 2
        self.seq_len = 128
        self.enc_vocab_size = c.enc.vocab_size
        self.dec_vocab_size = c.dec.vocab_size
        self.emb_dims = c.d_model
        self.transformer = TransformerEncoderDecoder(
            c.enc.vocab_size,
            c.dec.vocab_size,
            c.d_model,
            c.max_len,
            c.pos_dropout,
            c.enc.dropout,
            c.enc.num_heads,
            c.enc.num_block,
            c.enc.d_ff,
            c.enc.use_bias,
            c.dec.dropout,
            c.dec.num_heads,
            c.dec.num_block,
            c.dec.d_ff,
            c.dec.use_bias,
        )

    def test_shapes(self):
        """Test case to check number of learnable parameters in the layer."""
        enc_data = torch.randint(
            low=0,
            high=self.enc_vocab_size,
            size=[self.batch_size, self.seq_len],
            dtype=torch.int32,
        )
        dec_data = torch.randint(
            low=0,
            high=self.dec_vocab_size,
            size=[self.batch_size, self.seq_len],
            dtype=torch.int32,
        )

        enc_mask = torch.ones(size=[self.batch_size, 1, 1, self.seq_len])
        enc_mask[:, :, :, (self.seq_len // 2) :] = 0
        dec_mask = torch.tril(
            torch.ones(size=[self.batch_size, 1, self.seq_len, self.seq_len])
        )

        res = self.transformer(enc_data, dec_data, enc_mask, dec_mask)
        assert list(res.shape) == [self.batch_size, self.seq_len, self.dec_vocab_size]
