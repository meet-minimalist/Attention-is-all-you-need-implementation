##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-03-28 11:32:42 pm
# @copyright MIT License
#

import torch

import config as c
from model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    def setup_class(self):
        self.batch_size = 2
        self.seq_len = 128
        self.emb_size = c.d_model
        self.d_k = self.d_v = self.emb_size // c.enc.num_heads
        self.drop_prob = c.enc.dropout
        self.num_heads = c.enc.num_heads
        self.atten_block = MultiHeadAttention(
            self.emb_size, self.d_k, self.d_v, self.drop_prob, self.num_heads, bias=True
        )

    def test_shapes(self):
        """Test case to check output shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        res = self.atten_block(x)
        assert x.shape == res.shape

        # masked multi head attention
        mask = torch.tril(
            torch.ones(size=[self.batch_size, 1, self.seq_len, self.seq_len])
        )
        res = self.atten_block(x, mask=mask)
        assert x.shape == res.shape

        # multi head cross attention
        y = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        mask = torch.ones(size=[self.batch_size, 1, 1, self.seq_len])
        mask[:, :, :, (self.seq_len // 2) :] = 0
        res = self.atten_block(x, y, mask)
        assert x.shape == res.shape

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        for name, weight in self.atten_block.named_parameters():
            model_params = weight.numel()
            if "weight" in name:
                assert model_params == self.emb_size * self.emb_size
            elif "bias" in name:
                assert model_params == self.emb_size
            else:
                assert False, (
                    f"Weight {name} is extra added in the model "
                    "parameters. Please check."
                )
