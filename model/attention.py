##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-03-28 11:37:24 pm
# @copyright MIT License
#

import numpy as np
import torch
import torch.nn as nn


class PrepareaForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, bias: bool = True):
        """Initializer for QKV dense layer operation before actual attention
        calculations.

        Args:
            d_model (int, optional): Dimension for hidden states in encoder.
                Defaults to 512.
            num_heads (int, optional): Number of parallel heads in Multi head
                attention block. Defaults to 8.
            bias (bool, optional): Use bias for all the layers. Defaults to True.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        # Assuming d_k == d_v for simplicity.

        self.linear = nn.Linear(self.d_model, self.d_k * self.num_heads, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input     : [batch, seq_len, emb_size]
        # output    : [batch, seq_len, num_heads, d_k]

        x = self.linear(x)  # [batch, seq_len, self.d_k * self.num_heads]
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        drop_prob: float = 0.1,
        num_heads: int = 8,
        bias: bool = True,
    ):
        """Initializer for multi head attention block.

        Args:
            d_model (int, optional): Dimension for hidden states in encoder.
                Defaults to 512.
            d_k (int, optional): Dimensionality for Keys. Defaults to 64.
            d_v (int, optional): Dimensionality for Query. Defaults to 64.
            drop_prob (float, optional): Dropout rate used across multi head attention.
                Defaults to 0.1.
            num_heads (int, optional): Number of parallel heads in Multi head
                attention block. Defaults to 8.
            bias (bool, optional): Use bias for all the layers. Defaults to True.
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = d_model
        self.drop_prob = drop_prob

        self.query = PrepareaForMultiHeadAttention(
            self.d_model, self.num_heads, bias=bias
        )
        self.key = PrepareaForMultiHeadAttention(
            self.d_model, self.num_heads, bias=bias
        )
        self.value = PrepareaForMultiHeadAttention(
            self.d_model, self.num_heads, bias=True
        )

        self.linear_out = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(drop_prob)
        self.softmax = nn.Softmax(dim=-1)

        self.scale = 1 / np.sqrt(self.d_model)

    def __get_mask_q_k(
        self, q_k: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Function to get the mask to be used before softmax such that only the
        tokens before the current token gets attended by current token.

        Args:
            q_k (torch.Tensor): Tensor generated after matmul of Q and K.
            mask (torch.Tensor): Mask to be used to computed masked_q_k.
                Defaults to None.

        Returns:
            torch.Tensor: Masked tensor output.
        """
        # q_k : [batch, num_heads, seq_len_q, seq_len_k]
        # mask: [batch, 1, 1, seq_len] for encoder or [batch, 1, seq_len, seq_len] for decoder
        if mask is None:
            mask = torch.tril(q_k)

        masked_q_k = q_k.masked_fill(mask == 0, float("-inf"))
        return masked_q_k

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # x         : [batch, seq_len, emb_size]
        # y         : [batch, num_heads, seq_len, seq_len]. Optional parameter.
        #           : If this is provided then use it for cross attention.
        #           : i.e. Q and K vectors are generated using x which is encoder
        #           : representation and V vector is generated using y which is
        #           : decoder representation. In this case the mask shall be of
        #           : encoder mask as we need mask only for Q and K.
        # mask      : [batch, 1, 1, seq_len] for encoder or [batch, 1, seq_len, seq_len] for decoder
        # output    : [batch, seq_len, emb_size]

        # seq_len_q = seq_len_k = seq_len_v
        # The representation will be different. Dimensions will be same.
        q = self.query(x)  # [batch, seq_len_q, num_heads, d_k]
        k = self.key(x)  # [batch, seq_len_k, num_heads, d_k]
        if y is None:
            v = self.value(x)  # [batch, seq_len_v, num_heads, d_v]
        else:
            v = self.value(y)  # [batch, seq_len_v, num_heads, d_v]

        q = q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len_q, d_k]
        k = k.permute(0, 2, 3, 1)  # [batch, num_heads, d_k, seq_len_k]

        q_kT = q @ k  # [batch, num_heads, seq_len_q, seq_len_k]
        q_kT_scaled = q_kT * self.scale

        # We dont want to find attention for all the elements seq_len_q with all
        # the elements of seq_len_k. The token_idx i from q should only attend
        # the tokens upto ith index in k. That way we stop the q to look into
        # k's future token which is tokens after ith location.
        masked_q_kT_scaled = self.__get_mask_q_k(
            q_kT_scaled, mask
        )  # [batch, num_heads, seq_len_q, seq_len_k]
        # For that we removed upper triangle of [seq_len_q, seq_len_k] and fill
        # all the values with -inf. So when we apply softmax those values with
        # -inf will contribute very less.

        masked_q_kT_softmax = self.softmax(
            masked_q_kT_scaled
        )  # [batch, num_heads, seq_len_q, seq_len_k]
        # Applying softmax over seq_len_k dimension as we want each token index
        # of query to have attention with all the token of key at i or less than
        # i location. After that we will apply softmax over all the attention
        # values computed for q, which means at the seq_len_k dimension we will
        # apply softmax.
        masked_q_kT_softmax = self.dropout_layer(masked_q_kT_softmax)

        v = v.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len_v, d_v]

        # seq_len_v = seq_len_k and d_v = d_k
        attention_res = masked_q_kT_softmax @ v  # [batch, num_heads, seq_len_q, d_v]
        attention_res = attention_res.contiguous()
        attention_res = attention_res.permute(
            0, 2, 1, 3
        )  # [batch, seq_len_q, num_heads, d_v]
        attention_res = attention_res.reshape(
            attention_res.shape[0], attention_res.shape[1], -1
        )
        # [batch, num_heads, seq_len_q * d_v] = [batch, seq_len, emb_dims]

        attention_res = self.linear_out(attention_res)  # [batch, seq_len, emb_dims]

        return attention_res
