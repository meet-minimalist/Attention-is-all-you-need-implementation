##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-04-16 6:43:40 pm
# @copyright MIT License
#
from dataclasses import dataclass

d_model = 512
pos_dropout = 0.1
max_len = 3000


@dataclass
class EncoderConfig:
    dropout = 0.1
    num_heads = 8
    num_block = 6
    d_ff = 2048
    use_bias = True
    vocab_size = 30000


@dataclass
class DecoderConfig:
    dropout = 0.1
    num_heads = 8
    num_block = 6
    d_ff = 2048
    use_bias = True
    vocab_size = 40000


enc = EncoderConfig()
dec = DecoderConfig()
