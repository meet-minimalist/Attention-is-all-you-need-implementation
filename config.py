##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-04-16 6:43:40 pm
# @copyright MIT License
#
from dataclasses import dataclass

d_model = 512
vocab_size = 30000

pos_dropout = 0.1


@dataclass
class EncoderConfig:
    dropout = 0.1
    num_heads = 8
    num_block = 6
    d_ff = 2048
    use_bias = True


@dataclass
class DecoderConfig:
    dropout = 0.1
    num_heads = 8
    num_block = 6
    d_ff = 2048
    use_bias = True


enc = EncoderConfig()
dec = DecoderConfig()
