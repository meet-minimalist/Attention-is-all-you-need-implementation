##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-04-16 6:43:40 pm
# @copyright MIT License
#
from dataclasses import dataclass

d_model = 512
pos_dropout = 0.1  # Positional Embedding Dropout
max_len = 3000
tokenizer_en_path = "./tokenizer_data/bpe_iwslt2016_tokenizer_en.json"
tokenizer_de_path = "./tokenizer_data/bpe_iwslt2016_tokenizer_de.json"


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


@dataclass
class TrainingConfig:
    batch_size = 8
    seq_len = 128
    use_amp = False
    epochs = 10
    train_data_path = "./training_data/"
    loss_logging_frequency = 10
    lr_scheduler = "cosine"


enc = EncoderConfig()
dec = DecoderConfig()
train_cfg = TrainingConfig()
