##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-15 7:13:34 pm
# @copyright MIT License
#
from dataclasses import dataclass

d_model = 128
pos_dropout = 0.1  # Positional Embedding Dropout
max_len = 3000
tokenizer_path_en = "./tokenizer_data/bpe_iwslt2016_tokenizer_en.json"
tokenizer_path_de = "./tokenizer_data/bpe_iwslt2016_tokenizer_de.json"


@dataclass
class EncoderConfig:
    dropout = 0.1
    num_heads = 4
    num_block = 3
    d_ff = 256
    use_bias = True
    vocab_size = 30000


@dataclass
class DecoderConfig:
    dropout = 0.1
    num_heads = 4
    num_block = 3
    d_ff = 256
    use_bias = True
    vocab_size = 40000


@dataclass
class TrainingConfig:
    batch_size = 2
    seq_len = 128
    use_amp = False
    epochs = 10
    train_data_path = "./training_data/"
    loss_logging_frequency = 10
    lr_scheduler = "cosine"
    label_smoothing = 0.1
    init_lr = 1e-3
    burn_in_epochs = 2
    dataset = "iwslt2017"


enc = EncoderConfig()
dec = DecoderConfig()
train_cfg = TrainingConfig()
