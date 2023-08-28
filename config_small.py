##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-15 7:13:34 pm
# @copyright MIT License
#
from dataclasses import dataclass

d_model = 192
pos_dropout = 0.1  # Positional Embedding Dropout
max_len = 3000
tokenizer_path_en = "./tokenizer_data/bpe_tokenizer_en.json"
tokenizer_path_de = "./tokenizer_data/bpe_tokenizer_de.json"


@dataclass
class EncoderConfig:
    dropout: float = 0.1
    num_heads: int = 6
    num_block: int = 2
    d_ff: int = 256
    use_bias: bool = True
    vocab_size: int = 30000


@dataclass
class DecoderConfig:
    dropout: float = 0.1
    num_heads: int = 6
    num_block: int = 2
    d_ff: int = 256
    use_bias: bool = True
    vocab_size: int = 40000


@dataclass
class TrainingConfig:
    batch_size: int = 4
    seq_len: int = 128
    use_amp: bool = False
    epochs: int = 50
    train_data_path: str = "./training_data/"
    loss_logging_frequency: int = 100
    lr_scheduler: str = "cosine"
    label_smoothing: float = 0.1
    init_lr: float = 1e-3
    burn_in_epochs: int = 2
    dataset: str = "iwslt2017"
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    apply_grad_clipping: bool = False
    grad_clipping_max_norm: float = 0.0001
    track_gradients: bool = True


enc = EncoderConfig()
dec = DecoderConfig()
train_cfg = TrainingConfig()
