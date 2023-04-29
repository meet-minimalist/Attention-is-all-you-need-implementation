##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-04-28 11:44:21 pm
# @copyright MIT License
#

import torch


def get_src_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Function to get the padded mask from source sequence tensor.

    Args:
        seq (torch.Tensor): Source sequence Tensor.
        pad_idx (int): Padding token value.

    Returns:
        torch.Tensor: Source sequence mask.
    """
    # seq : [batch, seq_len]
    # mask: [batch, 1, 1, seq_len] for encoder

    batch_size = seq.shape[0]
    return (seq != pad_idx).view(batch_size, 1, 1, -1)


def get_trg_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Function to get the padded mask from target sequence tensor.

    Args:
        seq (torch.Tensor): Target sequence Tensor.
        pad_idx (int): Padding token value.

    Returns:
        torch.Tensor: Target sequence mask.
    """
    # seq : [batch, seq_len]
    # mask: [batch, 1, seq_len, seq_len]

    batch_size = seq.shape[0]
    seq_len = seq.shape[1]
    target_pad_mask = (seq != pad_idx).view(batch_size, 1, 1, -1)  # [b, 1, 1, s]
    target_no_look_forward_mask = torch.tril(torch.ones(size=(1, 1, seq_len, seq_len)))

    target_mask = target_pad_mask & target_no_look_forward_mask
    return target_mask
