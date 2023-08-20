##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-05 11:38:44 pm
# @copyright MIT License
#

import random
from typing import List, Tuple

import datasets
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader, Sampler

from model.utils import get_src_pad_mask, get_trg_pad_mask


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Function to load the tokenizer from given path and set its
    formatting.

    Args:
        tokenizer_path (str): Path of the trained tokenizer

    Returns:
        Tokenizer: Pretrained tokenizer object
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)

    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    return tokenizer


class BatchSamplerSimilarLength(Sampler):
    def __init__(
        self,
        dataset_iterator: datasets.Dataset,
        tokenizer_de: str,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
    ):
        """Initializer to load the dataset and sort as per the source sequences
        (german sequence) and prepare the indices in a bucketed manner where the
        sequences with similar lengths are grouped together.

        Args:
            dataset_iterator (datasets.Dataset): Dataset iterator.
            tokenizer_de (str): Source sequence tokenizer. e.g. German tokenizer.
            batch_size (int): Batch size to be used to compute upper limit of
                tokens.
            seq_len (int): Sequence length to be used to compute upper limit of
                tokens.
            shuffle (bool, optional): Shuffle the dataset before sorting and
                after getting the buckets. Defaults to True.
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens_in_batch = self.batch_size * self.seq_len
        self.shuffle = shuffle
        # get the indices and length

        # Considering only german sequences for sorting purposes.
        self.indices = [
            (i, len(tokenizer_de.encode(data["translation"]["de"]).ids))
            for i, data in enumerate(dataset_iterator)
        ]

        if self.shuffle:
            random.shuffle(self.indices)

        sorted_indices = sorted(self.indices, key=lambda x: x[1])

        self.all_batch_idx = []
        single_batch_idx = []
        cummulative_token_len = 0

        for idx, _ in sorted_indices:
            token_len = self.indices[idx][1]
            cummulative_token_len += token_len

            single_batch_idx.append(idx)

            if cummulative_token_len > self.total_tokens_in_batch:
                self.all_batch_idx.append(single_batch_idx)
                single_batch_idx.clear()
                cummulative_token_len = 0

        if self.shuffle:
            random.shuffle(self.all_batch_idx)

    def __iter__(self) -> List[int]:
        """Function will fetch list of indices to be used to generate a batch.

        Yields:
            List[int]: Yields list of indices for batch generation.
        """
        for batch_idx in self.all_batch_idx:
            yield batch_idx

    def __len__(self) -> int:
        """Function to get the total number of batches which can be generated.

        Returns:
            int: Number of batches from the given dataset.
        """
        return len(self.all_batch_idx)


class DataloaderHelper:
    def __init__(
        self,
        tokenizer_path_en: str,
        tokenizer_path_de: str,
        batch_size: int = 8,
        seq_len: int = 128,
        split: str = "train",
    ):
        """Initializer to load the dataset, prepare for training and convert it
        into dataloader format.

        Args:
            tokenizer_path_en (str): Path for english tokenizer.
            tokenizer_path_de (str): Path for German tokenizer.
            batch_size (int, optional): Batch size to be used. Defaults to 8.
            seq_len (int, optional): Sequence length to be used for combining
                sequences together. It will not produce all the samples of the
                batch as per this sequence length. But this value and batch size
                will provide some upper value which we can manage to load on GPU
                if all the tokens are non-pad tokens. Defaults to 128.
            split (str, optional): Type of dataset split. Defaults to "train".
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset_iterator = load_dataset(
            "iwslt2017", "iwslt2017-de-en", split=split
        )
        self.tokenizer_en = load_tokenizer(tokenizer_path_en)
        self.tokenizer_de = load_tokenizer(tokenizer_path_de)

    def get_iterator(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_iterator,
            batch_sampler=BatchSamplerSimilarLength(
                dataset_iterator=self.dataset_iterator,
                tokenizer_de=self.tokenizer_de,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                shuffle=True,
            ),
            collate_fn=self.collate_batch,
        )
        return dataloader

    def collate_batch(self, batch_data: List) -> Tuple[torch.Tensor]:
        """Function to tokenize sequences and prepare the mask for training.

        Args:
            batch_data (List): List of Tuple of source and target sequences.

        Returns:
            Tuple[torch.Tensor]: Tuple of source token, target token, source
                mask, target mask and target labels.
        """
        src_tokens, tar_tokens = [], []
        # src is german and tar is english.
        for data in batch_data:
            src_text = data["translation"]["de"]
            tar_text = data["translation"]["en"]
            src_tokens.append(src_text)
            tar_tokens.append(tar_text)

        src_tokens = torch.stack(
            [
                torch.tensor(tokenized_seq.ids)
                for tokenized_seq in self.tokenizer_de.encode_batch(src_tokens)
            ]
        )
        tar_tokens = torch.stack(
            [
                torch.tensor(tokenized_seq.ids)
                for tokenized_seq in self.tokenizer_en.encode_batch(tar_tokens)
            ]
        )
        tar_labels = tar_tokens[:, 1:]
        tar_tokens = tar_tokens[:, :-1]
        src_mask = get_src_pad_mask(src_tokens, self.tokenizer_de.token_to_id("[PAD]"))
        tar_mask = get_trg_pad_mask(tar_tokens, self.tokenizer_en.token_to_id("[PAD]"))
        return src_tokens, tar_tokens, src_mask, tar_mask, tar_labels


if __name__ == "__main__":
    tokenizer_path_en = "./tokenizer_data/bpe_iwslt2016_tokenizer_en.json"
    tokenizer_path_de = "./tokenizer_data/bpe_iwslt2016_tokenizer_de.json"
    bs = 8
    max_len = 128
    train_dataloader = DataloaderHelper(
        tokenizer_path_en, tokenizer_path_de, bs, max_len, "validation"
    )
    train_iter = train_dataloader.get_iterator()
    print("Dataloaded")
    op = next(iter(train_iter))
    print("Number of outputs: ", len(op))
    print("Shape of each outputs: ")
    print([s.shape for s in op])
