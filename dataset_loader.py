##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-05 11:38:44 pm
# @copyright MIT License
#

import random

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader, Dataset, Sampler
from torchtext.datasets import IWSLT2016


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
        dataset_iterator,
        tokenizer_de,
        batch_size,
        seq_len,
        indices=None,
        shuffle=True,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens_in_batch = self.batch_size * self.seq_len
        self.shuffle = shuffle
        # get the indices and length

        # Considering only german sequences for sorting purposes.
        self.indices = [
            (i, len(tokenizer_de.encode(src_text).ids))
            for i, (src_text, tar_text) in enumerate(dataset_iterator)
        ]

        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

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

    def __iter__(self):
        for batch_idx in self.all_batch_idx:
            yield batch_idx

    def __len__(self):
        return len(self.all_batch_idx)


class DataloaderHelper:
    def __init__(
        self,
        tokenizer_path_en,
        tokenizer_path_de,
        batch_size=8,
        seq_len=128,
        split="train",
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset_iterator = IWSLT2016(
            root="./",
            split=split,
            language_pair=("de", "en"),
            valid_set="tst2013",
            test_set="tst2014",
        )
        self.dataset_iterator = list(self.dataset_iterator)
        self.tokenizer_en = load_tokenizer(tokenizer_path_en)
        self.tokenizer_de = load_tokenizer(tokenizer_path_de)

    def get_iterator(self):
        dataloader = DataLoader(
            self.dataset_iterator,
            batch_sampler=BatchSamplerSimilarLength(
                dataset_iterator=self.dataset_iterator,
                tokenizer_en=self.tokenizer_en,
                tokenizer_de=self.tokenizer_de,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                shuffle=True,
            ),
            collate_fn=self.collate_batch,
        )
        return dataloader

    def collate_batch(self, batch_data):
        src_tokens, tar_tokens = [], []
        # src is german and tar is english.
        for src_text, tar_text in batch_data:
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
        return src_tokens, tar_tokens


if __name__ == "__main__":
    tokenizer_path_en = "./tokenizer_data/bpe_iwslt2016_tokenizer_en.json"
    tokenizer_path_de = "./tokenizer_data/bpe_iwslt2016_tokenizer_de.json"
    bs = 8
    max_len = 128
    train_dataloader = DataloaderHelper(
        tokenizer_path_en, tokenizer_path_de, bs, max_len, "valid"
    )
    train_iter = train_dataloader.get_iterator()
    print("Dataloaded")
    op = next(iter(train_iter))
