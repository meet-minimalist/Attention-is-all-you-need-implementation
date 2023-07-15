##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-05 12:15:47 am
# @copyright MIT License
#

import argparse
import os
import tempfile
from typing import Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe
from torchtext.datasets import IWSLT2016
from tqdm import tqdm


def get_tokenizers_and_trainers(vocab_size_eng: int, vocab_size_ger: int) -> Tuple:
    """Function to get the english tokenizer and german tokenizer.

    Args:
        vocab_size_eng (int): Vocab size for english corpus.
        vocab_size_ger (int): Vocab size for german corpus.

    Returns:
        Tuple: Tuple of English tokenizer, German tokenizer, english tokenizer
            trainer and german tokenizer trainer.
    """
    tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer_de = Tokenizer(BPE(unk_token="[UNK]"))
    trainer_en = BpeTrainer(
        vocab_size=vocab_size_eng,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )
    trainer_de = BpeTrainer(
        vocab_size=vocab_size_ger,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )
    tokenizer_en.pre_tokenizer = Whitespace()
    tokenizer_de.pre_tokenizer = Whitespace()
    return tokenizer_en, tokenizer_de, trainer_en, trainer_de


def get_dataset_iterators() -> Tuple[ShardingFilterIterDataPipe]:
    """Function to get the training iterators for IWSLT2016 dataset.

    Returns:
        Tuple[ShardingFilterIterDataPipe]: Tuple of train, validation and test
            iterators.
    """
    train_iter, valid_iter, test_iter = IWSLT2016(
        root="./", language_pair=("de", "en"), valid_set="tst2013", test_set="tst2014"
    )
    return train_iter, valid_iter, test_iter


def train_tokenizer(
    iterators: Tuple[ShardingFilterIterDataPipe],
    tokenizer_en: Tokenizer,
    tokenizer_de: Tokenizer,
    trainer_en: BpeTrainer,
    trainer_de: BpeTrainer,
    tokenizer_op_dir: str,
) -> None:
    """Function to train the tokenizer and save it on disk.

    Args:
        iterators (Tuple[ShardingFilterIterDataPipe]): Tuple of dataset
            iterators. It can contain train, valid and test split of dataset.
        tokenizer_en (Tokenizer): English tokenizer instance.
        tokenizer_de (Tokenizer): German tokenizer instance.
        trainer_en (BpeTrainer): English tokenizer trainer instance.
        trainer_de (BpeTrainer): German tokenizer trainer instance.
        tokenizer_op_dir (str): Tokenizer save directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer_paths_en = []
        tokenizer_paths_de = []
        for i, iterator in enumerate(iterators):
            data_path_en = os.path.join(temp_dir, f"{i}_en.data")
            data_path_de = os.path.join(temp_dir, f"{i}_de.data")
            if os.path.isfile(data_path_en):
                os.remove(data_path_en)
            if os.path.isfile(data_path_de):
                os.remove(data_path_de)

            en_lines = []
            de_lines = []
            for ger_line, eng_line in tqdm(iterator):
                de_lines.append("[SOS]" + ger_line.rstrip() + "[EOS]")
                en_lines.append("[SOS]" + eng_line.rstrip() + "[EOS]")

            with open(data_path_en, "w", encoding="utf-8") as f:
                f.writelines(en_lines)

            with open(data_path_de, "w", encoding="utf-8") as f:
                f.writelines(de_lines)

            tokenizer_paths_en.append(data_path_en)
            tokenizer_paths_de.append(data_path_de)
        tokenizer_en.train(tokenizer_paths_en, trainer_en)
        tokenizer_de.train(tokenizer_paths_de, trainer_de)
    tokenizer_en.save(os.path.join(tokenizer_op_dir, "bpe_iwslt2016_tokenizer_en.json"))
    tokenizer_de.save(os.path.join(tokenizer_op_dir, "bpe_iwslt2016_tokenizer_de.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help to train Tokenizer.")
    parser.add_argument(
        "--vocab_size_eng",
        default=30000,
        type=int,
        help="Max vocab size to keep for english sentences.",
    )
    parser.add_argument(
        "--vocab_size_ger",
        default=30000,
        type=int,
        help="Max vocab size to keep for german sentences.",
    )
    parser.add_argument(
        "--tokenizer_op_dir",
        default="./tokenizer_data/",
        type=str,
        help="Directory to store english and german tokenizer data.",
    )
    args = parser.parse_args()

    tokenizer_en, tokenizer_de, trainer_en, trainer_de = get_tokenizers_and_trainers(
        args.vocab_size_eng, args.vocab_size_ger
    )

    iterators = get_dataset_iterators()

    train_tokenizer(
        iterators,
        tokenizer_en,
        tokenizer_de,
        trainer_en,
        trainer_de,
        args.tokenizer_op_dir,
    )
