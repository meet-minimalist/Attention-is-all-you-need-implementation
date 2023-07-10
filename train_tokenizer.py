##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-05 12:15:47 am
# @copyright MIT License
#

import argparse
import os
import tempfile

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torchtext.datasets import IWSLT2016
from tqdm import tqdm


def train_tokenizer(
    iterators, tokenizer_en, tokenizer_de, trainer_en, trainer_de, tokenizer_op_dir
):
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

    tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer_de = Tokenizer(BPE(unk_token="[UNK]"))
    trainer_en = BpeTrainer(
        vocab_size=args.vocab_size_eng,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )
    trainer_de = BpeTrainer(
        vocab_size=args.vocab_size_ger,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )
    tokenizer_en.pre_tokenizer = Whitespace()
    tokenizer_de.pre_tokenizer = Whitespace()

    train_iter, valid_iter, test_iter = IWSLT2016(
        root="./", language_pair=("de", "en"), valid_set="tst2013", test_set="tst2014"
    )
    iterators = train_iter, valid_iter, test_iter

    train_tokenizer(
        iterators,
        tokenizer_en,
        tokenizer_de,
        trainer_en,
        trainer_de,
        args.tokenizer_op_dir,
    )
