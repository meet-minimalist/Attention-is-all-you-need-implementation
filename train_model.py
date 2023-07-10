##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-08 3:56:23 pm
# @copyright MIT License
#

import argparse
from typing import Tuple

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from dataset_loader import DataloaderHelper
from misc.checkpoint_handler import CheckpointHandler
from misc.summary_writer import SummaryHelper
from misc.utils import (
    LossAverager,
    get_ckpt_dir,
    get_exp_path,
    get_logger,
    get_lr_scheduler,
    get_summary_path,
    load_module,
    to_device,
)
from model.enc_dec_transformers import TransformerEncoderDecoder


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transformer = TransformerEncoderDecoder(
            cfg.enc.vocab_size,
            cfg.dec.vocab_size,
            cfg.d_model,
            cfg.max_len,
            cfg.pos_dropout,
            cfg.enc.dropout,
            cfg.enc.num_heads,
            cfg.enc.num_block,
            cfg.enc.d_ff,
            cfg.enc.use_bias,
            cfg.dec.dropout,
            cfg.dec.num_heads,
            cfg.dec.num_block,
            cfg.dec.d_ff,
            cfg.dec.use_bias,
        )

        self.train_dataloader = DataloaderHelper(
            cfg.tokenizer_path_en,
            cfg.tokenizer_path_de,
            cfg.train_cfg.batch_size,
            cfg.train_cfg.seq_len,
            "train",
        )
        self.valid_dataloader = DataloaderHelper(
            cfg.tokenizer_path_en,
            cfg.tokenizer_path_de,
            cfg.train_cfg.batch_size,
            cfg.train_cfg.seq_len,
            "train",
        )
        self.test_dataloader = DataloaderHelper(
            cfg.tokenizer_path_en,
            cfg.tokenizer_path_de,
            cfg.train_cfg.batch_size,
            cfg.train_cfg.seq_len,
            "train",
        )

        self.lr_scheduler = get_lr_scheduler(self.cfg.train_cfg.lr_scheduler)
        self.exp_path = get_exp_path(cfg.train_cfg.train_data_path)
        self.logger = get_logger(self.exp_path)
        self.ckpt_dir = get_ckpt_dir(self.exp_path)
        self.ckpt_handler = CheckpointHandler(self.ckpt_dir, "model", max_to_keep=3)

        self.epochs = self.cfg.train_cfg.epochs

        self.cpu = torch.device("cpu:0")
        self.cuda = torch.device("cuda:0")

    def __print_model_summary(self) -> None:
        """Function to print the model's summary.
        e.g. Each layer's shapes and parameters.
        """
        # src_tokens: [batch, seq_len]
        # dst_tokens: [batch, seq_len]
        # src_mask  : [batch, 1, 1, seq_len]
        # dst_mask  : [batch, 1, seq_len, seq_len]
        model_stats = summary(
            self.transformer, [(1, 128), (1, 64), (1, 1, 1, 128), (1, 1, 64, 128)]
        )
        for line in str(model_stats).split("\n"):
            self.logger.debug(line)

    def __train_batch_loop(
        self,
        iterator: DataLoader,
        opt: Optimizer,
        scalar: GradScaler,
        summ_writer: SummaryHelper,
        eps: int,
    ) -> None:
        """Function to run the training loop for single epoch based on given
        iterator.

        Args:
            iterator (DataLoader): Training dataloader.
            opt (Optimizer): Optimizer instance.
            scalar (GradScaler): Gradient scalar instance.
            summ_writer (SummaryHelper): Summary object to store data for
                tensorboard visualization.
            eps (int): Epoch number.
        """
        self.transformer.train()

        g_step = eps * len(iterator)
        for batch_num, (
            src_tokens,
            dst_tokens,
            src_mask,
            dst_mask,
            dst_labels,
        ) in enumerate(tqdm(iterator)):
            # src_tokens: [batch, seq_len]
            # dst_tokens: [batch, seq_len]
            # src_mask  : [batch, 1, 1, seq_len]
            # dst_mask  : [batch, 1, seq_len, seq_len]
            # dst_labels: [batch, seq_len]

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.cfg.train_cfg.use_amp):
                src_tokens, dst_tokens, src_mask, dst_mask = to_device(
                    [src_tokens, dst_tokens, src_mask, dst_mask, dst_labels], self.cuda
                )
                logits = self.transformer(src_tokens, dst_tokens, src_mask, dst_mask)
                loss, acc = self.compute_loss_and_accuracy(logits, dst_labels)

            scalar.scale(loss).backward()
            scalar.step(opt)
            scalar.update()

            lr = self.lr_scheduler.step(g_step, opt)
            opt.step()

            if (batch_num + 1) % self.cfg.train_cfg.loss_logging_frequency == 0:
                loss = to_device(loss, self.cpu)
                acc = to_device(acc, self.cpu)

                self.logger.info(
                    f"Epoch: {eps}/{self.epochs}, "
                    f"Batch No.: {batch_num}/{len(iterator)}, "
                    f"Loss: {loss:.4f}, "
                    f"Accuracy: {acc:.2f}, "
                    f"LR: {lr:.5f}"
                )

                summ_writer.add_summary(
                    {"loss": loss, "accuracy": acc, "lr": lr}, g_step
                )

    def __test_batch_loop(
        self, iterator: DataLoader, summ_writer: SummaryHelper, eps: int, g_step: int
    ) -> Tuple[float]:
        """Function to run the test loop at the end of each epoch.

        Args:
            iterator (DataLoader): Test dataloader.
            summ_writer (SummaryHelper): Summary object to store data for
                tensorboard visualization.
            eps (int): Epoch number.
            g_step (int): Total iteration number.

        Returns:
            Tuple[float]: Tuple of average test loss and average test accuracy.
        """
        self.transformer.eval()

        # Only test_loss and accuracy to be tracked.
        test_tracker = LossAverager(num_elements=2)

        with torch.no_grad():
            for src_tokens, dst_tokens, src_mask, dst_mask, dst_labels in tqdm(
                iterator
            ):
                # src_tokens: [batch, seq_len]
                # dst_tokens: [batch, seq_len]
                # src_mask  : [batch, 1, 1, seq_len]
                # dst_mask  : [batch, 1, seq_len, seq_len]
                # dst_labels: [batch, seq_len]

                src_tokens, dst_tokens, src_mask, dst_mask = to_device(
                    [src_tokens, dst_tokens, src_mask, dst_mask, dst_labels], self.cuda
                )
                logits = self.transformer(src_tokens, dst_tokens, src_mask, dst_mask)
                loss, acc = self.compute_loss_and_accuracy(logits, dst_labels)
                test_tracker([loss, acc])

            avg_test_loss, avg_test_acc = test_tracker.get_averages()

            self.logger.info(
                f"Avg. Test Loss: {avg_test_loss:.4f}, "
                + f"Avg. Test Accuracy: {avg_test_acc:.2f}"
            )

            summ_writer.add_summary(
                {"loss": avg_test_loss, "accuracy": avg_test_acc}, g_step
            )
        return avg_test_loss, avg_test_acc

    def __get_summary_writer(self, type: str) -> SummaryHelper:
        """Function to get the SummaryHelper object which will create and upated
        summary object during training.

        Args:
            type (str): Type of summary object. It can be "train" or "test".

        Returns:
            SummaryHelper: SummaryHelper instance as per given type.
        """
        summ_path = get_summary_path(self.exp_path, type)
        return SummaryHelper(summ_path)

    def __save_checkpoint(
        self, eps: int, g_step: int, loss: float, opt: Optimizer, scalar: GradScaler
    ) -> None:
        """Function to save the checkpoint along with model state, optimizer
        state, scalar state and other attributes corresponding to current epoch.

        Args:
            eps (int): Epoch number.
            g_step (int): Total iteration number.
            loss (float): Test Loss value
            opt (Optimizer): Optimizer instance.
            scalar (GradScaler): Gradient scalar instance.
        """
        checkpoint = {
            "epoch": eps,
            "global_step": g_step,
            "test_loss": loss,
            "model": self.transformer.state_dict(),
            "optimizer": opt.state_dict(),
            "scalar": scalar.state_dict(),
        }
        self.ckpt_handler.save(checkpoint)

    def __resume_training(
        self, resume: bool, resume_ckpt: str, opt: Optimizer, scalar: GradScaler
    ) -> Tuple[int]:
        """Function to restore the training parameters and return resume epoch
        number and global iteration number.

        Args:
            resume (bool): True for resuming the training else False.
            resume_ckpt (str): Path of resume checkpoint to be restored.
            opt (Optimizer): Optimizer instance.
            scalar (GradScaler): Gradient Scalar instance.

        Returns:
            Tuple[int]: Tuple of resume epoch number and global iteration count.
        """
        if resume:
            checkpoint = torch.load(resume_ckpt)
            self.transformer.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            scalar.load_state_dict(checkpoint["scalar"])
            resume_g_step = checkpoint["global_step"]
            resume_eps = checkpoint["epoch"]
            self.logger.info(f"Resuming training from {resume_eps} epochs.")
        else:
            resume_g_step = 1
            resume_eps = 1

        g_step = max(1, resume_g_step)
        return resume_eps, g_step

    def __get_loss_function(self):
        return None

    def train(self, resume: bool = False, resume_ckpt: str = None) -> None:
        """Function to be used to train the model based on provided config.

        Args:
            resume (bool, optional): Whether to resume the training. Defaults
                to False.
            resume_ckpt (str, optional): Checkpoint path to be used to resume
                training with. Defaults to None.
        """
        self.__print_model_summary()

        loss_fn = self.__get_loss_function()

        opt = torch.optim.Adam(self.transformer.parameters(), lr=0.0, weight_decay=0.0)
        scalar = torch.cuda.amp.GradScaler(enabled=self.cfg.train_cfg.use_amp)

        resume_eps, g_step = self.__resume_training(resume, resume_ckpt, opt, scalar)

        train_writer = self.__get_summary_writer("train")
        test_writer = self.__get_summary_writer("test")

        for eps in range(resume_eps, self.epochs + 1):
            train_iter = self.train_dataloader.get_iterator()
            # valid_iter = self.valid_dataloader.get_iterator()
            self.logger.info(f"Epoch: {eps}/{self.epochs} Started")
            self.__train_batch_loop(train_iter, opt, scalar, train_writer, eps)

            g_step = eps * len(train_iter)
            test_iter = self.test_dataloader.get_iterator()
            avg_test_loss, avg_test_acc = self.__test_batch_loop(
                test_iter, test_writer, eps, g_step
            )

            self.__save_checkpoint(eps, g_step, avg_test_loss, opt, scalar)
            self.logger.info(f"Epoch: {eps}/{self.epochs} completed")

        print("Training Completed.")
        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for training the model."
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path of config file to be used for training.",
    )
    args = parser.parse_args()

    config = load_module(args.config_path)
    trainer = Trainer(config)
    trainer.train()
