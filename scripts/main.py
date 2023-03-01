import argparse
import os
from typing import List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from torch.utils.data import DataLoader

from mtt.data import OnlineDataset, build_offline_datapipes, collate_fn
from mtt.models import Conv2dCoder
from mtt.simulator import Simulator


def init_simulator():
    return Simulator()


def get_trainer(params: argparse.Namespace) -> pl.Trainer:
    logger: Union[List[Logger], bool] = (
        False
        if params.no_log
        else [TensorBoardLogger(save_dir="./", name="tensorboard", version="")]
    )
    callbacks = [
        ModelCheckpoint(
            monitor="train/loss",
            dirpath="./checkpoints",
            filename="best",
            auto_insert_metric_name=False,
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        if not params.no_log
        else None,
        EarlyStopping(
            monitor="train/loss",
            patience=params.patience,
        ),
    ]
    callbacks = [c for c in callbacks if c is not None]
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=not params.no_log,
        precision=32,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
        check_val_every_n_epoch=1,
    )


def get_online_dataset(params: argparse.Namespace, n_steps=100):
    return OnlineDataset(
        length=params.input_length,
        init_simulator=init_simulator,
        n_steps=n_steps,
        **vars(params),
    )


def get_checkpoint_path() -> Union[str, None]:
    ckpt_path = "./checkpoints/best.ckpt"
    return ckpt_path if os.path.exists(ckpt_path) else None


def train(params: argparse.Namespace):
    train_dp, val_dp = build_offline_datapipes(
        "/nfs/general/mtt_data/train", map_location="cpu"
    )
    num_workers = min(torch.multiprocessing.cpu_count(), 4)
    train_loader = DataLoader(
        dataset=train_dp,
        collate_fn=collate_fn,
        batch_size=params.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        dataset=val_dp,
        collate_fn=collate_fn,
        batch_size=params.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    trainer = get_trainer(params)
    model = Conv2dCoder(**vars(params))
    trainer.fit(model, train_loader, val_loader, ckpt_path=get_checkpoint_path())


def test(params: argparse.Namespace):
    num_workers = min(torch.multiprocessing.cpu_count(), 4)
    test_loader = DataLoader(
        get_online_dataset(params),
        batch_size=params.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    trainer = get_trainer(params)
    model = Conv2dCoder(**vars(params))
    trainer.test(model, test_loader, ckpt_path=get_checkpoint_path())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("operation", choices=["train", "test"])
    parser.add_argument("--no_log", action="store_true")

    # data arguments
    group = parser.add_argument_group("Data")
    group.add_argument("--batch_size", type=int, default=16)

    # model arguments
    group = parser.add_argument_group("Model")
    group = Conv2dCoder.add_model_specific_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)
    group.add_argument("--patience", type=int, default=10)

    params = parser.parse_args()
    if params.operation == "train":
        train(params)
    elif params.operation == "test":
        test(params)
    else:
        raise ValueError(f"Unknown operation: {params.operation}")
