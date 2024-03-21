import argparse
import os
import sys
import typing
from functools import partial
from pathlib import Path
from typing import List

import optuna
import pytorch_lightning as pl
import torch
import wandb
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader, random_split
from wandb.wandb_run import Run

from mtt.data.image import OnlineImageDataset, build_image_dp, collate_image_fn
from mtt.data.sparse import SparseDataset
from mtt.models import KNN, Conv2dCoder, EncoderDecoder, SpatialTransformer
from mtt.models.sparse import SparseBase
from mtt.simulator import Simulator

models = {
    "conv2d": Conv2dCoder,
    "st": SpatialTransformer,
    "knn": KNN,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", type=str, choices=["train", "test", "study"])
    parser.add_argument("model", type=str, choices=models.keys())

    # program arguments
    group = parser.add_argument_group("General")
    group.add_argument("--no_log", action="store_true")
    group.add_argument("--log_dir", type=str, default="./logs")
    group.add_argument("--data_dir", type=str, default="./data/train")

    # model arguments
    model_name = sys.argv[2]
    group = parser.add_argument_group("Model Hyperparameters")
    models[model_name].add_model_specific_args(group)

    # data arguments
    group = parser.add_argument_group("Data")
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--files_per_epoch", type=int, default=1000)
    group.add_argument("--num_workers", type=int, default=os.cpu_count())
    group.add_argument("--input_length", type=int, default=20)
    group.add_argument("--slim", action="store_true")

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--patience", type=int, default=10)
    group.add_argument("--profiler", type=str, default=None)
    group.add_argument("--fast_dev_run", action="store_true")

    params = parser.parse_args()
    if params.operation == "train":
        train(make_trainer(params), params)
    elif params.operation == "test":
        test(make_trainer(params), params)
    elif params.operation == "study":
        study(params)
    else:
        raise ValueError(f"Unknown operation: {params.operation}")


def train(trainer: pl.Trainer, params: argparse.Namespace):
    torch.set_float32_matmul_precision("high")

    # common dataloader class
    dataloader_kwargs = dict(
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    model_cls = models[params.model]
    # load dataset for specific model
    if issubclass(model_cls, EncoderDecoder):
        # Prepare Image Dataset
        train_dataset = build_image_dp(
            params.data_dir, max_files=params.files_per_epoch
        )
        val_dataset = get_online_dataset(
            params,
            n_experiments=100 // dataloader_kwargs.get("num_workers", 1),
        )

        # collate function for images
        dataloader_kwargs["collate_fn"] = collate_image_fn

        # load model
        model = model_cls(**vars(params))

    elif issubclass(model_cls, SparseBase):
        # Prepare Sparse Dataset
        dataset = SparseDataset(params.data_dir, params.input_length, params.slim)
        train_dataset, val_dataset = random_split(
            dataset, [0.95, 0.05], generator=torch.Generator().manual_seed(42)
        )

        # collate function for sparse data
        dataloader_kwargs["collate_fn"] = SparseDataset.collate_fn

        # load model
        model = model_cls(measurement_dim=3, state_dim=2, pos_dim=2, **vars(params))

    else:
        raise ValueError(f"Unknown model: {params.model}.")

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)

    trainer.fit(model, train_loader, val_loader)


def test(trainer: pl.Trainer, params: argparse.Namespace):
    num_workers = min(torch.multiprocessing.cpu_count(), 4)
    test_loader = DataLoader(
        get_online_dataset(params, n_experiments=100 // num_workers),
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_image_fn,
    )
    trainer = make_trainer(params)
    model = Conv2dCoder(**vars(params))
    trainer.test(model, test_loader)


def study(params: argparse.Namespace):
    torch.set_float32_matmul_precision("high")
    study_name = "mtt-tomato"
    storage = os.environ["OPTUNA_STORAGE"]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5, max_resource=200, reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
        directions=["minimize"],
    )
    study.optimize(
        partial(objective, default_params=params),
        n_trials=1,
    )


def objective(trial: optuna.trial.Trial, default_params: argparse.Namespace):
    study_params = dict(
        lr=trial.suggest_float("lr", 1e-8, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-10, 1, log=True),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        upsampling=trial.suggest_categorical("upsampling", ["nearest", "transpose"]),
        n_encoder=3,
        n_hidden=4,
        n_channels=128,
        n_channels_hidden=1024,
        kernel_size=9,
        cardinality_weight=1e-7,
        activation="leaky_relu",
        optimizer="adamw",
    )
    params = argparse.Namespace(**{**vars(default_params), **study_params})

    # configure trainer
    logger = WandbLogger(
        project="mtt",
        log_model=True,
        group=trial.study.study_name,
    )
    callbacks: List[Callback] = [
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="./checkpoints",
            filename="best",
            auto_insert_metric_name=False,
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val/loss", patience=10),
        PyTorchLightningPruningCallback(trial, monitor="val/loss"),
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        max_epochs=params.max_epochs,
        devices=1,
    )
    train(trainer, params)

    # finish up
    trial.set_user_attr("wandb_id", logger.experiment.id)
    wandb.finish()
    return trainer.callback_metrics["val/loss"].item()


def init_simulator():
    return Simulator()


def make_trainer(params: argparse.Namespace, callbacks=[]) -> pl.Trainer:
    if params.no_log:
        logger = False
    else:
        # create loggers
        logger = WandbLogger(
            project="mtt", save_dir="logs", config=params, log_model=True
        )
        logger.log_hyperparams(params)
        typing.cast(Run, logger.experiment).log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                path.endswith(".py")
                and "logs" not in path
                and ("src" in path or "scripts" in path)
            ),
        )
        callbacks += [
            ModelCheckpoint(
                monitor="val/loss",
                dirpath="./checkpoints",
                filename="best",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        ]
    callbacks += [EarlyStopping(monitor="val/loss", patience=params.patience)]
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=not params.no_log,
        precision=32,
        devices=1,
        max_epochs=params.max_epochs,
        default_root_dir=".",
        profiler=params.profiler,
        fast_dev_run=params.fast_dev_run,
        log_every_n_steps=1 if params.slim else 50,
    )


def get_online_dataset(params: argparse.Namespace, n_experiments=1, n_steps=100):
    return OnlineImageDataset(
        **dict(
            **vars(params),
            length=params.input_length,
            init_simulator=init_simulator,
            n_experiments=n_experiments,
            n_steps=n_steps,
        )
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        # exit or the MPS server might be in an undefined state
        torch.cuda.synchronize()
