import argparse
import os
from functools import partial
from glob import glob
from typing import List, Union

import optuna
import pytorch_lightning as pl
import torch
import wandb
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from mtt.data import OnlineDataset, build_train_datapipe, collate_fn
from mtt.models import Conv2dCoder
from mtt.simulator import Simulator


def main():
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("operation", choices=["train", "test", "study"])
    parser.add_argument("--no_log", action="store_true")

    # data arguments
    group = parser.add_argument_group("Data")
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--files_per_epoch", type=int, default=1000)

    # model arguments
    group = parser.add_argument_group("Model")
    group = Conv2dCoder.add_model_specific_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)
    group.add_argument("--patience", type=int, default=10)
    group.add_argument(
        "--map_location",
        type=str,
        default="cpu",
        help="The torch device onto which data will be loaded: cpu or cuda.",
    )

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
    train_dp = build_train_datapipe(
        "/nfs/general/mtt_data/train",
        max_files=params.files_per_epoch,
        map_location=params.map_location,
    )
    dataloader_kwargs = dict(
        batch_size=params.batch_size,
        collate_fn=collate_fn,
    )
    if params.map_location == "cpu":
        dataloader_kwargs = dict(
            **dataloader_kwargs,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=min(torch.multiprocessing.cpu_count(), 4),
        )
    train_loader = DataLoader(train_dp, **dataloader_kwargs)

    # for validation we will generate samples online, to avoid overfitting
    val_dataset = get_online_dataset(
        params,
        n_experiments=100 // dataloader_kwargs.get("n_workers", 1),
    )
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)

    model = Conv2dCoder(**vars(params))
    trainer.fit(model, train_loader, val_loader, ckpt_path=get_checkpoint_path())


def test(trainer: pl.Trainer, params: argparse.Namespace):
    num_workers = min(torch.multiprocessing.cpu_count(), 4)
    test_loader = DataLoader(
        get_online_dataset(params, n_experiments=100 // num_workers),
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    trainer = make_trainer(params)
    model = Conv2dCoder(**vars(params))
    trainer.test(model, test_loader, ckpt_path=get_checkpoint_path())


def study(params: argparse.Namespace):
    torch.set_float32_matmul_precision("high")
    study_name = "mtt-less-cardinality"
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
        n_encoder=trial.suggest_int("n_encoder", 2, 6),
        n_hidden=trial.suggest_int("n_hidden", 1, 10),
        n_channels=trial.suggest_int("n_channels", 32, 256),
        n_channels_hidden=trial.suggest_int("n_channels_hidden", 32, 2048),
        kernel_size=trial.suggest_int("kernel_size", 3, 11),
        lr=trial.suggest_float("lr", 1e-8, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-10, 1, log=True),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        cardinality_weight=0.00001,
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
        accelerator="gpu",
        devices=1,
    )
    train(trainer, params)

    # finish up
    trial.set_user_attr("wandb_id", logger.experiment.id)
    wandb.finish()
    return trainer.callback_metrics["val/loss"].item()


def init_simulator():
    return Simulator()


def make_trainer(params: argparse.Namespace) -> pl.Trainer:
    if params.no_log:
        logger = False
    else:
        # resume wandb run if it exists
        wandb_kwargs = dict(
            project="mtt",
            log_model=True,
        )
        if os.path.exists("checkpoints/best.ckpt"):
            wandb_kwargs["resume"] = "must"
            wandb_kwargs["version"] = glob("wandb/run-*")[0].split("-")[-1]
            print(f"Resuming wandb run {wandb_kwargs['version']}.")

        # create loggers
        logger = [
            TensorBoardLogger(save_dir="./", name="tensorboard", version=""),
            WandbLogger(**wandb_kwargs),
        ]

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="./checkpoints",
            filename="best",
            auto_insert_metric_name=False,
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        if not params.no_log
        else None,
        EarlyStopping(
            monitor="val/loss",
            patience=params.patience,
        ),
    ]
    callbacks = [c for c in callbacks if c is not None]
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=not params.no_log,
        precision=32,
        accelerator="auto",
        devices=1,
        max_epochs=params.max_epochs,
        default_root_dir=".",
    )


def get_checkpoint_path() -> Union[str, None]:
    ckpt_path = "./checkpoints/best.ckpt"
    return ckpt_path if os.path.exists(ckpt_path) else None


def get_online_dataset(params: argparse.Namespace, n_experiments=1, n_steps=100):
    return OnlineDataset(
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
