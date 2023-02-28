import argparse
from typing import List

import wandb
import optuna
import pytorch_lightning as pl
import torch
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from mtt.data import build_offline_datapipes, collate_fn
from mtt.models import Conv2dCoder

BATCH_SIZE = 128


def objective(trial: optuna.trial.Trial) -> float:
    params = argparse.Namespace(
        n_encoder=trial.suggest_int("n_layers", 1, 3),
        n_hidden=trial.suggest_int("n_hidden", 1, 10),
        n_channels=trial.suggest_int("n_channels", 1, 128),
        n_channels_hidden=trial.suggest_int("n_channels_hidden", 1, 128),
        kernel_size=trial.suggest_int("kernel_size", 1, 11),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        activation=trial.suggest_categorical("activation", ["relu", "leaky_relu"]),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"]),
        lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
    )
    train_dp, val_dp = build_offline_datapipes("/nfs/general/mtt_data/train")
    train_loader = DataLoader(
        dataset=train_dp,
        batch_size=BATCH_SIZE,
        num_workers=min(torch.multiprocessing.cpu_count(), 32),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dp,
        batch_size=BATCH_SIZE,
        num_workers=min(torch.multiprocessing.cpu_count(), 32),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger = WandbLogger(
        project="mtt",
        log_model="best",
        group=trial.study.study_name,
        tags=[f"{}"],
    )
    callbacks: List[Callback] = [
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
        PyTorchLightningPruningCallback(trial, monitor="val/loss"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        max_epochs=1,
        accelerator="gpu",
        devices=1,
    )
    model = Conv2dCoder(**vars(params))
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    return trainer.callback_metrics["val/loss"].item()


if __name__ == "__main__":
    study_name = "mtt"  # Unique identifier of the study.
    storage = "postgresql://optuna:optuna@optuna-db.owerko.svc.cluster.local:5432"  # Storage option.
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
