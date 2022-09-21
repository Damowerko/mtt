import argparse
import os
from typing import Union

import pytorch_lightning as pl
from mtt.data import OnlineDataset
from mtt.models import Conv2dCoder, Conv3dCoder, EncoderDecoder
from mtt.simulator import Simulator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


def get_model_cls(model_type) -> EncoderDecoder:
    models = {
        "Conv2dCoder": Conv2dCoder,
        "Conv3dCoder": Conv3dCoder,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    assert issubclass(models[model_type], EncoderDecoder)
    return models[model_type]


def get_dataset(params: argparse.Namespace, n_steps=1000) -> OnlineDataset:
    init_simulator = lambda: Simulator(
        width=1000,
        n_targets=10,
        target_lifetime=5,
        clutter_rate=10,
        p_detection=0.95,
        sigma_motion=0.5,
        sigma_initial_state=(10.0, 10.0),
        n_sensors=3,
        sensor_range=500,
        noise_range=20.0,
        noise_bearing=0.2,
        dt=0.1,
    )
    return OnlineDataset(
        length=params.input_length,
        init_simulator=init_simulator,
        n_steps=n_steps,
        sigma_position=10,
        **vars(params),
    )


def get_trainer(params: argparse.Namespace) -> pl.Trainer:
    logger = (
        TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        if not params.no_log
        else None
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
        check_val_every_n_epoch=5,
    )


def get_checkpoint_path() -> Union[str, None]:
    ckpt_path = "./checkpoints/best.ckpt"
    return ckpt_path if os.path.exists(ckpt_path) else None


def train(params: argparse.Namespace):
    train_dataset = get_dataset(params, n_steps=1000)
    test_dataset = get_dataset(params, n_steps=100)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        num_workers=params.batch_size,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    trainer = get_trainer(params)
    model = Conv2dCoder(**vars(params))
    trainer.fit(model, train_loader, test_loader, ckpt_path=get_checkpoint_path())
    trainer.test(model, test_loader, ckpt_path=get_checkpoint_path())


def test(params: argparse.Namespace):
    dataset = get_dataset(params)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
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
    group.add_argument("--patience", type=int, default=4)

    params = parser.parse_args()
    if params.operation == "train":
        train(params)
    elif params.operation == "test":
        test(params)
    else:
        raise ValueError(f"Unknown operation: {params.operation}")
