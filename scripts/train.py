import argparse
from ast import arg
import os

import pytorch_lightning as pl
from mtt.data import OnlineDataset
from mtt.models import Conv2dCoder, Conv3dCoder, EncoderDecoder
from mtt.simulator import Simulator
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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


def train(params: argparse.Namespace):
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
    dataset = OnlineDataset(
        length=params.input_length,
        init_simulator=init_simulator,
        sigma_position=0.01,
        **vars(params),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.batch_size,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    model = Conv2dCoder(**vars(params))

    logger = (
        TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        if params.log
        else None
    )
    callbacks = [
        ModelCheckpoint(
            monitor="train/loss",
            dirpath="./checkpoints",
            filename="epoch={epoch}-loss={train/loss:0.4f}",
            auto_insert_metric_name=False,
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="train/loss",
            patience=params.patience,
        ),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
        check_val_every_n_epoch=5,
    )

    # check if checkpoint exists
    ckpt_path = "./checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--log", type=int, default=1)

    # data arguments
    group = parser.add_argument_group("Data")
    group.add_argument("--batch_size", type=int, default=16)
    group.add_argument("--n_steps", type=int, default=100)

    # model arguments
    group = parser.add_argument_group("Model")
    group = Conv2dCoder.add_model_specific_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)
    group.add_argument("--patience", type=int, default=4)

    params = parser.parse_args()
    train(params)
