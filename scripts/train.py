from typing import Dict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from mtt.models import EncoderDecoder, Conv2dCoder, Conv3dCoder
from mtt.sensor import Sensor
from mtt.simulator import Simulator
from mtt.data import OnlineDataset
import argparse
import os


def get_model_cls(model_type):
    models: Dict[str, EncoderDecoder] = {
        "Conv2dCoder": Conv2dCoder,
        "Conv3dCoder": Conv3dCoder,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type]


def train(params):
    init_simulator = lambda: Simulator(
        width=1000,
        p_initial=4,
        p_birth=params.p_birth,
        p_survival=0.95,
        p_clutter=params.p_clutter,
        p_detection=0.95,
        sigma_motion=5.0,
        sigma_initial_state=(50.0, 50.0, 2.0),
        n_sensors=1,
        noise_range=10.0,
        noise_bearing=0.1,
        dt=0.1,
    )
    dataset = OnlineDataset(
        length=params.input_length,
        init_simulator=init_simulator,
        sigma_position=0.01,
        **vars(params),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.batch_size,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    model = get_model_cls(params.model)(**vars(params))

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
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
        check_val_every_n_epoch=10,
    )

    # check if checkpoint exists
    ckpt_path = "./checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        default="Conv2dCoder",
        choices=["Conv2dCoder", "Conv3dCoder"],
        help="For the Conv3dCoder you need to lower the n_channels to something like 8 or less.",
    )

    # data arguments
    group = parser.add_argument_group("Data")
    group.add_argument("--batch_size", type=int, default=16)
    group.add_argument("--n_steps", type=int, default=100)
    group.add_argument("--p_clutter", type=float, default=1e-4)
    group.add_argument("--p_birth", type=float, default=2)

    # model arguments
    group = parser.add_argument_group("Model")
    # params = parser.parse_known_args()[0]
    group = get_model_cls("Conv2dCoder").add_model_specific_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)

    params = parser.parse_args()
    train(params)
