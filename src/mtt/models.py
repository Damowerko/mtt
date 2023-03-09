import os
import subprocess
from typing import Callable, Type

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.argparse import get_init_arguments_and_types

from mtt.data import StackedImageData
from mtt.peaks import find_peaks
from mtt.utils import compute_ospa

rng = np.random.default_rng()


def conv_output(shape, kernel_size, stride, padding, dilation):
    shape = np.asarray(shape)
    kernel_size = np.asarray(kernel_size)
    stride = np.asarray(stride)
    padding = np.asarray(padding)
    dilation = np.asarray(dilation)
    return tuple((shape + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)


def conv_transpose_output(shape, kernel_size, stride, padding, dilation):
    shape = np.asarray(shape)
    kernel_size = np.asarray(kernel_size)
    stride = np.asarray(stride)
    padding = np.asarray(padding)
    dilation = np.asarray(dilation)
    return tuple((shape - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1)


class EncoderDecoder(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        for base in cls.__bases__:
            if hasattr(base, "add_model_specific_args"):
                group = base.add_model_specific_args(group)  # type: ignore
        args = get_init_arguments_and_types(cls)  # type: ignore
        for name, types, default in args:
            if types[0] not in (int, float, str, bool):
                continue
            if types[0] == bool:
                group.add_argument(f"--{name}", dest=name, action="store_true")
            else:
                group.add_argument(f"--{name}", type=types[0], default=default)
        return group

    def __init__(
        self,
        img_size: int = 128,
        input_length: int = 20,
        output_length: int = 1,
        loss_fn: str = "l2",
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 0,
        ospa_cutoff: float = 500,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_shape = (input_length, img_size, img_size)
        self.output_shape = (output_length, img_size, img_size)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.ospa_cutoff = ospa_cutoff

    def forward(self, x):
        raise NotImplementedError

    def loss(self, output_img, target_img):
        if output_img.shape != target_img.shape:
            raise ValueError(
                f"Output shape {output_img.shape} != target shape {target_img.shape}."
            )
        if self.loss_fn == "l2":
            return F.mse_loss(output_img, target_img)
        elif self.loss_fn == "l1":
            return F.l1_loss(output_img, target_img)
        else:
            raise ValueError(f"Unknown loss {self.loss_fn}")

    def truncate_batch(self, batch: StackedImageData):
        input_img, target_img, info = batch
        target_img = target_img[:, -self.output_shape[0] :]
        info = [_info[-self.output_shape[0] :] for _info in info]
        return StackedImageData(input_img, target_img, info)

    def training_step(self, batch, *_):
        batch = self.truncate_batch(batch)
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        loss = self.loss(output_img, target_img)
        self.log("train/loss", loss, batch_size=input_img.shape[0])
        return loss

    def validation_step(self, batch, *_):
        batch = self.truncate_batch(batch)
        input_img, target_img, info = batch
        output_img = self(input_img)

        loss = self.loss(output_img, target_img)
        self.log("val/loss", loss, prog_bar=True, batch_size=input_img.shape[0])

        cardinality_mae = self.cardinality_errors(batch, output_img).abs().mean()
        self.log(
            "val/cardinality_mae",
            cardinality_mae,
            prog_bar=True,
            batch_size=input_img.shape[0],
        )
        return input_img[0, -1], target_img[0, -1], output_img[0, -1]

    def test_step(self, batch, *_):
        batch = self.truncate_batch(batch)
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        self.log(
            "test/loss",
            self.loss(output_img, target_img),
            batch_size=input_img.shape[0],
        )
        self.log(
            "test/ospa", self.ospa(batch), prog_bar=True, batch_size=input_img.shape[0]
        )
        return input_img[0, -1], target_img[0, -1], output_img[0, -1]

    def test_epoch_end(self, outputs):
        n_rows = min(5, len(outputs))
        idx = rng.choice(len(outputs), size=n_rows, replace=False)
        fig, ax = plt.subplots(
            n_rows, 3, figsize=(9, 3 * n_rows), sharex=True, sharey=True, squeeze=False
        )
        for i, j in enumerate(idx):
            input, target, output = outputs[j]
            assert isinstance(input, torch.Tensor)
            assert isinstance(target, torch.Tensor)
            assert isinstance(output, torch.Tensor)
            ax[i, 0].imshow(input.cpu().numpy())  # type: ignore
            ax[i, 1].imshow(target.cpu().numpy())  # type: ignore
            ax[i, 2].imshow(output.cpu().numpy())  # type: ignore
        plt.setp(ax, xticks=[], yticks=[])
        plt.subplots_adjust(wspace=0, hspace=0)
        if self.logger:
            self.logger.experiment.add_figure("images", fig, self.current_epoch)  # type: ignore

    def ospa(self, batch):
        # average ospa over batch
        input_img, target_img, info = batch
        output_img = self(input_img)
        assert output_img.shape == target_img.shape
        ospa_value = 0
        for i in range(output_img.shape[0]):
            img = output_img[i, -1].cpu().numpy()
            X = find_peaks(img, info[i][-1]["window"]).means
            Y = info[i][-1]["target_positions"]
            ospa_value += compute_ospa(X, Y, self.ospa_cutoff, p=2)
        return ospa_value / output_img.shape[0]

    def cardinality_from_image(self, image: torch.Tensor):
        return image.clamp_min(0).sum(dim=(-1, -2))

    def cardinality_errors(self, batch: StackedImageData, output_img: torch.Tensor):
        cardinality_estimates = self.cardinality_from_image(output_img).cpu()
        cardinality_truth = torch.zeros_like(cardinality_estimates)
        for i in range(cardinality_estimates.shape[0]):
            for j in range(cardinality_estimates.shape[1]):
                cardinality_truth[i, j] = len(batch.info[i][j]["target_positions"])
        cardinality_errors = cardinality_estimates - cardinality_truth
        return cardinality_errors

    def configure_optimizers(self):
        # pick optimizer
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        return optimizer


class Conv2dCoder(EncoderDecoder):
    def __init__(
        self,
        n_encoder: int = 3,
        n_hidden: int = 2,
        n_channels: int = 64,
        n_channels_hidden: int = 128,
        kernel_size: int = 9,
        dilation: int = 1,
        batch_norm: bool = True,
        activation: str = "leaky_relu",
        loss_valid_convolution: bool = False,
        **kwargs,
    ):
        """
        Args:
            n_encoder: number of encoder and decoder layers
            n_hidden: number of hidden layers
            n_channels: number of channels in the encoder layers
            n_channels_hidden: number of channels in the hidden layers
            kernel_size: kernel size of the convolutional layers
            dilation: dilation of the convolutional layers
            batch_norm: whether to use batch normalization
            activation: activation function to use, either "relu" or "leaky_relu"
            loss_valid_convolution: if true then compute loss only using parts of the output that did not use zero padding, similar to using padding="valid"
        """

        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.loss_valid_convolution = loss_valid_convolution
        self.n_encoder = n_encoder

        self.padding = ((kernel_size - 1) // 2,) * 2
        _kernel_size = (kernel_size, kernel_size)
        stride = (2, 2)
        _dilation = (dilation, dilation)
        _activation: Callable[..., nn.Module] = {
            "relu": lambda: nn.ReLU(inplace=True),
            "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        }[activation]

        # initialize the encoder and decoder layers
        # the input layer has one channel
        encoder_channels = [self.input_shape[0]] + [n_channels] * n_encoder
        encoder_shapes = [self.input_shape[1:]]
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            encoder_shapes.append(
                conv_output(
                    encoder_shapes[-1], _kernel_size, stride, self.padding, _dilation
                )
            )
            encoder_layers += [
                (nn.BatchNorm2d(encoder_channels[i]) if batch_norm else nn.Identity()),
                nn.Conv2d(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    _kernel_size,
                    stride,
                    self.padding,
                    _dilation,
                ),
                _activation(),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

        # the decoder layers are analogous to the encoder layers but in reverse
        decoder_shapes = encoder_shapes[::-1]
        decoder_channels = [n_channels] * n_encoder + [self.output_shape[0]]
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            # specify output_padding to resolve output shape ambiguity when stride > 1
            input_shape = np.asarray(decoder_shapes[i], dtype=np.int32)
            desired_shape = np.asarray(decoder_shapes[i + 1], dtype=np.int32)
            actual_shape = conv_transpose_output(
                input_shape, kernel_size, stride, self.padding, dilation
            )
            output_padding = desired_shape - actual_shape

            decoder_layers += [
                (nn.BatchNorm2d(decoder_channels[i]) if batch_norm else nn.Identity()),
                nn.ConvTranspose2d(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size,
                    stride,
                    self.padding,
                    tuple(output_padding),
                    dilation=dilation,
                ),
                _activation(),
            ]
        self.decoder = nn.Sequential(*decoder_layers)

        # initialize the hidden layers
        hidden_channels = [n_channels] + [n_channels_hidden] * n_hidden + [n_channels]
        hidden_layers = []
        for i in range(n_hidden + 1):
            hidden_layers += [
                (nn.BatchNorm2d(hidden_channels[i]) if batch_norm else nn.Identity()),
                nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], 1),
                _activation(),
            ]
        self.hidden = nn.Sequential(*hidden_layers)

    def loss(self, output_img: torch.Tensor, target_img: torch.Tensor):
        if not self.loss_valid_convolution:
            return super().loss(output_img, target_img)
        layer_padding = self.padding
        total_padding = [p * self.n_encoder for p in layer_padding]
        return super().loss(
            output_img[
                ...,
                total_padding[0] : -total_padding[0],
                total_padding[1] : -total_padding[1],
            ],
            target_img[
                ...,
                total_padding[0] : -total_padding[0],
                total_padding[1] : -total_padding[1],
            ],
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.hidden(x)
        x = self.decoder(x)
        return x


def load_model(model_cls: Type[EncoderDecoder], run_id: str) -> EncoderDecoder:
    run_path = subprocess.run(
        ["guild", "ls", run_id], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    run_path = run_path.splitlines()[0][:-1]
    checkpoint_path = os.path.join(run_path, "checkpoints/best.ckpt")
    return model_cls.load_from_checkpoint(checkpoint_path).cuda()
