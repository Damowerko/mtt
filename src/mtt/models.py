from black import out
import pytorch_lightning as pl
from pytorch_lightning.utilities.argparse import get_init_arguments_and_types
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, loss="l2", **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")

    def forward(self, x):
        raise NotImplementedError

    def loss(self, output_img, target_img):
        if self.hparams.loss == "l2":
            return F.mse_loss(output_img, target_img)
        elif self.hparams.loss == "l1":
            return F.l1_loss(output_img, target_img)
        else:
            raise ValueError(f"Unknown loss {self.hparams.loss}")

    def training_step(self, batch, batch_idx):
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        loss = self.loss(output_img, target_img)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        loss = self.loss(output_img, target_img)
        self.log("val/loss", loss)
        self.log("hp_metric", loss)  # this is for tensorboard
        return input_img[0, 0, -1], target_img[0, 0, -1], output_img[0, 0, -1]

    def validation_epoch_end(self, outputs):
        n_rows = min(5, len(outputs))
        idx = rng.choice(len(outputs), size=n_rows, replace=False)
        fig, ax = plt.subplots(
            n_rows, 3, figsize=(9, 3 * n_rows), sharex=True, sharey=True, squeeze=False
        )
        for i, j in enumerate(idx):
            input, target, output = outputs[j]
            ax[i, 0].imshow(input.cpu().numpy())
            ax[i, 1].imshow(target.cpu().numpy())
            ax[i, 2].imshow(output.cpu().numpy())
        plt.setp(ax, xticks=[], yticks=[])
        plt.subplots_adjust(wspace=0, hspace=0)
        self.logger.experiment.add_figure("images", fig, self.current_epoch)


class Conv3dCoder(EncoderDecoder):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: int = 0,
        length: int = 20,
        img_size: int = 128,
        n_encoder: int = 3,
        n_hidden: int = 2,
        n_channels: int = 64,
        kernel_time: int = 5,
        kernel_space: int = 9,
        dilation_time: int = 1,
        dilation_space: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore="kwargs")

        self.input_shape = (length, img_size, img_size)
        kernel_size = (kernel_time, kernel_space, kernel_space)
        stride = (1, 2, 2)
        padding = (kernel_time // 2, kernel_space // 2, kernel_space // 2)
        dilation = (dilation_time, dilation_space, dilation_space)

        # initialize the encoder and decoder layers
        # the input layer has one channel
        encoder_channels = [1] + [n_channels] * n_encoder
        encoder_shapes = [self.input_shape]
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            encoder_shapes.append(
                conv_output(encoder_shapes[-1], kernel_size, stride, padding, dilation)
            )
            encoder_layers += [
                nn.Conv3d(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                ),
                nn.ReLU(True),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

        # the decoder layers are analogous to the encoder layers but in reverse
        decoder_shapes = encoder_shapes[::-1]
        decoder_channels = encoder_channels[::-1]
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            # specify output_padding to resolve output shape ambiguity when stride > 1
            input_shape = decoder_shapes[i]
            desired_shape = decoder_shapes[i + 1]
            actual_shape = conv_transpose_output(
                input_shape, kernel_size, stride, padding, dilation
            )
            output_padding = np.asarray(desired_shape) - np.asarray(actual_shape)

            decoder_layers += [
                nn.ConvTranspose3d(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size,
                    stride,
                    padding,
                    output_padding,
                    dilation=dilation,
                ),
                nn.ReLU(True) if i < len(decoder_channels) - 2 else nn.Identity(),
            ]
        self.decoder = nn.Sequential(*decoder_layers)

        # initialize the hidden layers
        self.embedding_shape = encoder_shapes[-1]
        self.embedding_size = np.prod(self.embedding_shape)
        self.hiden_layers = []
        for i in range(n_hidden):
            self.hiden_layers += [
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(True),
            ]
        self.hidden = nn.Sequential(*self.hiden_layers)

    @staticmethod
    def add_model_specific_args(group):
        args = get_init_arguments_and_types(Conv3dCoder)
        for name, types, default in args:
            if types[0] not in (int, float, str, bool):
                continue
            group.add_argument(f"--{name}", type=types[0], default=default)
        return group

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.embedding_size)
        x = self.hidden(x)
        x = x.view((-1, self.hparams.n_channels) + self.embedding_shape)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
