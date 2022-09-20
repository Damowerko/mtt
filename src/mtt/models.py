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
    @classmethod
    def add_model_specific_args(cls, group):
        for base in cls.__bases__:
            if hasattr(base, "add_model_specific_args"):
                group = base.add_model_specific_args(group)  # type: ignore
        args = get_init_arguments_and_types(cls)
        for name, types, default in args:
            if types[0] not in (int, float, str, bool):
                continue
            group.add_argument(f"--{name}", type=types[0], default=default)
        return group

    def __init__(
        self,
        img_size: int = 128,
        input_length: int = 20,
        output_length: int = 20,
        loss_fn: str = "l2",
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 0,
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

    def forward(self, x):
        raise NotImplementedError

    def loss(self, output_img, target_img):
        # target_img = target_img[0, -self.hparams.output_length - 1 :]
        if self.loss_fn == "l2":
            return F.mse_loss(output_img, target_img)
        elif self.loss_fn == "l1":
            return F.l1_loss(output_img, target_img)
        else:
            raise ValueError(f"Unknown loss {self.loss_fn}")

    def training_step(self, batch, batch_idx):
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        assert output_img.shape == target_img.shape
        loss = self.loss(output_img, target_img)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        assert output_img.shape == target_img.shape
        loss = self.loss(output_img, target_img)
        self.log("val/loss", loss, prog_bar=True)
        return input_img[0, -1], target_img[0, -1], output_img[0, -1]

    def validation_epoch_end(self, outputs):
        n_rows = min(1, len(outputs))
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
        self.logger.experiment.add_figure("images", fig, self.current_epoch)  # type: ignore

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )


class Conv2dCoder(EncoderDecoder):
    def __init__(
        self,
        n_encoder: int = 3,
        n_hidden: int = 2,
        n_channels: int = 64,
        n_channels_hidden: int = 128,
        kernel_size: int = 9,
        dilation: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        padding = ((kernel_size - 1) // 2,) * 2
        _kernel_size = (kernel_size, kernel_size)
        stride = (2, 2)
        _dilation = (dilation, dilation)

        # initialize the encoder and decoder layers
        # the input layer has one channel
        encoder_channels = [self.input_shape[0]] + [n_channels] * n_encoder
        encoder_shapes = [self.input_shape[1:]]
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            encoder_shapes.append(
                conv_output(
                    encoder_shapes[-1], _kernel_size, stride, padding, _dilation
                )
            )
            encoder_layers += [
                nn.Conv2d(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    _kernel_size,
                    stride,
                    padding,
                    _dilation,
                ),
                nn.ReLU(True),
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
                input_shape, kernel_size, stride, padding, dilation
            )
            output_padding = desired_shape - actual_shape

            decoder_layers += [
                nn.ConvTranspose2d(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size,
                    stride,
                    padding,
                    tuple(output_padding),
                    dilation=dilation,
                ),
                nn.ReLU(True) if i < len(decoder_channels) - 2 else nn.Identity(),
            ]
        self.decoder = nn.Sequential(*decoder_layers)

        # initialize the hidden layers
        hidden_channels = [n_channels] + [n_channels_hidden] * n_hidden + [n_channels]
        hidden_layers = []
        for i in range(n_hidden + 1):
            hidden_layers += [
                nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], 1),
                nn.ReLU(True),
            ]
        self.hidden = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(-1, self.embedding_size)
        x = self.hidden(x)
        # x = x.view((-1, self.hparams.n_channels) + self.embedding_shape)
        x = self.decoder(x)
        return x


class Conv3dCoder(EncoderDecoder):
    def __init__(
        self,
        n_encoder: int = 3,
        n_hidden: int = 2,
        n_channels: int = 8,
        kernel_time: int = 3,
        kernel_space: int = 9,
        dilation_space: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        padding = (kernel_time,) + ((kernel_space - 1) // 2,) * 2
        kernel_size = (kernel_time, kernel_space, kernel_space)
        stride = (1, 2, 2)
        dilation = (1, dilation_space, dilation_space)

        # initialize the encoder and decoder layers
        # the input layer has one channel
        encoder_channels = [1] + [n_channels] * n_encoder
        encoder_shapes = [self.input_shape[1:]]
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
        decoder_channels = [n_channels] * n_encoder + [1]
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            # specify output_padding to resolve output shape ambiguity when stride > 1
            input_shape = np.asarray(decoder_shapes[i], dtype=np.int32)
            desired_shape = np.asarray(decoder_shapes[i + 1], dtype=np.int32)
            actual_shape = conv_transpose_output(
                input_shape, kernel_size, stride, padding, dilation
            )
            output_padding = desired_shape - input_shape

            decoder_layers += [
                nn.ConvTranspose3d(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size,
                    stride,
                    padding,
                    tuple(output_padding),
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

    def forward(self, x):
        x = x.view((-1, 1) + self.input_shape).tranpose(1, 2)
        x = self.encoder(x)
        x = x.view(-1, self.embedding_size)
        x = self.hidden(x)
        x = x.view((-1, self.n_channels) + self.embedding_shape)
        x = self.decoder(x)
        x = x.view((-1,) + self.output_shape)
        return x
