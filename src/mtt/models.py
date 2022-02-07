import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvEncoderDecoder(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        input_size=(20, 256, 256),
        n_encoder=5,
        n_hidden=2,
        n_channels=128,
        kernel_size=(3, 6, 6),
        stride=(1, 2, 2),
        padding=(1, 2, 2),
        dilation=(1, 1, 1),
    ):
        super().__init__()
        self.save_hyperparameters()

        # initialize the encoder and decoder layers
        channels = [1] + [n_channels] * n_encoder
        encoder_layers = []
        decoder_layers = []
        for i in range(len(channels) - 1):
            encoder_layers += [
                nn.Conv3d(channels[i], channels[i + 1], kernel_size, stride, padding),
                nn.LeakyReLU(True),
            ]
            decoder_layers += [
                nn.ConvTranspose3d(
                    channels[-i - 1], channels[-i - 2], kernel_size, stride, padding
                ),
                nn.LeakyReLU(True),
            ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # find the dimensions of the embedding
        self.embedding_shape = list(input_size)
        for _ in range(n_encoder):
            for i in range(3):
                self.embedding_shape[i] = (
                    self.embedding_shape[i]
                    + 2 * padding[i]
                    - dilation[i] * (kernel_size[i] - 1)
                    - 1
                ) // stride[i] + 1
        self.embedding_shape = tuple(self.embedding_shape)
        self.embedding_size = np.prod(self.embedding_shape)

        # initialize the hidden layers
        self.hiden_layers = [
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(True),
        ] * n_hidden
        self.hidden = nn.Sequential(*self.hiden_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.embedding_size)
        x = self.hidden(x)
        x = x.view((-1, self.hparams.n_channels) + self.embedding_shape)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )

    def training_step(self, batch):
        input_img, target_img, *_ = batch
        output_img = self(input_img)
        loss = self.loss(output_img, target_img)
        self.log("train/loss", loss)
        return loss

    def train_dataloader(self):
        return super().train_dataloader()

    def loss(self, input_img, target_img):
        return F.mse_loss(input_img, target_img)
