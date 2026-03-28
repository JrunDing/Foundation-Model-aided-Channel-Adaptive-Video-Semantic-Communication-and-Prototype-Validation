"""
@Brief: Key frame transmission network
"""

import torch
import torch.nn as nn
from Quantization import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filter = 128


class KeyFrameEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, padding=1), nn.PReLU(),
                                     nn.Conv2d(16, 32, 3, 2, padding=2), nn.PReLU(),
                                     nn.Conv2d(32, 64, 3, 2, padding=2), nn.PReLU(),
                                     nn.Conv2d(64, 128, 3, 1, padding=2), nn.PReLU(),
                                     nn.Conv2d(128, 128, 3, 1, padding=2), nn.PReLU(),
                                     nn.Conv2d(128, filter, 3, 1, padding=2), nn.Sigmoid())

    def forward(self, x):
        x = x/255.
        x = self.encoder(x)
        return x


class KeyFrameDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(nn.ConvTranspose2d(filter, 128, 3, 1, padding=2), nn.PReLU(),
                                     nn.ConvTranspose2d(128, 128, 3, 1, padding=2), nn.PReLU(),
                                     nn.ConvTranspose2d(128, 64, 3, 1, padding=2), nn.PReLU(),
                                     nn.ConvTranspose2d(64, 32, 3, 2, padding=2), nn.PReLU(),
                                     nn.ConvTranspose2d(32, 16, 3, 2, padding=2, output_padding=1), nn.PReLU(),
                                     nn.ConvTranspose2d(16, 3, 3, 2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.decoder(x)
        x = x*255.
        return x


class KeyFrameCodec(nn.Module):

    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.encoder = KeyFrameEncoder()
        self.q = QuantizationLayer(2)
        self.dq = DequantizationLayer(2)
        self.decoder = KeyFrameDecoder()

    def channel(self, x, SNRdB):
        """
        avg = torch.sum(x ** 2, dim=[1, 2, 3], dtype=torch.float32)/(self.filter*56*56)
        noise_std = torch.sqrt(avg*(1/(10**(SNRdB/10))))
        noise_r = torch.randn(self.batch, self.filter, 56, 56).to(device)
        return torch.multiply(noise_r.T, noise_std).T + x
        """
        avg = torch.sum(x ** 2, dim=[1, 2, 3], dtype=torch.float32)/(32*60*80)
        noise_std = torch.sqrt(avg*(1/(10**(SNRdB/10))))
        noise_r = torch.randn(self.batch, 32, 60, 80).to(device)
        return torch.multiply(noise_r.T, noise_std).T + x

    def forward(self, x):
        x = self.encoder(x)
        shape = x.shape
        x = x.reshape(self.batch, -1)
        x = self.q(x)
        # print(x.shape)
        x = self.dq(x)

        # x = self.channel(x, SNRdB)
        x = x.reshape(shape)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    inuu = torch.normal(0, 1, (1, 3, 240, 320)).to(device)
    codec = KeyFrameCodec(1).to(device)

    y = codec(inuu)
    print(y.shape)

