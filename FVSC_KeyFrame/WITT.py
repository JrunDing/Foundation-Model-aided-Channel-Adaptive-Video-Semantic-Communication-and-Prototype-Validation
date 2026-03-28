"""
@Brief: WITT model
"""

from WITT_decoder import *
from WITT_encoder import *
from random import choice
import torch.nn as nn
from Quantization import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Witt(nn.Module):
    def __init__(self, batch, image_size, C):
        super(Witt, self).__init__()
        self.batch = batch
        self.encoder = WITT_Encoder(image_size, 2, 3, [128, 256], [2, 4], [4, 8], C)
        self.decoder = WITT_Decoder(image_size, [256, 128], [4, 2], [8, 4], C)
        # self.H = self.W = 0
        self.q = nn.Sequential(nn.Sigmoid(), QuantizationLayer(2))
        self.dq = DequantizationLayer(2)

    def forward(self, input_image):

        B, _, H, W = input_image.shape

        """
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        """
        x = self.encoder(input_image)

        shape = x.shape
        x = x.reshape(self.batch, -1)
        x = self.q(x)
        # print(x.shape)
        x = self.dq(x)
        x = x.reshape(shape)
        recon_image = self.decoder(x)
        return recon_image


if __name__ == "__main__":
    # Example
    inuu = torch.normal(0, 1, (1, 3, 240, 320)).to(device)
    codec = Witt(1, (240, 320), C=1).to(device)

    y = codec(inuu)
    print(y.shape)


