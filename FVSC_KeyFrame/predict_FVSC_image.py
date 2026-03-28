"""
@Brief: Test the performance of FVSC key frame transmission
"""

import torch
import torch.nn as nn
import argparse
import torchvision
import numpy as np
import PIL.Image as Image
from dataset_image import UCF101_Image
from torch.utils.data import DataLoader
from FVSC_image import KeyFrameCodec


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):

    test_data = UCF101_Image('../data/UCF101/UCF-101-TEST-IMAGE')
    test_data_loader = DataLoader(test_data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = KeyFrameCodec().to(device)
    model.encoder.load_state_dict(torch.load('../checkpoints/enc_epoch19_loss8.95.pth'))
    model.q.load_state_dict(torch.load('../checkpoints/q_epoch19_loss8.95.pth'))
    model.dq.load_state_dict(torch.load('../checkpoints/dq_epoch19_loss8.95.pth'))
    model.decoder.load_state_dict(torch.load('../checkpoints/dec_epoch19_loss8.95.pth'))
    model.eval()

    PSNR = 0
    with torch.no_grad():
        for src_img in test_data_loader:
            src_img = src_img.to(device)
            compressed_img = model(args.batch, src_img)

            loss_fn = torch.nn.MSELoss().to(device)
            loss = loss_fn(src_img, compressed_img)
            print(loss)
            psnr = 10 * torch.log10(255 * 255 / loss).item()
            PSNR += psnr
            break
    print(PSNR/len(test_data_loader))
    print(PSNR)

    compressed_img = np.rint(torch.squeeze(compressed_img).transpose(0, 1).transpose(1, 2).cpu().numpy()).astype(np.uint8)
    PIL_compressed_img = Image.fromarray(compressed_img)
    PIL_compressed_img.save('1.png', lossless=True)

    src_img = np.rint(torch.squeeze(src_img).transpose(0, 1).transpose(1, 2).cpu().numpy()).astype(np.uint8)
    PIL_compressed_img = Image.fromarray(src_img)
    PIL_compressed_img.save('2.png', lossless=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=1, type=int)
    # parser.add_argument('--SNRdB', default=10, type=int)

    args_ = parser.parse_args()

    main(args_)


