"""
@Brief: Test the performance of one frame  RONI
"""
import torch
import torch.nn as nn
import argparse
import torchvision
from model_FVSC_BA import *
from dataset_FVSC import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_one_frame():
    """
    @Brief: Test the performance of one frame   RONI
    """
    test_data = UCF101(path="../data/UCF101/UCF-101-TRA-S-BACK-IMG-1/")
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    frameCodec_model = FrameCodec_BA().to(device)
    frameCodec_model.videocompressor.load_state_dict(torch.load('../checkpoints/BACKGROUND/***.pth'))
    frameCodec_model.videodecompressor.load_state_dict(torch.load('../checkpoints/BACKGROUND/***.pth'))
    frameCodec_model.eval()

    lpipsLossAlex = lpips.LPIPS(net='alex').to(device)

    for test_data_ in test_data_loader:
        ref = test_data_[0].to(device)
        frame = test_data_[1].to(device)

        with torch.no_grad():  
            recovered_frame = frameCodec_model(ref, frame)  

            output_numpy_frame = np.rint(torch.squeeze(frame).transpose(0, 1).transpose(1, 2).cpu().numpy()).astype(np.uint8)
            PIL_frame = Image.fromarray(output_numpy_frame)

            output_numpy_recovered_frame = np.rint(torch.squeeze(recovered_frame).transpose(0, 1).transpose(1, 2).cpu().numpy()).astype(np.uint8)
            PIL_recovered_frame = Image.fromarray(output_numpy_recovered_frame)

            PIL_frame.save('./frame.jpg', quality=95)
            PIL_recovered_frame.save('./recovered_frame.jpg', quality=95)

            psnr_value = compare_psnr(output_numpy_frame, output_numpy_recovered_frame, data_range=255)
            ssim_value = compare_ssim(output_numpy_frame, output_numpy_recovered_frame, data_range=255, channel_axis=2)

            lpips_value = lpipsLossAlex(frame, recovered_frame).item()
            print("PSNR:", psnr_value)
            print("SSIM:", ssim_value)
            print("lpips:", lpips_value)

            break


if __name__ == "__main__":
    run_one_frame()

