"""
@Brief: FVSC model for background/RONI
"""
from subnet import *
from dataset_FVSC import *
from torch.utils.data import DataLoader
from Quantization import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from SSIM import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Semantic_encoder_BA(nn.Module):
    def __init__(self, in_channels=5):
        self.out_channel_N = 6  
        self.out_channel_M = 6  
        super(Semantic_encoder_BA, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels, self.out_channel_N, 3, stride=2, padding=1)  
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + self.out_channel_N) / 6)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(self.out_channel_N)  
        self.conv2 = nn.Conv2d(self.out_channel_N, self.out_channel_N, 3, stride=2, padding=1)  
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(self.out_channel_N)  
        self.conv3 = nn.Conv2d(self.out_channel_N, self.out_channel_N, 3, stride=2, padding=1)  
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(self.out_channel_N)  
        self.conv4 = nn.Conv2d(self.out_channel_N, self.out_channel_M, 3, stride=2, padding=1)  
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (self.out_channel_M + self.out_channel_N) / (self.out_channel_N + self.out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        x = self.gdn1(self.conv1(x)) 
        x = self.gdn2(self.conv2(x)) 
        x = self.gdn3(self.conv3(x))  
        x = self.conv4(x) 
        x = self.Sig(x)
        return x


class Semantic_decoder_BA(nn.Module):
    def __init__(self, out_channels=5):
        self.out_channel_N = 6  
        self.out_channel_M = 6  
        self.input_channel = self.out_channel_M
        super(Semantic_decoder_BA, self).__init__()  
        self.deconv1 = nn.ConvTranspose2d(self.input_channel, self.out_channel_N, 3, stride=2, padding=1, output_padding=1) 
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (
            math.sqrt(2 * 1 * (self.out_channel_M + self.out_channel_N) / (self.out_channel_M + self.out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(self.out_channel_N, inverse=True) 
        self.deconv2 = nn.ConvTranspose2d(self.out_channel_N, self.out_channel_N, 3, stride=2, padding=1, output_padding=1)  
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(self.out_channel_N, inverse=True)  
        self.deconv3 = nn.ConvTranspose2d(self.out_channel_N, self.out_channel_N, 3, stride=2, padding=1, output_padding=1)  
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(self.out_channel_N, inverse=True)  
        self.deconv4 = nn.ConvTranspose2d(self.out_channel_N, out_channels, 3, stride=2, padding=1, output_padding=1)  
        torch.nn.init.xavier_normal_(self.deconv4.weight.data,
                                     (math.sqrt(2 * 1 * (self.out_channel_N + 3) / (self.out_channel_N + self.out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class VideoCompressor_BA(nn.Module):

    def __init__(self):
        super(VideoCompressor_BA, self).__init__()
        self.opticFlow = ME_Spynet()  
        self.warpnet = Warp_net()  

        self.semantic_encoder = Semantic_encoder_BA(5) 
        self.Q = QuantizationLayer(3)
        self.DQ = DequantizationLayer(3)

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe):
        batch_size = input_image.shape[0]
        estmv = self.opticFlow(input_image, referframe)  
        prediction, _ = self.motioncompensation(referframe, estmv) 
        res_infor = input_image - prediction 
        transmission_infor = torch.cat([estmv, res_infor], dim=1)  
        semantic_feature = self.semantic_encoder(transmission_infor)
        semantic_feature_shape = semantic_feature.shape
        semantic_feature = semantic_feature.reshape(batch_size, -1)
        quant_semantic_feature = self.Q(semantic_feature)

        return quant_semantic_feature, semantic_feature_shape


class VideoDecompressor_BA(nn.Module):

    def __init__(self):
        super(VideoDecompressor_BA, self).__init__()
        self.mvDecoder = Synthesis_mv_net()  
        self.warpnet = Warp_net()  
        self.resDecoder = Synthesis_net() 
        self.respriorDecoder = Synthesis_prior_net()  
        self.semantic_decoder = Semantic_decoder_BA()
        self.Q = QuantizationLayer(3)
        self.DQ = DequantizationLayer(3)

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, referframe, quant_semantic_feature, semantic_feature_shape):
        quant_semantic_feature = self.DQ(quant_semantic_feature)
        quant_semantic_feature = quant_semantic_feature.reshape(semantic_feature_shape)
        semantic_de_feature = self.semantic_decoder(quant_semantic_feature)  
        out_res_infor = semantic_de_feature[:, 2:, :, :]
        out_mv = semantic_de_feature[:, :2, :, :]

        receive_predict_frame, _ = self.motioncompensation(referframe, out_mv)
        reconstructed_frame = receive_predict_frame + out_res_infor
        clipped_recon_image = reconstructed_frame.clamp(0., 1.)

        return clipped_recon_image


class FrameCodec_BA(nn.Module):
    """
    :brief: Frame codec for RONI. It is almost identical to FrameCodec, except for the loss function and compression ratio.
    """
    def __init__(self):
        super(FrameCodec_BA, self).__init__()
        self.videocompressor = VideoCompressor_BA()
        self.videodecompressor = VideoDecompressor_BA()
        self.ssimLoss = SSIM()
        self.mseLoss = nn.MSELoss(reduction='mean')

    def forward(self, referframe, input_image):
        input_image = input_image/255.
        referframe = referframe/255.
        quant_semantic_feature, semantic_feature_shape = self.videocompressor(input_image, referframe)  
        resultFrame = self.videodecompressor(referframe, quant_semantic_feature, semantic_feature_shape) 
        resultFrame = resultFrame*255

        return resultFrame

    def loss_fn(self, recovered_frame, frame):
        mseTmp = self.mseLoss(frame, recovered_frame)  
        lambda_ = 500
        ssimTmp = lambda_ - self.ssimLoss(frame, recovered_frame)*lambda_  
        loss = mseTmp + ssimTmp
        return loss


if __name__ == "__main__":
    model = FrameCodec_BA().to(device)
    ref_frame = torch.randint(0, 255, (1, 3, 240, 320)).to(device)
    input_frame = torch.randint(0, 255, (1, 3, 240, 320)).to(device)
    result = model(ref_frame, input_frame)
