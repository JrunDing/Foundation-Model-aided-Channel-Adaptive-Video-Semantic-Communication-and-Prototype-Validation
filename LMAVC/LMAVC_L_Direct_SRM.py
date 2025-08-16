
import os
import numpy as np
import warnings
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import time
from utils.results_ope import cleanFiles
from metrics import Metrics
from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, triang_ldpc_systematic_encode
from physical_mimo_ofdm import MIMO_OFDM
from dataset_LMAVC import *
from utils.mergeVideo import *
from model_LMAVC_SRM import *
from model_SRM import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
    GoP = 5
    cleanFiles("Results_L/Results")
    metrics = Metrics()

    encoder = VideoCompressor().to(device)
    remapper = Semantic_Remapping(1).to(device)

    decoder = VideoDecompressor().to(device)

    encoder.eval()
    decoder.eval()
    remapper.eval()

    parentDirectory = os.listdir("../data/UCF101/UCF-101-TEST-S/") 
    videoPath = [] 
    for k in parentDirectory: videoPath.append(os.listdir(f"../data/UCF101/UCF-101-TEST-S/{k}"))

    SNRdBList = [0, 5, 10, 15, 20, 25, 30]
    noVideoTrans = 0
    nextVideoFlag = False  
    breakDataCount = 0  
    for i in range(len(parentDirectory)):
        for j in range(len(videoPath[i])):

            mimo_ofdm = MIMO_OFDM(2, 2, 64, "Perfect", "ZF",
                                  moduType='QPSK')  

            nextVideoFlag = False
            srcObjectVideoPath = f"../data/UCF101/UCF-101-TEST-OBJECT-S/{parentDirectory[i]}/{videoPath[i][j]}" 
            srcVideoPath = f"../data/UCF101/UCF-101-TEST-S/{parentDirectory[i]}/{videoPath[i][j]}"  
            FPS = eval(os.popen(f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {srcVideoPath}").readline()[:-1])  
            widthAndHeight = os.popen(f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of default=nokey=1:noprint_wrappers=1 {srcVideoPath}").readlines()
            width = int(widthAndHeight[0][:-1])  
            height = int(widthAndHeight[1][:-1])  

            srcFrameNum = int(os.popen(
                f'ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 {srcVideoPath}').readlines()[0][:-1])  

            srcCapture = cv.VideoCapture(srcVideoPath)

            txBits = np.array([])
            I_frame = []  
            I_frame_Num = srcFrameNum // GoP + 1 
            srcBits = np.array([])

            if srcCapture.isOpened():
                for frame_idx_tx in range(srcFrameNum):
                    retSrc, imgSrc = srcCapture.read() 
                    if not retSrc:
                        nextVideoFlag = True  
                        break  
                    if type(imgSrc) is not np.ndarray:
                        nextVideoFlag = True
                        break
                    imgSrc = torch.unsqueeze(torch.tensor(imgSrc, dtype=torch.float).permute(2, 0, 1), dim=0) 
                    if frame_idx_tx % GoP == 0:
                        I_frame.append(imgSrc)
                    frameShape = imgSrc.shape 

                    with torch.no_grad():
                        if frame_idx_tx % GoP == 0:
                            semantics, semantics_shape = encoder((imgSrc / 255.).to(device), (
                                                         I_frame[frame_idx_tx//GoP]/255.).to(device)) 

                            semantics = remapper.Remapping(semantics, mimo_ofdm.H_freq)  
                            bits = remapper.Q(semantics) 
                            BitsNum_PerFrame = len(bits.reshape(-1))
                            last_frame_tx = I_frame[frame_idx_tx//GoP]
                        else:  
                            semantics, semantics_shape = encoder((imgSrc / 255.).to(device), (
                                                        last_frame_tx / 255.).to(device))
                            semantics = remapper.Remapping(semantics, mimo_ofdm.H_freq)  

                            bits = remapper.Q(semantics)
                            BitsNum_PerFrame = len(bits.reshape(-1)) 
                            last_frame_tx = imgSrc  

                    srcBitsTmp = np.array(bits.reshape(-1).cpu(), dtype=np.int8) 
                    srcBits_PerFrame = len(srcBitsTmp) 
                    srcBits = np.concatenate((srcBits, srcBitsTmp)).astype(np.uint8)  
                srcCapture.release()
            else:
                print('Metric SSIM: Open video failed!')
                continue
            if nextVideoFlag:
                breakDataCount += 1
                continue
            txBits = srcBits
            for SNRdB in SNRdBList:
                print('SNR={}dB'.format(SNRdB))
                rxBits = mimo_ofdm.MIMO_OFDM_Simulation(txBits, SNRdB)  
                if os.path.exists("../data/temp/LMAVC_L_temp.avi"):
                    os.remove("../data/temp/LMAVC_L_temp.avi")
                fourcc = cv.VideoWriter_fourcc(*'FFV1')  
                video_writer = cv.VideoWriter("../data/temp/LMAVC_L_temp.avi", fourcc, FPS, (width, height), True)
                for frame_idx_rx in range(srcFrameNum):
                    I_frame_ = (I_frame[frame_idx_rx//GoP]/255.).to(device) 
                    rxBits_PerFrame = torch.unsqueeze(torch.tensor(rxBits[frame_idx_rx*srcBits_PerFrame:(frame_idx_rx+1)*srcBits_PerFrame], dtype=torch.float), dim=0).to(device)  
                    with torch.no_grad():
                        if frame_idx_rx % GoP == 0: 
                            rxBits_PerFrame = remapper.DQ(rxBits_PerFrame)  
                            rxBits_PerFrame = remapper.Deremapping(rxBits_PerFrame, mimo_ofdm.H_freq)  
                            frame_tensor = decoder(I_frame_, rxBits_PerFrame, semantics_shape) 
                            last_frame_rx = frame_tensor
                        else:  
                            rxBits_PerFrame = remapper.DQ(rxBits_PerFrame)  
                            rxBits_PerFrame = remapper.Deremapping(rxBits_PerFrame, mimo_ofdm.H_freq)  
                            frame_tensor = decoder(last_frame_rx, rxBits_PerFrame, semantics_shape)  
                            last_frame_rx = frame_tensor

                    frame_numpy = np.array((torch.squeeze(frame_tensor) * 255).cpu().permute(1, 2, 0)).astype(np.uint8) 
                    frame_current = np.clip(frame_numpy, 0, 255)  
                    video_writer.write(frame_current)
                video_writer.release()



