
import torch
import cv2 as cv
import numpy as np
import lpips
import os
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Metrics(object):
    def __init__(self):
        self.lpipsLossAlex = lpips.LPIPS(net='alex').to(device)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

    def cal_psnr(self, srcVideo, dstVideo, srcFrameNum):

        psnrNpList = np.array([])
        if srcVideo.isOpened() and dstVideo.isOpened():
            for frame in range(srcFrameNum):
                retSrc, imgSrc = srcVideo.read() 
                retDst, imgDst = dstVideo.read()
                if not retDst: break  
                psnrTmp = compare_psnr(imgSrc, imgDst, data_range=255)
                if psnrTmp == "shapeNotEqual":
                    psnrNpList = np.concatenate((psnrNpList, np.array([0])))
                else:
                    psnrNpList = np.concatenate((psnrNpList, np.array([psnrTmp])))
            PSNR = np.array([np.mean(psnrNpList)])
            print('PSNR = {}dB'.format(PSNR))
            return PSNR
        else:
            # raise RuntimeError('Metric PSNR: Open video failed!')
            print('Metric PSNR: Open video failed!')
            return np.array([0])

    def cal_ssim(self, srcVideo, dstVideo, srcFrameNum):
        ssimNpList = np.array([])
        if srcVideo.isOpened() and dstVideo.isOpened():
            for frame in range(srcFrameNum):
                retSrc, imgSrc = srcVideo.read()  
                retDst, imgDst = dstVideo.read()
                if not retDst: break  
                ssimTmp = compare_ssim(imgSrc, imgDst, data_range=255, channel_axis=2)
                if ssimTmp == "shapeNotEqual":
                    ssimNpList = np.concatenate((ssimNpList, np.array([0])))
                else:
                    ssimNpList = np.concatenate((ssimNpList, np.array([ssimTmp])))
            SSIM = np.array([np.mean(ssimNpList)])
            print('SSIM = {}'.format(SSIM))
            return SSIM
        else:
            # raise RuntimeError('Metric SSIM: Open video failed!')
            print('Metric SSIM: Open video failed!')
            return np.array([0])

    def cal_lpips(self, srcVideo, dstVideo, srcFrameNum):
        lpipsNpList = np.array([])
        if srcVideo.isOpened() and dstVideo.isOpened():
            for frame in range(srcFrameNum):
                retSrc, imgSrc = srcVideo.read()  
                retDst, imgDst = dstVideo.read()
                if not retDst: break  
                imgSrc = torch.unsqueeze(torch.tensor(imgSrc), dim=0).transpose(3, 1).transpose(2, 3).to(device)
                imgDst = torch.unsqueeze(torch.tensor(imgDst), dim=0).transpose(3, 1).transpose(2, 3).to(device)
                lpipsNpList = np.concatenate((lpipsNpList, np.array([self.lpipsLossAlex(imgSrc, imgDst).to("cpu").item()])))
            LPIPS = np.array([np.mean(lpipsNpList)])
            print('LPIPS = {}'.format(LPIPS))
            return LPIPS
        else:
            # raise RuntimeError('Metric LPIPS: Open video failed!')
            print('Metric LPIPS: Open video failed!')
            return np.array([1])

    def cal_object_mse(self, video_, srcFrameNum, refVideo):
        obMSEList = np.array([])
        if video_.isOpened() and refVideo.isOpened():
            for frame in range(srcFrameNum):
                retVid, imgVid = video_.read()
                if imgVid is None:
                    continue
                retRef, imgRef = refVideo.read()  
                if imgRef is None:
                    continue
                if not retRef: break  

                maskRef = imgRef.copy() 
                maskRef[np.sum(maskRef, axis=2) <= 10] = np.array([0, 0, 0]) 
                maskRef[np.sum(maskRef, axis=2) > 10] = np.array([1, 1, 1])

                imgVid = imgVid*maskRef
                imgRef = imgRef*maskRef

                imgRefTmp = np.sum(imgRef, axis=2) 
                num = 320*240*3 - np.sum(imgRefTmp == 0)*3  
                mseTmp = np.sum((imgRef - imgVid) ** 2) / num 

                obMSEList = np.concatenate((obMSEList, np.array([mseTmp])))
            MSE = np.array([np.mean(obMSEList)])
            print('OBJ MSE = {}'.format(MSE))
            return MSE
        else:
            # raise RuntimeError('Metric SSIM: Open video failed!')
            print('Metric OBJ MSE: Open video failed!')
            return np.array([65025])

    def cal_object_mse_woSegGPT(self, video_, srcFrameNum, refVideo):
        obMSEList = np.array([])
        if video_.isOpened() and refVideo.isOpened():
            for frame in range(srcFrameNum):
                retVid, imgVid = video_.read()  
                if imgVid is None:
                    continue
                retRef, imgRef = refVideo.read() 
                if imgRef is None:
                    continue
                if not retRef: break  

                maskRef = imgRef.copy() 
                maskRef[np.sum(maskRef, axis=2) <= 10] = np.array([0, 0, 0]) 
                maskRef[np.sum(maskRef, axis=2) > 10] = np.array([1, 1, 1])

                imgRefTmp = np.sum(maskRef, axis=2)
                num = 320*240*3 - np.sum(imgRefTmp == 0)*3  
                imgVid = imgVid*maskRef 
                imgRef = imgRef*maskRef

                mseTmp = np.sum((imgVid - imgRef) ** 2) / num  

                obMSEList = np.concatenate((obMSEList, np.array([mseTmp])))
            MSE = np.array([np.mean(obMSEList)])
            print('OBJ MSE = {}'.format(MSE))
            return MSE
        else:
            # raise RuntimeError('Metric SSIM: Open video failed!')
            print('Metric OBJ MSE: Open video failed!')
            return np.array([65025])


if __name__ == "__main__":

    metrics = Metrics()
    srcCapture = cv.VideoCapture('ob.avi')
    dstCapture = cv.VideoCapture('merge.avi')
    srcVideoPath = 'ob.avi'

    a = np.array([])
    srcFrameNum = int(os.popen(
        f'ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 {srcVideoPath}').readlines()[
                          0][:-1])

    mse = metrics.cal_object_mse_woSegGPT(dstCapture, srcFrameNum, srcCapture)
    dstCapture.release()
    srcCapture.release()
    print(mse)


