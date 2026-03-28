"""
@Brief: Use SegGPT to make image dataset from source video dataset. You need to git clone the SegGPT project and learn
how to use it first. You can also use other models to implement segmentation.
"""

import os
import argparse

import torch
import numpy as np
import cv2 as cv

from seggpt_engine import inference_image, inference_video
import models_seggpt

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


if __name__ == '__main__':

    path = "data/UCF101-TRA-S/"
    prompt_image = ["refs/src_2.jpg"] 
    prompt_target = ["refs/mask_2.jpg"]
    output_dir = "data/temp/"  

    device = torch.device("cuda")
    model = prepare_model('checkpoint/seggpt_vit_large.pth', 'seggpt_vit_large_patch16_input896x448', 'instance').to(device)
    print('Model loaded.')

    dataset_num = 0
    parentDirectory = os.listdir(path) 
    videoPath = []  
    cnt1 = 0 
    cnt2 = 0  
    count_video = 0
    for k in parentDirectory: videoPath.append(os.listdir(f"data/UCF101-TRA-S/{k}"))
    for i in range(len(parentDirectory)):
        for j in range(len(videoPath[i])):
            input_video = f"data/UCF101-TRA-S/{parentDirectory[i]}/{videoPath[i][j]}" 

            vid_name = os.path.basename(input_video)
            object_out_path = os.path.join(output_dir, "object" + '.avi')
            background_out_path = os.path.join(output_dir, "background" + '.avi')

            inference_video(model, device, input_video, 0, prompt_image, prompt_target, object_out_path, background_out_path)
            print('Finish Seg.')

            cap = cv.VideoCapture(object_out_path)  
            cnt_tmp1 = 0
            if cap.isOpened():
                while cnt_tmp1 < 10:
                    _, frame = cap.read()
                    if not _: break  
                    if cnt_tmp1 % 2 == 0:
                        os.mkdir(f"data/object/{int(cnt1 / 2 + 1)}")
                        cv.imwrite(f'data/object/{int(cnt1 / 2 + 1)}/1.png', frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        cv.imwrite(f'data/object/{int((cnt1 - 1) / 2 + 1)}/2.png', frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
                    cnt1 += 1
                    cnt_tmp1 += 1
                cap.release()
            else:
                print('Open video failed!')

            cap = cv.VideoCapture(background_out_path) 
            cnt_tmp2 = 0
            if cap.isOpened():
                while cnt_tmp2 < 10:
                    _, frame = cap.read()  
                    if not _: break  
                    if cnt_tmp2 % 2 == 0:
                        os.mkdir(f"data/background/{int(cnt2 / 2 + 1)}")
                        cv.imwrite(f'data/background/{int(cnt2 / 2 + 1)}/1.png', frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        cv.imwrite(f'data/background/{int((cnt2 - 1) / 2 + 1)}/2.png', frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
                    cnt2 += 1
                    cnt_tmp2 += 1
                cap.release()
            else:
                print('Open video failed!')

            count_video += 1
            print("Already segment video number:", count_video)

