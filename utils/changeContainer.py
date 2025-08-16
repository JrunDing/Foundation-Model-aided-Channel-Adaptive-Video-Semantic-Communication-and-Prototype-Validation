"""
@Brief: Change video container.
"""

import os


def mp4_2_avi():
    parentDirectory = os.listdir("../data/UCF101/DEBUG/")
    videoPath = []  
    for k in parentDirectory: videoPath.append(os.listdir(f"../data/UCF101/DEBUG/{k}"))

    for i in range(len(parentDirectory)):
        for j in range(len(videoPath[i])):

            srcVideoPath = f"../data_lmavc/UCF101/DEBUG/{parentDirectory[i]}/{videoPath[i][j]}"  
            print(srcVideoPath)
            os.system(f'ffmpeg -i {srcVideoPath} -vcodec copy -acodec copy {srcVideoPath[:-3]}avi')

            os.remove(srcVideoPath)


def avi_2_mp4():
    parentDirectory = os.listdir("../data/UCF101/DEBUG/")
    videoPath = []  
    for k in parentDirectory: videoPath.append(os.listdir(f"../data/UCF101/DEBUG/{k}"))

    for i in range(len(parentDirectory)):
        for j in range(len(videoPath[i])):
            srcVideoPath = f"../data_lmavc/UCF101/DEBUG/{parentDirectory[i]}/{videoPath[i][j]}" 
            print(srcVideoPath)
            os.system(f'ffmpeg -i {srcVideoPath} -vcodec copy -acodec copy {srcVideoPath[:-3]}mp4')
            os.remove(srcVideoPath)


if __name__ == "__main__":

    avi_2_mp4()
