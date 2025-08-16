
import os


def rmAudio():
    parentDirectory = os.listdir("../data/UCF101/UCF-101-VAL/")
    videoPath = []  
    for k in parentDirectory: videoPath.append(os.listdir(f"UCF101/UCF101-VAL/{k}"))

    for i in range(len(parentDirectory)):
        for j in range(len(videoPath[i])):

            srcVideoPath = f"UCF101/UCF101-VAL/{parentDirectory[i]}/{videoPath[i][j]}"  
            print(srcVideoPath)
            os.system(f'ffmpeg -i {srcVideoPath} -c:v copy -an {srcVideoPath[:-3]}mp4')
            os.remove(srcVideoPath)


if __name__ == "__main__":
    rmAudio()
