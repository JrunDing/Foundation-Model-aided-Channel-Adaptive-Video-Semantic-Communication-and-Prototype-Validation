# Introduction

​		This repository includes the code and demos of paper "Foundation Model-Aided Channel-Adaptive Video Semantic Communication and Prototype Validation".  

> Abstract: The increasing demand for services such as live streaming and virtual reality places significant pressure on wireless communication systems. Enhancing system performance or reducing bandwidth consumption is critical for delivering high-quality video experiences. Semantic communication, which focuses on the transmission of meaning, offers a promising solution. However, existing approaches are often limited to single scenarios, rely on simple channels, lack adaptability to dynamic wireless environments, and remain untested in practical air interfaces. To address these challenges, we propose a foundation model-aided universal video semantic communication framework designed for pixel-wise reconstruction across diverse scenarios. This framework enables the transmission of entire videos using joint source-channel coding (JSCC) based on optical flow estimation and leverages multiple-input multiple-output orthogonal frequency division multiplexing (MIMO-OFDM) for efficient semantic delivery in 3rd generation partnership project (3GPP) standard channels. In scenarios requiring full transmission for regions of interest and selective transmission for other areas, the framework employs a foundation model for segmentation, followed by JSCC and delivery. Furthermore, we introduce a channel condition number-adaptive semantic remapping method based on an attention mechanism to mitigate the effects of wireless fading. To validate our approach, we implement the framework on a testbed and develop two online demonstrations. Simulations and over-the-air experiments confirm significant improvements in video quality and substantial reductions in bandwidth overhead compared to existing methods.

# Dependencies and file introduction

​		`conda create -n FVSC python==3.10.15` and use `pip install -r requirements.txt` to install dependencies. 

​		[FFmpeg](https://ffmpeg.org/download.html) is required.

​		[SegGPT](https://github.com/baaivision/Painter) is utilized in our work to implement video segmentation. You may also choose to use other more advanced models.

```
data: used to store data
    ---Make_FVSC_Dataset.py: a script used to build ROI/RONI data
demos: two demo videos
FVSC: main model files
    ---subnet: some networks inherited from optical flow estimation
    ---channel.py: load mimo channel
    ---dataset_FVSC.py: dataset class
    ---model_FVSC.py: FVSC model class
    ---model_FVSC_BA/OB/SRM.py: FVSC models for RONI, ROI, and SRM, respectively
    ---model_SRM.py: SRM model
    ---physical_mimo_ofdm.py: MIMO OFDM simulation
    ---predict_FVSC_BA/OB.py：test one frame RONI/ROI
    ---Quantization.py: quantization module
    ---test_FVSC.py: test script for FVSC
    ---test_FVSC_SRM.py: test script for FVSC with SRM
    ---train_FVSC.py: train script for FVSC
    ---train_FVSC_BA/OB/SRM.py: train scripts for RONI/ROI/SRM
FVSC_KeyFrame: code for key frame transmission (image tranmission)
SegGPT_Server: use SSH and FTP to interact with servers
utils：some utilities
metrics.py: python class for performance testing
requirements.txt: software dependencies
```

# Demo introduction

- File introduction: This project contains two demos, including the video semantic communication with and without foundation model SegGPT. You can download these two demo videos from the `demos/` path. The `VSC.mp4` is the demo video of $2\times2$ MIMO-OFDM video semantic communication. The `FVSC.mp4` is  the demo video of $2\times 2$ MIMO-OFDM SegGPT-aided video semantic communication. 

- Equipments: Equipments involved mainly includes two Nvidia Jetson Xavier NX, one NI USRP-2974R, four omni-directional antennas, one router and one Nvidia RTX 4090 server. 
- Methods: These demos are implemented using multithreading technology. **In VSC, the transmitter** consists of three threads. The main thread is responsible for playing the video to be transmitted. Sub-thread 1 continuously reads video frames, encodes them, and places the encoded results into a queue. Sub-thread 2 continuously reads from the queue to send control information and service data. **In FVSC, the transmitter** consists of four threads. The main thread is responsible for playing both the original video to be transmitted and the segmented video. Sub-thread 1 continuously segments the original video, which involves transmitting it to the server, waiting for the server to complete the segmentation, and receiving the segmented video. Sub-thread 2 continuously reads the segmented video frames, encodes them, and places the encoded results into a queue. Sub-thread 3 continuously reads from the queue to send control information and service data. **The receiver** program for both VSC and FVSC is same and consists of three threads. The main thread is responsible for playing the decoded video. Sub-thread 1 continuously receives control and service data and places it into a queue. Sub-thread 2 continuously reads data from the queue and decodes it to obtain the original video frames.

- Advantages: These two demos simply demonstrate the practical feasibility of the video semantic communication paradigm. The system architecture is flexible and easy to deploy algorithms. It allows for near-arbitrary air interface testing of physical layer and upper-layer AI communication algorithms.
- Disadvantages: The system's continuous video playback has an FPS of only 3-4. After troubleshooting and identifying the issue, it was found that the runtime of the convolution neural network is unstable. Sometimes it takes a few milliseconds(FPS is in the tens), while at other times it takes several hundred milliseconds(FPS in the single digits, even less than 1). As shown below(in seconds). Even when directly deployed on a 4090 server, this issue persists. This issue is beyond the scope of this study, but attempts will be made to address it in subsequent studies.

<img src="https://s21.ax1x.com/2025/01/07/pE9HhCD.png" style="zoom:67%;" />