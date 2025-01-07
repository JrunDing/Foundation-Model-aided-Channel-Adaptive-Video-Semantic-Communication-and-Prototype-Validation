# Introduction

This repo includes the code and demos of paper "Foundation Model-Aided Channel-Adaptive Video Semantic Communication and Prototype Validation". The code will be available upon the acceptance of paper.

# Demo introduction

- File introduction: This project contains two demos, including the video semantic communication with and without foundation model SegGPT. You can download these two demo videos from the `demos/` path. The `VSC.mp4` is the demo video of $2\times2$ MIMO-OFDM video semantic communication. The `FVSC.mp4` is  the demo video of $2\times 2$ MIMO-OFDM SegGPT-aided video semantic communication. 

- Equipments: Equipments involved mainly includes two Nvidia Jetson Xavier NX, one NI USRP-2974R, four omni-directional antennas, one router and one Nvidia RTX 4090 server. 
- Ideas for design: It is implemented using multithreading technology. **In VSC, the transmitter **consists of three threads. The main thread is responsible for playing the video to be transmitted. Sub-thread 1 continuously reads video frames, encodes them, and places the encoded results into a queue. Sub-thread 2 continuously reads from the queue to send control information and service data. **In FVSC, the transmitter** consists of four threads. The main thread is responsible for playing both the original video to be transmitted and the segmented video. Sub-thread 1 continuously segments the original video, which involves transmitting it to the server, waiting for the server to complete the segmentation, and receiving the segmented video. Sub-thread 2 continuously reads the segmented video frames, encodes them, and places the encoded results into a queue. Sub-thread 3 continuously reads from the queue to send control information and service data. **The receiver** program for both VSC and FVSC is same and consists of three threads. The main thread is responsible for playing the decoded video. Sub-thread 1 continuously receives control and service data and places it into a queue. Sub-thread 2 continuously reads data from the queue and decodes it to obtain the original video frames.

- Advantages: These two demos simply demonstrate the practical feasibility of the video semantic communication paradigm. The system architecture is flexible and easy to deploy algorithms. It allows for near-arbitrary air interface testing of physical layer and upper-layer AI communication algorithms.
- Disadvantages: The system's continuous video playback has an FPS of only 3-4. After troubleshooting and identifying the issue, it was found that the runtime of the convolution neural network is unstable. Sometimes it takes a few milliseconds(FPS is in the tens), while at other times it takes several hundred milliseconds(FPS in the single digits, even less than 1). As shown below(in seconds). Even when directly deployed on a 4090 server, this issue persists. This issue is beyond the scope of this study, but attempts will be made to address it in subsequent studies.

![](https://s21.ax1x.com/2025/01/07/pE9HB34.png)