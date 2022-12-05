
# Towards Asteroid Detection in Microlensing Surveys with Deep Learning 

This repository contains the classification and object detection models used for locating asteroid tracklets in composite difference images as described in [Towards Asteroid Detection in Microlensing Surveys with Deep Learning by Cowan et al.](https://arxiv.org/abs/2211.02239) 

We do not have permission to share the data, however we have shared the trained weights for each of the network architectures. 

The classification networks were trained on a machine running Ubuntu 18.04 with a NVIDIA Quadro M4000 GPU (8 GB, 2.5 TFLOPS). The code was written in
Python 3.6 in the Jupyter Notebook environment. TensorFlow GPU 2.4.1 (CUDA 11.0), along with the Keras deep learning API, was used for creating the CNN
models tested.

The [YOLOv4](https://github.com/AlexeyAB/darknet) model was trained via the Google Colaboratory with a hosted GPU runtime environment. 
