# Handpose regognition programmm

![](https://i.imgur.com/gDYoqN2.jpg)

A real-time object recognition application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).
This Repo is based on [datitran](https://github.com/datitran) [Object-Detector-App](https://github.com/datitran/object_detector_app).
We trained our model on different hand poses using the [IBM cloud-annotation traing CLI](https://github.com/cloud-annotations/training)
To trigger system action we used multiple poses.

## Getting Started
1. `conda env create -f environment.yml`
2. `python object_detection_multithreading.py`


## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

