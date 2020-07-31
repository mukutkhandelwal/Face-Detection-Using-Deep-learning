# Face-Detection-Using-Deep-learning

An End To End deep learning face detection  application with openCV with pre-trained deeplearning face detector present in the library which is present in the Deep neural network(DNN) module of openCV.

Using the  dnn module with the Caffe models there are 2 files needed :
1. .prototxt file which contains the model architecture
2. .caffemodel which contains the weights for thw model 

These can be downloaded From [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

### Here are some sample images from video_detection File

![elon musk](/image/elonmusk.jpg)

![face masl](/image/mask.jpg)

### How to run

for windows

for face detection in image

python face_detection.py --image [path image name] --prototxt [prototxt file] --model [caffe model file]

for face detection in video

python face_detection.py --prototxt [prototxt file] --model [caffe model file]

### Credits

This file was made with the help from the [blog](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
