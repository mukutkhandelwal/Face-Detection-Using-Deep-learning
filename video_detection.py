# required libraries

import numpy as np
import argparse
import imutils
import time
import cv2

# creating the command line argument for passing the image,model,weights of model and optional Confidence
ap = argparse.ArgumentParser()
# ap.add_argument('-i','--image',required=True,help = 'path of img')
ap.add_argument('-p','--prototxt',required=True,help = 'path to prototxt file ')
ap.add_argument('-m','--model',required=True,help='path to model')
ap.add_argument('-c','--confidence',type = float,default= 0.5,help='min filter')
args = vars(ap.parse_args())

# loading the model and weights
print("[info] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

# assigning the video cam to the vs
print("[info] starting video stream")
vs= cv2.VideoCapture(0)
time.sleep(2)

# looping ove the frame
while True:
    # taking the videostream into frame and resizing it into the frame of 400 X 400
    _,frame = vs.read()
    frame = imutils.resize(frame,400)
    (h,w) = frame.shape[:2]

    # taking the frame size and converting into blob object
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

    # passing the blob to the model and predecting the face
    net.setInput(blob)
    detections = net.forward()

    # making the sequare box and writing the probability of the face detected in the video stream
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence< args['confidence']:
            continue
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype("int")


        text = "{:.2f}%".format(confidence*100)
        y = startY-10 if startY-10>10 else startY +10
        # drawing the rectangle
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)

        # writing the probablity
        cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    # press "q" to quit from the video frame
    if key ==ord("q"):
        break

# releasing the video and destroying the video
vs.release()
cv2.destroyAllWindows()
# vs.()
