# required libraries

import numpy as np
import argparse
import cv2

# creating the command line argument for passing the image,model,weights of model and optional Confidence
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help = 'path of img')
ap.add_argument('-p','--prototxt',required=True,help = 'path to prototxt file ')
ap.add_argument('-m','--model',required=True,help='path to model')
ap.add_argument('-c','--confidence',type = float,default= 0.5,help='min filter')
args = vars(ap.parse_args())

# loading the model and weights
print("[info] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])


# reading the image
image = cv2.imread(args['image'])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0))


# passing the blob to the  neural net and getting the detection
print("[info] detecting faces")
net.setInput(blob)
detection = net.forward()

for i in range(0,detection.shape[2]):
    # taking confidence/probablity from the detected area
    confidence = detection[0,0,i,2] 
    # removing the weak confidence by ensuring minimum confidence reaches
    if confidence> args['confidence']:
        # compute the x and y coordinates 
        box = detection[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype('int')

        # drawing the box along with the face
        text = '{:.2f}%'.format(confidence*100)
        y = startY-10 if startY -10 > 10 else startY + 10
        cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

# Displaying the Detected image
cv2.imshow("output",image)
cv2.waitKey(0)



