from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from cv2 import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from cv2 import cv2
import os
import argparse

ap = argparse.ArgumentParser()

args = {
	'dataset': 'dataset',
	'model': 'mask_detector.model',
	'proto': 'deploy.prototxt.txt',
	'modelpath': 'res10_300x300_ssd_iter_140000.caffemodel'
}

#load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args['proto'], args['modelpath'])
model = load_model(args['model'])
print("[INFO] starting video stream...")

#READ VIDEO
vs = cv2.VideoCapture('video3.mp4')
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

# Define the codec and filename.
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('video_out.avi',fourcc,10,(frame_width,frame_height))

while True:

	#RESIZE AND GRAB DIMENSIONS
	ret,frame = vs.read()
	frame = imutils.resize(frame,width=400)
	(h,w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()


	#DETECT FACE AND MASK OR NO MASK
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < 0.5:
			continue
		box = detections[0,0,i,3:7]*np.array([w,h,w,h])
		(startX,startY,endX,endY)=box.astype("int")

		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		# extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
		face = frame[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pass the face through the model to determine if the face
		# has a mask or not
		(mask, withoutMask) = model.predict(face)[0]

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 0, 255) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}".format(label)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
		cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
	
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# lower_green = np.array([40, 60, 60])
	# upper_green = np.array([80, 255, 255])
	# mask = cv2.inRange(hsv, lower_green, upper_green)
	# mask = cv2.GaussianBlur(mask, (7, 7), 0)
	# res = cv2.bitwise_and(frame, frame, mask=mask)
	out.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
vs.release()
out.release()
cv2.destroyAllWindows()
#vs.stop()
