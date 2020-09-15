from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from cv2 import cv2
import os
import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

args = {
    'dataset': 'dataset',
    'model': 'mask_detector.model',
    'proto': 'deploy.prototxt.txt',
    'modelpath': 'res10_300x300_ssd_iter_140000.caffemodel'
}

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args['proto'], args['modelpath'])
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
args['dataset'] = 'with_mask203.jpg'
image = cv2.imread(args['dataset'])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(
    image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

model = load_model(args['model'])
# passing blob through the network to detect and pridiction
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence and prediction
    confidence = detections[0, 0, i, 2]

    # filter detections by confidence greater than the minimum     #confidence
    print(confidence)
    if confidence > 0.75:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        face = image[startY:endY, startX:endX]
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
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 1)
        

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
