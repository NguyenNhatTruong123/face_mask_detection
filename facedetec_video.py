from cv2 import cv2
from face_recognition_models import dlibrary
cap=cv2.VideoCapture('video.mp4')
face=[]

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB   
    # color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face=dlibrary()
    for top, right, bottom, left in face:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,  
        255), 2)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break