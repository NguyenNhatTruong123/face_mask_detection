from cv2 import cv2
import numpy as np

img = cv2.imread('with_mask203.jpg')
face_cascade = cv2.CascadeClassifier('C:/Users/truongnn/work_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
mouth = cv2.CascadeClassifier('C:/Users/truongnn/work_env/Lib/site-packages/cv2/data/haarcascade_smile.xml')

# cv2.imshow('image',img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img_gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    #cv2.putText(img, 'No Mask', (x, y-10), cv2.FONT_ITALIC, 0.8, (0,0,255), 1)
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_color=img[y:y+h, x:x+w]
    is_mouth = mouth.detectMultiScale(roi_gray)
    for (a,b,c,d) in is_mouth:
        cv2.rectangle(roi_color, (a, b), (a+c, b+d), (0, 0, 255), 1)
    # if re:
    #     cv2.putText(img, 'No Mask', (x, y-10), cv2.FONT_ITALIC, 0.8, (0,0,255), 1)
    # else:
    #     cv2.putText(img, 'Masked', (x, y-10), cv2.FONT_ITALIC, 0.8,  (0,0,255), 1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
