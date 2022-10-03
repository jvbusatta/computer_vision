from typing import Any
import cv2
from matplotlib.scale import scale_factory #OpenCV

image = cv2.imread('D:\PYTHON\AULAS\curso_python_a_z\computer_vision\computer_vision\imagem_01.png', cv2.IMREAD_COLOR)

cv2.imshow('image', image)
cv2.waitKey(0)
print(image.shape)

face_detector = cv2.CascadeClassifier('D:\PYTHON\AULAS\curso_python_a_z\computer_vision\computer_vision\haarcascade_frontalface_default.xml')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('image_gray', image_gray)
cv2.waitKey(0)

detector = face_detector.detectMultiScale(image_gray, scaleFactor=1.3, minSize=(30,30))
print(detector)

#show number of faces detected
print('Number of faces detected: ', len(detector))


for (x,y,l,a) in detector:

    #print(x,y,l,a)
    cv2.rectangle(image, (x,y), (x + l, y + a), (0,255,0), 2)

cv2.imshow('image_detector',image)

cv2.waitKey(0)

#----------------------
##detecting people body
#----------------------

body_image = cv2.imread('D:\PYTHON\AULAS\curso_python_a_z\computer_vision\computer_vision\pessoas.jpg', cv2.IMREAD_COLOR)

cv2.imshow('body_image', body_image)
cv2.waitKey(0)
print(body_image.shape)

body_detector = cv2.CascadeClassifier('D:/PYTHON/AULAS/curso_python_a_z/computer_vision/computer_vision/fullbody.xml')
body_image_gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('body_image_gray', body_image_gray)
cv2.waitKey(0)

detector_body = body_detector.detectMultiScale(body_image_gray, scaleFactor=1.1, minSize=(50,50))
print(len(detector_body))

#show number of bodies detected
print('Numbers of bodies detected: ', len(detector_body))

for (x,y,l,a) in detector_body:

    cv2.rectangle(body_image, (x,y), (x+l, y+a), (0,255,0), 2)

cv2.imshow('body_image_detector', body_image)
cv2.waitKey(0)

