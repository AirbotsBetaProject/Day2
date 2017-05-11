import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)  # open camera
pic_num = 1
# read necessary cascade files
face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
face_cascade1 = cv2.CascadeClassifier(
    'haarcascades/haarcascade_profileface.xml')
face_cascade2 = cv2.CascadeClassifier(
    'haarcascades/lbpcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
eye_cascade1 = cv2.CascadeClassifier(
    'haarcascades/haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')
for i in range(0, 100):  # capture 100 frames
    ret, frame = cap.read()
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect objects using cascade files, get an output vector of rectangles
    # indicating object coordinates in the images
    faces = face_cascade.detectMultiScale(framegray, 1.3, 5)
    for (x, y, w, h) in faces:  # for all the detected objects
        # draw a blue-coloured rectangle with thickness 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = framegray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # detect eyes on the face using different cascades
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        eyes1 = eye_cascade1.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes1:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        mouth = mouth_cascade.detectMultiScale(roi_gray)  # detect mouth
        for (ex, ey, ew, eh) in mouth:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)

    # detect facial objects in profile face using haarcascades
    faces1 = face_cascade1.detectMultiScale(framegray, 1.3, 5)
    for (x, y, w, h) in faces1:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = framegray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        eyes1 = eye_cascade1.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes1:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in mouth:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)

    # detect facial objects in profile face using lbpcascades
    faces2 = face_cascade2.detectMultiScale(framegray, 1.3, 5)
    for (x, y, w, h) in faces2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = framegray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        eyes1 = eye_cascade1.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes1:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in mouth:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)

    # write images into output folder
    cv2.imwrite('Demo3/'+str(pic_num)+'.jpg', frame)
    pic_num += 1

cap.release()
