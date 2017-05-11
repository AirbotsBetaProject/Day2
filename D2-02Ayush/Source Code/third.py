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

    # detecting profile face using haarcascade
    faces1 = face_cascade1.detectMultiScale(framegray, 1.3, 5)
    for (x, y, w, h) in faces1:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
    if len(faces1) == 0:  # if not detected, try once by flipping the image
        frame2 = cv2.flip(framegray, 1)
        faces1 = face_cascade1.detectMultiScale(frame2, 1.3, 5)
        # draw a shifted rectangle for original frame
        for (x, y, w, h) in faces1:
            cv2.rectangle(
                frame, (x+(frame.shape[0])/2, y), (x+w+(frame.shape[0])/2, y+h), (255, 0, 0), 1)

    # detect profile faces using lbpcascade
    faces2 = face_cascade2.detectMultiScale(framegray, 1.3, 5)
    for (x, y, w, h) in faces2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
    if len(faces2) == 0:
        frame2 = cv2.flip(framegray, 1)
        faces2 = face_cascade2.detectMultiScale(frame2, 1.3, 5)
        for (x, y, w, h) in faces2:
            cv2.rectangle(
                frame, (x+(frame.shape[0])/2, y), (x+w+(frame.shape[0])/2, y+h), (255, 0, 0), 1)

    # write images into output folder
    cv2.imwrite('Demo4/'+str(pic_num)+'.jpg', frame)
    pic_num += 1

cap.release()
