import cv2

import numpy as np

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # load face cascade

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  # load eyes cascade

mouth_cascade = cv2.CascadeClassifier(
    'haarcascade_mcs_mouth.xml')  # load mouth cascade

profileface_cascade = cv2.CascadeClassifier(
    'haarcascade_profileface.xml')  # load profile cascade

cap = cv2.VideoCapture(0)  # capture video
i = 0
while True:
    ret, img = cap.read()  # capture image from video
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting face in rectangular shape
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            gray, 1.3, 5)  # detecting eye greenin
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            mouth = mouth_cascade.detectMultiScale(gray, 1.7, 11)
            for (mx, my, mw, mh) in mouth:
                # detecting mouth in rectangle in black
                cv2.rectangle(img, (mx, my), (mx+mw, my+mh), (0, 0, 0), 1)
                profilefaces = profileface_cascade.detectMultiScale(
                    gray, 1.3, 7)  # detecting profile face
                for (px, py, pw, ph) in profilefaces:
                    # detecting profile in rectangle in red
                    cv2.rectangle(img, (px, py), (px+pw, py+ph), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.imwrite("img" + str(i)+'.jpg', img)  # save image in jpg format
    i += 1
    cv2.waitKey(30)
    k=cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
