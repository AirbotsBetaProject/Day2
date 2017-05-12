import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
proface_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    proface = proface_cascade.detectMultiScale(gray, 1.3, 5)
    for (a,b,c,d) in proface:
        cv2.rectangle (frame, (a,b), (a+c,b+d), (250,250,250), 3)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (250,0,0), 2)
        roi_gray = gray [y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            
    cv2.imshow('frame', frame)
    '''cv2.imwrite('F:\day2\demo\\'+ str(i+8000)+'.jpg',gray)
    i = i+1
    plt.hist(gray.ravel(),256,[0,256]);
    plt.savefig('F:\day2\demo\\'+ str(i)+'.jpg')'''
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


