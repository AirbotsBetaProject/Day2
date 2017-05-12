import numpy as np
import cv2
#reading cascade files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('eyes.xml')
#opening webcam
cap = cv2.VideoCapture(0)
while 1:
    # Capture frame-by-frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face using cascade files
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        #detecting eye in face domain 
        eye=eye_cascade.detectMultiScale(roi_gray)
        for(eye_x,eye_y,eye_w,eye_h)in eye:
            #drawing rectangle around eyes of thickness 2
            cv2.rectangle(roi_color,(eye_x,eye_y),(eye_x+eye_w,eye_y+eye_h),(0,0,255),2)    
    cv2.imshow('img',img)
    #saving frames
    cv2.imwrite('C:\Users\\ravindra\\Desktop\\intern\\day2-ravi\\output\\'+str(x)+'.jpg',img)
    x=x+1
    #press q to exit
    if cv2.waitKey(30) & 0xFF== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

