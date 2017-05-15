import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        mouth=mouth_cascade.detectMultiScale(roi_gray)
        for(m_x,m_y,m_w,m_h)in mouth:
            cv2.rectangle(roi_color,(m_x,m_y),(m_x+m_w,m_y+m_h),(0,0,255),2)
    cv2.imshow('img',img)
    cv2.imwrite('C:\Users\ravindra\Desktop\intern\day2-ravi\output'+'FRAME'+str(i)+'.jpg',frame)
    plt.hist(gray.ravel(),32,[0,256])
    plt.savefig('C:\Users\ravindra\Desktop\intern\day2-ravi\output'+str(i)+'.jpg')
    plt.gcf().clear()
    i=i+1
    if cv2.waitKey(30) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
