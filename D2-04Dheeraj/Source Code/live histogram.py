import numpy as np
import cv2
import matplotlib.pyplot as plt
i=0

f_c=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
e_c=cv2.CascadeClassifier('haarcascade_eye.xml')
m_c=cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fa=f_c.detectMultiScale(gray,1.3,5)
    for(z,x,c,v) in fa:
        cv2.rectangle(frame,(z,x),(z+c,x+v),(0,255,0),3)
        roi_g=gray[x:x+v,z:z+c]
        roi_c=frame[x:x+v,z:z+c]
        eyes=e_c.detectMultiScale(roi_g)
        mouth=m_c.detectMultiScale(roi_g)
        for(qz,qx,qc,qv) in eyes:
            cv2.rectangle(roi_c,(qz,qx),(qz+qc,qx+qv),(255,0,1),2)
        for(wz,wx,wc,wv) in mouth:
            cv2.rectangle(roi_c,(wz,wx),(wz+wc,wx+wv),(125,0,0),2)
    cv2.imshow('frame',frame)
    cv2.imwrite('G:\softies\python\day2\demo\\'+'FRAME'+str(i)+'.jpg',frame)
    plt.hist(gray.ravel(),32,[0,256])
    plt.savefig('G:\softies\python\day2\demo\\'+str(i)+'.jpg')
    plt.gcf().clear()
    i=i+1
    if cv2.waitKey(50)& 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
