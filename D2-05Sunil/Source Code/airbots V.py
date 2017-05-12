import cv2
import numpy as np
import matplotlib.pyplot as plt 

cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.imwrite('F:\day2\demo\\'+ str(i+8000)+'.jpg',gray)
    i = i+1
    plt.hist(gray.ravel(),256,[0,256]);
    plt.savefig('F:\day2\demo\\'+ str(i)+'.jpg')
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
