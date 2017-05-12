import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.imwrite(str(i+8000)+'.jpg',gray)
    plt.hist(gray.ravel(),256,[0,256]);
    plt.savefig(str(i)+'.jpg')
    plt.gcf().clear()
    i = i+1
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
