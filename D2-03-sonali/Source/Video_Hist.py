import cv2


import numpy as np


from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)  # capture video
i = 0
while True:
    ret, frame = cap.read()  # capture frame fromvideo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# convert it into grey scale
    cv2.imshow('gray', gray)  # show  the frames in greay scale
    cv2.imwrite('C:\Users\Sonali Singh\Desktop\Airbots\day2\D2-03Sonali\Source/'+str(i+1000)+'.jpg', gray)
    # store the frames from a video in given file location
    plt.hist(gray.ravel(), 16, [0, 256])  # plot the histogram
    plt.savefig('C:\Users\Sonali Singh\Desktop\Airbots\day2\D2-03Sonali\Source/'+str(i)+'.jpg')
    # save the histogram for frames in given file location

    plt.gcf().clear()# clean the histogram plot
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
