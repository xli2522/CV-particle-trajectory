import cv2
import numpy as np

"""
    June 2019
    by Xiyuan Li
"""

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    original = frame

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(frame, 100, 150)

    ret,inverted = cv2.threshold(canny,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    #result = cv2.addWeighted(original, 0.7, inverted, 0.3, 0)
    cv2.imshow('result', inverted)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


