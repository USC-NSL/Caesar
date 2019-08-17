import numpy as np
from time import time, sleep 
import cv2


SECOND_VIDEO = True 
FPS = 10
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480

fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap0 = cv2.VideoCapture(0)
out0 = cv2.VideoWriter('v1.avi',fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

if SECOND_VIDEO:
    cap1 = cv2.VideoCapture(1)
    out1 = cv2.VideoWriter('v2.avi',fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))


cur = time()
period = 1. / FPS

while(True):
    # Capture frame-by-frame
    ret, frame = cap0.read()
    if not ret:
        print("video0 ended")
        break 
    
    cv2.imshow('frame0', frame)
    out0.write(frame)

    if not SECOND_VIDEO:
        continue 

    ret, frame = cap1.read()
    if not ret:
        print("video1 ended")
        break 
    out1.write(frame)

    cv2.imshow('frame1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(max(period - (time() - cur), 0))
    cur = time()

# When everything done, release the capture
cap0.release()
cap1.release()

cv2.destroyAllWindows()

print('dones')