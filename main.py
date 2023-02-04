import cv2
import numpy as np

cap = cv2.VideoCapture('test.MP4') # Replace with your video file

orb = cv2.ORB_create()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        resized_frame = cv2.resize(frame, (640, 320)) 

        key_points = orb.detect(resized_frame, None)
        key_points, des = orb.compute(resized_frame, key_points)

        feature_image = cv2.drawKeypoints(resized_frame, key_points, None, (0,0,255), flags = 0)

        cv2.imshow('Frame', resized_frame)
        cv2.imshow('Frame', feature_image)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()


cv2.destroyAllWindows()