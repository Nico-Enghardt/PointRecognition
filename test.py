import numpy as np
import cv2
import skvideo.io

webcam = cv2.VideoCapture(0)

images = []

i = 0

while i<300:

    istrue, frame = webcam.read()
    images.append(frame)

    i = i+1
    print(i)


images = np.array(images)

np.save("Videos/TestHand.npy",images)

skvideo.io.vwrite("Videos/TestHand.mp4",images)
