import os
import cv2
import time
import numpy as np
import findDot
import distance
import compressors
import skvideo.io

compressor = "hsv160"
datasetName = "Hsv160"

lastCenterPoint,lastHeightCenter = findDot.adjustCenterPoint()


top = cv2.VideoCapture(2)
bottom = cv2.VideoCapture(0)

startTime = time.time()
dauer = 30

frames = []
centerPoints = []

while time.time() - startTime < dauer:
    
    # Read from Camera, find Blue CenterPoint
    istrue,topFrame = top.read()
    istrue,bottomFrame = bottom.read()
    if not istrue:
        break

    editedFrame,center = findDot.findDotCenter(topFrame, testing=False)
    rubbish, heightCenter = findDot.findDotCenter(bottomFrame,testing=True,lower=True)
    editedFrame = cv2.circle(editedFrame,lastCenterPoint,5,(0,200,100),-1)
    bottomFrame = cv2.circle(bottomFrame,lastHeightCenter,5,(0,200,100),-1)

    cv2.imshow("editedFrame",editedFrame)
    cv2.imshow("bottomFrame",cv2.resize(bottomFrame,(320,240)))
    
    print(time.time()-startTime,center,heightCenter,distance.distance(lastCenterPoint,center),distance.distance(lastHeightCenter,heightCenter))

    if distance.distance(lastCenterPoint,center) > 60 or distance.distance(lastHeightCenter,heightCenter)> 40 or center[0] == -100:
        print("Invalid position calculated.\n")
        cv2.waitKey(1)
        continue

    # Add Frame without Point and found Centerpoint to their arrays
    frames.append(editedFrame)
    centerPoints.append([center[0],center[1],heightCenter[1]])

    lastCenterPoint = center
    lastHeightCenter = heightCenter
    cv2.waitKey(1)

frames = compressors.compressVideo(frames,compressor=compressor)

# Some Report
print(str(frames.shape[0]) + " Pictures taken \n")
print(frames.shape)

fileName = 'Datasets/'+datasetName+"/"+time.strftime("%m-%d--%H:%M.%S",time.localtime()) + "-"+str(frames.shape[0])

skvideo.io.vwrite(fileName+".mp4",frames)
np.save(fileName+".npy",centerPoints)

cv2.destroyAllWindows()