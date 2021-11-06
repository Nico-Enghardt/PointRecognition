import os
import numpy as np
import skvideo.io
import random

def readDataset(path):

    files = os.listdir(path);

    pictures = []
    labels = np.empty((1,3))

    for file in files:

        format = file[-3:]

        if format=="mp4":
            video = skvideo.io.vread(path+"/"+file)
            for frame in video:
                frame = frame[:,:,:2]
                pictures.append(frame.flatten())
            print(len(pictures))
        
        if format=="npy":
            coordinates = np.load(path+"/"+file)
            zs = coordinates[:,2]
            coordinates[:,2] = zs - np.min(zs)
            labels = np.concatenate((labels,coordinates))


    random.shuffle(pictures)
    
    return np.array(pictures), labels[1:,:]  # Delete first row (random inintialisation of np.empty), # Convert both arrays to numpy format

def readDatasetSize(path):
    files = os.listdir(path);

    labels = np.empty((1,3))

    for file in files:
    
        format = file[-3:]
        
        if format=="npy":
            labels = np.concatenate((labels,np.load(path+"/"+file)))

    return labels.shape[0]