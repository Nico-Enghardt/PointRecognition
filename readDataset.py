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
            labels = np.concatenate((labels,np.load(path+"/"+file)))

            ##if local:
            ##    break;

    random.shuffle(pictures)
    
    return np.array(pictures), labels[1:,:]  # Delete first row (random inintialisation of np.empty), # Convert both arrays to numpy format