import os
import numpy as np
import skvideo.io
import random

def readDatasetTraining(path,shuffling,onlyFile=False):

    if not onlyFile: files = os.listdir(path);
    else: 
        files = [path]
        
    pictures = np.empty((1,38400))
    labels = np.empty((1,3))

    for file in files:

        if not onlyFile: pathToFile = path+"/"+file
        else: pathToFile = files[0]

        format = file[-3:]

        if format=="mp4":
            pictures = np.concatenate((pictures,loadVideo(pathToFile)))
        
        if format=="npy":
            labels = np.concatenate((labels,loadNumpy(pathToFile)))


    #if shuffling: random.shuffle(pictures)
    
    return pictures[1:,:], labels[1:,:]  # Delete first row (random inintialisation of np.empty), # Convert both arrays to numpy format

def loadVideo(pathToFile,flatten=True):
    pictures = []
    video = skvideo.io.vread(pathToFile)
    for frame in video:
        frame = frame[:,:,:2]
        if flatten: frame = frame.flatten()
        pictures.append(frame)
        
    print("Number of frames:" +  str(len(pictures)))
    
    return np.array(pictures,dtype="float32")
    
def loadNumpy(pathToFile):
    
    labels = np.empty((1,3))
    coordinates = np.load(pathToFile)
    zs = coordinates[:,2]
    coordinates[:,2] = zs - np.min(zs)
    return np.array(coordinates,dtype="float32")
    
def readDatasetSize(path):
    files = os.listdir(path);

    labels = np.empty((1,3))

    for file in files:
    
        format = file[-3:]
        
        if format=="npy":
            labels = np.concatenate((labels,np.load(path+"/"+file)))

    return labels.shape[0]