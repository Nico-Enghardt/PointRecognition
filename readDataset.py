import os
import numpy as np
import skvideo
skvideo.setFFmpegPath("C:/Programme/ffmpeg/bin")
import skvideo.io
import random
import time

def readDatasetTraining(path,shuffleMode="shuffleBatches",percentageDataset=1,onlyFile=False):
    
    trainingPictures,trainingLabels = readFromFolder(path + "/Training")
    testPictures,testLabels = readFromFolder(path + "/Testing")
    
    seed = time.time()  # Set a random seed
    
    if shuffleMode == "shuffleDataset":
        pictures = np.concatenate((trainingPictures,testPictures))
        labels = np.concatenate((trainingLabels,testLabels))

        random.Random(seed).shuffle(pictures)  # Shuffle according to seed
        random.Random(seed).shuffle(labels)
        
        split = int(0.8 * len(pictures))
        
        trainingPictures, testPictures = pictures[:split,:],pictures[split:,:]
        trainingLabels, testLabels = labels[:split,:],labels[split:,:]
     
        
    elif shuffleMode == "shuffleBatches":
        random.Random(seed).shuffle(trainingPictures)
        random.Random(seed).shuffle(trainingLabels)
        
    splitPercentage = int(len(trainingPictures)*percentageDataset)

    trainingPictures,trainingLabels = pictures[:splitPercentage,:], labels[:splitPercentage,:]

    return trainingPictures,trainingLabels,testPictures,testLabels

def readFromFolder(path):

    files = os.listdir(path); 
        
    pictures = np.empty((1,38400))
    labels = np.empty((1,3))

    for file in files:

        pathToFile = path+"/"+file

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