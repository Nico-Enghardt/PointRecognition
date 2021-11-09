import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import tensorflow as tf
import numpy as np
import cv2
from readDataset import *
import loss
import createModelArtifact as createModel
import time

local = False
if platform.node()=="kubuntu20nico2":
    local = True


modelName = None
datasetName = "Huegray160"
architecture = (4000,1000,100)
max_epochs = 750
batch_size = 2800
regularization_factor =  0.5
learning_rate = 0.000001
shuffling = True;

run = wandb.init(job_type="model-training", config={"epochs":0,"learning_rate":learning_rate,"batch-size":batch_size,"regularization":regularization_factor,"architecture":architecture,"shuffling":shuffling})

# Define DatasetArtifact

datasetArtifact = run.use_artifact(datasetName+":latest")

# Load Model --------------------------------------------------------------------------------------------

if modelName:

    modelArtifact = run.use_artifact(modelName+":latest")
    model_directory = modelArtifact.download()

    model = tf.keras.models.load_model(model_directory,custom_objects={"loss3D":loss.loss3D,"heightError":loss.heightError,"planeError":loss.planeError})

else:
    model, imageShape = createModel.createModel(architecture,datasetArtifact,run.config["learning_rate"],regularization_factor)

# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetFolder = "Datasets/"+datasetName
if not local:
    datasetFolder = datasetArtifact.download()

trainingPictures,trainingLabels = readDataset(datasetFolder+"/Training")
testPictures, testLabels = readDataset(datasetFolder+"/Testing")


pictures = np.concatenate((trainingPictures,testPictures))
labels = np.concatenate((trainingLabels,testLabels))

seed = time.time()  # Set a random seed
print(random.Random(seed).randint(1,10))
random.Random(seed).shuffle(pictures)  # Shuffle according to seed
random.Random(seed).shuffle(labels)
print(random.Random(seed).randint(1,10))
split = int(0.8*len(pictures))

trainingPictures,testPictures = pictures[:split,:],pictures[split:,:]
trainingLabels,testLabels = labels[:split,:],labels[split:,:]

# Fit model to training data --------------------------------------------------------------------------------------------

e = 0

while e < max_epochs:
    print("Epoch: "+ str(e))
    #model.fit(x=trainingPictures,y=trainingLabels,batch_size=batch_size,verbose=1)

    i = 0
    metrics = []

    while i < len(trainingPictures):
        range = [i,i+batch_size]
        
        if i+batch_size > len(trainingPictures):
            range[1]=len(trainingPictures)
        currPictures = np.array(trainingPictures[range[0]:range[1]])
        currLabels = np.array(trainingLabels[range[0]:range[1]])

        i=i+batch_size
        model.fit(x=currPictures,y=currLabels,verbose=1)
        
        currMetrics = model.history.history

        metrics.append([currMetrics["loss"][0],currMetrics["loss3D"][0],currMetrics["heightError"][0],currMetrics["planeError"][0]])

    #acc = model.history.history["mean_squared_error"][0]
    metrics = np.average(metrics,axis=0)
    wandb.log({"loss":metrics[0],"acc3D":metrics[1],"heightError":metrics[2],"planeError":metrics[3]})

    if (e % 5 == 0):
        metrics = model.evaluate(x=testPictures,y=testLabels,batch_size=batch_size,verbose=2)
        wandb.log({"testLoss":metrics[0],"testAcc3D":metrics[1],"testHeightError":metrics[2],"testPlaneError":metrics[3]},commit=False)
        
    e = e+1
    


model.summary()

# Test model's predictions
predictions = model.predict(trainingPictures)
print("\n Predictions:")
print(predictions[:10,:])
print("\n")

# Save Model online and finish run –------------------------------------------------------------------------------------

model.save("artifacts/"+modelName)

modelArtifact = wandb.Artifact(modelName,type="model")
modelArtifact.add_dir("artifacts/"+modelName)

run.log_artifact(modelArtifact)

wandb.finish()

