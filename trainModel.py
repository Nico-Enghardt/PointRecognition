import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import tensorflow as tf
import numpy as np
import cv2
from readDataset import *


local = False
if platform.node()=="kubuntu20nico2":
    local = True


modelName = "Arachne"
datasetName = "Huegray160"
epochs = 1000

run = wandb.init(job_type="model-training", config={"epochs":epochs,"learning_rate":0.0000003})

# Load Model --------------------------------------------------------------------------------------------

modelArtifact = run.use_artifact(modelName+":latest")
model_directory = modelArtifact.download()

if local:
    model_directory = "./Models/"+modelName

model = tf.keras.models.load_model(model_directory)

# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetArtifact = run.use_artifact(datasetName+":latest")

datasetFolder = "Datasets/"+datasetName
if not local:
    datasetFolder = datasetArtifact.download()

trainingPictures,trainingLabels = readDataset(datasetFolder+"/Training")
testPictures, testLabels = readDataset(datasetFolder+"/Testing")

# Fit model to training data --------------------------------------------------------------------------------------------

e = 0
batch_size = 1000;

while e < run.config["epochs"]:

    #model.fit(x=trainingPictures,y=trainingLabels,batch_size=batch_size,verbose=1)

    i = 0
    mse = []

    while i < len(trainingPictures):
        range = [i,i+batch_size]
        
        if i+batch_size > len(trainingPictures):
            range[1]=len(trainingPictures)
        currPictures = np.array(trainingPictures[range[0]:range[1]])
        currLabels = np.array(trainingLabels[range[0]:range[1]])

        i=i+batch_size
        model.fit(x=currPictures,y=currLabels,verbose=1)
        mse.append(model.history.history["mean_squared_error"][0])

    #acc = model.history.history["mean_squared_error"][0]
    acc = np.average(mse)

    if (e % 5 == 0):
        testing = model.evaluate(x=testPictures,y=testLabels,batch_size=batch_size,verbose=2)[0]
        wandb.log({"acc":acc,"test":testing})
    else :
        wandb.log({"acc":acc})
    
    e = e+1
    cv2.waitKey(1)
    
    wandb.log({"acc":np.average(mse)})


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

