import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import tensorflow as tf
import numpy as np
from readDataset import *
import loss
import createModelArtifact

local = False
if platform.node()=="kubuntu20nico2":
    local = True

 
modelName = None
datasetName = "Huegray160"
architecture = (4000,1000,100)
max_epochs = 900
batch_fraction = 1/2
regularization_factor =  0.5
learning_rate = 0.000001
shuffleMode = None;
percentageDataset = 1;

run = wandb.init(job_type="model-training", config={"epochs":0,"learning_rate":learning_rate,"batch_fraction":batch_fraction,"regularization":regularization_factor,"architecture":architecture,"shuffleMode":shuffleMode})

# Define DatasetArtifact

datasetArtifact = run.use_artifact(datasetName+":latest")

# Load Model --------------------------------------------------------------------------------------------

if modelName:

    modelArtifact = run.use_artifact(modelName+":latest")
    model_directory = modelArtifact.download()

    model = tf.keras.models.load_model(model_directory,custom_objects={"loss3D":loss.loss3D,"heightError":loss.heightError,"planeError":loss.planeError})

else:
    model, imageShape = createModelArtifact.createModel(architecture,datasetArtifact,run.config["learning_rate"],regularization_factor)

# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetFolder = "Datasets/"+datasetName
if not local:
    datasetFolder = datasetArtifact.download()

trainingPictures,trainingLabels,testPictures,testLabels = readDatasetTraining(datasetFolder,shuffleMode)

run.config["trainingExamples"] = trainingLabels.shape[0];
run.config["testExamples"] = testLabels.shape[0];

# Fit model to training data --------------------------------------------------------------------------------------------

e = 0
batch_size = int(batch_fraction*trainingPictures.shape[0])
run.config["batch-size"] = batch_size;

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

modelName = "radiantFirework"

model.save("artifacts/"+modelName)

modelArtifact = wandb.Artifact(modelName,type="model")
modelArtifact.add_dir("artifacts/"+modelName)

run.log_artifact(modelArtifact)

wandb.finish()

