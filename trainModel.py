import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import tensorflow as tf
import numpy as np
import split
import math
import cv2
import skvideo.io
import sys

run = wandb.init(job_type="model-training", config={"epochs":2000,"learning_rate":0.0000003})

modelName = "Sparta"
trainingsetName = "Hsv640"
splitSettings = (.7,0,.3)

# Load Model --------------------------------------------------------------------------------------------

modelArtifact = run.use_artifact(modelName+":latest")
model_directory = modelArtifact.download()

model = tf.keras.models.load_model(model_directory)

# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetArtifact = run.use_artifact(trainingsetName+":latest")

datasetFolder = "Datasets/"+trainingsetName

files = os.listdir(datasetFolder)

pictures = []
labels = np.empty((1,3))

for file in files:

    format = file[-3:]

    if format=="mp4":
        video = skvideo.io.vread(datasetFolder+"/"+file)
        for frame in video:
            pictures.append(frame)
        print(len(pictures))
    
    if format=="npy":
        labels = np.concatenate((labels,np.load(datasetFolder+"/"+file)))

labels = labels[1:,:]  # Delete first row (random inintialisation of np.empty)

# Trenne inputs von outputs ab (Preparation for training)
trainingInputs,trainingLabels = split.splitDataset(pictures,labels,splitSettings,mode="training")

# Fit model to training data --------------------------------------------------------------------------------------------
e = 0
while e < run.config["epochs"]:
    model.fit(trainingInputs,trainingLabels,verbose=0)
    wandb.log({"acc":model.history.history["mean_squared_error"][0]})
    e += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

model.summary()

# Test model's predictions
predictions = model.predict(trainingInputs)
print("\n Predictions:")
print(predictions[:10,:])
print("\n")

# Prediction Quality on Training Set

testInputs,testLabels = split.splitLabels(split.splitDataset(data,splitSettings ,mode="testing"))
prediction = model(testInputs,testLabels)

mse = tf.keras.losses.MeanSquaredError()
quality = math.sqrt(mse(prediction,testLabels))

print(f"\n The networt predicts with {quality} pixels of accuracy.")

# Save Model online and finish run –------------------------------------------------------------------------------------

model.save("artifacts/"+modelName)

modelArtifact = wandb.Artifact(modelName,type="model")
modelArtifact.metadata = {"compressionType":"gray8045","lastQuality":quality}
modelArtifact.add_dir("Models/"+modelName)

run.log_artifact(modelArtifact)

wandb.finish()

