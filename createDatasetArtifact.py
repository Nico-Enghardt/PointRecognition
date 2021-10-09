import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np

run = wandb.init(project="PointRecognition",job_type="dataset-creation")

datasetName = "Hsv640"
compressor = "hsv"
shape = (480,640,3)

dataset = wandb.Artifact(datasetName,type="dataset")

for file in os.listdir("Datasets/" + datasetName):
    print(file)
    dataset.add_file("Datasets/" + datasetName + "/"+file)

dataset.metadata = {"compressor":compressor,"shape":shape}

run.log_artifact(dataset)
