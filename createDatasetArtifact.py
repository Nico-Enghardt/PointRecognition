import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np

run = wandb.init(project="PointRecognition",job_type="dataset-creation")

datasetName = "Huegray160"
compressor = "huegray160"
shape = (120,160,2)

dataset = wandb.Artifact(datasetName,type="dataset")

dataset.add_dir("Datasets/" + datasetName)

dataset.metadata = {"compressor":compressor,"shape":shape}

run.log_artifact(dataset)
