import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np
from readDataset import readDatasetSize

run = wandb.init(project="PointRecognition",job_type="dataset-creation")

datasetName = "Huegray160"
path = "Datasets/" + datasetName


compressor = "huegray160"
shape = (120,160,2)


trainingSize = readDatasetSize(path+"/Training");
testingSize = readDatasetSize(path+"/Testing");
percentTesting = testingSize/(testingSize+trainingSize)

dataset = wandb.Artifact(datasetName,type="dataset")

dataset.add_dir(path)

dataset.metadata = {"compressor":compressor,"shape":shape,"trainingSize":trainingSize,"testingSize":testingSize,"percentTesting":percentTesting}

run.log_artifact(dataset)
