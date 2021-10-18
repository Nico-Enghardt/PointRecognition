import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np
import tensorflow as tf
import compressors


run = wandb.init(project="PointRecognition",job_type="model-creation")
datasetArtifact = run.use_artifact('nico-enghardt/PointRecognition/Huegray160:latest', type='dataset')

modelName = input("What shall be the new model's name? Type here:  ")

learningRate = 0.0000003;
imageShape = datasetArtifact.metadata["shape"]
imageShape = imageShape[0]*imageShape[1]*imageShape[2]

architecture = (3600,1000,10,3)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=imageShape),
    tf.keras.layers.Dense(3600),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(3,bias_initializer=tf.keras.initializers.RandomNormal(mean=400,stddev=100))
])

meanSquaredError = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),loss=meanSquaredError,metrics=tf.keras.metrics.MeanSquaredError(name="mean_squared_error"))
model.build()

model.save("Models/"+modelName)

modelArtifact = wandb.Artifact(modelName,type="model",description=str(architecture))
modelArtifact.add_dir("Models/"+modelName)

modelArtifact.metadata = {"dataset":datasetArtifact.name,"input_type":imageShape}

run.log_artifact(modelArtifact)