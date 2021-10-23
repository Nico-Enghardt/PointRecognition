import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np
import tensorflow as tf
import loss





def createModel(architecture,datasetArtifact,learningRate):

    # Get Input shape from datset Artifact
    imageShape = datasetArtifact.metadata["shape"]
    imageShape = imageShape[0]*imageShape[1]*imageShape[2]

    # Initialize Model with Layers
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=imageShape))
    for layerSize in architecture:
        model.add(tf.keras.layers.Dense(layerSize,kernel_regularizer=tf.keras.regularizers.L1(0.01),
))
    model.add(tf.keras.layers.Dense(3,bias_initializer=tf.keras.initializers.RandomNormal(mean=400,stddev=100)))

    customLoss = loss.loss3D
    customMetrics = [loss.heightError,loss.planeError]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),loss=customLoss,metrics=customMetrics)
    model.build()

    model.summary()

    return model,imageShape


if __name__ == '__main__':

    # Creating a fresh model, saving it to WandB

    modelName = input("What shall be the new model's name? Type here:  ")

    run = wandb.init(project="PointRecognition",job_type="model-creation")
    datasetArtifact = run.use_artifact('nico-enghardt/PointRecognition/Huegray160:latest', type='dataset')

    learningRate = 0.0000003;
    architecture = (3600,1000,10)

    model,imageShape = createModel(architecture,datasetArtifact,learningRate)    

    modelArtifact = wandb.Artifact(modelName,type="model",description=str(architecture))
    model.save("Models/"+modelName)
    modelArtifact.add_dir("Models/"+modelName)

    modelArtifact.metadata = {"dataset":datasetArtifact.name,"architecture":architecture,"input_type":imageShape}

    run.log_artifact(modelArtifact)