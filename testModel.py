import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import compressors
import numpy
import tensorflow as tf
import wandb
import numpy as np


def drawCross(image,position,color=(237, 74, 28)):
    cv2.line(image,(position[0]-10,position[1]-10),(position[0]+10,position[1]+10),color)
    cv2.line(image,(position[0]+10,position[1]-10),(position[0]-10,position[1]+10),color)
    return image

run = wandb.init(job_type="model-evaluation")

# Model laden

modelName = "kassiopeia-1000-100-10:latest"

modelArtifact = run.use_artifact(modelName)
model_directory = modelArtifact.download()

model = tf.keras.models.load_model(model_directory)


# Webcam laden
webcam = cv2.VideoCapture(0)

while True:

    istrue,frame = webcam.read()
    if not istrue:
        break

    compressed = compressors.compress(np.expand_dims(frame,axis=0),"gray8045") #model.metadata["compressor"])


    # Komprimierte Bilder durch Model packen

    fingerCoords = model(np.expand_dims(compressed,axis=0))

    x = fingerCoords[0,0].numpy()
    y = fingerCoords[0,1].numpy()

    x,y = int(x),int(y)

    # Ergebnis und eigentliches Ergebnis anzeigen

    print(x,y)

    cv2.imshow("Result",drawCross(frame, (x,y)))

    cv2.waitKey(20)

    

    