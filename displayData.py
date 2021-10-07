import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def drawCross(image,position,color=(237, 74, 28)):
    cv2.line(image,(position[0]-10,position[1]-10),(position[0]+10,position[1]+10),color)
    cv2.line(image,(position[0]+10,position[1]-10),(position[0]-10,position[1]+10),color)
    return image

model = tf.keras.models.load_model('Models/LastModel')
model.summary()

run = wandb.init(
    project="ManusDisplay"
    )

dataset = np.load("Data/Training/08-31--15:06.11.npy")

y = dataset[:,:2]
X = dataset[:,2:]

X = np.int32(X)

predictions = model.predict(X)
predictions = np.array(predictions,dtype='int')
X = np.uint8(X)

collection = []

for imageNr in range(len(X)):
    array = X[imageNr]
    
    img = np.reshape(array,(45,80))
    cv2.imshow("Image",img)

    upscaled = cv2.resize(img,(640,460))

    upscaled = cv2.cvtColor(upscaled,cv2.COLOR_GRAY2BGR)

    upscaled = drawCross(upscaled,(y[imageNr,0],y[imageNr,1]))

    print(predictions[imageNr])
    upscaled = drawCross(upscaled,(predictions[imageNr,0],predictions[imageNr,1]),color=(0,250,75))

    cv2.imshow("Upscaled",upscaled)


    wandbImg = wandb.Image(Image.fromarray(img)) #jpg)
    wandbUpscaled = wandb.Image(Image.fromarray(upscaled))
    collection.append([imageNr,wandbImg,wandbUpscaled])
    cv2.waitKey(1)

pictureDisplay = wandb.Table(columns=["count","Original","Solution"],data=collection)
run.log({"PictureDisplay":pictureDisplay})

wandb.finish()
