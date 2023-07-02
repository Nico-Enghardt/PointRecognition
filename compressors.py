import cv2
import numpy as np

def outputConfig():
    return ["centerX","centerY"]

def compressVideo(video,compressor):
    # Splittet Video in Frames auf

    video = list(video)
    for number in range(len(video)):
        video[number] = compress(video[number],compressor)

     # Python List to Numpy Matrix (PictureNumber x PixelZahl)

    return np.array(video)
 
def compress(picture,compressor):

    # Verteilt Compressing-Anfragen

    picture = np.squeeze(picture)

    if(compressor=="gray8045"):
        return COMPgray8045(picture)

    if(compressor=="hsv160"):
        return COMPhsv160(picture)

    if(compressor=="hsv"):
        return COMPhsv(picture)


def COMPgray8045(picture): # Remove color channels, reduce size to 80x45, flatten out matrix to arrray

    matrix = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
    smallMatrix = cv2.resize(matrix,(80,45))
    return smallMatrix.flatten()
    
def COMPhsv160(picture): # Reduce size to 160x120, only use hue and grayscale values (leave saturation), flatten out matrix to array
    
    resized = cv2.resize(picture,(160,120))
    hsv = cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)

    return hsv

def COMPhsv(picture):

    return cv2.cvtColor(picture,cv2.COLOR_BGR2HSV)

def size(compressor_type,training):

    pixels = None

    if(compressor_type=="gray8045"):
        pixels = 80*45

    if(compressor_type=="hsv160"):
        pixels = 160*120

    if(compressor_type=="hsv"):
        return (640,480,3)

    if training:
        return (1,len(outputConfig())+pixels)

    return (pixels,)