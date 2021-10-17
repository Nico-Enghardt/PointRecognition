import cv2
import numpy as np

hue = 0
sat = 0
val = 0

hueVar = 0
satVar = 0
valVar = 0

def readColorSettings():
    global hue,sat,val,hueVar,satVar,valVar
    import colorSettings
    settings = colorSettings.colorSettings()
    hue,sat,val = settings["hue"],settings["sat"],settings["val"]
    hueVar,satVar,valVar = settings["hueVar"],settings["satVar"],settings["valVar"]

def writeColorSettings():
    import os
    settingsFile = "def colorSettings(): \n  return {"
    settingsFile = settingsFile + f"'hue':{hue}, \n  'sat':{sat}, \n  'val':{val}, \n  'hueVar':{hueVar}, \n  'satVar':{satVar}, \n  'valVar':{valVar}" + "}"

    f = open("colorSettings.py","w")
    f.write(settingsFile)
    f.close()

readColorSettings()

def extractMass(img):
    (contours, hierarchy) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []

    for j in range(0,len(contours)):
        areas.append({'area':cv2.contourArea(contours[j]), 'position':contours[j]})
    
    def ar(x):
        return x['area']
    
    areas.sort(key=ar,reverse=True)

    if len(areas) > 0:
        moments = cv2.moments(areas[0]["position"]);
        if moments["m00"] > 0:
            x = moments["m10"]/moments["m00"]
            y = moments["m01"]/moments["m00"]
            return (round(x),round(y))
    
    return (-100,-100)

def changeHue(x):
    global hue
    hue = x

def changeSat(x):
    global sat
    sat = x

def changeVal(x):
    global val
    val = x

def changeHueVar(x):
    global hueVar
    hueVar = x

def changeSatVar(x):
    global satVar
    satVar = x

def changeValVar(x):
    global valVar
    valVar = x

def adjustCenterPoint():
    topWebcam = cv2.VideoCapture(0)
    bottomWebcam = cv2.VideoCapture(2)
    cv2.imshow("blueDots",np.zeros((10,10)))
    cv2.createTrackbar('Hue','blueDots',hue,255,changeHue)
    cv2.createTrackbar('Sat','blueDots',sat,255,changeSat)
    cv2.createTrackbar('Val','blueDots',val,255,changeVal)
    cv2.createTrackbar('Hue-Variance','blueDots',hueVar,255,changeHueVar)
    cv2.createTrackbar('Sat-Variance','blueDots',satVar,255,changeSatVar)
    cv2.createTrackbar('Val-Variance','blueDots',valVar,255,changeValVar)

    while True:
        isTrue,topFrame = topWebcam.read()
        isTrue,bottomFrame = bottomWebcam.read()
    
        editedFrame,center = findDotCenter(topFrame,testing=True)

        cv2.imshow("TopFrame",editedFrame)
        
        rubbish,heightCenter = findDotCenter(bottomFrame,testing=True)

        print("Adjusting:",center,heightCenter)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    writeColorSettings()

    topWebcam.release()
    bottomWebcam.release()

    cv2.destroyAllWindows()

    return center,heightCenter

def findDotCenter(img,testing=False):
    
    # Konvertiere Kamera-Feed in HSV-Format
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Extrahiere den gesuchten Punkt aus dem Bild
    blueDots = cv2.inRange(imgHSV, (hue-hueVar,sat-satVar,val-valVar),(hue+hueVar,sat+satVar,val+valVar))

    if testing:
        cv2.imshow("blueDots",blueDots)
   
    center = extractMass(blueDots)

    # Center an der richtigen Stelle
    cv2.circle(img,center,6,(200,200,0),thickness=-1)

    # Generiere überdeckenden Kreis in Hautfarbe
    innerRadius = 7
    outerRadius = 9

    mask = np.zeros(img.shape,'uint8')
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    cv2.circle(mask,center,outerRadius,255,thickness=-1)
    cv2.circle(mask,center,innerRadius,0,thickness=-1)

    skinColor = cv2.mean(img,mask)
    cv2.circle(img,center,int(innerRadius*0.5+outerRadius*0.5),skinColor,-1)
    if testing:
        cv2.imshow("GrayScaleWithoutDot",img)

    return img,center


if __name__ == '__main__':
    adjustCenterPoint()

    
            

