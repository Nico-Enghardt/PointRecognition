import compressors

def splitDataset(pictures,labels,trainValTest=(.9,0,0.1),mode="training"):

    laenge = len(pictures)

    trainValSchnitt = int(laenge*trainValTest[0])
    valTestSchnitt = int(laenge*(trainValTest[0]+trainValTest[1]))

    if(mode=="training"):
        print(f"Of {laenge} Pictures, {trainValSchnitt} are used for {mode}.")
        return pictures[:trainValSchnitt],labels[:trainValSchnitt,:]

    if(mode=="validation"):
        print(f"Of {laenge} Pictures, {valTestSchnitt-trainValSchnitt} are used for {mode}.")
        return pictures[trainValSchnitt:valTestSchnitt],labels[trainValSchnitt:valTestSchnitt,:]

    if(mode=="testing"):
        print(f"Of {laenge} Pictures, {laenge-valTestSchnitt} are used for {mode}.")
        return pictures[valTestSchnitt:],labels[valTestSchnitt:,:]

    

    return print("No right mode as input.")

def splitLabels(matrix):
    
    ausgaenge = len(compressors.outputConfig())

    labels = matrix[:,:ausgaenge]
    inputs = matrix[:,ausgaenge:]

    return inputs,labels