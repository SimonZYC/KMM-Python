
# TODO: we can combine the two files into one
def loadPoints(filename):
    input = open(filename, "r")
    info = input.readline().split()
    
    nData = int(info[0]) # number of instances
    nDim = int(info[1])  # number of features
    
    data = [[0]*nDim for i in range(nData)]


    for i in range(nData):
        info = input.readline().split()
        for j in range(nDim):
            data[i][j] = float(info[j]) 
    return data 

def normalize(data):
    nData=len(data)
    nDim=len(data[0])

    normalizer=[0]*nDim

    for j in range(nDim):
        for i in range(nData):
            normalizer[j]=max(normalizer[j],data[i][j])

    for i in range(nData):
        for j in range(nDim):
            data[i][j]=data[i][j]/float(normalizer[j])*30



def loadClusters(filename): 
    input = open(filename, "r") 
    info = input.readline() 
    nData = int(info)
    
    clusters = [0] * nData    
    for i in range(nData):
        info = input.readline()
        clusters[i] = int(info)
    
    return clusters

