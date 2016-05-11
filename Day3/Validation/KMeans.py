import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm

# data = np.loadtxt("s3.txt")
# k = 15 # number of clusters
data = np.loadtxt("Aggregation.txt")
k = 7

npoints = data.shape[0]
print("\tStarting KMeans algorithm for k = ",k)
MyK = KMeans(k, npoints)
MyK.init_pp(data)
MyK.clusterize()

cmap = cm.get_cmap('nipy_spectral')
for j in range(MyK.MyData.shape[0]):
    conv = float(MyK.membership[j]) / float(k-1)        
    MyColor = cmap(conv)
    plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], '.', c=MyColor)

MyTitle = "Case of k = "
MyTitle += str(k)    
plt.title(MyTitle)

plt.show()
# MyTitle = "KMeansk"+str(k)+".png"
# plt.savefig(MyTitle)

# COMPUTE F-VARIANCE TEST
N = np.zeros(k)
XMean = np.zeros(2)
XCMean = np.zeros((k,2))

# compute xmean for each cluster
for j in range(npoints):
    N[MyK.membership[j]] += 1
    XMean[0] += MyK.MyData[j, 0]/npoints
    XMean[1] += MyK.MyData[j, 1]/npoints

    XCMean[MyK.membership[j], 0] += data[j, 0]
    XCMean[MyK.membership[j], 1] += data[j, 1]

for j in range(k):
    XCMean[j] /= N[j]
    
numerator = 0.

for j in range(npoints):
    numerator += MySquareDistance(data[j,0:2], XCMean[MyK.membership[j],:])

numerator *= k
denom = 0.

for j in range(k):
    denom += N[j] * MySquareDistance(MyK.centers[j,:], XMean[:])

print("\tMy FRatio = ", numerator/denom)


# COMPUTE NORMALIZED MUTUAL INFORMATION
MyProbability = np.zeros(k)
GroundT = np.zeros(k)
MyMat = np.zeros((k, k))

for j in range(npoints):
    GroundT[data[j,2]-1] += 1.
    MyMat[data[j,2]-1, MyK.membership[j]] += 1.

MyMat /= float(npoints)

for j in range(k):
    MyProbability[j] = float(N[j]) / float(npoints)
    GroundT[j] /= float(npoints)
    
# print(np.sort(GroundT))
# print(np.sort(MyProbability))
# print(MyMat)

MutualInfo = 0.
Hk = 0.
Hg = 0.

for i in range(k):
    Hg += GroundT[i] * np.log(GroundT[i])
    Hk += MyProbability[i] * np.log(MyProbability[i])
    for j in range(k):
        if MyMat[i,j] > 0.:
            MutualInfo += MyMat[i,j] * np.log(MyMat[i,j]/(GroundT[i] * MyProbability[j])) 

print("\tMy Normalized Mutual Information =", -2.*MutualInfo/(Hg+Hk))
