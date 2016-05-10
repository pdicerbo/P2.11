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
