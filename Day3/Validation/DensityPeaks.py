import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def MySquareDistance(pt, center):
    return ((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

def MyDistance(pt, center):
    return np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

# data = np.loadtxt("s3.txt")
# dc = 57500
# DeltaCut = 51970
data = np.loadtxt("Aggregation.txt")
dc = 2.3 # value of cut radius
DeltaCut = 6.

npoints = data.shape[0]
DcSquare = dc * dc

Density = np.zeros(npoints)
MyDelta = np.zeros(npoints)
MyNearestDense = np.zeros(npoints, dtype = int)
MyAssign = np.zeros(npoints, dtype = int)
DMax = 0.
RhoT = 25. #100.

for j in np.arange(npoints):

    for i in np.arange(j+1, npoints):
        d = MySquareDistance(data[i,:], data[j,:])

        if d <= DcSquare:
            Density[i] += 1
            Density[j] += 1

        if d > DMax:
            DMax = d

SortingMask = Density.argsort()[::-1]
DMax *= 1.1

for k in np.arange(1, npoints):
    dmin = DMax
    NearestIndex = 0
    for j in np.arange(k):
        dtmp = MySquareDistance(data[SortingMask[j], :], data[SortingMask[k], :])
        if dtmp <= dmin:
            dmin = dtmp
            NearestIndex = j

    MyDelta[k] = np.sqrt(dmin)
    MyNearestDense[k] = NearestIndex
    
MyDelta[0] = MyDelta.max()*1.1

plt.figure()
for j in range(npoints):
    plt.plot(Density[SortingMask[j]], MyDelta[j], 'b.')
plt.show()
plt.close("all")


# optionally, I could take the value of DeltaCut also from raw_input
MyCenters = np.where((MyDelta >= DeltaCut) & (np.sort(Density)[::-1] >= RhoT))
print("Centers:")
print(data[SortingMask[MyCenters]])
NCenters = len(data[SortingMask[MyCenters]])

for j in range(NCenters):
    # MyAssign[SortingMask[MyCenters[0][j]]] = j + 1
    MyAssign[SortingMask[MyCenters[0][j]]] = data[SortingMask[MyCenters[0][j]],2] # j + 1
    
for j in np.arange(npoints):
    if MyAssign[SortingMask[j]] < 1:
        MyAssign[SortingMask[j]] = MyAssign[SortingMask[MyNearestDense[j]]]
    
plt.figure()
cmap = cm.get_cmap('nipy_spectral')
for j in range(npoints):
    MyColor = cmap(float(MyAssign[j]-1) / float(NCenters))
    plt.plot(data[j,0], data[j,1], '.', c = MyColor)


# plot centers
for j in range(NCenters):
    MyColor = cmap(float(MyAssign[SortingMask[MyCenters[0][j]]]-1) / float(NCenters))
    plt.plot(data[SortingMask[MyCenters[0][j]], 0], data[SortingMask[MyCenters[0][j]], 1],
             'o', c = MyColor, label = "Cluster "+str(j))
lgd = plt.legend(fontsize = 10, borderpad=0., markerscale=.7, numpoints = 1, bbox_to_anchor=(1.2,1.))
plt.show()
# plt.savefig("MyDensityPeakS3.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.savefig("MyDensityPeak.png", bbox_extra_artists=(lgd,), bbox_inches='tight')


# COMPUTE F-VARIANCE TEST
N = np.zeros(NCenters)
XMean  = np.zeros(2)
XCMean = np.zeros((NCenters,2))
# compute xmean
for j in range(npoints):
    N[MyAssign[j]-1] += 1
    XMean[0] += data[j, 0]/npoints
    XMean[1] += data[j, 1]/npoints

    XCMean[MyAssign[j]-1, 0] += data[j, 0]
    XCMean[MyAssign[j]-1, 1] += data[j, 1]

for j in range(NCenters):
    XCMean[j,:] /= N[j]
    
numerator = 0.

for j in range(npoints):
    numerator += MySquareDistance(data[j,0:2], XCMean[MyAssign[j]-1,:])

numerator *= NCenters

denom = 0.

for j in range(NCenters):
    denom += N[j] * MySquareDistance(data[SortingMask[MyCenters[0][j]], :], XMean)
    
print("\tMy FRatio =", numerator/denom)

# COMPUTE NORMALIZED MUTUAL INFORMATION

MyProbability = np.zeros(NCenters)
GroundT = np.zeros(NCenters)
MyMat = np.zeros((NCenters, NCenters))

for j in range(npoints):
    GroundT[data[j,2]-1] += 1.
    MyMat[data[j,2]-1, MyAssign[j]-1] += 1.

MyMat /= float(npoints)

for j in range(NCenters):
    MyProbability[j] = float(N[j]) / float(npoints)
    GroundT[j] /= float(npoints)
    
# print(np.sort(GroundT))
# print(np.sort(MyProbability))
# print(MyMat)

MutualInfo = 0.
Hk = 0.
Hg = 0.

for i in range(NCenters):
    Hg += GroundT[i] * np.log(GroundT[i])
    Hk += MyProbability[i] * np.log(MyProbability[i])
    for j in range(NCenters):
        if MyMat[i,j] > 0.:
            MutualInfo += MyMat[i,j] * np.log(MyMat[i,j]/(GroundT[i] * MyProbability[j])) 

print("\tMy Normalized Mutual Information =", -2.*MutualInfo/(Hg+Hk))
