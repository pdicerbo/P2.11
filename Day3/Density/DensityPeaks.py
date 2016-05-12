import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def MySquareDistance(pt, center):
    return ((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

def MyDistance(pt, center):
    return np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

data = np.loadtxt("Aggregation.txt")
dc = 2.3 # value of cut radius
DeltaCut = 6.

# data = np.loadtxt("s3.txt")
# dc = 57500
# DeltaCut = 51970

npoints = data.shape[0]
DcSquare = dc * dc

Density = np.zeros(npoints)
MyDelta = np.zeros(npoints)
MyNearestDense = np.zeros(npoints, dtype = int)
MyAssign = np.zeros(npoints, dtype = int)
DMax = 0.
RhoT = 25.

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
    MyAssign[SortingMask[MyCenters[0][j]]] = j + 1
    
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
