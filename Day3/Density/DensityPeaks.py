import numpy as np
import matplotlib.pyplot as plt

def MySquareDistance(pt, center):
    return ((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

def MyDistance(pt, center):
    return np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

# data = np.loadtxt("s3.txt")
# dc = 57500
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
    MyNearestDense[j] = NearestIndex
    
MyDelta[0] = MyDelta[MyDelta < DMax].max()*1.1

plt.figure()
for j in range(npoints):
    plt.plot(Density[SortingMask[j]], MyDelta[j], 'b.')
plt.show()
plt.close("all")


MyCenters = np.where(MyDelta >= DeltaCut)
print(data[SortingMask[MyCenters]])

plt.figure()
for j in range(npoints):
    plt.plot(data[j,0], data[j,1], 'b.')
plt.show()
