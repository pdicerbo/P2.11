import numpy as np
import matplotlib.pyplot as plt

def MySquareDistance(pt, center):
    return ((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

# data = np.loadtxt("s3.txt")
# dc = 57500
data = np.loadtxt("Aggregation.txt")
dc = 2.5 # value of cut radius

npoints = data.shape[0]
DcSquare = dc * dc

Density = np.zeros(npoints)
MyDelta = np.zeros(npoints)
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
    for j in np.arange(k):
        dtmp = MySquareDistance(data[SortingMask[j], :], data[SortingMask[k], :])
        if dtmp <= dmin:
            dmin = dtmp
            
    MyDelta[k] = np.sqrt(dmin)

MyDelta[0] = MyDelta[MyDelta < DMax].max()*1.1

plt.figure()
for j in range(npoints):
    plt.plot(Density[SortingMask[j]], MyDelta[j], 'b.')
plt.show()
plt.close("all")

plt.figure()
for j in range(npoints):
    plt.plot(data[j,0], data[j,1], 'b.')
plt.show()
