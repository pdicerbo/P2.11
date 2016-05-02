import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MyTextFile = np.genfromtxt("ionosphere.data", delimiter = ",", dtype = 'str')
# print(MyData.shape)

MyData = np.array(MyTextFile[:,:-1], dtype = float)
MyMeasure = MyTextFile[:,-1]

MeanValues = np.mean(MyData, axis = 0)
NMes  = MyData.shape[0]
NFeat = MyData.shape[1]
CorrelationMatrix = np.zeros((NFeat,NFeat))

# print(MeanValues.shape)

for i in range(NFeat):
    for j in range(NFeat):
        tmp = 0.
        for k in range(NMes):
            tmp += (MyData[k,i] - MeanValues[i]) * (MyData[k,j] - MeanValues[j])

        CorrelationMatrix[i,j] = tmp / NMes

# print(CorrelationMatrix)

val, vec = np.linalg.eig(CorrelationMatrix)

MyIndex = np.argsort(val)[::-1]
MyVal = val[MyIndex]
MyVec = vec[:,MyIndex]

plt.figure()
plt.plot(np.arange(0,NFeat-1), np.log(MyVal[:-1]), '-o') # remove eigenvalue 0
plt.xlabel("$\lambda$ value")
plt.ylabel("log($\lambda$)")
# plt.show()
plt.savefig("lambda.png")
plt.close("all")

Dim = 2

if Dim > NFeat:
    print("\n\tError! Reduction Dimension > Number of Feature\n")
    print("\tSet Dim = NFeat")
    Dim = NFeat

# construct new basis

# first case: Dim = 2
y = np.zeros((NMes, Dim))
for j in range(Dim):
    for i in range(NMes):
        tmp = 0.
        
        for k in range(NFeat):
            tmp += MyData[i,k]*MyVec[k,j]

        y[i, j] = tmp
    
plt.figure()

for j in range(y.shape[0]):
    if MyMeasure[j] == 'g':
        plt.scatter(y[j,0], y[j,1], c = 'r')
    elif MyMeasure[j] == 'b':
        plt.scatter(y[j,0], y[j,1], c = 'b')

# plt.show()
plt.savefig("2D.png")
plt.close("all")

Dim = 3

y = np.zeros((NMes, Dim))
for j in range(Dim):
    for i in range(NMes):
        tmp = 0.
        
        for k in range(NFeat):
            tmp += MyData[i,k]*MyVec[k,j]

        y[i, j] = tmp

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in range(y.shape[0]):
    if MyMeasure[j] == 'g':
        ax.scatter(y[j,0], y[j,1], y[j,2], c = 'r')
    elif MyMeasure[j] == 'b':
        ax.scatter(y[j,0], y[j,1], y[j,2], c = 'b')

# plt.show()
plt.savefig("3D.png")
