import numpy as np
import matplotlib.pyplot as plt

MyData = np.loadtxt("leaf.csv", delimiter = ",")
print(MyData.shape)

NAttribute = MyData.shape[1]

MyIndex = np.arange(2, NAttribute, 1, dtype = int)
# print(MyIndex)

# array in which store the mean values of each attribute
MeanVals = np.mean(MyData, axis = 0)
# print("\n\tPrint mean values:\n\n", MeanVals)

LCC = np.zeros(NAttribute-2)

for index in MyIndex:
    tmp = 0.
    xdenom = 0.
    ydenom = 0.

    for k in range(MyData.shape[0]):
        tmp += (MyData[k,index] - MeanVals[index])*(MyData[k,0] - MeanVals[0])
        xdenom += (MyData[k,index] - MeanVals[index])**2
        ydenom += (MyData[k,0] - MeanVals[0])**2

    LCC[index-2] = np.abs(tmp) / np.sqrt(xdenom*ydenom)

print("\n\tLinear Correlation Coefficient:\n\n", LCC)
plt.figure()
plt.plot(MyIndex, LCC, '-o')
plt.xlabel("Attribute")
plt.ylabel("Linear Correlation Coefficient")
plt.show()
plt.savefig("correlation.png")
plt.close("all")
