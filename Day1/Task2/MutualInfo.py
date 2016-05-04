import numpy as np
import matplotlib.pyplot as plt

classes = ['republican', 'democrat']
attr = ['y', 'n', '?']

MyTextFile = np.genfromtxt("house-votes-84.data", delimiter = ",", dtype = 'str')
MyClasses = MyTextFile[:,0]
MyData = MyTextFile[:,1:]

NFeat = MyTextFile.shape[1]-1

MutualInfo = np.zeros((len(attr)*len(classes)+5, NFeat))

for MyAttr in range(NFeat):
    NDem = 0
    NRep = 0
    DemY = 0
    DemN = 0
    DemQ = 0
    RepY = 0
    RepN = 0
    RepQ = 0
    NY   = 0
    NN   = 0
    NQ   = 0

    for j in range(MyData.shape[0]):
        
        if MyClasses[j] == classes[0]:
            NRep += 1

            if MyData[j, MyAttr] == attr[0]:
                RepY += 1
                NY += 1
            if MyData[j, MyAttr] == attr[1]:
                RepN += 1
                NN += 1
            if MyData[j, MyAttr] == attr[2]:
                RepQ += 1
                NQ += 1
            
        if MyClasses[j] == classes[1]:
            NDem += 1

            if MyData[j, MyAttr] == attr[0]:
                DemY += 1
                NY += 1
            if MyData[j, MyAttr] == attr[1]:
                DemN += 1
                NN += 1
            if MyData[j, MyAttr] == attr[2]:
                DemQ += 1
                NQ += 1
                
    MutualInfo[0, MyAttr] = RepY/(NDem+NRep)
    MutualInfo[1, MyAttr] = RepN/(NDem+NRep)
    MutualInfo[2, MyAttr] = RepQ/(NDem+NRep)
    MutualInfo[3, MyAttr] = DemY/(NDem+NRep)
    MutualInfo[4, MyAttr] = DemN/(NDem+NRep)
    MutualInfo[5, MyAttr] = DemQ/(NDem+NRep)
    MutualInfo[6, MyAttr] = NRep/(NDem+NRep)
    MutualInfo[7, MyAttr] = NDem/(NDem+NRep)
    MutualInfo[8, MyAttr] = NY/(NDem+NRep)
    MutualInfo[9, MyAttr] = NN/(NDem+NRep)
    MutualInfo[10, MyAttr]= NQ/(NDem+NRep)
    
# print(MutualInfo)

Results = np.zeros(NFeat)

for MyAttr in range(NFeat):

    # RepY
    Results[MyAttr] = MutualInfo[0, MyAttr] * np.log(MutualInfo[0, MyAttr]/
                                                     (MutualInfo[8, MyAttr]*MutualInfo[6, MyAttr]))

    Results[MyAttr] += MutualInfo[1, MyAttr] * np.log(MutualInfo[1, MyAttr]/
                                                      (MutualInfo[9, MyAttr]*MutualInfo[6, MyAttr]))

    Results[MyAttr] += MutualInfo[2, MyAttr] * np.log(MutualInfo[2, MyAttr]/
                                                      (MutualInfo[10, MyAttr]*MutualInfo[6, MyAttr]))
    
    Results[MyAttr] += MutualInfo[3, MyAttr] * np.log(MutualInfo[3, MyAttr]/
                                                      (MutualInfo[8, MyAttr]*MutualInfo[7, MyAttr]))
    
    Results[MyAttr] += MutualInfo[4, MyAttr] * np.log(MutualInfo[4, MyAttr]/
                                                      (MutualInfo[9, MyAttr]*MutualInfo[7, MyAttr]))

    Results[MyAttr] += MutualInfo[5, MyAttr] * np.log(MutualInfo[5, MyAttr]/
                                                     (MutualInfo[10, MyAttr]*MutualInfo[7, MyAttr]))
    
# print(Results)

plt.plot(np.arange(1,NFeat+1,1), Results, '-o')
plt.xlabel("Attribute")
plt.ylabel("Mutual Information")
# plt.show()
plt.savefig("correlation_mutual.png")
