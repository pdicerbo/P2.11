import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm
# import colorsys

def main():
    data = np.loadtxt("s3.txt")

    k = 15
    NIter = 10
    
    npoints = data.shape[0]
    ObjFunc  = np.zeros(NIter)
    MyAssign = np.zeros((NIter, npoints))

    for NIt in range(NIter):
        print("\n\tStarting iteration ", NIt+1, " of ", NIter)

        MyK = KMeans(k,npoints)

        # MyK.init_system(data)
        MyK.init_pp(data)
        MyK.clusterize()
        ObjFunc[NIt] = MyK.objective_func()
        MyAssign[NIt,:] = np.copy(MyK.membership)
        
        print("\tNumber of iteration: ", MyK.n_iter)
        print("\tObjectiveFuncion: ", ObjFunc[NIt])

    MinIndex = np.argmin(ObjFunc)
    print("\n\n\tObjective Function minimum: ", ObjFunc.min())
    print("\tPlotting...")

    plt.figure()

    # cmap = cm.get_cmap('jet')
    cmap = cm.get_cmap('nipy_spectral')
    # cmap = cm.get_cmap('hsv')
    
    for j in range(MyK.MyData.shape[0]):
        conv = float(MyAssign[MinIndex,j]) / float(k-1)        
        MyColor = cmap(conv)
        plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], '.', c=MyColor)
        
    # plt.show()
    plt.savefig("MyClusters.png")

main()
