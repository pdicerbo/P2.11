import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm
# import colorsys

def main():
    data = np.loadtxt("s3.txt")

    print("\n\tPerform scree plot")
    kmin = 2
    kmax = 20
    counter = 0
    MyAssign = [] # list in which store assignation
    ObjFunc  = np.zeros(kmax-kmin+1)
    npoints = data.shape[0]

    plt.figure()
    cmap = cm.get_cmap('nipy_spectral')
    MyRange = np.arange(kmin, kmax+1, dtype = int)

    for k in MyRange:
        print("\tStarting computation for k = ",k)
        MyK = KMeans(k, npoints)
        MyK.init_pp(data)
        MyK.clusterize()
        ObjFunc[counter] = MyK.objective_func()
        MyAssign.append(MyK.membership)

        for j in range(MyK.MyData.shape[0]):
            conv = float(MyAssign[counter][j]) / float(k-1)        
            MyColor = cmap(conv)
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], '.', c=MyColor)
            MyTitle = "Case of k = "
            MyTitle += str(k)
            plt.title(MyTitle)
        # plt.show()
        MyTitle = "Images/k"
        MyTitle += str(k)
        MyTitle += ".png"
        plt.savefig(MyTitle)
        counter += 1
        del MyK

    plt.close("all")
    plt.figure()
    MinIndex = np.argmin(ObjFunc)

    print("\n\tPlot ObjectiveFunction")
    plt.plot(MyRange, ObjFunc, 'o-')
    plt.title("Objective Function")
    plt.xlabel("k")
    plt.ylabel("ObjFunc")
    plt.show()
    
    # k = 15
    k = MyRange[MinIndex]
    print("\n\tChoose k = ", k)
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
        del MyK
        
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
