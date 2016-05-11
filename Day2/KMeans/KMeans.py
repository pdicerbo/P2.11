import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm

def main():
    data = np.loadtxt("s3.txt")

    print("\n\tPerform scree plot")
    kmin = 2
    kmax = 20
    counter = 0
    MyAssign = [] # list in which store assignation values
    ObjFunc  = np.zeros(kmax-kmin+1)
    npoints = data.shape[0]

    cmap = cm.get_cmap('nipy_spectral')
    MyRange = np.arange(kmin, kmax+1, dtype = int)

    for k in MyRange:
        print("\tStarting computation for k = ",k)
        MyK = KMeans(k, npoints)
        MyK.init_pp(data)
        MyK.clusterize()
        ObjFunc[counter] = MyK.objective_func()
        MyAssign.append(MyK.membership)

        plt.figure()

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
        plt.close("all")
        counter += 1
        del MyK

    print("\n\tPlot ObjectiveFunction")
    plt.figure()

    plt.plot(MyRange, ObjFunc, 'o-')
    plt.title("Objective Function")
    plt.xlabel("k")
    plt.ylabel("ObjFunc")
    plt.show()
    
    k = 15
    print("\n\tChoose k = ", k)
    print("\tPlotting...")

    plt.figure()

    cmap = cm.get_cmap('nipy_spectral')
    # cmap = cm.get_cmap('jet')
    # cmap = cm.get_cmap('hsv')
    
    for j in range(npoints):
        conv = float(MyAssign[k-kmin][j]) / float(k-1)      
        MyColor = cmap(conv)
        plt.plot(data[j,0], data[j,1], '.', c=MyColor)
        
    # plt.show()
    MyTitle = "Number of clusters = " + str(k)
    plt.title(MyTitle)
    plt.savefig("MyClusters.png")

main()
