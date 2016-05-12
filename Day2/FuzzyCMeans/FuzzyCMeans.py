import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm

def main():

    data = np.loadtxt("s3.txt")

    kmin = 2
    kmax = 18

    npoints = data.shape[0]
    ObjFunc  = np.zeros(kmax - kmin + 1)
    MyAssign = []

    for k in np.arange(kmin, kmax+1):
        
        print("\n\tStarting computation with", k, " centers (kmax =", kmax, ")")

        MyK = FuzzyCMeans(k, npoints)

        MyK.init_system(data)
        MyK.clusterize()
        ObjFunc[k-kmin] = MyK.objective_func()
        MyAssign.append(MyK.membership)
        
        print("\tNumber of iteration: ", MyK.n_iter)
        print("\tObjectiveFuncion: ", ObjFunc[k-kmin])

    # ObjectiveFunction plot
    plt.figure()
    plt.plot(np.arange(kmin, kmax+1), ObjFunc, 'o-')
    plt.title("Objective Function")
    plt.xlabel("k")
    plt.ylabel("ObjFunc")
    plt.show()
    plt.close("all")
    
    ck = 15
    print("\n\tChoose k =", ck)
    print("\tPlotting...")

    cmap = cm.get_cmap('nipy_spectral')

    for k in range(ck):
        plt.figure()

        MyTitle = "Number of Cluster =" + str(ck)+", case of k ="+str(k)
        plt.title(MyTitle)
        print("\tPlotting k =", k)

        for j in range(npoints):
            conv = MyAssign[ck-kmin][j,k]
            MyColor = cmap(conv)
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], '.', c=MyColor)

        
        # saving k-th picture in Images/k#.png
        MyTitle = "Images/k"
        MyTitle += str(k)
        MyTitle += ".png"
        plt.savefig(MyTitle)
        plt.close("all")

main()
