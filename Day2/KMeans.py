import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm
import colorsys

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

        MyK.init_system(data)
        MyK.clusterize()
        ObjFunc[NIt] = MyK.objective_func()
        MyAssign[NIt,:] = np.copy(MyK.membership)
        
        print("\tNumber of iteration: ", MyK.n_iter)
        print("\tObjectiveFuncion: ", ObjFunc[NIt])

    MinIndex = np.argmin(ObjFunc)
    print("\n\n\tObjective Function minimum: ", ObjFunc.min())
    print("\tPlotting...")

    # h = np.zeros(k, dtype = float)
    # # ind = 0
    # # for p in pdef:
    # #     h[ind] = ((p) / (pmax-pmin))*250.
    # #     ind += 1
    # h = [l/(k+1) for l in range(k+1)]
    # cdef = [colorsys.hsv_to_rgb(x/360., 1., 1.) for x in h]
    # h = np.linspace(0, 1, k+2)

    plt.figure()

    for j in range(MyK.MyData.shape[0]):

        if MyAssign[MinIndex,j] == 0:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'r.') #color = cdef[MyAssign[MinIndex,j]])
        if MyAssign[MinIndex,j] == 1:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'b.')
        if MyAssign[MinIndex,j] == 2:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'g.')
        if MyAssign[MinIndex,j] == 3:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'k.')
        if MyAssign[MinIndex,j] == 4:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'c.')
        if MyAssign[MinIndex,j] == 5:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'm.')
        if MyAssign[MinIndex,j] == 6:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'y.')
        if MyAssign[MinIndex,j] == 7:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'r*')
        if MyAssign[MinIndex,j] == 8:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'b*')
        if MyAssign[MinIndex,j] == 9:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'g*')
        if MyAssign[MinIndex,j] == 10:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'k*')
        if MyAssign[MinIndex,j] == 11:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'c*')
        if MyAssign[MinIndex,j] == 12:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'm*')
        if MyAssign[MinIndex,j] == 13:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'y*')
        if MyAssign[MinIndex,j] == 14:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'ko')

        # s = str(h[1+MyAssign[MinIndex,j]])
        # plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], color = h[1+MyAssign[MinIndex,j]])
        
    # plt.plot(data[:,0], data[:,1], '.')
    plt.show()

main()
