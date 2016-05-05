import numpy as np
import matplotlib.pyplot as plt
from MyModule import *
import matplotlib.cm as cm
import colorsys

def main():
    data = np.loadtxt("s3.txt")

    k = 15
    npoints = data.shape[0]

    MyK = KMeans(k,npoints)

    MyK.init_system(data)
    MyK.clusterize()
    # print(MyK.centers)


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

        #     # print(MyK.membership[j], type(MyK.membership[j]))

        if MyK.membership[j] == 0:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'r.') #color = cdef[MyK.membership[j]])
        if MyK.membership[j] == 1:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'b.')
        if MyK.membership[j] == 2:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'g.')
        if MyK.membership[j] == 3:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'k.')
        if MyK.membership[j] == 4:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'c.')
        if MyK.membership[j] == 5:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'm.')
        if MyK.membership[j] == 6:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'y.')
        if MyK.membership[j] == 7:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'r.')
        if MyK.membership[j] == 8:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'b.')
        if MyK.membership[j] == 9:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'g.')
        if MyK.membership[j] == 10:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'k.')
        if MyK.membership[j] == 11:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'c.')
        if MyK.membership[j] == 12:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'm.')
        if MyK.membership[j] == 13:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'y.')
        if MyK.membership[j] == 14:
            plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], 'k.')

        # s = str(h[1+MyK.membership[j]])
        # plt.plot(MyK.MyData[j,0], MyK.MyData[j,1], color = h[1+MyK.membership[j]])
        
    # plt.plot(data[:,0], data[:,1], '.')
    plt.show()

main()
