import numpy as np

def MyDistance(pt, center):
    return np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

def MySquareDistance(pt, center):
    return ((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

class KMeans:
    def __init__(self, k, npt):
        self.k = k
        self.n_iter = 0
        self.membership = np.zeros(npt, dtype = int)
        self.new_member = np.zeros(npt, dtype = int)
        self.centers = np.zeros((k, 2))
        self.MyData  = np.empty(0)

    def __del__(self):
        # also with this statements
        # the array will not be deleted....
        del self.membership
        del self.new_member
        del self.centers
        del self.MyData
        
    # "standard" K-means initialization
    def init_system(self, data):
        NData = data.shape[0]
        MyArr = np.random.random_integers(0, NData, self.k) # extract k random points from data

        self.MyData = np.copy(data)

        for j in np.arange(self.k):
            self.centers[j, 0] = data[MyArr[j], 0]
            self.centers[j, 1] = data[MyArr[j], 1]

    # K-means++ initialization
    def init_pp(self, data):
        NData = data.shape[0]
        MyRand = np.random.random_integers(0, NData-1, 1) # extract 1 random point (first center)

        self.MyData = np.copy(data)
        NCenter = 1
        self.centers[0, :] = data[MyRand, 0:2]

        MyArr = np.zeros(data.shape[0], dtype = float)
        
        while NCenter < self.k:
            dtot = 0.
            counter = 0
            check = 0
            for pt in data:
                dmin = MySquareDistance(pt, self.centers[0,:])

                for j in np.arange(1, NCenter):  # if NCenter == 1 np.arange(1, 1) is empty
                    dtmp = MySquareDistance(pt, self.centers[j,:])
                    if dtmp < dmin:
                        dmin = dtmp
                        
                dtot += dmin
                MyArr[counter] = dmin
                counter += 1

            MyArr /= dtot
            while check == 0:
                MyRand  = np.random.random_integers(0, NData-1, 1)
                NewRand = np.random.random()
                if NewRand < MyArr[MyRand]:
                    self.centers[NCenter, :] = data[MyRand, 0:2]
                    NCenter += 1
                    check = 1
                
    def clusterize(self):
        self.n_iter += 1
        counter = 0
        for pt in self.MyData:
            MyArg = 0

            dmin = MyDistance(pt, self.centers[MyArg,:])

            for j in np.arange(1, self.k):
                dtmp = MyDistance(pt, self.centers[j,:])
                if dtmp < dmin:
                    dmin = dtmp
                    MyArg = j

            self.new_member[counter] = MyArg
            counter += 1
            
        for j in np.arange(self.k):
            NElem = 0
            XMean = np.zeros(2)
            counter = 0

            MyMask = np.where(self.new_member == j)            
            for pt in self.MyData[MyMask]:
                XMean[0] += pt[0]
                XMean[1] += pt[1]
                NElem += 1
                                    
            if NElem > 0:
                self.centers[j, 0] = XMean[0] / float(NElem)
                self.centers[j, 1] = XMean[1] / float(NElem)
            else:
                self.centers[j, 0] = XMean[0]
                self.centers[j, 1] = XMean[1]

        if np.array_equal(self.membership, self.new_member):
            return
        else:
            self.membership = np.copy(self.new_member)
            self.new_member = np.zeros(self.membership.shape, dtype = int)
            self.clusterize()
            
    def objective_func(self):
        MyVal = 0.

        for j in range(self.membership.shape[0]):
            MyVal += MySquareDistance(self.MyData[j], self.centers[self.membership[j],:])
        return MyVal
