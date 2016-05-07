import numpy as np

def MyDistance(pt, center):
    return np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

def MySquareDistance(pt, center):
    return ((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

class FuzzyCMeans:
    def __init__(self, k, m, npt):
        self.k = k
        self.m = m
        self.threshold = 1.e-3
        self.n_iter = 0
        self.MyDim = 0
        self.NPts = 0
        self.membership = np.zeros((npt, k), dtype = float)
        self.new_membership = np.zeros((npt, k), dtype = float)
        self.centers = np.zeros((k, 2))
        self.MyData  = np.empty(0)
        
    # Fuzzy CMeans initialization
    def init_system(self, data):
        NData = data.shape[0]
        self.NPts = NData
        self.MyDim = data.shape[1]
        self.MyData = np.copy(data)

        for j in np.arange(NData):

            MyArr = np.random.random(self.k)

            self.membership[j,:] = np.random.random(self.k)
     
    def clusterize(self):
        self.n_iter += 1
        print("iteration ", self.n_iter)

        # compute centers:
        for j in range(self.k):
            ColSum = np.sum(self.membership[:,j]**self.m)
            TmpCenter = np.zeros(self.MyDim)
            
            for i in range(self.NPts):
                TmpCenter += self.membership[i,j]**self.m * self.MyData[i,:]
                
            self.centers[j,:] = TmpCenter / ColSum

        # update U matrix
        for i in range(self.NPts):
            for j in range(self.k):
                tmp_sum = 0.
                MyD = MyDistance(self.MyData[i,:], self.centers[j,:])
                for k in range(self.k):
                    tmp_sum += (MyD / MyDistance(self.MyData[i,:], self.centers[k,:]))**(2./(self.m - 1.))

                self.new_membership[i,j] = 1./tmp_sum

        MyDiff = np.fabs(self.membership - self.new_membership)

        if MyDiff.max() < self.threshold:
            return
        else:
            print("MyDiff.max()", MyDiff.max())
            self.membership = np.copy(self.new_membership)
            self.new_membership = np.zeros((self.MyData.shape[0], self.k))
            self.clusterize()
        
    def objective_func(self):
        MyVal = 0.

        for j in range(self.k):
            for i in range(self.NPts):
                MyVal += self.membership[i,j]**self.m * (MySquareDistance(self.MyData[i,:], self.centers[j,:]))
            
        return MyVal
