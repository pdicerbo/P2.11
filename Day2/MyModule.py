import numpy as np

def MyDistance(pt, center):
    return np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)

class KMeans:
    def __init__(self, k, npt):
        self.k_ = k
        # self.coord = np.zeros((k, npt))
        # self.new_coork = np.zeros((k, npt))
        self.membership = np.zeros(npt, dtype = int)
        self.new_member = np.zeros(npt, dtype = int)
        self.centers = np.zeros((k, 2))
        self.MyData  = np.empty(0) #zeros((npt, 2))
        self.new_centers = np.zeros((k, 2))

    def init_system(self, data):
        NData = data.shape[0]
        MyArr = np.random.random_integers(0, NData, self.k_) # extract k_ random points from data

        self.MyData = np.copy(data)

        # print("self.MyData.shape", self.MyData.shape, " Data.shape", data.shape)
        # print("self.MyData", self.MyData, "data ", data)

        for j in np.arange(self.k_):
            self.centers[j, 0] = data[MyArr[j], 0]
            self.centers[j, 1] = data[MyArr[j], 1]

        counter = 0
        
        for pt in data:
            MyArg = 0

            dmin = MyDistance(pt, self.centers[MyArg,:])

            for j in np.arange(1, self.k_):
                dtmp = MyDistance(pt, self.centers[j,:])
                if dtmp < dmin:
                    dmin = dtmp
                    MyArg = j

            # print(MyArg)
            self.membership[counter] = MyArg
            counter += 1

        # first centroid update
        counter = 0
        for j in np.arange(self.k_):
            NElem = 0
            XMean = np.zeros(2)
            counter = 0
            for pt in data:
                if self.membership[counter] == j:
                    # print(pt, j)
                    XMean[0] += pt[0]
                    XMean[1] += pt[1]
                    NElem += 1
                    
                counter += 1
                
            if NElem > 0:
                self.centers[j, 0] = XMean[0] / float(NElem)
                self.centers[j, 1] = XMean[1] / float(NElem)
            else:
                self.centers[j, 0] = XMean[0]
                self.centers[j, 1] = XMean[1]
        
    def clusterize(self):

        # print("\n\n\tstarting with:\n")
        # print(self.membership)
        # print(self.new_member)
        counter = 0
        for pt in self.MyData:
            MyArg = 0

            dmin = MyDistance(pt, self.centers[MyArg,:])

            for j in np.arange(1, self.k_):
                dtmp = MyDistance(pt, self.centers[j,:])
                if dtmp < dmin:
                    dmin = dtmp
                    MyArg = j

            self.new_member[counter] = MyArg
            counter += 1
            
        for j in np.arange(self.k_):
            NElem = 0
            XMean = np.zeros(2)
            counter = 0
            for pt in self.MyData:
                if self.new_member[counter] == j:
                    XMean[0] += pt[0]
                    XMean[1] += pt[1]
                    NElem += 1
                    
                counter += 1
                
            if NElem > 0:
                self.centers[j, 0] = XMean[0] / float(NElem)
                self.centers[j, 1] = XMean[1] / float(NElem)
            else:
                self.centers[j, 0] = XMean[0]
                self.centers[j, 1] = XMean[1]

        # for j in range(self.membership.shape[0]):
        #     print(self.membership[j], self.new_member[j])
            
        if np.array_equal(self.membership, self.new_member):
            print("true!:\n")
            # for j in range(self.membership.shape[0]):
            #     print(self.membership[j], self.new_member[j])
            # print(self.new_member)
            return
        else:
            self.membership = np.copy(self.new_member)
            self.new_member = np.zeros(self.membership.shape, dtype = int)
            self.clusterize()
