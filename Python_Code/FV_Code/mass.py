from sklearn.decomposition import sparse_encode
import mesh
from mesh import Mesh
import numpy as np
import math
import scipy
from scipy import sparse

class Mass():
    def __init__(self,N,mid,silength):
        self.mid=mid
        self.si=silength
        self.N=N
    def Construct(self):
        alpha=np.empty(shape=(self.N))
        diag=np.empty(shape=(self.N+1))
        M=np.empty(shape=(self.N+1,self.N+1))
        off=np.empty(shape=(self.N))
        kappa=np.piecewise(self.mid,[self.mid<math.pi/6, self.mid>=math.pi/6],[1,1e-4])
        # kappa=.00001
        alpha=kappa/self.si
        diag[0],diag[-1]=alpha[0],alpha[-1]
        diag[1:-1]=alpha[:-1]+alpha[1:]
        off=-alpha
        M=sparse.csc_matrix(sparse.diags(diag,offsets=0)+sparse.diags(off,offsets=1)+sparse.diags(off,offsets=-1))
        return M

        