import mesh
from mesh import Mesh
import numpy as np
import scipy
from scipy import sparse


class Reaction():
    def __init__(self,N,x,cv):
        self.N=N
        self.cv=cv
        self.x=x
    def Construct(self):
        reac=lambda x: 0*x
        beta=reac(self.x)*self.cv
        r=sparse.csc_matrix(sparse.diags(beta,offsets=0))
        return r
