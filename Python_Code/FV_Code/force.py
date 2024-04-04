import numpy as np
from mesh import Mesh

class Force(Mesh):
    def __init__(self,a,b,N,t_0=0,t_m=0,M=0):
        super().__init__(a,b,N,t_0,t_m,M)
    def Construct(self):
        self.F  = np.empty((self.N+1,self.M+1))
        force = lambda x,t: 0
        self.F = force(self.x,self.t)*self.cv()
        return self.F
