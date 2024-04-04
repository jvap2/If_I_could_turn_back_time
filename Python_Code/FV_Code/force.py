import numpy as np
from mesh import Mesh

class Force():
    def __init__(self, mesh):
        super().__init__(mesh)
    def Construct(self):
        self.F  = np.empty((self.N+1,self.M+1))
        force = lambda x,t: 0
        self.F = force(self.x,self.t)*self.cv
        return self.F
