from mesh import Mesh
import numpy as np

class Mass():
    def __init__(self, mesh):
        super().__init__(mesh)
    def Construct(self):
        self.M = np.empty((self.N+1,self.N+1))
        