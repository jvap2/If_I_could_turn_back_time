import numpy as np
from mesh import Mesh

class Reaction():
    def __init__(self, mesh):
        '''
        This class will generate a sparse diagonal matrix containing the coefficients 
        from the reaction component of our equation. The value Beta has the form r(x_i)delta x_i.

        
        
        '''
        super().__init__(mesh)
    def Construct(self):
        reaction = lambda x: x
        beta = reaction(self.x)*self.cv()
        R = np.diag(beta,0)
        return R