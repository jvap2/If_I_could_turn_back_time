import numpy as np
from mesh import Mesh

class Reaction(Mesh):
    def __init__(self,a,b,N,t_0=0,t_m=0,M=0):
        '''
        This class will generate a sparse diagonal matrix containing the coefficients 
        from the reaction component of our equation. The value Beta has the form r(x_i)delta x_i.

        
        
        '''
        super().__init__(a,b,N,t_0,t_m,M)
    def Construct(self):
        reaction = lambda x: x
        beta = reaction(self.x)*self.cv()
        R = np.diag(beta,0)
        return R