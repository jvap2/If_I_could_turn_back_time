from mesh import Mesh
import numpy as np

class Mass():
    def __init__(self, mesh):
        '''
        This is going to output a tridiagonal matrix which contains the mass matrix
        for the diffusion portion. In construct, alpha is defined as k(x_{i+1/2})/h_i
        Along the diagonal, M[0,0]=alpha[0] and M[N,N]=alpha[N]. The subdiagonal
        M[i,i-1]=-alpha[i-1] and the superdiagonal M[i,i+1]=-alpha[i]. The rest of the
        diagonal will have the form M[i,i]=alpha[i-1]+alpha[i] for i=1,2,...,N-1.
        
        '''
        super().__init__(mesh)
    def Construct(self):
        self.M = np.empty((self.N+1,self.N+1))
        kappa = lambda x: -2*x
        alpha = self.kappa(self.mesh.midpoint())/self.mesh.get_silengths()
        self.M[0,0] = alpha[0]
        self.M[-1,-1] = alpha[-1]
        self.M[1:-1,1:-1] = np.diag(alpha[:-1]+alpha[1:],0)+np.diag(-alpha[:-1],-1)+np.diag(-alpha[1:],1)
        return self.M

