from mesh import Mesh
import numpy as np

class Mass(Mesh):
    def __init__(self,a,b,N,t_0=0,t_m=0,M=0):
        '''
        This is going to output a tridiagonal matrix which contains the mass matrix
        for the diffusion portion. In construct, alpha is defined as k(x_{i+1/2})/h_i
        Along the diagonal, M[0,0]=alpha[0] and M[N,N]=alpha[N]. The subdiagonal
        M[i,i-1]=-alpha[i-1] and the superdiagonal M[i,i+1]=-alpha[i]. The rest of the
        diagonal will have the form M[i,i]=alpha[i-1]+alpha[i] for i=1,2,...,N-1.
        
        '''
        super().__init__(a,b,N,t_0,t_m,M)
        self.M = np.empty((self.N+1,self.N+1))
    def Construct(self):
        kappa = lambda x: -2*x
        alpha = kappa(self.midpoint())/self.get_silengths()
        print(np.shape(alpha))
        ME = np.empty(self.N+1)
        ME[0],ME[-1] = alpha[0],alpha[-1]
        ME[1:-1] = alpha[:-1]+alpha[1:]
        print(np.shape(self.M))
        print(np.shape(np.diag(ME,0)))
        print(np.shape(np.diag(-alpha[:-1],-1)))
        self.M = np.diag(ME,0)+np.diag(-alpha,-1)+np.diag(-alpha,1)
        return self.M

