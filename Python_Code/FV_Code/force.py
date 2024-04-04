
import numpy as np


class Force():
    def __init__(self,N,x,cv):
        self.N=N
        self.x=x
        self.cv=cv
    def Construct(self):
        f=np.empty(shape=(self.N+1,1))
        force=lambda x:x-x**2
        f=force(self.x)*self.cv
        return f

