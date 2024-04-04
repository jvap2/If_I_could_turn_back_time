from re import A
import numpy as np


class Mesh():
    def __init__(self, a,b,N):
        self.a=a
        self.b=b
        self.N=N
    def xpoints(self): 
        x_mesh=np.empty(shape=(self.N+1,1))
        x_mesh=np.linspace(self.a,self.b,self.N+1)
        return x_mesh
    def midpoints(self):
        x_mid=np.empty(shape=(self.N,1))
        x_mid=(self.xpoints()[:-1]+self.xpoints()[1:])/2
        return x_mid
    def silengths(self):
        h=np.empty(shape=(self.N,1))
        h=self.xpoints()[1:]-self.xpoints()[:-1]
        return h 
    def cvlengths(self):
        cv=np.empty(shape=(self.N+1))
        cv[0],cv[-1]=self.midpoints()[0]-self.xpoints()[0],self.xpoints()[-1]-self.midpoints()[-1]
        cv[1:self.N]=self.midpoints()[1:]-self.midpoints()[:-1]
        return cv

