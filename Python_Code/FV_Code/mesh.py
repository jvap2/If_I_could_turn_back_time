import numpy as np


class Mesh():
    def __init__(self,a,b,N,t_0=0,t_m=0,M=0):
        self.a = a
        self.b = b
        self.N = N
        self.t_0 = t_0
        self.t_m = t_m
        self.M = M
        self.h = (b-a)/N
        self.k = (t_m-t_0)/M
        self.x = np.linspace(a,b,N+1)
        self.t = np.linspace(t_0,t_m,M+1)
    def NumSubIntervals(self):
        return self.N
    def NumTimeSteps(self):
        return self.M
    def get_silengths(self):
        return self.x[1:]-self.x[:-1]
    def get_time_steps(self):
        return self.t[1:]-self.t[:-1]
    def get_mesh(self):
        return self.x
    def get_time(self):
        return self.t
    def get_u(self):
        return self.u
    def set_u(self,u):
        self.u = u
    def get_h(self):
        return self.h
    def get_k(self):
        return self.k
    def midpoint(self):
        return (self.x[1:]+self.x[:-1])/2
        
