from mesh import Mesh
from force import Force
from mass import Mass
from reaction import Reaction
import numpy as np

class Solve():
    def __init__(self,mesh,force,mass,reaction):
        self.mesh = mesh
        self.force = force
        self.mass = mass
        self.reaction = reaction
    def Construct(self):
        self.M = self.mass.Construct()
        self.R = self.reaction.Construct()
        self.F = self.force.Construct()
        return self.M,self.R,self.F
    def Solve(self):
        self.Construct()
        self.u = np.zeros((self.mesh.N+1,self.mesh.M+1))
        self.u[:,0] = 0
        self.u[0,:] = self.mesh.get_a()
        self.u[-1,:] = self.mesh.get_b()
        for i in range(1,self.mesh.M+1):
            self.u[1:-1,i] = np.linalg.solve(self.M+self.R,self.F)
        return self.u