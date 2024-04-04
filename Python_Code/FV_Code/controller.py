from errno import EPFNOSUPPORT
import mesh
from mesh import Mesh
import boundgui
from boundgui import First_View
import tkinter as tk
from solve import Solution
import numpy as np
import matplotlib.pyplot as plt
import math


class Controller():
    def __init__(self):
        self.root=tk.Tk()#create root for GUI
        self.root.title("Adaptive Mesh")
        self.root.geometry("450x500")
        self.view=First_View(self.root)#This makes the Controller control the GUI
        self.view.mesh_button.bind("<Button>", self.make_sol)
    def run(self):
        self.root.mainloop()##This puts it in inf loop to stay there until destroyed
    def make_sol(self,event):
        a=float(self.view.A.get())
        b=float(self.view.B.get())
        N=int(self.view.N.get())
        g_A=float(self.view.g_A.get())
        g_B=float(self.view.g_B.get())
        sol=Solution(a,b,N)
        x=sol.m.xpoints()
        u=np.empty(shape=(N+1,1))
        choice=self.view.bound_type.get()
        gamma_r=0
        gamma_l=0
        epsilon=.00001
        s= lambda x,e: -(1+x)+(.25*x+1)*np.cos(x/math.sqrt(e))
        if(choice=='Right Dirichelet'):
            gamma_l=input("Enter \u03b3 for the left bound: ")
            type=1
        elif(choice=='Left Dirichlet'):
            gamma_r=input("Enter \u03b3 for the right bound: ")
            type=2
        elif(choice=='Neither'):
            gamma_l=input("Enter \u03b3 for the left bound: ")
            gamma_r=input("Enter \u03b3 for the right bound: ")
            type=3
        else:
            type=0
        x,u=sol.AdaptiveMesh(gamma_l,gamma_r,g_A,g_B,type)
        plt.figure()
        plt.plot(x,u,label='Numerical')
        plt.plot(x,s(x,epsilon),label='Asymptotic')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Approximate Solution')
        plt.legend()
        plt.grid()
        plt.show()


def True_Sol(x_mesh,epsilon):
    # c_1=(4+math.sqrt(epsilon)*(1+math.exp(-.5)+math.cos(1/(2*math.sqrt(epsilon)))))/(math.sin(1/(2*math.sqrt(epsilon)))-(1/(8*math.sqrt(epsilon)))*math.cos(1/(2*math.sqrt(epsilon)))+.25*math.sin(1/(2*math.sqrt(epsilon))))
    # print(c_1)
    k_1=4/(math.sqrt(epsilon)*math.sin(1/(2*math.sqrt(epsilon))))+1/(math.tan(1/(2*math.sqrt(epsilon))))+math.exp(-1/(2*math.sqrt(epsilon)))/math.sin(1/(2*math.sqrt(epsilon)))
    right=lambda x: 1+2*math.sqrt(epsilon)*(1-x)/math.sqrt(epsilon)+math.sqrt(epsilon)*np.exp((x-1)/math.sqrt(epsilon))
    # left=lambda x: -1-2*x+c_1*np.sin(x/math.sqrt(epsilon))-.5*c_1*(x**2/math.sqrt(epsilon))*np.cos(x/math.sqrt(epsilon))+.5*c_1*x*np.sin(x/math.sqrt(epsilon))-math.sqrt(epsilon)*np.cos(x/math.sqrt(epsilon))
    left = lambda x: -1-2*x+k_1*math.sqrt(epsilon)*np.sin(x/math.sqrt(epsilon))-math.sqrt(epsilon)*np.cos(x/math.sqrt(epsilon))
    sol=np.piecewise(x_mesh,[x_mesh<.5,x_mesh>=.5],[left,right])
    return sol

