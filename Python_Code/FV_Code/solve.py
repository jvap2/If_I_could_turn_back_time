from mesh import Mesh
from reaction import Reaction
from force import Force
from mass import Mass
import numpy as np
from scipy.sparse import linalg
import math
from scipy.special import hermite
from scipy.interpolate import interp1d, PchipInterpolator

class Solution():
    def __init__(self,a,b,N):
        self.m=Mesh(a,b,N)
        self.x=self.m.xpoints()
        self.mid=self.m.midpoints()
        self.cv=self.m.cvlengths()
        self.sil=self.m.silengths()
        self.F=Force(N,self.x,self.cv)
        self.R=Reaction(N,self.x,self.cv)
        self.M=Mass(N,self.mid,self.sil)
        self.N=N
        self.a=a
        self.b=b
    def Sol(self, gamma_left,gamma_right,g_a,g_b,type):
        M=self.M.Construct()
        R=self.R.Construct()
        F=self.F.Construct()
        Mat=M+R
        if type==0:
            Mat[0,0],Mat[0,1]=1,0
            Mat[self.N,self.N],Mat[self.N,self.N-1]=1,0
            F[0],F[-1]=g_a,g_b
        elif type==1:
            Mat[0,0],Mat[0,1]=1,0
            Mat[self.N,self.N]-=gamma_right
            F[0]=g_a
            F[self.N]-=g_b
        elif type==2:
            Mat[self.N,self.N],Mat[self.N,self.N-1]=1,0
            Mat[0,0]-=gamma_left
            F[0]-=g_a
            F[-1]=g_b
        else:
            Mat[0,0]-=gamma_left
            Mat[self.N,self.N]-=gamma_right
            F[0]-=g_a
            F[self.N]-=g_b
        solution=np.zeros(shape=(self.N+1,1))
        solution=linalg.spsolve(Mat,F)
        return solution
    def AdaptiveMesh(self,gamma_left,gamma_right,g_a,g_b,type):
        M=self.M.Construct()
        R=self.R.Construct()
        F=self.F.Construct()
        x=self.x
        h=self.sil
        x_new=np.empty(shape=(self.N+1))
        denom=np.empty(shape=(self.N,1))
        new_mid=np.empty(shape=(self.N))
        dif=np.zeros(shape=(self.N-1))
        I=np.zeros(shape=(self.N+1))
        new_cv=np.empty(shape=(self.N+1))
        L=np.array([i/self.N for i in range(0,self.N+1)])
        I[0]=0
        Mat=M+R
        error=1
        tol=1e-5
        p=0
        x_in=x
        while error>tol and p<10:
            if type==0:
                Mat[0,0],Mat[0,1]=1,0
                Mat[self.N,self.N],Mat[self.N,self.N-1]=1,0
                F[0],F[-1]=g_a,g_b
            elif type==1:
                Mat[0,0],Mat[0,1]=1,0
                Mat[self.N,self.N]-=gamma_right
                F[0]=g_a
                F[self.N]-=g_b
            elif type==2:
                Mat[self.N,self.N],Mat[self.N,self.N-1]=1,0
                Mat[0,0]-=gamma_left
                F[0]-=g_a
                F[-1]=g_b
            else:
                Mat[0,0]-=gamma_left
                Mat[self.N,self.N]-=gamma_right
                F[0]-=g_a
                F[self.N]-=g_b
            C=0
            u=linalg.spsolve(Mat,F)
            dif=np.abs((x[1:-1]-x[:-2])**2+(u[1:-1]-u[:-2])**2-((x[2:]-x[1:-1])**2-(u[2:]-u[1:-1])**2))
            error=np.linalg.norm(dif)
            print(error)
            if error<tol:
                break
            poly=PchipInterpolator(x=x_in,y=x)
            ##Need to be piecewise hermite
            x=poly(x_in)
            x[0]=self.a
            x[-1]=self.b
            print(x)
            new_mid=(x[1:]+x[:-1])/2
            new_cv[0],new_cv[-1]=new_mid[0]-x[0],x[-1]-new_mid[-1]
            new_cv[1:self.N]=new_mid[1:]-new_mid[:-1]
            h=x[1:]-x[:-1]
            F_obj=Force(self.N,x,new_cv)
            F=F_obj.Construct()
            R_obj=Reaction(self.N,np.squeeze(np.asarray(x)),np.squeeze(np.asarray(new_cv)))
            R=R_obj.Construct()
            M_obj=Mass(self.N,np.squeeze(np.asarray(new_mid)),np.squeeze(np.asarray(h)))
            M=M_obj.Construct()
            Mat=M+R
            p+=1
        return u,x
            
            

