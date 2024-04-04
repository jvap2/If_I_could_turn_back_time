from solve import Solve

# Path: Python_Code/FV_Code/test.py

if __name__ == '__main__':
    from mesh import Mesh
    from force import Force
    from mass import Mass
    from reaction import Reaction   

    a = 0
    b = 1
    N = 10
    t_0 = 0
    t_m = 0
    M = 0
    mesh = Mesh(a,b,N,t_0,t_m,M)
    force = Force(a,b,N,t_0,t_m,M)
    mass = Mass(a,b,N,t_0,t_m,M)
    reaction = Reaction(a,b,N,t_0,t_m,M)
    solve = Solve(mesh,force,mass,reaction)
    u = solve.Solve()
    print(u)
