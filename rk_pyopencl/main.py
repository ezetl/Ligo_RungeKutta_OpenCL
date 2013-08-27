import numpy as np

FLOAT = np.float32

def main(iters=0):
    # Initial Conditions
    omega = FLOAT(0.004)
    S1ux = FLOAT(0.7071067811865476)
    S1uy = FLOAT(0.7071067811865476)
    S1uz = 0
    S2ux = 0
    S2uy = FLOAT(0.7071067811865476)
    S2uz = FLOAT(0.7071067811865476)
    LNx = LNy = 0
    LNz = FLOAT(1)
    # Number of variables
    N = 10
    # Time range. time integrator stops automatically, t2 is just large
    t1 = 0
    t2 = FLOAT(1e10)
    # Tolerance for solver
    tol = FLOAT(1e-7)
    # Max time step
    hmax = 10

    ### Aca comenzaria el ciclo para cada una de las condiciones iniciales, 
    ### como solo tengo una, pongo una
    

if __name__=="__main__":
    main()