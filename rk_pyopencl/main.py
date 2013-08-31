# coding: utf-8
import sys
import numpy as np
import pyopencl as cl

from ode45 import Ode45



def main(iters=0):
    # Time range. time integrator stops automatically, t2 is just large
    t1 = 0
    t2 = FLOAT(1e10)

    ode45 = Ode45()
    ode45.execute()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("You must enter the number of iterations")
    ite = int(sys.argv[1])
    main(ite)
    sys.exit(0)