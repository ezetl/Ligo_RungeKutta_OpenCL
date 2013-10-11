# coding: utf-8
import sys
import numpy as np
import pyopencl as cl

from ode45 import Ode45, FLOAT



def main():
    # Time range. time integrator stops automatically, t2 is just large   
    ode45 = Ode45()
    ode45.execute()

if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #   sys.exit("You must enter the number of iterations")
    #ite = int(sys.argv[1])
    main()
    sys.exit(0)
