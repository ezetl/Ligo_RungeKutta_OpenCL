# coding: utf-8
import sys
import numpy as np
import pyopencl as cl

from ode45 import Ode45, FLOAT



def main():
    ode45 = Ode45()
    ode45.execute()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Simulación interrumpida.")
    sys.exit(0)
