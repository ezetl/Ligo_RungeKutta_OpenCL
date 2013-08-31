# coding: utf-8
import sys
import numpy as np
import pyopencl as cl


FLOAT = np.float32
RANGE_IC = 10  # number of initial conditions to test
NUM_VBLS = 10  # number of variables
TOL = 1e-7  #  tolerance for solver
HMAX = 10  # max time step


class Ode45:
    """
    This class wraps everything related with setting up the opencl enviroment 
    and running the Runge-Kutta code in the devices.
    """

    def __init__(self):
        self.opencl_init()

    def opencl_init(self):
        """
        Creates context, queue, etc.
        """
        # Get platform
        ### TODO: multiple devices
        self.platform = cl.get_platforms()[0]
        # Get GPU
        self.device = self.platform.get_devices()[0]
        # Create context
        self.ctx = cl.Context([self.device])
        # Create queue
        self.queue = cl.CommandQueue(self.ctx)

    def load_program(self, program):
        with open(program, 'r') as f:
            code = f.read()
            self.program = cl.Program(self.ctx, code).build()

    def data_init(self):
        """
        This creates some arrays in the device with data used for further 
        calculations.
        """
        ### ACA CREA LAS Aij, Bij, probablemente las Ki


    def generate_init_cond(self):
        """
        This method should generate initial conditions.
        For now, we only use one initial condition.

        self.init_cond =  numpy.array containing values of initial condition:
                [omega, S1ux, S1uy, S1uz, S2ux, S2uy, S2uz, LNx, LNy, LNz]
        """
        ### (VER TODO PARA VER UNA IDEA DE COMO IMPLEMENTAR ESTA FUNCION)
        omega = FLOAT(0.004)
        S1ux = FLOAT(0.7071067811865476)
        S1uy = FLOAT(0.7071067811865476)
        S1uz = FLOAT(0)
        S2ux = FLOAT(0)
        S2uy = FLOAT(0.7071067811865476)
        S2uz = FLOAT(0.7071067811865476)
        LNx = LNy = FLOAT(0)
        LNz = FLOAT(1)
        initial_cond = [omega, S1ux, S1uy, S1uz, S2ux, S2uy, S2uz, LNx, LNy, LNz]
        self.init_cond = np.array(initial_cond, dtype=FLOAT)


    def execute(self):
        """
        Once the initial data is created, this execute the algorithm
        """
        self.load_program("ode45.cl")
        # Do magic stuff