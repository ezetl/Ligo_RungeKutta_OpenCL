# coding: utf-8
import sys
import numpy as np
import pyopencl as cl


FLOAT = np.float32
INT = np.int32
RANGE_IC = 10  # number of initial conditions to test
NUM_VBLS = 10  # number of variables
TOL = 1e-7  #  tolerance for solver
HMAX = 10  # max time step
STEPS = 7
mf = cl.mem_flags

class Ode45:
    """
    This class wraps everything related with setting up the opencl enviroment
    and running the algorithm in the devices.
    """

    def __init__(self):
        self.opencl_init()
        self.init_cond = []

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
        f = open(program, 'r')
        code = f.read()
        self.code = code
        self.program = cl.Program(self.ctx, code).build()

    def data_init(self):
        """
        This creates some arrays in the device with data used for further
        calculations.
        """
        a = np.array(
                [0.,0.,0.,0.,0.,0.,
                 1./5.,0.,0.,0.,0.,0.,
                 3./40., 9./40.,0.,0.,0.,0.,
                 44./45., -56./15., 32./9.,0.,0.,0.,
                 19372./6561., -25360./2187., 64448./6561., -212./729.,0.,0.,
                 9017./3168., -355./33., 46732./5247., 49./176., -5103./18656.,0.,
                 35./384., 0., 500./1113., 125./192., -2187./6784., 11./84.
                ], dtype=FLOAT)
        #4th order b coeffs
        b4 = np.array([5179./57600., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.],
                dtype=FLOAT)
        #5th order b coeffs
        self.b5_host = np.array([35./384., 500./1113., 125./192., -2187./6784., 11./84.], dtype=FLOAT)
        #auxiliar arrays
        self.k_host = np.zeros(shape=(NUM_VBLS*STEPS,), dtype=FLOAT)
        self.ytemp_host = np.zeros(shape=(NUM_VBLS,), dtype=FLOAT)
        self.y_host = np.zeros(shape=(NUM_VBLS,), dtype=FLOAT)
        #Copy arrays to device
        mf = cl.mem_flags
        self.a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        self.b4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b4)
        self.b5 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b5_host)
        self.y = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.y_host)
        self.k = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.k_host)
        self.ytemp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.ytemp_host)
        #After this,the buffers should be accesible from kernels


    def generate_init_cond(self):
        """
        This method should generate initial conditions.  For now, we only use one initial condition.

        self.init_cond =  numpy.array containing values of initial condition:
                [omega, S1ux, S1uy, S1uz, S2ux, S2uy, S2uz, LNx, LNy, LNz]
        OJO: ahora las initial_conditions se generan aparte y se guardan en un archivo.
        modificar esto para que levante el archivo y lo guarde por aca
        """
        #TODO: modificar para que cargue todas las condiciones iniciales
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
        self.init_cond.append(np.array(initial_cond, dtype=FLOAT))


    def execute(self):
        """
        This calls data_init and then executes the algorithm
        """
        t1 = FLOAT(0)
        t2 = FLOAT(1e10)
        error = 0 #error of rhs
        hmin = (t2-t1)/1e20
        hh = (t2-t1)/100
        hh = min(HMAX, hh)
        hh = max(hmin, hh)
        self.data_init()
        self.generate_init_cond()
        self.load_program("ode45.cl")
        global_size = NUM_VBLS #es la cantidad de work items que voy a usar (? creo que seria la cantidad de blocks en cuda
        # el local size lo pongo como None, tal vez sean los threads

        f_rhs = self.program.f_rhs
        f_rhs.set_scalar_arg_dtypes([None, None, INT, INT, INT, INT])

        #TODO: hacer lo mismo que arriba pero con el rk_step

        for cond in self.init_cond:
            cond = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cond)
            #Calculate f_rhs with initial values. The number 0 is because we want 
            #to use the first portion of self.k array
            f_rhs(self.queue, (global_size,), None, cond, self.k, NUM_VBLS, STEPS, 0, error)
            while(True):
                #TODO: chequear estado
                for i in range(1,STEPS+1): # cantidad de steps, son 7
                    #hacer algo por el estilo...
                    self.program.rk_step(self.queue, (global_size,), None, self.b5)
                    f_rhs(self.queue, (global_size,), None, self.ytemp, self.k, NUM_VBLS, STEPS, i, error)
                # hacer lo de 4º y 5º orden
                self.program.rk_step(self.queue, (global_size,), None, self.b5)
                self.program.rk_step(self.queue, (global_size,), None, self.b5)
                #TODO: chequear delta y otras cosas
                #TODO: updatear el array de resultado
            # Do magic stuff
        #un printeo para chequear cosas
        self.print_array(self.k_host, self.k)


    def print_array(self, arr_like, arr_device):
        """
        This copy an array from device and print.
        arr_like must be an array in host with the same style arr_device
        arr_device is the array that is currently in device
        """
        c = np.empty_like(arr_like)
        cl.enqueue_read_buffer(self.queue, arr_device, c).wait()
        print c
