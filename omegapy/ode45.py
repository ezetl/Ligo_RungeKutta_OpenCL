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

    def __init__(self, batch=1, nvars=10):
        self.opencl_init()
        self.init_cond = []
        # TODO: hacer chequeos sobre el batch?? no se me ocurrue que podria
        # llegar a pasar
        self.batch = batch
        self.nvars = nvars
        self.global_size = self.batch * self.nvars
        self.local_size = nvars

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
        b5_host = np.array([35./384., 500./1113., 125./192., -2187./6784., 11./84.], dtype=FLOAT)
        #auxiliar arrays
        k_host = np.zeros(shape=(self.global_size*STEPS,), dtype=FLOAT)
        ytemp_host = np.zeros(shape=(NUM_VBLS,), dtype=FLOAT)

        y_host = np.zeros(shape=(NUM_VBLS*self.batch,), dtype=FLOAT)
        y4_host = np.zeros(shape=(NUM_VBLS*self.batch,), dtype=FLOAT)
        y5_host = np.zeros(shape=(NUM_VBLS*self.batch,), dtype=FLOAT)
        #Copy arrays to device
        mf = cl.mem_flags
        self.a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        self.b4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b4)
        self.b5 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b5_host)
        self.y = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y_host)
        self.y4 = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y4_host)
        self.y5 = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y5_host)
        self.k = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=k_host)
        self.ytemp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ytemp_host)

        self.t1 = FLOAT(0)
        self.t2 = FLOAT(1e10)
        self.hmin = (self.t2-self.t1)/1e20
        delta = -1e10
        yinf = -1e10
        hh = (self.t2-self.t1)/100
        hh = min(HMAX, hh)
        hh = max(self.hmin, hh)
        #TODO: chequear como se usaba el error en el rhs
        error_host = np.zeros(shape=(self.batch,), dtype=FLOAT)
        h_host = np.array([hh]*self.batch, dtype=FLOAT)
        time_host = np.array([self.t1]*self.batch, dtype=FLOAT)
        stop_host = np.zeros(shape=(self.batch,), dtype=INT)
        n_ok_host = np.zeros(shape=(self.batch,), dtype=INT)
        n_bad_host = np.zeros(shape=(self.batch,), dtype=INT)
        self.error =  cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=error_host)
        self.h = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_host)
        self.time = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_host)
        self.stop = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=stop_host)
        self.n_ok = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=n_ok_host)
        self.n_bad = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=n_bad_host)
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
        assert(self.t2 > self.t1)
        assert(TOL > 0)

        self.nsteps = 0 # Number of steps used in the process

        self.data_init()
        self.generate_init_cond()
        self.load_program("ode45.cl")

        # Set scalar arguments for OpenCL kernels
        check_step = self.program.check_step
        check_step.set_scalar_arg_dtypes([None, None, None, FLOAT, INT])

        f_rhs = self.program.f_rhs
        f_rhs.set_scalar_arg_dtypes([None, None, INT, INT, INT, INT])

        rk_step = self.program.rk_step
        rk_step.set_scalar_arg_dtypes([None, None, None, None, INT, INT, INT, FLOAT])


        for cond in self.init_cond:
            cond = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cond)
            while(True):
                self.nsteps += 1
                check_step(self.queue, (self.global_size,), (self.local_size), self.h, self.time, self.stops, self.t2, self.hmin )
                #This copy the h array and check if we need to stop. TODO: find a way to do this in gpu and avoid copying arrays in every step

                #Calculate f_rhs with initial values. The number 0 is because we want
                #to use the first portion of self.k array
                f_rhs(self.queue, (self.global_size,), (self.local_size,), self.y, self.k, self.nvars, STEPS, 0, self.error)
                for i in range(1,STEPS+1): # cantidad de steps, es del 1 al 7
                    rk_step(self.queue, (self.global_size,), (self.local_size,), self.ytemp, self.y, self.k, self.a, self.h, i, STEPS, self.nvars)
                    f_rhs(self.queue, (self.global_size,), (self.local_size,), self.ytemp, self.k, NUM_VBLS, STEPS, i, self.error)
                # 4º y 5º order
                self.program.rk_step(self.queue, (self.global_size,), (self.local_size,), self.y4, self.y, self.k, self.b4, self.h, STEPS, 0)
                self.program.rk_step(self.queue, (self.global_size,),(self.local_size,), self.y5, self.y, self.k, self.b4, self.h, STEPS-1, 0)
                #TODO: chequear delta y otras cosas
                #TODO: updatear el array de resultado

    def copy_array(self, arr_like, arr_device):
        """
        This copy an array from device to host and returns it.
        """
        c = np.empty_like(arr_like)
        cl.enqueue_read_buffer(self.queue, arr_device, c).wait()
        return c

    def print_array(self, arr_like, arr_device):
        """
        This copy an array from device and print.
        arr_like must be an array in host with the same style arr_device
        arr_device is the array that is currently in device
        """
        c = self.copy_array(arr_like, arr_device)
        print c

