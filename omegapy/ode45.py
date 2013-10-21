# coding: utf-8
import sys
import numpy as np
import pyopencl as cl



np.set_printoptions(suppress=True)
FLOAT = np.float32
INT = np.int32
TOL = 1e-7  #  tolerance for solver
DELTA = -1e10
YINF = DELTA
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
        # Constants
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
        self.k_host = np.zeros(shape=(self.global_size*STEPS,), dtype=FLOAT)
        self.ytemp_host = np.zeros(shape=(self.nvars,), dtype=FLOAT)
        #states
        self.y_host = np.zeros(shape=(self.global_size,), dtype=FLOAT)
        self.y4_host = np.zeros(shape=(self.global_size,), dtype=FLOAT)
        self.y5_host = np.zeros(shape=(self.global_size,), dtype=FLOAT)

        mf = cl.mem_flags

        self.t1 = FLOAT(0)
        self.t2 = FLOAT(1e10)
        self.hmin = (self.t2-self.t1)/1e20
        self.hmax = FLOAT(10)
        self.final_omega = FLOAT(0.1)
        hh = (self.t2-self.t1)/100
        hh = min(self.hmax, hh)
        hh = max(self.hmin, hh)

        self.tau_host = np.zeros(shape=(self.batch,), dtype=FLOAT)
        self.delta_host = np.array([DELTA]*self.batch, dtype=FLOAT)
        self.error_host = np.zeros(shape=(self.batch,), dtype=FLOAT)
        self.h_host = np.array([hh]*self.batch, dtype=FLOAT)
        time_host = np.array([self.t1]*self.batch, dtype=FLOAT)
        self.stop_host = np.zeros(shape=(self.batch,), dtype=INT)
        n_ok_host = np.zeros(shape=(self.batch,), dtype=INT)
        n_bad_host = np.zeros(shape=(self.batch,), dtype=INT)

        #Copy arrays from host to device
        self.a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        self.b4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b4)
        self.b5 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b5_host)
        self.y = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.y_host)
        self.y4 = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.y4_host)
        self.y5 = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.y5_host)
        self.k = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.k_host)
        self.ytemp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.ytemp_host)
        self.tau = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.tau_host)
        self.delta = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.delta_host)
        self.error =  cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.error_host)
        self.h = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_host)
        self.time = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_host)
        self.stop = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.stop_host)
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
        self.data_init()

        assert(self.t2 > self.t1)
        assert(TOL > 0)

        self.nsteps = 0 # Number of steps used in the process

        self.generate_init_cond()
        self.load_program("ode45.cl")

        # Set scalar arguments for OpenCL kernels
        check_step = self.program.check_step
        check_step.set_scalar_arg_dtypes([None, None, None, FLOAT, INT])

        f_rhs = self.program.f_rhs
        f_rhs.set_scalar_arg_dtypes([None, None, None, INT, INT, INT])

        rk_step = self.program.rk_step
        rk_step.set_scalar_arg_dtypes([None, None, None, None, None, INT, INT, INT, INT, INT])

        evaluate_step = self.program.evaluate_step
        evaluate_step.set_scalar_arg_dtypes([None, None, None, None, None, FLOAT, INT])

        update_variables = self.program.update_variables
        update_variables.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, FLOAT, FLOAT, FLOAT, INT])

        for cond in self.init_cond:
            #TODO: CAMBIAR ESTO POR EL ARRAY QUE CONTENGA TODO EL BATCH CON CONDICIONES INICIALES (NO NECESARIAMENTE ES UNA SOLA)
            self.y = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cond)
            while(True):
                check_step(self.queue, (self.global_size,), (self.local_size,), self.h, self.time, self.stop, self.t2, self.hmin)
                #This copy the stop array and check if we need to stop.
                #TODO: find a way to do this in gpu and avoid copying arrays in every step
                stop = self.copy_array(self.stop_host, self.stop)

#                if self.nsteps%19000==0:
#                    print self.nsteps
#                    self.print_array(self.y_host, self.y)

                if any(stop):
                    #TODO: separar esto para los errores y la condicion de terminacion
                    print "break"
                    self.print_array(self.stop_host, self.stop)

                    break
                #Calculate f_rhs with initial values. The number 0 is because we want
                #to use the first portion of self.k array
                f_rhs(self.queue, (self.global_size,), (self.local_size,), self.y, self.k, self.stop, self.nvars, STEPS, INT(0))
                for i in range(1,STEPS): # cantidad de steps, es del 1 al 7
                    rk_step(self.queue, (self.global_size,), (self.local_size,), self.ytemp, self.y, self.k, self.a, self.h, i, STEPS, self.nvars, 6, 1)
                    #print "ytmp"
                    #self.print_array(self.ytemp_host, self.ytemp)
                    f_rhs(self.queue, (self.global_size,), (self.local_size,), self.ytemp, self.k, self.stop, self.nvars, STEPS, i)
                # 4º y 5º order
                self.program.rk_step(self.queue, (self.global_size,), (self.local_size,), self.y4, self.y, self.k, self.b4, self.h, INT(STEPS-1), INT(STEPS), INT(self.nvars), INT(6), INT(0))
                self.program.rk_step(self.queue, (self.global_size,),(self.local_size,), self.y5, self.y, self.k, self.b5, self.h, INT(STEPS-2), INT(STEPS), INT(self.nvars), INT(5), INT(0))
                evaluate_step(self.queue, (self.global_size,), (self.local_size,), self.y, self.y4, self.y5, self.tau, self.delta, TOL, self.nvars)
                #print "antes del update"
                #self.print_array(self.y_host, self.y)
                update_variables(self.queue, (self.global_size,), (self.local_size,), self.y5, self.delta, self.tau, self.time, self.h, self.y, self.n_ok, self.n_bad, self.stop, TOL, self.hmax, self.final_omega, self.nvars)
#                print "despues del update"
#                self.print_array(self.y_host, self.y)
                self.nsteps += 1
                if(self.nsteps>=19000 and self.nsteps%1000==0):
                        print "paso {}".format(self.nsteps)
                        self.print_array(self.y_host, self.y)
            print "res"
            self.print_array(self.y_host, self.y)

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

