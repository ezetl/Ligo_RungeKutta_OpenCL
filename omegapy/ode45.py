# coding: utf-8
from itertools import islice
import numpy as np
import pyopencl as cl


np.set_printoptions(suppress=True)
INT = np.int32
TOL = 1e-7  # Tolerance for solver
DELTA = -1e10
YINF = DELTA
STEPS = 7
INIT_COND = "init_cond.dat"
mf = cl.mem_flags
FLOAT = np.float32

class Ode45:
    """
    This class wraps everything related with setting up the opencl enviroment
    and running the algorithm in the devices.
    """

    def __init__(self, batch=1, nvars=10, initial_conditions=INIT_COND):
        assert (initial_conditions is not None) and \
            (initial_conditions != ""), \
            "You have to pass the file containing initial conditions."
        self.init_cond_file = INIT_COND
        self.init_cond = []
        self.batch = batch
        self.nvars = nvars
        self.global_size = self.batch * self.nvars
        self.local_size = nvars
        self.init_states_batchs = []  # saves the batchs with initial states
        self.opencl_init()

    def opencl_init(self):
        """
        Creates context, queue, etc.
        """
        global FLOAT
        # Get platform
        ### TODO: multiple devices
        self.platform = cl.get_platforms()[0]
        # Get GPU
        self.device = self.platform.get_devices()[0]
        # Use double precision if available
        if self.device.double_fp_config:
            FLOAT = np.float64
        else:
            FLOAT = np.float32
        # Create context
        self.ctx = cl.Context([self.device])
        # Create queue
        self.queue = cl.CommandQueue(self.ctx)

    def load_program(self, program):
        program_file = open(program, 'r')
        self.program = cl.Program(self.ctx, program_file.read()).build()
        program_file.close()

    def data_init(self):
        """
        This creates some arrays in the device with data used for further
        calculations.
        """
        # Dormand-Prince coefficients
        a = np.array([0., 0., 0., 0., 0., 0.,
                      1. / 5., 0., 0., 0., 0., 0.,
                      3. / 40., 9. / 40., 0., 0., 0., 0.,
                      44. / 45., -56. / 15., 32. / 9., 0., 0., 0.,
                      19372. / 6561., -25360. / 2187., 64448. / 6561.,
                      -212. / 729., 0., 0.,
                      9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176.,
                      -5103. / 18656., 0.,
                      35. / 384., 0., 500. / 1113., 125. / 192.,
                      -2187. / 6784., 11. / 84.], dtype=FLOAT)
        # 4th order b coeffs
        b4 = np.array([5179. / 57600., 7571. / 16695., 393. / 640.,
                      -92097. / 339200., 187. / 2100., 1. / 40.], dtype=FLOAT)
        # 5th order b coeffs
        b5_host = np.array([35. / 384., 500. / 1113., 125. / 192.,
                           -2187. / 6784., 11. / 84.], dtype=FLOAT)
        # auxiliar arrays
        self.k_host = np.zeros(shape=(self.global_size * STEPS,), dtype=FLOAT)
        self.ytemp_host = np.zeros(shape=(self.nvars,), dtype=FLOAT)
        # states, y4, and y5 are used for 4º and 5º order calculations
        self.y_host = np.zeros(shape=(self.global_size,), dtype=FLOAT)
        self.y4_host = np.zeros(shape=(self.global_size,), dtype=FLOAT)
        self.y5_host = np.zeros(shape=(self.global_size,), dtype=FLOAT)

        # mem flags, used to copy data to the device
        mf = cl.mem_flags

        # Some constants
        self.t1 = FLOAT(0)
        self.t2 = FLOAT(1e10)
        self.hmin = (self.t2 - self.t1) / 1e20
        self.hmax = FLOAT(10)
        self.final_omega = FLOAT(0.1)
        hh = (self.t2 - self.t1) / 100
        hh = min(self.hmax, hh)
        hh = max(self.hmin, hh)

        # Arrays for errors and other stuff
        self.tau_host = np.zeros(shape=(self.batch,), dtype=FLOAT)
        self.delta_host = np.array([DELTA] * self.batch, dtype=FLOAT)
        self.error_host = np.zeros(shape=(self.batch,), dtype=FLOAT)
        self.h_host = np.array([hh] * self.batch, dtype=FLOAT)
        time_host = np.array([self.t1] * self.batch, dtype=FLOAT)
        self.stop_host = np.zeros(shape=(self.batch,), dtype=INT)
        n_ok_host = np.zeros(shape=(self.batch,), dtype=INT)
        n_bad_host = np.zeros(shape=(self.batch,), dtype=INT)

        # Copy all arrays from host to device
        self.a = self.copy_to_device(mf.READ_ONLY, a)
        self.b4 = self.copy_to_device(mf.READ_ONLY, b4)
        self.b5 = self.copy_to_device(mf.READ_ONLY, b5_host)
        self.y = self.copy_to_device(mf.READ_WRITE, self.y_host)
        self.y4 = self.copy_to_device(mf.READ_WRITE, self.y4_host)
        self.y5 = self.copy_to_device(mf.READ_WRITE, self.y5_host)
        self.k = self.copy_to_device(mf.READ_WRITE, self.k_host)
        self.ytemp = self.copy_to_device(mf.READ_WRITE, self.ytemp_host)
        self.tau = self.copy_to_device(mf.READ_WRITE, self.tau_host)
        self.delta = self.copy_to_device(mf.READ_WRITE, self.delta_host)
        self.error = self.copy_to_device(mf.READ_WRITE, self.error_host)
        self.h = self.copy_to_device(mf.READ_WRITE, self.h_host)
        self.time = self.copy_to_device(mf.READ_WRITE, time_host)
        self.stop = self.copy_to_device(mf.READ_WRITE, self.stop_host)
        self.n_ok = self.copy_to_device(mf.READ_WRITE, n_ok_host)
        self.n_bad = self.copy_to_device(mf.READ_WRITE, n_bad_host)
        #After this,the buffers should be accesible from kernels

    def generate_init_cond(self, initial_conditions):
        """
        This method loads initial states from a file initial_conditions.
        """
        try:
            init_file = open(initial_conditions, "r")
            with open("init_cond.dat", "r") as f:
                while True:
                    next_line_slice = list(islice(f, self.batch))
                    if not next_line_slice:
                        break
                    next_line_slice = [float(x) for elem in next_line_slice
                                       for x in elem.replace('\n', '').split()]
                    next_line_slice = np.array(next_line_slice, dtype=FLOAT)
                    self.init_states_batchs.append(next_line_slice)
            init_file.close()
        except IOError:
            print("Initial conditions file {} does not \
                    exist.".format(initial_conditions))

    def execute(self):
        """
        This calls data_init and then executes the algorithm
        """
        self.data_init()
        assert(self.t2 > self.t1)
        assert(TOL > 0)
        self.nsteps = 0  # Number of steps used in the process
        self.generate_init_cond(INIT_COND)
        self.load_program("ode45.cl")

        # Set scalar arguments for OpenCL kernels
        check_step = self.program.check_step
        check_step.set_scalar_arg_dtypes([None] * 3 + [FLOAT, INT])

        f_rhs = self.program.f_rhs
        f_rhs.set_scalar_arg_dtypes([None] * 3 + [INT, INT, INT])

        rk_step = self.program.rk_step
        rk_step.set_scalar_arg_dtypes([None] * 5 + [INT, INT, INT, INT, INT])

        evaluate_step = self.program.evaluate_step
        evaluate_step.set_scalar_arg_dtypes([None] * 5 + [FLOAT, INT])

        update_variables = self.program.update_variables
        update_variables.set_scalar_arg_dtypes([None] * 9 +
                                               [FLOAT, FLOAT, FLOAT, INT])

        for state_batch in self.init_states_batchs:
            print "este es el batch"
            print state_batch
            # global_size can change if the last batch is smaller than the
            # original
            batch_size = min(self.batch, len(state_batch) / self.nvars)
            global_s = batch_size * self.nvars
            self.y = self.copy_to_device(mf.READ_WRITE, state_batch)
            # since global_size can change, so does stop, because it has one
            # cell per work group
            self.stop_host = np.zeros(shape=(batch_size,), dtype=INT)
            self.stop = self.copy_to_device(mf.READ_WRITE, self.stop_host)
            # count of steps for each batch
            # TODO: count of steps for each state, must use an array
            self.nsteps = 0
            while True:
                check_step(self.queue, (global_s,), (self.local_size,),
                           self.h, self.time, self.stop, self.t2, self.hmin)
                # This copy the stop array and check if we need to stop.
                # TODO: find a way to do this in gpu and avoid copying arrays
                # in every step (events?)
                stop = self.copy_array(self.stop_host, self.stop)

                if any(stop):
                    # TODO: separar esto para los errores y cond de terminacion
                    print("break")
                    print("steps {}".format(self.nsteps))
                    self.print_array(self.stop_host, self.stop)
                    break
                # Calculate f_rhs with initial values. The number 0 is because
                # we want to use the first portion of self.k array
                f_rhs(self.queue, (global_s,), (self.local_size,),
                      self.y, self.k, self.stop, self.nvars, STEPS, INT(0))
                for i in range(1, STEPS):  # cantidad de steps, es del 1 al 7
                    rk_step(self.queue, (global_s,),
                            (self.local_size,), self.ytemp, self.y, self.k,
                            self.a, self.h, i, STEPS, self.nvars, 6, 1)
                    f_rhs(self.queue, (global_s,), (self.local_size,),
                          self.ytemp, self.k, self.stop, self.nvars, STEPS, i)
                # 4º y 5º order
                rk_step(self.queue, (global_s,), (self.local_size,),
                        self.y4, self.y, self.k, self.b4, self.h,
                        INT(STEPS - 1), INT(STEPS), INT(self.nvars), INT(6),
                        INT(0))
                rk_step(self.queue, (global_s,), (self.local_size,),
                        self.y5, self.y, self.k, self.b5, self.h,
                        INT(STEPS - 2), INT(STEPS), INT(self.nvars), INT(5),
                        INT(0))
                evaluate_step(self.queue, (global_s,),
                              (self.local_size,), self.y, self.y4, self.y5,
                              self.tau, self.delta, TOL, self.nvars)
                update_variables(self.queue, (global_s,),
                                 (self.local_size,), self.y5, self.delta,
                                 self.tau, self.time, self.h, self.y,
                                 self.n_ok, self.n_bad, self.stop, TOL,
                                 self.hmax, self.final_omega, self.nvars)
                self.nsteps += 1
            print("res")
            self.print_array(state_batch, self.y)

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
        print(c)

    def copy_to_device(self, flags=mf.READ_WRITE, hostbuf=None):
        """
        This method copy a host buffer to device. The flags ar OR'ed with
        mf.COPY_HOST_PTR.
        """
        assert (hostbuf is not None)
        return cl.Buffer(self.ctx, flags | mf.COPY_HOST_PTR, hostbuf=hostbuf)
