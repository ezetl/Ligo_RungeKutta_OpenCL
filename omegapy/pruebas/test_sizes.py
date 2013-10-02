# coding: utf-8
import sys
import numpy as np
import pyopencl as cl

FLOAT = np.float32
INT = np.int32


class test_sizes:
    """
    Pequeño programa para testear como maneja opencl los work groups y work items
    """
    def __init__(self, batch=1, nvars=10):
        self.opencl_init()
        self.batch=batch
        self.nvars=nvars

        self.global_size=batch * nvars
        self.local_size=nvars

    def opencl_init(self):
        """
        Crear context, queue, etc.
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

    def load_data(self):
        """
        Creamos dos arrays, a y h, cada elemento de h se corresponde con un grupo
        de elementos de a. h podria ser, por ejemplo, el error de un calculo
        sobre un grupo de elementos sobre a.
        """
        #array de todos 1
        self.a_host = np.array([1]*(self.batch*self.nvars), dtype=INT)

        self.h_host = np.array(range(self.batch), dtype=INT)
        self.res_host = np.zeros(shape=(self.batch*self.nvars), dtype=INT)
        mf = cl.mem_flags
        self.a = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.a_host)
        self.h = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_host)
        self.res = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.res_host)

    def do_something(self):
        self.program.test(self.queue, (self.global_size,), (self.local_size,), self.a, self.h, self.res)
        self.print_array(self.a_host, self.a)
        self.print_array(self.h_host, self.h)
        self.print_array(self.res_host, self.res)

    def print_array(self, arr_like, arr_device):
        """
        This copy an array from device and print.
        arr_like must be an array in host with the same style arr_device
        arr_device is the array that is currently in device
        """
        c = np.empty_like(arr_like)
        cl.enqueue_read_buffer(self.queue, arr_device, c).wait()
        print c

def main():
    test = test_sizes(batch=10, nvars=13)
    test.load_program("test.cl")
    test.load_data()
    test.do_something()


if __name__=="__main__":
    main()
    sys.exit(0)
