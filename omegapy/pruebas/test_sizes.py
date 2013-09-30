# coding: utf-8
import sys
import numpy as np
import pyopencl as cl

FLOAT = np.float32


class test_sizes:
    """
    Pequeño programa para testear como maneja opencl los work groups y work items
    """
    def __init__(self, batch=1, nvars=10):
        self.opencl_init()
        self.batch=batch
        self.nvars=nvars

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

    def load_data(self):
        """
        Creamos dos arrays, a y h, cada elemento de h se corresponde con un grupo
        de elementos de a. h podria ser, por ejemplo, el error de un calculo
        sobre un grupo de elementos sobre a.
        """
        a_host = np.array([1]*(self.batch*self.nvars), dtype=FLOAT)
        h_host = np.array(range(self.batch), dtype=FLOAT)
        mf = cl.mem_flags
        self.a = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_host)
        self.h = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_host)

    def do_something(self):
        
def main():
    test = test_sizes()
    test.load_data()
    test.do_something()


if __name__=="__main__":
    main()
    sys.exit(0)
