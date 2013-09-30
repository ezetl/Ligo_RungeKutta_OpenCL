# coding: utf-8
import pyopencl as cl
import numpy
import numpy.linalg as la

cl.get_platforms()
#Obtener plataforma
nvidia = cl.get_platforms()[0]
#Obtener placa de video
gforce = nvidia.get_devices()[0]
#Crear context
ctx = cl.Context([gforce])
#Crear queue
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

a = numpy.random.rand(10000000).astype(numpy.float32)
b = numpy.random.rand(10000000).astype(numpy.float32)

a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
prg = cl.Program(ctx, """
     __kernel void sum(__global const float *a,
         __global const float *b, __global float *c)
             {
                   int gid = get_global_id(0);
                         c[gid] = a[gid] + b[gid];
                             }
                                 """).build()
for i in range(1, 1000):
    prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)
a_plus_b = numpy.empty_like(a)
cl.enqueue_copy(queue, a_plus_b, dest_buf)

print(la.norm(a_plus_b - (a+b)), la.norm(a_plus_b))
