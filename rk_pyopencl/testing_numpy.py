import pyopencl as cl
import numpy
import numpy.linalg as la

a = numpy.random.rand(10000000).astype(numpy.float32)
b = numpy.random.rand(10000000).astype(numpy.float32)
c = numpy.zeros_like(a)

for i in range(1, 1000):
    c = numpy.add(a, b)

print(la.norm(c - (a+b)), la.norm(c))
