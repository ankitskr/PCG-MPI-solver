"""
from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = cp.arange(10, dtype='i')
recvbuf = cp.empty_like(sendbuf)
assert hasattr(sendbuf, '__cuda_array_interface__')
assert hasattr(recvbuf, '__cuda_array_interface__')
comm.Allreduce(sendbuf, recvbuf)

assert cp.allclose(recvbuf, sendbuf*size)


"""
#------------------------------------------------------------------


import numpy as np
import cupy as cp
import time

"""
Results for a single GPU device:
Single Precision (Float): ~300 MDOF, ~20 Speedup 
Double Precision (Float): ~210-220 MDOF, ~19 Speedup
Memory required for saving element matrices of 50M elements is ~230 GB in FEM
"""

Nele = 50000000
NDof = int(4*Nele)
Nedof= 26

npFloatType = np.float64
cpFloatType = cp.float64


### Numpy and CPU (Creating Data using CPU)
s = time.time()
K_cpu = np.random.rand(Nedof,Nedof).astype(npFloatType)
U_cpu = np.random.rand(NDof).astype(npFloatType)
EleDof_cpu = np.random.randint(0, NDof, size=[Nedof, Nele]).astype(np.uint32)
#EleSign_cpu = np.random.randint(0, 2, size=[Nedof, Nele]).astype(bool)
w_cpu = np.random.rand(Nele).astype(npFloatType)
N1_cpu = EleDof_cpu.ravel()
e = time.time()
print('t0_cpu', e - s)



### Numpy and CPU (Computing on CPU)
s = time.time()
Uele_cpu = U_cpu[EleDof_cpu]
F_cpu = np.dot(K_cpu, w_cpu*Uele_cpu)
A_cpu = np.bincount(N1_cpu, weights=F_cpu.ravel(), minlength=NDof)
e = time.time()
print('t1_cpu', e - s)




### CuPy and GPU (Transferring Data to GPU)
s = time.time()
U_gpu = cp.asarray(U_cpu)
K_gpu = cp.asarray(K_cpu)
EleDof_gpu = cp.asarray(EleDof_cpu)
#EleSign_gpu = cp.asarray(EleSign_cpu)
w_gpu = cp.asarray(w_cpu)
N1_gpu = cp.asarray(N1_cpu)
cp.cuda.Stream.null.synchronize()
e = time.time()
print('t0_gpu', e - s)




### CuPy and GPU (Computing on GPU)
s = time.time()
Uele_gpu = U_gpu[EleDof_gpu]
Uele_gpu *= w_gpu
F_gpu = cp.dot(K_gpu, Uele_gpu)
del Uele_gpu
B_gpu = cp.bincount(N1_gpu, weights=F_gpu.ravel(), minlength=NDof)
cp.cuda.Stream.null.synchronize()
e = time.time()
print('t1_gpu', e - s)

B_cpu = cp.asnumpy(B_gpu)
cp.cuda.Stream.null.synchronize()



print('ds', np.linalg.norm(A_cpu-B_cpu))