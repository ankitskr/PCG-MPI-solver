"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from mpi4py import MPI
# import cupy as cp

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
Single Precision (Float): ~400-420 MDOF, 20-21 Speedup 
Double Precision (Float): ~240-250 MDOF, 19-20 Speedup
"""

Nele = 5000000
NDof = int(4*Nele)
Nedof= 24
NTypes = 10

npFloatType = np.float32
cpFloatType = cp.float32


### Numpy and CPU (Creating Data using CPU)
s = time.time()
K_cpu = np.random.rand(Nedof,Nedof).astype(npFloatType)
U_cpu = np.random.rand(NDof).astype(npFloatType)
EleDof_cpu = np.random.randint(0, NDof, size=[Nedof, Nele]).astype(np.uint32)
w_cpu = np.random.rand(Nele).astype(npFloatType)
N1_cpu = EleDof_cpu.ravel()
e = time.time()
print('t0_cpu', e - s)



### Numpy and CPU (Computing on CPU)

A_cpu = np.zeros(NDof, dtype=npFloatType)

s = time.time()
for i in range(NTypes): 
    A_cpu += np.bincount(N1_cpu, weights=np.dot(K_cpu, w_cpu*U_cpu[EleDof_cpu]).ravel(), minlength=NDof)
e = time.time()
print('t1_cpu', e - s)





### CuPy and GPU (Transferring Data to GPU)
s = time.time()

U_gpu = cp.array(U_cpu, copy=True)
K_gpu = cp.array(K_cpu, copy=True)
w_gpu = cp.array(w_cpu, copy=True)
EleDof_gpu = cp.array(EleDof_cpu, copy=True)

K_gpuList = [K_gpu*cp.random.rand(1) for i in range(NTypes)]
w_gpuList = [w_gpu*cp.random.rand(1) for i in range(NTypes)]
EleDof_gpuList = [cp.array(EleDof_gpu, copy=True) for i in range(NTypes)]
N1_gpuList = [EleDof_gpuList[i].ravel() for i in range(NTypes)]

del K_gpu, w_gpu, EleDof_gpu

cp.cuda.Stream.null.synchronize()
e = time.time()
print('t0_gpu', e - s)




### CuPy and GPU (Computing on GPU)

#Testing parallel streams for octree types (Doesn't seem to work) ----------------------------
#https://github.com/cupy/cupy/blob/master/examples/stream/map_reduce.py
device = cp.cuda.Device()
memory_pool = cp.cuda.MemoryPool()
cp.cuda.set_allocator(memory_pool.malloc)

s = time.time()
b_gpu = []
map_streams = []
stop_events = []
reduce_stream = cp.cuda.stream.Stream()
for i in range(NTypes):
    map_streams.append(cp.cuda.stream.Stream())

for i in range(NTypes):   
    stream = map_streams[i]
    
    K_gpu = K_gpuList[i]
    w_gpu = w_gpuList[i]
    EleDof_gpu = EleDof_gpuList[i]
    N1_gpu = N1_gpuList[i]
    
    Uele_gpu = cp.array(U_gpu[EleDof_gpu], copy=True)
    
    with stream:
        B_gpu_i = cp.bincount(N1_gpu, weights=cp.dot(K_gpu, w_gpu*Uele_gpu).ravel(), minlength=NDof)
        b_gpu.append(B_gpu_i)
    stop_event = stream.record()
    stop_events.append(stop_event)


# Block the `reduce_stream` until all events occur. This does not block host.
# This is not required when reduction is performed in the default (Stream.null) stream unless streams are created with `non_blocking=True` flag.
for i in range(NTypes):
    reduce_stream.wait_event(stop_events[i])
   
#Reduce
with reduce_stream:
    B_gpu = cp.sum(cp.asarray(b_gpu), axis=0)  

print('t1_gpu11', time.time() - s)
device.synchronize()

#print('total GB', memory_pool.total_bytes()/1024**3)
for stream in map_streams: # Free all blocks in the memory pool of streams
    memory_pool.free_all_blocks(stream=stream)    
    
    
    
    
    
#Testing for-loop for octree types -----------------------------------------------------------
s = time.time()
B_gpu = cp.zeros(NDof,dtype=cpFloatType)
for i in range(NTypes): 
 
    K_gpu = K_gpuList[i]
    w_gpu = w_gpuList[i]
    EleDof_gpu = EleDof_gpuList[i]
    N1_gpu = N1_gpuList[i]
    Uele_gpu = U_gpu[EleDof_gpu]
    
    B_gpu += cp.bincount(N1_gpu, weights=cp.dot(K_gpu, w_gpu*Uele_gpu).ravel(), minlength=NDof)
    
print('t1_gpu12', time.time() - s)    
    

    
    
    
    
    

s = time.time()
B_cpu = cp.asnumpy(B_gpu)
print('t1_gpu2', time.time() - s)

s = time.time()
cp.cuda.Stream.null.synchronize()
print('t1_gpu3', time.time() - s)


print('ds', np.linalg.norm(A_cpu-B_cpu))