

import numpy as np
import cupy as cp
import time


#Hex
"""
Nele = 4000000
NDof = int(3.5*Nele)
Nedof= 24
NTypes = 5
NT=10
"""

#Polyhex
Nele = 100000 #polyhex needs 1/8 times Nele of Hex for similar accuracy
NDof = int(22*Nele)
Nedof= 78
NTypes = 5
NT = 10

npFloatType = np.float64
cpFloatType = cp.float64


### Numpy and CPU (Creating Data using CPU)
s = time.time()
K_cpu = np.random.rand(Nedof,Nedof).astype(npFloatType)
U_cpu = np.random.rand(NDof).astype(npFloatType)
EleDof_cpu = np.random.randint(0, NDof, size=[Nedof, Nele]).astype(np.uint32)
w_cpu = np.random.rand(Nele).astype(npFloatType)
N1_cpu = EleDof_cpu.ravel()
e = time.time()
#print('t0_cpu', e - s)



### Numpy and CPU (Computing on CPU)

s = time.time()
for t in range(NT):
    A_cpu = np.zeros(NDof, dtype=npFloatType)
    for i in range(NTypes): 
        A_cpu += np.bincount(N1_cpu, weights=np.dot(K_cpu, w_cpu*U_cpu[EleDof_cpu]).ravel(), minlength=NDof)
print('t1_cpu', time.time() - s)




### CuPy and GPU (Transferring Data to GPU)
s = time.time()

#Enable unified memory
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)


U_gpu = cp.asarray(U_cpu)
K_gpu = cp.asarray(K_cpu)
w_gpu = cp.asarray(w_cpu)
EleDof_gpu = cp.asarray(EleDof_cpu)

K_gpuList = [cp.array(K_gpu, copy=True) for i in range(NTypes)]
w_gpuList = [cp.array(w_gpu,copy=True) for i in range(NTypes)]
EleDof_gpuList = [cp.array(EleDof_gpu, copy=True) for i in range(NTypes)]
N1_gpuList = [EleDof_gpuList[i].ravel() for i in range(NTypes)]

del K_gpu, w_gpu, EleDof_gpu

cp.cuda.Stream.null.synchronize()
e = time.time()
#print('t0_gpu', e - s)




### CuPy and GPU (Computing on GPU)
    

#In-bin
s = time.time()
for t in range(NT):
    B_gpu = cp.zeros(NDof,dtype=cpFloatType)
    for i in range(NTypes): 
        K_gpu = K_gpuList[i]
        w_gpu = w_gpuList[i]
        EleDof_gpu = EleDof_gpuList[i]
        N1_gpu = N1_gpuList[i]
        
        B_gpu += cp.bincount(N1_gpu, weights=cp.dot(K_gpu, w_gpu*U_gpu[EleDof_gpu]).ravel(), minlength=NDof)
        
print('t1_gpu0', time.time() - s)    
    

#Out-bin
N1_Flat_gpu = cp.hstack(N1_gpuList)
NCount = len(N1_Flat_gpu)
        
s = time.time()
for t in range(NT):
    Flat_B = cp.zeros(NCount, dtype=cpFloatType)
    I=0
    for i in range(NTypes): 
        K_gpu = K_gpuList[i]
        w_gpu = w_gpuList[i]
        EleDof_gpu = EleDof_gpuList[i]
        N1_gpu = N1_gpuList[i]
        
        N = len(N1_gpu)
        Flat_B[I:I+N]=K_gpu.dot(w_gpu*U_gpu[EleDof_gpu]).ravel()
        I += N
            
    B_gpu0 = np.bincount(N1_Flat_gpu, weights=Flat_B, minlength=NDof)

print('t1_gpu1', time.time() - s)    

print('ds_outbin', cp.linalg.norm(B_gpu-B_gpu0))
   
cp.cuda.Stream.null.synchronize()
