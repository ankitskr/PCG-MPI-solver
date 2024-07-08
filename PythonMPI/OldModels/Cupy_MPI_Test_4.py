import numpy as np
import cupy as cp
import time


for i in range(1,11):

    Nele = 2400000*i #polyhex needs 1/8 times Nele of Hex for similar accuracy
    NDof = int(22*Nele)
    Nedof= 78
        
    npFloatType = np.float32
    cpFloatType = cp.float32

    K_cpu = np.random.rand(Nedof,Nedof).astype(npFloatType)
    U_cpu = np.random.rand(NDof).astype(npFloatType)
    EleDof_cpu = np.random.randint(0, NDof, size=[Nedof, Nele]).astype(np.uint32)
    w_cpu = np.random.rand(Nele).astype(npFloatType)
    
    U_gpu = cp.asarray(U_cpu)
    K_gpu = cp.asarray(K_cpu)
    w_gpu = cp.asarray(w_cpu)
    EleDof_gpu = cp.asarray(EleDof_cpu)
    N1_gpu = EleDof_gpu.ravel()
    cp.cuda.Stream.null.synchronize()
  
    B_gpu = cp.zeros(NDof,dtype=cpFloatType)
    s = time.time()
    B_gpu += cp.bincount(N1_gpu, weights=cp.dot(K_gpu, w_gpu*U_gpu[EleDof_gpu]).ravel(), minlength=NDof)
    dt = time.time() - s
    dtdN = dt/i
    print('t1_gpu',i, np.round(dt,2), np.round(dtdN,2))   
        

cp.cuda.Stream.null.synchronize()
