# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:33:01 2020

@author: z5166762
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from time import time, sleep
import numpy as np
import mpi4py
from mpi4py import MPI
from numpy.random import rand

"""
def calcFint(MP_U, ElemDofVectorList, ElemDofVectorList_Flat, Ke, ElemList_Level, MP_NDOF):
    ElemList_U =   MP_U[ElemDofVectorList] 
    ElemList_Fint =   np.dot(Ke, ElemList_Level*ElemList_U)    
    MP_FintVec = np.zeros(MP_NDOF)
    MP_FintVec += np.bincount(ElemDofVectorList_Flat, weights=ElemList_Fint.ravel(), minlength=MP_NDOF)    
    return MP_FintVec*1e-4
    
    
def calcU(MP_U, MP_FintVec, MP_FextVec):
    MP_U = MP_U + 0.5*(MP_FextVec - MP_FintVec)
    return MP_U
    

    
def calcFint0(MP_U, ElemDofVectorList, Ke):

    ElemList_U =   MP_U[ElemDofVectorList] 
    ElemList_Fint =   np.dot(Ke, ElemList_U)   
    
    return ElemList_Fint, ElemList_U
    
"""

Comm = MPI.COMM_WORLD
N_Workers = Comm.Get_size()
Rank = Comm.Get_rank()
p = int(np.log2(N_Workers)) #N_MP = 2**p
if Rank==0:    print('N_Workers', N_Workers)


ModelName =         sys.argv[1]
ScratchPath =       sys.argv[2]    
MeshSizeFac =       int(sys.argv[3])
Run =               int(sys.argv[4])

#Creating directories
ResultPath = ScratchPath + 'Results/'
OutputFileName = ResultPath + ModelName
if Rank==0:    
    if not os.path.exists(ResultPath):   
        os.makedirs(ResultPath)        

Total_NElem = 2**MeshSizeFac
Total_NDOF = Total_NElem*3.2
MP_NElem = int(Total_NElem/N_Workers)
Elem_NDof    = 24
MP_NDOF      = int(MP_NElem*3.2)

Ke = np.random.rand(Elem_NDof, Elem_NDof)
#ElemList_Level = np.random.rand(MP_NElem)
ElemDofVectorList = np.random.randint(0, MP_NDOF, [Elem_NDof,MP_NElem])
#ElemDofVectorList_Flat = ElemDofVectorList.ravel()
#MP_FextVec = np.random.rand(MP_NDOF)
MP_U = np.random.rand(MP_NDOF)

t0 = time()
for t in range(1000):
    #MP_FintVec = calcFint(MP_U, ElemDofVectorList, ElemDofVectorList_Flat, Ke, ElemList_Level, MP_NDOF)
    #MP_U = calcU(MP_U, MP_FintVec, MP_FextVec)    
    
    
    #ElemList_Fint, ElemList_U = calcFint0(MP_U, ElemDofVectorList, Ke) 

    ElemList_U =   MP_U[ElemDofVectorList] 
    ElemList_Fint =   np.dot(Ke, ElemList_U) 
    MP_U *= 0.999
    
CalcTime = time()-t0
    
 
VarSizeData = {}
VarSizeData['Ke']                     = N_Workers*Ke.nbytes
VarSizeData['ElemDofVectorList']     = N_Workers*ElemDofVectorList.nbytes
VarSizeData['MP_U']                 = N_Workers*MP_U.nbytes  
VarSizeData['ElemList_Fint']         = N_Workers*ElemList_Fint.nbytes
VarSizeData['ElemList_U']             = N_Workers*ElemList_U.nbytes
    
MPList_CalcTime = Comm.gather(CalcTime , root=0)

if Rank == 0:          

    Mean_CalcTime     = np.mean(MPList_CalcTime)
    TimeData = {'Mean_CalcTime': Mean_CalcTime,
                'Total_NElem':  Total_NElem,
                'Total_NDOF':   Total_NDOF}
    
    print('TimeData', TimeData)
    np.savez_compressed(OutputFileName + '_MP' +  str(p) + '_Sz' + str(MeshSizeFac) + '_R' + str(Run), TimeData = TimeData, VarSizeData = VarSizeData)
    
    print(max(MP_U), min(MP_U))   
