# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:10:24 2020

@author: z5166762
"""

#Setting Threads = 1
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


#Setting Garbage collection
import gc
gc.collect()
gc.set_threshold(5600, 20, 20)
#gc.disable()


#Importing Libraries
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from time import time, sleep
import numpy as np
from numpy.linalg import norm
import os.path
import shutil

import pickle
import zlib

import mpi4py
from mpi4py import MPI

#mpi4py.rc.threads = False

#import logging
#from os.path import abspath
#from Cython_Array.Array_cy import apply_sum
from scipy.io import savemat
from GeneralFunc import configTimeRecData


#Defining Constants
eps = np.finfo(float).eps




def calcMPQ(MP_P, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData):        
    
    #Calculating Local Q Vector
    MP_NDOF = len(MP_P)    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList)    
    if QCalcMode in ['infor', 'inbin']:
        MP_LocQ = np.zeros(MP_NDOF, dtype=float)    
    elif QCalcMode == 'outbin':            
        Flat_ElemLocDof = np.zeros(NCount, dtype=int)
        Flat_ElemQ = np.zeros(NCount)
        I=0
        
    
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]        
        Ke = RefTypeGroup['ElemStiffMat']
        ElemList_LocDofVector = RefTypeGroup['ElemList_LocDofVector']
        ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
        ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
        ElemList_Level = RefTypeGroup['ElemList_Level']
        
        ElemList_P =   MP_P[ElemList_LocDofVector]
        ElemList_P[ElemList_SignVector] *= -1.0        
        ElemList_Q =   np.dot(Ke, ElemList_Level*ElemList_P)
        ElemList_Q[ElemList_SignVector] *= -1.0
        
        if QCalcMode == 'inbin':            
            MP_LocQ += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_Q.ravel(), minlength=MP_NDOF)
        elif QCalcMode == 'infor':
            apply_sum(ElemList_LocDofVector, MP_LocQ, ElemList_Q)            
        elif QCalcMode == 'outbin':
            N = len(ElemList_LocDofVector_Flat)
            Flat_ElemLocDof[I:I+N]=ElemList_LocDofVector_Flat
            Flat_ElemQ[I:I+N]=ElemList_Q.ravel()
            I += N
            
    if QCalcMode == 'outbin':    MP_LocQ = np.bincount(Flat_ElemLocDof, weights=Flat_ElemQ, minlength=MP_NDOF)
         
    
    #Calculating Overlapping Q Vectors
    MP_OvrlpQList = []
    N_NbrMP = len(MP_OvrlpLocalDofVecList)    
    for j in range(N_NbrMP):    MP_OvrlpQList.append(MP_LocQ[MP_OvrlpLocalDofVecList[j]])    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping Q
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        SendReq     = Comm.Isend(MP_OvrlpQList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpQList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
     
    #Calculating Q
    MP_Q = MP_LocQ
    for j in range(N_NbrMP):        
        MP_Q[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpQList[j]    
    updateTime(MP_TimeRecData, 'dT_Calc')
     
    return MP_Q


    

def MPI_SUM(MP_RefVar, MP_TimeRecData):
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    Glob_RefVar = Comm.allreduce(MP_RefVar, op=MPI.SUM)
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    return Glob_RefVar

    
    

def updateTime(MP_TimeRecData, Ref):
    
    t1 = time()
    MP_TimeRecData[Ref] += t1 - MP_TimeRecData['t0']
    MP_TimeRecData['t0'] = t1
    
    


if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    #-------------------------------------------------------------------------------
    #print(Initializing ModelData..')
    MP_TimeRecData = {'dT_FileRead':   0.0,
                      'dT_Calc':        0.0,
                      'dT_CommWait':    0.0,
                      't0':            time()}
    
                         
    p                   = int(np.log2(N_Workers)) #N_MP = 2**p
    ModelName           = sys.argv[1]
    ScratchPath         = sys.argv[2] 
    R0                  = sys.argv[3]
    
    #Creating directories
    ResultPath          = ScratchPath + 'Results_Run' + str(R0) + '/'
    if Rank==0:    
        if not os.path.exists(ResultPath):            
            os.makedirs(ResultPath)        
    
    #Reading Model Data Files
    OutputFileName                      = ResultPath + ModelName + '_MP' +  str(p)
    PyDataPath                          = ScratchPath + 'ModelData/' + 'MP' +  str(p)  + '/'
    RefMeshPart_FileName                = PyDataPath + str(Rank) + '.zpkl'
    Cmpr_RefMeshPart                    = open(RefMeshPart_FileName, 'rb').read()
    RefMeshPart                         = pickle.loads(zlib.decompress(Cmpr_RefMeshPart))
    
    MP_SubDomainData                     = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector                     = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList              = RefMeshPart['OvrlpLocalDofVecList']
    MP_WeightVector                      = RefMeshPart['WeightVector']
    MP_NDOF                              = RefMeshPart['NDOF']
    MP_RefLoadVector                     = RefMeshPart['RefLoadVector']
    MP_FixedLocalDofVector               = RefMeshPart['FixedLocalDofVector']
    MP_InvLumpedMassVector               = RefMeshPart['InvLumpedMassVector']
    MPList_NDofVec                      = RefMeshPart['MPList_NDofVec']
    GathDofVector                       = RefMeshPart['GathDofVector']
    GlobData                            = RefMeshPart['GlobData']
    
    GlobNDOF                            = GlobData['GlobNDOF']
    ExportFlag                          = GlobData['ExportFlag']
    #MaxIter                             = GlobData['MaxIter']    
    #Tol                                 = GlobData['Tol']
    #QCalcMode                           = GlobData['QCalcMode'].lower()
    
    Tol = 1e-6
    MaxIter = 10000
    QCalcMode = 'outbin'
    
    if not QCalcMode in ['inbin', 'infor', 'outbin']:  raise ValueError("QCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    if Rank==0:   print('GlobNDOF', GlobNDOF)
    
    #Checking Tolerance
    if Tol < eps:        
        raise Warning('PCG : Too Small Tolerance')
        Tol = eps        
    elif Tol > 1:        
        raise Warning('PCG : Too Big Tolerance')
        Tol = 1 - eps
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    #Initializing Variables
    N_NbrMP                             = len(MP_NbrMPIdVector)
    N_OvrlpLocalDofVecList              = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    MP_InvOvrlpQList                    = [np.zeros(N_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    N_NbrDof                            = np.sum(N_OvrlpLocalDofVecList)
    NCount                              = 0
    N_Type                              = len(MP_SubDomainData['StrucDataList'])
    for j in range(N_Type):             NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
    
    """
    Ref: https://www.cs.cornell.edu/~bindel/class/cs6210-f16/lec/2016-10-17.pdf
    """
       
    #Initial Settings    
    Flag                                        = 1
    Iter                                        = 0 
    
    #Initializing Variables
    MP_V                                        = np.ones(MP_NDOF)/np.sqrt(GlobNDOF)
    MP_V[MP_FixedLocalDofVector]                = 0.0
    
    #A = InvM*K
    MP_Q                                       = calcMPQ(MP_V, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
    MP_AV                                      = MP_InvLumpedMassVector*MP_Q
    
    MP_SqAV                                    = np.dot(MP_AV, MP_AV*MP_WeightVector)
    NormAV                                     = np.sqrt(MPI_SUM(MP_SqAV, MP_TimeRecData))

    MP_Lambda                                   = np.dot(MP_V, MP_AV*MP_WeightVector)
    Lambda                                      = MPI_SUM(MP_Lambda, MP_TimeRecData)
    
    if Rank == 0:   
        ResVec                       = np.zeros(GlobNDOF+1)   
        print('Mode', QCalcMode)
    
    
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')
    
    
    for i in range(MaxIter):
        
        MP_V                                        = MP_AV/NormAV
        MP_V[MP_FixedLocalDofVector]                = 0.0
        
        MP_Q                                        = calcMPQ(MP_V, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
        MP_AV                                       = MP_InvLumpedMassVector*MP_Q
        
        
        MP_SqAV                                     = np.dot(MP_AV, MP_AV*MP_WeightVector)
        NormAV                                      = np.sqrt(MPI_SUM(MP_SqAV, MP_TimeRecData))
        
        Lambda_1                                    = Lambda
        MP_Lambda                                   = np.dot(MP_V, MP_AV*MP_WeightVector)
        Lambda                                      = MPI_SUM(MP_Lambda, MP_TimeRecData)
        
        NormR                                       = np.abs((Lambda - Lambda_1)/Lambda)
        
        if Rank==0:
            if i%20==0:    print(i, [NormR, Tol], Lambda)            
            ResVec[i]                               = NormR

        if NormR <= Tol:
            
            Flag                                    = 0
            Iter                                    = i
            break
            
        
    
    #Truncate the zeros from resvec
    if Rank == 0:            
        ResVec     = ResVec[:i+1]
        
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    #Gathering Results
    SendReq = Comm.Isend(MP_V, dest=0, tag=Rank)    
    if Rank == 0:        
        N_MeshParts                 = len(MPList_NDofVec)
        MPList_V                     = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]        
        for j in range(N_Workers):  Comm.Recv(MPList_V[j], source=j, tag=j)            
        V                             = np.zeros(GlobNDOF, dtype=float)
        V[GathDofVector]             = np.hstack(MPList_V)        
    SendReq.Wait()
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('Iter', Iter) 
        print('Lambda', Lambda)
        print('tcr', 2*np.pi/Lambda**0.5)
    
    #Saving CPU Time
    MP_TimeRecData['dT_Total_Verify'] = t0_End - t0_Start 
    MP_TimeRecData['t0_Start'] = t0_Start
    MP_TimeRecData['t0_End'] = t0_End
    MP_TimeRecData['MP_NCount'] = NCount
    MP_TimeRecData['MP_NDOF'] = MP_NDOF
    MP_TimeRecData['N_NbrDof'] = N_NbrDof    
    MPList_TimeRecData = Comm.gather(MP_TimeRecData, root=0)
    
    if Rank == 0:        
        TimeData = configTimeRecData(MPList_TimeRecData)
        np.savez_compressed(OutputFileName +  '_' + QCalcMode, TimeData = TimeData)
    
        #Exporting Results
        if ExportFlag == 1:        
            ExportData = {'V': V, 'Lambda': Lambda, 'Flag': Flag, 'Iter': Iter, 'ResVec': ResVec, 'TimeData': TimeData}
            savemat(OutputFileName+'.mat', ExportData)
        
        
    
       
       

