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


#Defining Constants
eps = np.finfo(float).eps




def calcMPQ(MP_P, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList):
    
    t0 = time()
    dtcalc = 0.0
    dtcommwait = 0.0
    
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
    dtcalc,t0 = updateTime(dtcalc,t0)
    
    
    """
    #Communicating Overlapping Q
    RecvReqList = []
    for j in range(N_NbrMP):    
        NbrMP_Id     = MP_NbrMPIdVector[j]
        RecvReq     = Comm.Irecv([MP_InvOvrlpQList[j], MPI.DOUBLE], source=NbrMP_Id, tag=NbrMP_Id)
        RecvReqList.append(RecvReq)
        
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        Comm.Send([MP_OvrlpQList[j], MPI.DOUBLE], dest=NbrMP_Id, tag=Rank)
        
    MPI.Request.Waitall(RecvReqList)    
    dtcommwait,t0 = updateTime(dtcommwait,t0)
    """
    
    """
    #Communicating Overlapping Q
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        SendReq     = Comm.Isend([MP_OvrlpQList[j], MPI.DOUBLE], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id = MP_NbrMPIdVector[j]
        Comm.Recv([MP_InvOvrlpQList[j], MPI.DOUBLE], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    dtcommwait,t0 = updateTime(dtcommwait,t0)
    """
    
    """
    #Communicating Overlapping Q
    RecvReqList = []
    for j in range(N_NbrMP):    
        NbrMP_Id     = MP_NbrMPIdVector[j]
        RecvReq     = Comm.Irecv(MP_InvOvrlpQList[j], source=NbrMP_Id, tag=NbrMP_Id)
        RecvReqList.append(RecvReq)
        
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        Comm.Send(MP_OvrlpQList[j], dest=NbrMP_Id, tag=Rank)
        
    MPI.Request.Waitall(RecvReqList)    
    dtcommwait,t0 = updateTime(dtcommwait,t0)
    """
    
    
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
    dtcommwait,t0 = updateTime(dtcommwait,t0)
    
    
    #Calculating Q
    MP_Q = MP_LocQ
    for j in range(N_NbrMP):        
        MP_Q[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpQList[j]    
    dtcalc,t0 = updateTime(dtcalc,t0)
    
    return MP_Q, dtcalc, dtcommwait, t0




def updateTime(dtRef,t0):
    
    t1 = time()
    dtRef += t1-t0
    t0 = t1
    
    return dtRef, t0
    


if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    #-------------------------------------------------------------------------------
    #print(Initializing ModelData..')
    
    dT_FileRead         = 0.0
    dT_Calc             = 0.0
    dT_CommWait         = 0.0    
    t0 = time()
    
    p                   = int(np.log2(N_Workers)) #N_MP = 2**p
    ModelName           = sys.argv[1]
    ScratchPath         = sys.argv[2] 
    
    #Creating directories
    ResultPath          = ScratchPath + 'Results/'
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
    MP_InvDiagPreCondVector0             = RefMeshPart['InvDiagPreCondVector0']
    MP_InvDiagPreCondVector1             = RefMeshPart['InvDiagPreCondVector1']
    MP_DispConstraintVector              = RefMeshPart['DispConstraintVector']
    MP_X0                                = RefMeshPart['X0']
    MPList_NDofVec                      = RefMeshPart['MPList_NDofVec']
    GathDofVector                       = RefMeshPart['GathDofVector']
    GlobData                            = RefMeshPart['GlobData']
    
    GlobNDOF                            = GlobData['GlobNDOF']
    ExportFlag                          = GlobData['ExportFlag']
    PlotFlag                             = GlobData['PlotFlag']
    RefDofIdList                         = GlobData['RefDofIdList']
    MaxIter                             = GlobData['MaxIter']    
    Tol                                 = GlobData['Tol']
    ExistDP0                            = GlobData['ExistDP0']
    ExistDP1                            = GlobData['ExistDP1']
    QCalcMode                           = GlobData['QCalcMode'].lower()
    
    if not QCalcMode in ['inbin', 'infor', 'outbin']:  raise ValueError("QCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    #Checking Tolerance
    if Tol < eps:        
        raise Warning('PCG : Too Small Tolerance')
        Tol = eps        
    elif Tol > 1:        
        raise Warning('PCG : Too Big Tolerance')
        Tol = 1 - eps
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    dT_FileRead,t0 = updateTime(dT_FileRead,t0)    
    t0_Start = time()
    
    #Initializing Variables
    N_NbrMP                             = len(MP_NbrMPIdVector)
    N_OvrlpLocalDofVecList              = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    MP_InvOvrlpQList                     = [np.zeros(N_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    
    NCount                              = 0
    N_Type                              = len(MP_SubDomainData['StrucDataList'])
    for j in range(N_Type):             NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
    
    #Converting Dirichlet BC  into Newmann BC
    MP_X                                = np.zeros(MP_NDOF, dtype=float)
    MP_X[MP_FixedLocalDofVector]         = MP_DispConstraintVector
    MP_FintFixed, d1, d2, d3            = calcMPQ(MP_X, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList)
    MP_RefLoadVector                    -= MP_FintFixed
    
    #Initializing Variables
    MP_X                                 = MP_X0
    MP_X[MP_FixedLocalDofVector]          = 0.0
    MP_XMin                              = MP_X                                  #Iterate which has minimal residual so far
    MP_SqRefLoad                         = np.dot(MP_RefLoadVector, MP_RefLoadVector*MP_WeightVector)
    dT_Calc,t0                          = updateTime(dT_Calc,t0)
    SqRefLoad                           = Comm.allreduce(MP_SqRefLoad, op=MPI.SUM)
    dT_CommWait,t0                      = updateTime(dT_CommWait,t0)            
    NormRefLoadVector                   = np.sqrt(SqRefLoad) #n2b
    TolB                                = Tol*NormRefLoadVector
    
    #Check for all zero right hand side vector => all zero solution
    if NormRefLoadVector == 0:                      # if rhs vector is all zeros        
        X                               = np.zeros(GlobNDOF, dtype=float);  # then  solution is all zeros
        Flag                            = 0;                                   # a valid solution has been obtained
        RelRes                          = 0;                                   # the relative residual is actually 0/0
        Iter                            = 0;                                   # no iterations need be performed
        ResVec                          = [0];                               # resvec(1) = norm(b-A*x) = norm(0)
        
        #Exporting Files
        if ExportFlag == 1:        
            ExportData = {'X': X, 'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec}
            savemat(OutputFileName+'.mat', ExportData)        
        exit()
    
    #Initial Settings    
    Flag                                = 1
    Rho                                 = 1.0 #Dummy
    Stag                                = 0  # stagnation of the method
    MoreSteps                           = 0
    MaxStagSteps                        = 3
    MaxMSteps                           = min([int(GlobNDOF/50),5,GlobNDOF-MaxIter]);
    iMin                                = 0
    Iter                                = 0 
    
    MP_RefLoadVector0, d1, d2, d3        = calcMPQ(MP_X, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList)
    MP_R                                 = MP_RefLoadVector - MP_RefLoadVector0
    MP_R[MP_FixedLocalDofVector]          = 0.0    
    MP_SqR                               = np.dot(MP_R, MP_R*MP_WeightVector)
    dT_Calc,t0                          = updateTime(dT_Calc,t0)    
    NormR                               = np.sqrt(Comm.allreduce(MP_SqR, op=MPI.SUM))
    dT_CommWait,t0                          = updateTime(dT_CommWait,t0)
    NormRMin                            = NormR
    
    #Checking if the initial guess is a good enough solution
    if NormR <= TolB:
        Flag                            = 0;
        RelRes                          = NormR/NormRefLoadVector;
        Iter                            = 0;
        ResVec                          = [NormR];
        
        #Exporting Files
        if ExportFlag == 1:        
            ExportData = {'X': X, 'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec}
            savemat(OutputFileName+'.mat', ExportData)        
        exit()
    
    if Rank == 0:   
        ResVec                       = np.zeros(GlobNDOF+1)        
        ResVec[0]                    = NormR    
        
        print('Mode', QCalcMode)
        print(0, [NormR, TolB], [Stag, MaxStagSteps])
                
    
    
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')
    
    for i in range(MaxIter):
        
        #Calculating Z
        if ExistDP0: #Diagonal Preconditioner DP0            
            MP_Y                         = MP_InvDiagPreCondVector0*MP_R
            if np.any(np.isinf(MP_Y)):                
                Flag = 2
                break        
        else:    MP_Y                     = MP_R
            
        if ExistDP1: #Diagonal Preconditioner DP1            
            MP_Z                         = MP_InvDiagPreCondVector1*MP_Y
            if np.any(np.isinf(MP_Z)):                
                Flag = 2
                break        
        else:    MP_Z                     = MP_Y
        
        #Calculating Rho
        Rho_1                           = Rho
        MP_Rho                           = np.dot(MP_Z,MP_R*MP_WeightVector)
        dT_Calc,t0                      = updateTime(dT_Calc,t0)
        Rho                             = Comm.allreduce(MP_Rho, op=MPI.SUM)
        dT_CommWait,t0                  = updateTime(dT_CommWait,t0)
        if Rho == 0 or np.isinf(Rho):
            Flag = 4
            break
        
        #Calculating P and Beta
        if i == 0:
            MP_P                         = MP_Z
        else:
            Beta                        = Rho/Rho_1   
            if Beta == 0 or np.isinf(Beta):
                Flag = 4
                break
            MP_P                            = MP_Z + Beta*MP_P         
            
        #Calculating Q
        dT_Calc,t0                      = updateTime(dT_Calc,t0)
        MP_Q, dtcalc, dtcommwait, t0     = calcMPQ(MP_P, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList)
        dT_Calc                         += dtcalc
        dT_CommWait                     += dtcommwait
        
        #Calculating PQ and Alpha
        MP_PQ                            = np.dot(MP_P,MP_Q*MP_WeightVector)
        dT_Calc,t0                      = updateTime(dT_Calc,t0)
        PQ                              = Comm.allreduce(MP_PQ, op=MPI.SUM)
        dT_CommWait,t0                  = updateTime(dT_CommWait,t0)
        if PQ <= 0 or np.isinf(PQ):
            Flag = 4
            break
        else:    Alpha                   = Rho/PQ    
        if np.isinf(Alpha):
            Flag = 4
            break
        
        #Calculating X
        MP_X                            += Alpha*MP_P
        MP_X[MP_FixedLocalDofVector]     = 0.0
        
        #Calculating R
        MP_R                            -= Alpha*MP_Q
        MP_R[MP_FixedLocalDofVector]     = 0.0
        
        #Calculating Convergence Variables
        MP_SqP                           = np.dot(MP_P, MP_P*MP_WeightVector)
        MP_SqX                           = np.dot(MP_X, MP_X*MP_WeightVector)
        MP_SqR                           = np.dot(MP_R, MP_R*MP_WeightVector)
        dT_Calc,t0                      = updateTime(dT_Calc,t0)
        SqVars                          = Comm.allreduce(np.array([MP_SqP, MP_SqX, MP_SqR]), op=MPI.SUM)
        dT_CommWait,t0                  = updateTime(dT_CommWait,t0)
        [NormP, NormX, NormR]           = np.sqrt(SqVars)
        NormR_Act                       = NormR
        
        #Stagnation
        if (NormP*abs(Alpha) < eps*NormX):  Stag += 1
        else:                               Stag = 0
        
        if Rank==0:
            #savemat(OutputFileName+'_Log.mat', {'NormR':NormR, 'TolB': TolB, 'i': i})
            if i%1==0:    print(i, [NormR, TolB], [Stag, MaxStagSteps], [dT_CommWait, dT_Calc])            
            ResVec[i+1]                    = NormR_Act

        if NormR <= TolB or Stag >= MaxStagSteps or MoreSteps > 0:
            MP_Fint, d1, d2, d3              = calcMPQ(MP_X, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList)
            MP_R                             = MP_RefLoadVector - MP_Fint
            MP_R[MP_FixedLocalDofVector]      = 0.0
            MP_SqR_Act                       = np.dot(MP_R, MP_R*MP_WeightVector)
            dT_Calc,t0                      = updateTime(dT_Calc,t0)
            NormR_Act                       = np.sqrt(Comm.allreduce(MP_SqR_Act, op=MPI.SUM))
            dT_CommWait,t0                  = updateTime(dT_CommWait,t0)
                    
            if Rank==0:                     ResVec[i+1] = NormR_Act                
            
            #Converged
            if NormR_Act <= TolB: #Act = Actual                
                Flag = 0
                Iter = i
                break            
            else:            
                if Stag >= MaxStagSteps and MoreSteps == 0: Stag = 0
                MoreSteps           += 1            
                #Stagnated
                if MoreSteps >= MaxMSteps: 
                    raise Warning('PCG : TooSmallTolerance')
                    Flag = 3
                    Iter = i
                    break
        
        #Update Minimal Norm Quanitites
        if NormR_Act < NormRMin:            
            NormRMin                    = NormR_Act
            MP_XMin                      = np.array(MP_X)
            iMin                        = i
        
        if Stag >= MaxStagSteps:
            Flag = 3
            break
    
    
    dT_Calc,t0                          = updateTime(dT_Calc,t0)
    
    #Finalizing Results
    if Flag == 0:
        RelRes = NormR_Act/NormRefLoadVector
    else:        
        MP_Fint, d1, d2, d3             = calcMPQ(MP_XMin, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList)
        MP_R                            = MP_RefLoadVector - MP_Fint
        MP_R[MP_FixedLocalDofVector]     = 0.0
        MP_SqR                          = np.dot(MP_R, MP_R*MP_WeightVector)
        NormR                          = np.sqrt(Comm.allreduce(MP_SqR, op=MPI.SUM))        
        if NormR < NormR_Act:            
            MP_X = MP_XMin
            Iter = iMin
            RelRes = NormR/NormRefLoadVector        
        else:            
            Iter = i
            RelRes = NormR_Act/NormRefLoadVector
    
    #Truncate the zeros from resvec
    if Rank == 0:            
        if Flag <= 1 or Flag ==3:        ResVec     = ResVec[:i+2]
        else:                            ResVec     = ResVec[:i+1]
    
    #Applying Dirichlet BC
    MP_X[MP_FixedLocalDofVector]      = MP_DispConstraintVector
    dT_Calc,t0                      = updateTime(dT_Calc,t0)
            
    #Gathering Results
    SendReq = Comm.Isend(MP_X, dest=0, tag=Rank)    
    if Rank == 0:        
        N_MeshParts                 = len(MPList_NDofVec)
        MPList_X                     = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]        
        for j in range(N_Workers):  Comm.Recv(MPList_X[j], source=j, tag=j)            
        X                             = np.zeros(GlobNDOF, dtype=float)
        X[GathDofVector]             = np.hstack(MPList_X)        
    SendReq.Wait()    
    dT_CommWait,t0                     = updateTime(dT_CommWait,t0)

    #Saving CPU Time
    t0_End = time()
    dT_Total_Verify = t0_End - t0_Start
    dT_Total = dT_Calc + dT_CommWait
    MP_TimeRecordList = [dT_FileRead, dT_Calc, dT_CommWait, dT_Total_Verify, t0_Start, t0_End]
    MPList_TimeRecordList = Comm.gather(MP_TimeRecordList, root=0)
    
    if Rank == 0 :  
        TimeRecList_Max = np.round(np.max(MPList_TimeRecordList, axis=0), 6)        
        TimeRecList_Min = np.round(np.min(MPList_TimeRecordList, axis=0), 6)        
        TimeRecList_Mean = np.round(np.mean(MPList_TimeRecordList, axis=0), 6)   
        
        TotalTime = TimeRecList_Max[5] - TimeRecList_Min[4]
        MaxCommWaitTime = TimeRecList_Max[2]
        MinCommWaitTime = TimeRecList_Min[2]
        Delta_CommWaitTime = MaxCommWaitTime - MinCommWaitTime
        
        print('-----------------')
        print('Total Time:', TotalTime, TimeRecList_Mean[3])
        print('\n')
        
        #Saving Data File
        TimeData = {'TotalTime': TotalTime, 'MaxCommWaitTime': MaxCommWaitTime, 'Delta_CommWaitTime': Delta_CommWaitTime}        
        np.savez_compressed(OutputFileName + '_' + QCalcMode, TimeData = TimeData)
    
        #Exporting Results
        if ExportFlag == 1:        
            ExportData = {'X': X, 'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec, 'TimeData': TimeData}
            savemat(OutputFileName+'.mat', ExportData)
        
        if PlotFlag == 1:
            PlotData = {'RefDofIdList': RefDofIdList+1,'X': X[RefDofIdList]}
            savemat(OutputFileName+'_Plot.mat', PlotData)
            print('PlotData', PlotData)
            print('\n')
        
        print('----------')
        print('Iter', Iter)
        print('Flag', Flag)
        print('RelRes', RelRes)
        print('RelTol', Tol)
        print('TimeData', TimeData)
        print('----------\n')
        
        #Printing Results
        print('Max displacement')
        np.set_printoptions(precision=12)
        MaxXn = np.max(X)
        I = np.where(X==MaxXn)[0][0]
        print('Result', I, X[I-5:I+5])
       

