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
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    #Initializing Variables
    N_NbrMP                             = len(MP_NbrMPIdVector)
    N_OvrlpLocalDofVecList              = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    MP_InvOvrlpQList                     = [np.zeros(N_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    N_NbrDof                            = np.sum(N_OvrlpLocalDofVecList)
    NCount                              = 0
    N_Type                              = len(MP_SubDomainData['StrucDataList'])
    for j in range(N_Type):             NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
    
    #Converting Dirichlet BC  into Newmann BC
    MP_X                                    = np.zeros(MP_NDOF, dtype=float)
    MP_X[MP_FixedLocalDofVector]             = MP_DispConstraintVector
    MP_FintFixed                              = calcMPQ(MP_X, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
    MP_RefLoadVector                        -= MP_FintFixed
    MP_RefLoadVector[MP_FixedLocalDofVector] = 0.0    
    
    #Initializing Variables
    MP_X                                 = MP_X0
    MP_X[MP_FixedLocalDofVector]          = 0.0
    MP_XMin                              = MP_X                                  #Iterate which has minimal residual so far
    MP_SqRefLoad                         = np.dot(MP_RefLoadVector, MP_RefLoadVector*MP_WeightVector)
    SqRefLoad                             = MPI_SUM(MP_SqRefLoad, MP_TimeRecData)
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
    Flag                                        = 1
    Rho                                         = 1.0 #Dummy
    Stag                                        = 0  # stagnation of the method
    MoreSteps                                   = 0
    MaxStagSteps                                = 3
    MaxMSteps                                   = min([int(GlobNDOF/50),5,GlobNDOF-MaxIter]);
    iMin                                        = 0
    Iter                                        = 0 
    
    MP_RefLoadVector0                             = calcMPQ(MP_X, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
    MP_R                                         = MP_RefLoadVector - MP_RefLoadVector0
    MP_R[MP_FixedLocalDofVector]                  = 0.0    
    MP_SqR                                       = np.dot(MP_R, MP_R*MP_WeightVector)
    NormR                                         = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
    
    
    NormRMin                                    = NormR
    
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
        Rho                             = MPI_SUM(MP_Rho, MP_TimeRecData)
    
        
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
        MP_Q                                = calcMPQ(MP_P, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
        
        #Calculating PQ and Alpha
        MP_PQ                            = np.dot(MP_P,MP_Q*MP_WeightVector)
        PQ                               = MPI_SUM(MP_PQ, MP_TimeRecData)
        if PQ <= 0 or np.isinf(PQ):
            Flag = 4
            break
        else:    Alpha                   = Rho/PQ    
        if np.isinf(Alpha):
            Flag = 4
            break
        
        #Calculating Convergence Variables
        MP_SqP                           = np.dot(MP_P, MP_P*MP_WeightVector)
        MP_SqX                           = np.dot(MP_X, MP_X*MP_WeightVector)
        [NormP, NormX]                    = np.sqrt(MPI_SUM(np.array([MP_SqP, MP_SqX]), MP_TimeRecData))
        
        #Stagnation
        if (NormP*abs(Alpha) < eps*NormX):  Stag += 1
        else:                               Stag = 0
        
        #Calculating X
        MP_X                            += Alpha*MP_P
        MP_X[MP_FixedLocalDofVector]     = 0.0
        
        #Calculating R
        MP_R                            -= Alpha*MP_Q
        MP_R[MP_FixedLocalDofVector]     = 0.0
        
        #Calculating Convergence Variables
        MP_SqR                          = np.dot(MP_R, MP_R*MP_WeightVector)
        SqR                               = MPI_SUM(MP_SqR, MP_TimeRecData)
        NormR                           = np.sqrt(SqR)
        NormR_Act                       = NormR
        
        if Rank==0:
            #savemat(OutputFileName+'_Log.mat', {'NormR':NormR, 'TolB': TolB, 'i': i})
            if i%1==0:    print(i, [NormR, TolB], [Stag, MaxStagSteps])            
            ResVec[i+1]                    = NormR_Act

        if NormR <= TolB or Stag >= MaxStagSteps or MoreSteps > 0:
            MP_Fint                             = calcMPQ(MP_X, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
            MP_R                                 = MP_RefLoadVector - MP_Fint
            MP_R[MP_FixedLocalDofVector]         = 0.0
            MP_SqR_Act                           = np.dot(MP_R, MP_R*MP_WeightVector)
            NormR_Act                           = np.sqrt(MPI_SUM(MP_SqR_Act, MP_TimeRecData))
        
                    
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
    
        
    #Finalizing Results
    if Flag == 0:
        RelRes = NormR_Act/NormRefLoadVector
    else:        
        MP_Fint                            = calcMPQ(MP_XMin, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
        MP_R                                = MP_RefLoadVector - MP_Fint
        MP_R[MP_FixedLocalDofVector]         = 0.0
        MP_SqR                              = np.dot(MP_R, MP_R*MP_WeightVector)
        NormR                               = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
        
        if NormR < NormR_Act:            
            MP_X = MP_XMin
            Iter = iMin
            RelRes = NormR/NormRefLoadVector        
        else:            
            Iter = i
            RelRes = NormR_Act/NormRefLoadVector
    
    Iter += 1 #So that the iteration matches with Matlab
    
    #Truncate the zeros from resvec
    if Rank == 0:            
        if Flag <= 1 or Flag ==3:        ResVec     = ResVec[:i+2]
        else:                            ResVec     = ResVec[:i+1]
    
    #Applying Dirichlet BC
    MP_X[MP_FixedLocalDofVector]      = MP_DispConstraintVector
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    #Gathering Results
    SendReq = Comm.Isend(MP_X, dest=0, tag=Rank)    
    if Rank == 0:        
        N_MeshParts                 = len(MPList_NDofVec)
        MPList_X                     = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]        
        for j in range(N_Workers):  Comm.Recv(MPList_X[j], source=j, tag=j)            
        X                             = np.zeros(GlobNDOF, dtype=float)
        X[GathDofVector]             = np.hstack(MPList_X)        
    SendReq.Wait()
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    if Rank==0:    print('Analysis Finished Sucessfully..')
    t0_End = time()
    
    #Saving CPU Time
    MP_TimeRecData['dT_Total_Verify'] = t0_End - t0_Start 
    MP_TimeRecData['t0_Start'] = t0_Start
    MP_TimeRecData['t0_End'] = t0_End
    MP_TimeRecData['MP_NDOF'] = MP_NDOF
    MP_TimeRecData['N_NbrDof'] = N_NbrDof
    
    MPList_TimeRecData = Comm.gather(MP_TimeRecData, root=0)
    
    if Rank == 0 :
                      
        MPList_dT_FileRead     = np.array([MP_TimeRecData['dT_FileRead'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_dT_Calc         = np.array([MP_TimeRecData['dT_Calc'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_dT_CommWait     = np.array([MP_TimeRecData['dT_CommWait'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_dT_Total_Verify     = np.array([MP_TimeRecData['dT_Total_Verify'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_t0_Start     = np.array([MP_TimeRecData['t0_Start'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_t0_End         = np.array([MP_TimeRecData['t0_End'] for MP_TimeRecData in MPList_TimeRecData])
        
        ts0 = np.min(MPList_t0_Start)
        te1 = np.max(MPList_t0_End)      
        TotalTime = te1 - ts0
        
        MPList_dT_Wait0 = MPList_t0_Start - ts0
        MPList_dT_Wait1 = te1 - MPList_t0_End
        MPList_dT_Wait = MPList_dT_Wait0 + MPList_dT_Wait1
        
        MPList_dT_CommWait += MPList_dT_Wait
        
        MaxCommWaitTime = np.max(MPList_dT_CommWait)
        MinCommWaitTime = np.min(MPList_dT_CommWait)
        
        MPList_NDOF     = np.array([MP_TimeRecData['MP_NDOF'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_N_NbrDof = np.array([MP_TimeRecData['N_NbrDof'] for MP_TimeRecData in MPList_TimeRecData])
        #I = np.argsort(MPList_NDOF)
        #LoadUnbalanceData = [MPList_NDOF[I], MPList_dT_Calc[I], MPList_dT_CommWait[I], MPList_N_NbrDof[I], MPList_dT_CalcTest[I], MPList_dT_CommWaitTest[I]]
        LoadUnbalanceData = [MPList_NDOF, MPList_dT_Calc, MPList_dT_CommWait, MPList_N_NbrDof]
        
        
        MPList_TotalTime = np.zeros(N_Workers)
        for i in range(N_Workers):            
            MPList_TotalTime[i] = MPList_dT_Calc[i] + MPList_dT_CommWait[i]
        
        Mean_FileReadTime     = np.mean(MPList_dT_FileRead)
        Mean_CalcTime         = np.mean(MPList_dT_Calc)
        Mean_CommWaitTime     = np.mean(MPList_dT_CommWait)
        TotalTime_Analysis     = Mean_CalcTime + Mean_CommWaitTime
        
        Mean_TotalTime         = np.mean(MPList_dT_Total_Verify)
        
        
        print('-----------------')
        print('Verify Total Time:', norm(TotalTime - MPList_TotalTime), np.abs(TotalTime-TotalTime_Analysis))
        print('Total Time:', TotalTime, ['Mean', Mean_TotalTime])
        print('\n')
        
        #Saving Data File
        TimeData = {'TotalTime'                : TotalTime, 
                    'MaxCommWaitTime'        : MaxCommWaitTime, 
                    'MinCommWaitTime'        : MinCommWaitTime, 
                    'Mean_FileReadTime'        : Mean_FileReadTime,
                    'Mean_CalcTime'            : Mean_CalcTime,
                    'Mean_CommWaitTime'        : Mean_CommWaitTime,
                    'RelRes'                    : RelRes,
                    'Iter'                        : Iter,
                    'LoadUnbalanceData'            :LoadUnbalanceData}
                    
        np.savez_compressed(OutputFileName + '_' + QCalcMode, TimeData = TimeData)
    
        #Exporting Results
        if ExportFlag == 1:        
            ExportData = {'X': X, 'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec, 'TimeData': TimeData}
            savemat(OutputFileName+'.mat', ExportData)
        
        
        np.set_printoptions(precision=12)
        
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
        print('----------\n')
        
        #Printing Results
        print('Max displacement')
        MaxXn = np.max(X)
        I = np.where(X==MaxXn)[0][0]
        print('Result', I, X[I-5:I+5])
       
       

