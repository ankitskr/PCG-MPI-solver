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
        #ElemList_Level = RefTypeGroup['ElemList_Level']
        #ElemList_Cm = RefTypeGroup['ElemList_Cm']
        ElemList_Ck = RefTypeGroup['ElemList_Ck']
        
        ElemList_P =   MP_P[ElemList_LocDofVector]
        ElemList_P[ElemList_SignVector] *= -1.0        
        #ElemList_Q =   np.dot(Ke, ElemList_Level*ElemList_P)
        ElemList_Q =  np.dot(Ke, ElemList_Ck*ElemList_P)
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
    
    

def exportDispVecData(DispVecPath, X, ExportData, Splits = 1):
                   
    J = Splits
    N = int(len(X))
    Nj = int(N/J)
    for j in range(J):
        if j==0:
            N1 = 0; N2 = Nj;
        elif j == J-1:
            N1 = N2; N2 = N;
        else:
            N1 = N2; N2 = (j+1)*Nj;
        
        X_j = {'RefX': X[N1:N2]}
        savemat(DispVecPath+'DispVec_'+str(j+1)+'.mat', X_j)
    
    savemat(DispVecPath+'AnalysisData.mat', ExportData)   
    savemat(DispVecPath+'J.mat', {'J':J})   




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
    
    N_MshPrt =          N_Workers
    ModelName =         sys.argv[1]
    ScratchPath =       sys.argv[2]
    R0 =                sys.argv[3]
    SpeedTestFlag =     int(sys.argv[4])
    PBS_JobId =         sys.argv[5]
    
    #Creating directories
    ResultPath          = ScratchPath + 'Results_Run' + str(R0)
    if SpeedTestFlag == 1:  ResultPath += '_SpeedTest/'
    else:                   ResultPath += '/'
        
    PlotPath = ResultPath + 'PlotData/'
    DispVecPath = ResultPath + 'DispVecData/'
    if Rank==0:    
        if not os.path.exists(ResultPath):            
            os.makedirs(PlotPath)        
            os.makedirs(DispVecPath)
    
 
    OutputFileName = DispVecPath + ModelName
    PlotFileName = PlotPath + ModelName
    
    #Reading Model Data Files
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(N_MshPrt)  + '/'
    RefMeshPart_FileName = PyDataPath + str(Rank) + '.zpkl'
    Cmpr_RefMeshPart = open(RefMeshPart_FileName, 'rb').read()
    RefMeshPart = pickle.loads(zlib.decompress(Cmpr_RefMeshPart))
    
    MP_SubDomainData                     = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector                     = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList              = RefMeshPart['OvrlpLocalDofVecList']
    MP_WeightVector                      = RefMeshPart['WeightVector']
    MP_NDOF                              = RefMeshPart['NDOF']
    MP_RefLoadVector                     = RefMeshPart['RefLoadVector']
    MP_LocDof_eff                       = RefMeshPart['LocDof_eff']
    MP_InvDiagPreCondVector0             = RefMeshPart['InvDiagPreCondVector0']
    MP_InvDiagPreCondVector1             = RefMeshPart['InvDiagPreCondVector1']
    MP_X0                                = RefMeshPart['X0']
    MPList_NDof_eff                      = RefMeshPart['MPList_NDof_eff']
    GathDof_eff                       = RefMeshPart['GathDof_eff']
    GlobData                            = RefMeshPart['GlobData']
    
    GlobNDof_eff                        = GlobData['GlobNDof_eff']
    GlobNDof                            = GlobData['GlobNDof']
    ExportFlag                          = GlobData['ExportFlag']
    MaxIter                             = GlobData['MaxIter']    
    Tol                                 = GlobData['Tol']
    ExistDP0                            = GlobData['ExistDP0']
    ExistDP1                            = GlobData['ExistDP1']
    QCalcMode                           = GlobData['QCalcMode'].lower()
    
    
    if SpeedTestFlag==1:  
        ExportFlag = 0; QCalcMode = 'outbin';
    
    if Rank == 0:    print(ExportFlag, QCalcMode)
    
    if not QCalcMode in ['inbin', 'infor', 'outbin']:  raise ValueError("QCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    
    MP_RefLoadVector = MP_RefLoadVector[MP_LocDof_eff]
    MP_X0 = MP_X0[MP_LocDof_eff]
    MP_WeightVector = MP_WeightVector[MP_LocDof_eff]
    
    if ExistDP0: #Diagonal Preconditioner DP0
        MP_InvDiagPreCondVector0 = MP_InvDiagPreCondVector0[MP_LocDof_eff]
    
    if ExistDP1: #Diagonal Preconditioner DP1            
        MP_InvDiagPreCondVector1 = MP_InvDiagPreCondVector1[MP_LocDof_eff]
    
    
    
    #Checking Tolerance
    if Tol < eps:        
        raise Warning('PCG : Too Small Tolerance')
        Tol = eps        
    elif Tol > 1:        
        raise Warning('PCG : Too Big Tolerance')
        Tol = 1 - eps
    
    #Initializing Variables
    N_NbrMP                             = len(MP_NbrMPIdVector)
    N_OvrlpLocalDofVecList              = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    MP_InvOvrlpQList                     = [np.zeros(N_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    N_NbrDof                            = np.sum(N_OvrlpLocalDofVecList)
    NCount                              = 0
    N_Type                              = len(MP_SubDomainData['StrucDataList'])
    for j in range(N_Type):             NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
    
    #Initializing Variables
    MP_X                                 = MP_X0
    MP_XMin                              = MP_X                                  #Iterate which has minimal residual so far
    MP_SqRefLoad                         = np.dot(MP_RefLoadVector, MP_RefLoadVector*MP_WeightVector)
    SqRefLoad                            = MPI_SUM(MP_SqRefLoad, MP_TimeRecData)
    NormRefLoadVector                    = np.sqrt(SqRefLoad) #n2b
    TolB                                 = Tol*NormRefLoadVector
    
    #Check for all zero right hand side vector => all zero solution
    if NormRefLoadVector == 0:                      # if rhs vector is all zeros        
        X                               = np.zeros(GlobNDof_eff);  # then  solution is all zeros
        Flag                            = 0;                                   # a valid solution has been obtained
        RelRes                          = 0;                                   # the relative residual is actually 0/0
        Iter                            = 0;                                   # no iterations need be performed
        ResVec                          = [0];                               # resvec(1) = norm(b-A*x) = norm(0)
        
        #Exporting Files
        if ExportFlag == 1:        
            ExportData = {'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec}
            exportDispVecData(DispVecPath, X, ExportData)
        exit()
    
    #Initial Settings    
    Flag                                        = 1
    Rho                                         = 1.0 #Dummy
    Stag                                        = 0  # stagnation of the method
    MoreSteps                                   = 0
    MaxStagSteps                                = 3
    MaxMSteps                                   = min([int(GlobNDof_eff/50),5,GlobNDof_eff-MaxIter]);
    iMin                                        = 0
    Iter                                        = 0 
    
    MP_X_Unq                                     = np.zeros(MP_NDOF)
    MP_P_Unq                                     = np.zeros(MP_NDOF)
    
    MP_X_Unq[MP_LocDof_eff]                      = MP_X
    MP_RefLoadVector0_Unq                        = calcMPQ(MP_X_Unq, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
    MP_RefLoadVector0                              = MP_RefLoadVector0_Unq[MP_LocDof_eff]
    MP_R                                        = MP_RefLoadVector - MP_RefLoadVector0
    MP_SqR                                      = np.dot(MP_R, MP_R*MP_WeightVector)
    NormR                                       = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
    NormRMin                                    = NormR
    
    #Checking if the initial guess is a good enough solution
    if NormR <= TolB:
        Flag                            = 0;
        RelRes                          = NormR/NormRefLoadVector;
        Iter                            = 0;
        ResVec                          = [NormR];
        
        #Exporting Files
        if ExportFlag == 1:        
            ExportData = {'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec}
            exportDispVecData(DispVecPath, X, ExportData)
        exit()
    
    if Rank == 0:   
        ResVec                       = np.zeros(GlobNDof_eff+1)        
        ResVec[0]                    = NormR    
        
        print('Mode', QCalcMode)
        print(0, [NormR, TolB], [Stag, MaxStagSteps])
                
    
    
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')    
    
    for i in range(MaxIter):
        
        #Calculating Z
        if ExistDP0: #Diagonal Preconditioner DP0            
            MP_Y           = MP_InvDiagPreCondVector0*MP_R
            if np.any(np.isinf(MP_Y)):                
                Flag = 2
                break        
        else:    MP_Y      = MP_R
            
        if ExistDP1: #Diagonal Preconditioner DP1            
            MP_Z           = MP_InvDiagPreCondVector1*MP_Y
            if np.any(np.isinf(MP_Z)):                
                Flag = 2
                break        
        else:    MP_Z      = MP_Y
        
        #Calculating Rho
        Rho_1                           = Rho
        MP_Rho                          = np.dot(MP_Z,MP_R*MP_WeightVector)
        Rho                             = MPI_SUM(MP_Rho, MP_TimeRecData)
    
        
        if Rho == 0 or np.isinf(Rho):
            Flag = 4
            break
        
        #Calculating P and Beta
        if i == 0:
            MP_P         = MP_Z
        else:
            Beta                        = Rho/Rho_1   
            if Beta == 0 or np.isinf(Beta):
                Flag = 4
                break
            MP_P         = MP_Z + Beta*MP_P         
            
        #Calculating Q
        MP_P_Unq[MP_LocDof_eff]              = MP_P    
        MP_Q_Unq                            = calcMPQ(MP_P_Unq, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
        MP_Q                                 = MP_Q_Unq[MP_LocDof_eff]
        
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
        MP_X                              += Alpha*MP_P
        
        #Calculating R
        MP_R                              -= Alpha*MP_Q
        
        #Calculating Convergence Variables
        MP_SqR                          = np.dot(MP_R, MP_R*MP_WeightVector)
        SqR                             = MPI_SUM(MP_SqR, MP_TimeRecData)
        NormR                           = np.sqrt(SqR)
        NormR_Act                       = NormR
        
        if Rank==0:
            #savemat(OutputFileName+'_Log.mat', {'NormR':NormR, 'TolB': TolB, 'i': i})
            if i%1==0:    print(i, [NormR, TolB], [Stag, MaxStagSteps])            
            ResVec[i+1]                    = NormR_Act

        if NormR <= TolB or Stag >= MaxStagSteps or MoreSteps > 0:
            MP_X_Unq[MP_LocDof_eff]             = MP_X
            MP_Fint_Unq                         = calcMPQ(MP_X_Unq, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
            MP_Fint                               = MP_Fint_Unq[MP_LocDof_eff]
            MP_R                                = MP_RefLoadVector - MP_Fint
            MP_SqR_Act                          = np.dot(MP_R, MP_R*MP_WeightVector)
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
        MP_XMin_Unq[MP_LocDof_eff]          = MP_XMin
        MP_Fint_Unq                         = calcMPQ(MP_XMin_Unq, QCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpQList, MP_TimeRecData)
        MP_Fint                               = MP_Fint_Unq[MP_LocDof_eff]
        MP_R                                = MP_RefLoadVector - MP_Fint
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
    
    
    
    if Rank==0:    print('Analysis Finished Sucessfully..')
    t0_End = time()
    
    
    #Saving CPU Time
    MP_TimeRecData['dT_Total_Verify'] = t0_End - t0_Start 
    MP_TimeRecData['t0_Start'] = t0_Start
    MP_TimeRecData['t0_End'] = t0_End
    MP_TimeRecData['MP_NCount'] = NCount
    MP_TimeRecData['MP_NDOF'] = MP_NDOF
    MP_TimeRecData['N_NbrDof'] = N_NbrDof    
    MPList_TimeRecData = Comm.gather(MP_TimeRecData, root=0)
    
    
    #Gathering Results
    SendReq = Comm.Isend(MP_X, dest=0, tag=Rank)    
    if Rank == 0:        
        N_MeshParts                 = len(MPList_NDof_eff)
        MPList_X                     = [np.zeros(MPList_NDof_eff[i]) for i in range(N_MeshParts)]        
        for j in range(N_Workers):  Comm.Recv(MPList_X[j], source=j, tag=j)            
        X                             = np.zeros(GlobNDof, dtype=float)
        X[GathDof_eff]             = np.hstack(MPList_X)        
    SendReq.Wait()
    
    
    if Rank == 0 :
        TimeData = configTimeRecData(MPList_TimeRecData)
        TimeData['PBS_JobId'] = PBS_JobId
        TimeData['RelRes'] = RelRes
        TimeData['Iter'] = Iter
        
        TimeDataFileName = OutputFileName + '_MP' +  str(N_MshPrt) + '_' + QCalcMode + '_TimeData'
        np.savez_compressed(TimeDataFileName, TimeData = TimeData)
        
        #Exporting Results
        if ExportFlag == 1:
            ExportData = {'Flag': Flag, 'RelRes': RelRes, 'Iter': Iter, 'ResVec': ResVec, 'TimeData': TimeData}
            exportDispVecData(DispVecPath, X, ExportData)
        
        
        np.set_printoptions(precision=12)
        
        print('----------')
        print('Iter', Iter)
        print('Flag', Flag)
        print('RelRes', RelRes)
        print('RelTol', Tol)
        print('----------\n')
        
       
       

