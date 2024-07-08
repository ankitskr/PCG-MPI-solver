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



def calcMPFint(MP_Un, RefMeshPart, MP_TimeRecData, A0=0.0, B0=1.0):
    
    #Reading Variables
    MP_SubDomainData                     = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector                     = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList              = RefMeshPart['OvrlpLocalDofVecList']
    Flat_ElemLocDof                      = RefMeshPart['Flat_ElemLocDof']
    NCount                               = RefMeshPart['NCount']
    FintCalcMode                         = RefMeshPart['GlobData']['FintCalcMode']
    
    
    #Calculating Local Fint Vector for Octree cells
    MP_NDOF = len(MP_Un)    
    MP_LocFintVec = np.zeros(MP_NDOF, dtype=float)   
    if FintCalcMode == 'outbin':
        Flat_ElemFint = np.zeros(NCount, dtype=float)
        I=0     
        
        
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        ElemList_LocDofVector = RefTypeGroup['ElemList_LocDofVector']
        ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
        
        Ke = RefTypeGroup['ElemStiffMat']; Me = RefTypeGroup['ElemMassMat']
        ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
        #ElemList_Level = RefTypeGroup['ElemList_Level']
        #ElemList_LevelCubed = RefTypeGroup['ElemList_LevelCubed']
        ElemList_Cm = RefTypeGroup['ElemList_Cm']
        ElemList_Ck = RefTypeGroup['ElemList_Ck']
        
        ElemList_Un =   MP_Un[ElemList_LocDofVector]
        ElemList_Un[ElemList_SignVector] *= -1.0      
        """
        if B0==0.0:     ElemList_Fint =  A0*np.dot(Me, ElemList_LevelCubed*ElemList_Un)
        elif A0==0.0:   ElemList_Fint =  B0*np.dot(Ke, ElemList_Level*ElemList_Un)
        else:           ElemList_Fint =  A0*np.dot(Me, ElemList_LevelCubed*ElemList_Un) + B0*np.dot(Ke, ElemList_Level*ElemList_Un)
        """
        if B0==0.0:     ElemList_Fint =  A0*np.dot(Me, ElemList_Cm*ElemList_Un)
        elif A0==0.0:   ElemList_Fint =  B0*np.dot(Ke, ElemList_Ck*ElemList_Un)
        else:           ElemList_Fint =  A0*np.dot(Me, ElemList_Cm*ElemList_Un) + B0*np.dot(Ke, ElemList_Ck*ElemList_Un)
        
        ElemList_Fint[ElemList_SignVector] *= -1.0
        
        if FintCalcMode == 'inbin':            
            MP_LocFintVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_Fint.ravel(), minlength=MP_NDOF)        
        elif FintCalcMode == 'infor':
            apply_sum(ElemList_LocDofVector, MP_LocFintVec, ElemList_Fint)            
        elif FintCalcMode == 'outbin':
            N = len(ElemList_LocDofVector_Flat)
            Flat_ElemFint[I:I+N]=ElemList_Fint.ravel()
            I += N

    
    if FintCalcMode == 'outbin':
        MP_LocFintVec = np.bincount(Flat_ElemLocDof, weights=Flat_ElemFint, minlength=MP_NDOF)
    
    
    #Calculating Overlapping Fint Vectors
    MP_OvrlpFintVecList = []
    MP_InvOvrlpFintVecList = []
    N_NbrMP = len(MP_OvrlpLocalDofVecList)    
    for j in range(N_NbrMP):
        MP_OvrlpFintVec = MP_LocFintVec[MP_OvrlpLocalDofVecList[j]]
        MP_OvrlpFintVecList.append(MP_OvrlpFintVec)   
        
        N_NbrDof_j = len(MP_OvrlpLocalDofVecList[j]);
        MP_InvOvrlpFintVecList.append(np.zeros(N_NbrDof_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping Fint
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        SendReq     = Comm.Isend(MP_OvrlpFintVecList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpFintVecList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
     
    #Calculating Fint
    MP_FintVec = MP_LocFintVec
    for j in range(N_NbrMP):
        MP_FintVec[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpFintVecList[j]
                    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
     
    return MP_FintVec



    

def MPI_SUM(MP_RefVar, MP_TimeRecData):
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    Glob_RefVar = Comm.allreduce(MP_RefVar, op=MPI.SUM)
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    return Glob_RefVar

    
    

def updateTime(MP_TimeRecData, Ref):
    
    t1 = time()
    MP_TimeRecData[Ref] += t1 - MP_TimeRecData['t0']
    MP_TimeRecData['t0'] = t1
    
    



def plotDispVecData(PlotFileName, TimeList, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList, TimeList_PlotDispVector.T)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    



def getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData, RecTime = True):
    
    if RecTime: updateTime(MP_TimeRecData, 'dT_Calc')
    SendReq = Comm.Isend(MP_Un, dest=0, tag=Rank)            
    if Rank == 0:                
        for j in range(N_Workers):    Comm.Recv(MPList_Un[j], source=j, tag=j)                
    SendReq.Wait()
    RecTime:    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    GlobUnVector = []
    if Rank == 0:
        GathUnVector = np.hstack(MPList_Un)
        GlobUnVector = np.zeros(GlobNDof, dtype=float)
        GlobUnVector[GathDofVector] = GathUnVector
            
    return GlobUnVector



def exportDispVecData(OutputFileName, ExportCount, Time_dT, GlobUnVector, Splits = 1):
    
    DispVecFileName = OutputFileName+'_'+str(ExportCount)+'_'
    
    if Rank == 0:
        
        J = Splits
        N = int(len(GlobUnVector))
        Nj = int(N/J)
        for j in range(J):
            if j==0:
                N1 = 0; N2 = Nj;
            elif j == J-1:
                N1 = N2; N2 = N;
            else:
                N1 = N2; N2 = (j+1)*Nj;
            
            DispData_j = {'T': Time_dT, 'U': GlobUnVector[N1:N2]}
            savemat(DispVecFileName+str(j+1)+'.mat', DispData_j)






def PCG(MP_X0, MP_Fc, RefMeshPart, MP_TimeRecData, A0=None, B0=None):

    MP_WeightVector                      = RefMeshPart['WeightVector']
    MP_NDOF                              = RefMeshPart['NDOF']
    MP_LocDof_eff                        = RefMeshPart['LocDof_eff']
    MP_LumpedMassVector                  = RefMeshPart['LumpedMassVector']
    MP_InvDiagPreCondVector0             = RefMeshPart['InvDiagPreCondVector0']
    MP_InvDiagPreCondVector1             = RefMeshPart['InvDiagPreCondVector1']
    MPList_RefPlotDofIndicesList         = RefMeshPart['MPList_RefPlotDofIndicesList']
    GathDofVector                        = RefMeshPart['GathDofVector']
    GlobData                             = RefMeshPart['GlobData']
    
    GlobNDof_eff                        = GlobData['GlobNDof_eff']
    GlobNDof                            = GlobData['GlobNDof']
    MaxIter                             = GlobData['MaxIter']    
    Tol                                 = GlobData['Tol']
    ExistDP0                            = GlobData['ExistDP0']
    ExistDP1                            = GlobData['ExistDP1']
    UseLumpedMass                       = GlobData['UseLumpedMass']
    fb1a                                 = GlobData['fb1a']
    
    if A0==None and B0==None:
        if UseLumpedMass:   A0 = 0.0; B0 = 1.0
        else:               A0 = 1.0; B0 = fb1a
    
    #Initializing Variables
    MP_Fc                                = MP_Fc[MP_LocDof_eff]
    MP_X0                                = MP_X0[MP_LocDof_eff]
    MP_X                                 = MP_X0
    MP_XMin                              = MP_X                                  #Iterate which has minimal residual so far
    MP_SqRefLoad                         = np.dot(MP_Fc, MP_Fc*MP_WeightVector)
    SqRefLoad                            = MPI_SUM(MP_SqRefLoad, MP_TimeRecData)
    NormRefLoadVector                    = np.sqrt(SqRefLoad) #n2b
    TolB                                 = Tol*NormRefLoadVector
    
    #Check for all zero right hand side vector => all zero solution
    if NormRefLoadVector == 0:                      # if rhs vector is all zeros        
        MP_X_Unq                        = np.zeros(MP_NDOF);  # then  solution is all zeros
        MP_X_Unq[MP_LocDof_eff]         = MP_X
        Flag                            = 0;                                   # a valid solution has been obtained
        RelRes                          = 0;                                   # the relative residual is actually 0/0
        Iter                            = 0;                                   # no iterations need be performed
        ResVec                          = [0];                               # resvec(1) = norm(b-A*x) = norm(0)
        
        return MP_X_Unq, Flag, RelRes, Iter
    
    
    #Initial Settings    
    Flag                                        = 1
    Rho                                         = 1.0 #Dummy
    Stag                                        = 0  # stagnation of the method
    MoreSteps                                   = 0
    MaxStagSteps                                = 3
    MaxMSteps                                   = min([int(GlobNDof_eff/50),5,GlobNDof_eff-MaxIter]);
    iMin                                        = 0
    Iter                                        = 0 
    
    MP_X_Unq                                    = np.zeros(MP_NDOF)
    MP_P_Unq                                    = np.zeros(MP_NDOF)
    
    MP_X_Unq[MP_LocDof_eff]                     = MP_X
    MP_RefLoadVector0_Unq                       = calcMPFint(MP_X_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
    if UseLumpedMass:      
        MP_RefLoadVector0_Unq                   = MP_LumpedMassVector*MP_X_Unq + fb1a*MP_RefLoadVector0_Unq
    MP_RefLoadVector0                           = MP_RefLoadVector0_Unq[MP_LocDof_eff]
    MP_R                                        = MP_Fc - MP_RefLoadVector0
    MP_SqR                                      = np.dot(MP_R, MP_R*MP_WeightVector)
    NormR                                       = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
    NormRMin                                    = NormR
    
    #Checking if the initial guess is a good enough solution
    if NormR <= TolB:
        Flag                            = 0;
        RelRes                          = NormR/NormRefLoadVector;
        Iter                            = 0;
        #ResVec                          = [NormR];
        return MP_X_Unq, Flag, RelRes, Iter
    
    """
    if Rank == 0:   
        ResVec                       = np.zeros(GlobNDof_eff+1)        
        ResVec[0]                    = NormR    
        
        #print(0, [NormR, TolB], [Stag, MaxStagSteps])
    """
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
        MP_Q_Unq                             = calcMPFint(MP_P_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
        if UseLumpedMass:      
            MP_Q_Unq                         = MP_LumpedMassVector*MP_P_Unq + fb1a*MP_Q_Unq
        MP_Q                                 =  MP_Q_Unq[MP_LocDof_eff]
        
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
        
        #Calculating R
        MP_R                              -= Alpha*MP_Q
        
        #Calculating Convergence Variables
        MP_SqP                           = np.dot(MP_P, MP_P*MP_WeightVector)
        MP_SqX                           = np.dot(MP_X, MP_X*MP_WeightVector)
        MP_SqR                           = np.dot(MP_R, MP_R*MP_WeightVector)
        [NormP, NormX, NormR]            = np.sqrt(MPI_SUM(np.array([MP_SqP, MP_SqX, MP_SqR]), MP_TimeRecData))
        
        #Stagnation
        if (NormP*abs(Alpha) < eps*NormX):  Stag += 1
        else:                               Stag = 0
        
        #Calculating X
        MP_X                              += Alpha*MP_P
        
        NormR_Act                       = NormR
        
        """
        if Rank==0:
            #savemat(OutputFileName+'_Log.mat', {'NormR':NormR, 'TolB': TolB, 'i': i})
            #if i%1==0:    print(i, [NormR, TolB], [Stag, MaxStagSteps])            
            ResVec[i+1]                    = NormR_Act
        """
        
        if NormR <= TolB or Stag >= MaxStagSteps or MoreSteps > 0:
            MP_X_Unq[MP_LocDof_eff]             = MP_X
            MP_Fint_Unq                         = calcMPFint(MP_X_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
            if UseLumpedMass:      
                MP_Fint_Unq                     = MP_LumpedMassVector*MP_X_Unq + fb1a*MP_Fint_Unq
            MP_Fint                             = MP_Fint_Unq[MP_LocDof_eff]
            MP_R                                = MP_Fc - MP_Fint
            MP_SqR_Act                          = np.dot(MP_R, MP_R*MP_WeightVector)
            NormR_Act                           = np.sqrt(MPI_SUM(MP_SqR_Act, MP_TimeRecData))
        
                    
            #if Rank==0:                     ResVec[i+1] = NormR_Act                
            
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
        MP_X_Unq[MP_LocDof_eff]             = MP_XMin
        MP_Fint_Unq                         = calcMPFint(MP_X_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
        if UseLumpedMass:      
            MP_Fint_Unq                     = MP_LumpedMassVector*MP_X_Unq + fb1a*MP_Fint_Unq
        MP_Fint                             = MP_Fint_Unq[MP_LocDof_eff]
        MP_R                                = MP_Fc - MP_Fint
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
    
    """
    #Truncate the zeros from resvec
    if Rank == 0:            
        if Flag <= 1 or Flag ==3:        ResVec     = ResVec[:i+2]
        else:                            ResVec     = ResVec[:i+1]
    """
    
    return MP_X_Unq, Flag, RelRes, Iter
    




if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    """
    --> Map MPI processes by NUMA nodes as explained below:
    https://opus.nci.org.au/display/Help/nci-parallel
    
    --> Check NUMA connection using: 
    $ numactl --hardware
    $ lscpu
    
    Ref: https://superuser.com/questions/916516/is-the-amount-of-numa-nodes-always-equal-to-sockets
    
    -->UPDATE INTRODUCTION CHAPTER
    """
    
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
    MP_WeightVector                      = RefMeshPart['WeightVector']
    MP_LocDof_eff                        = RefMeshPart['LocDof_eff']
    MP_DiagMVector                      = RefMeshPart['DiagMVector']
    MP_DPDiagKVector                      = RefMeshPart['DPDiagKVector']
    MP_DPDiagMVector                      = RefMeshPart['DPDiagMVector']
        
    MP_NDOF                              = RefMeshPart['NDOF']
    MP_RefLoadVector                     = RefMeshPart['RefLoadVector']
    MPList_NDofVec                       = RefMeshPart['MPList_NDofVec']
    MP_RefPlotData                       = RefMeshPart['RefPlotData']
    MPList_RefPlotDofIndicesList         = RefMeshPart['MPList_RefPlotDofIndicesList']
    GathDofVector                       = RefMeshPart['GathDofVector']
    GlobData                             = RefMeshPart['GlobData']
    
    AlphaN                              = GlobData['Alpha']
    Beta                                = GlobData['Beta']
    Gamma                               = GlobData['Gamma']
    GlobNDof                            = GlobData['GlobNDof']
    MaxTime                             = GlobData['MaxTime']
    dt                                  = GlobData['TimeStepSize']
    DeltaLambdaList                     = GlobData['DeltaLambdaList']
    dT_Export                           = GlobData['dT_Export']
    ExportFrms                             = GlobData['ExportFrms']
    PlotFlag                            = GlobData['PlotFlag']
    ExportFlag                          = GlobData['ExportFlag']
    FintCalcMode                        = GlobData['FintCalcMode'].lower()
    Tol                                 = GlobData['Tol']
    
    
    if SpeedTestFlag==1:  
        PlotFlag = 0; ExportFlag = 0; MaxTime=25*dt; FintCalcMode = 'outbin';
    
    if not FintCalcMode in ['inbin', 'infor', 'outbin']:  
        raise ValueError("FintCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    ExportKeyFrm = round(dT_Export/dt)
    RefMaxTimeStepCount = int(np.ceil(MaxTime/dt)) + 1
    
    if Rank==0:
        print(ExportFlag, FintCalcMode)
        print('dt', dt)
        print('ExportKeyFrm',ExportKeyFrm)   
    
    #A small sleep to avoid hanging
    sleep(Rank*1e-4)
    
    #Barrier so that all processes start at same time
    Comm.barrier() 
    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    #Initializing Variables
    MPList_Un = []
    if Rank==0:
        N_MeshParts = len(MPList_NDofVec)
        MPList_Un = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]
            
        TimeList_Flag = np.zeros(RefMaxTimeStepCount)
        TimeList_RelRes = np.zeros(RefMaxTimeStepCount)
        TimeList_Iter = np.zeros(RefMaxTimeStepCount)
    
    fg1             = Gamma*dt;
    fg2             = (1.0-Gamma)*dt;
    fb1             = Beta*dt*dt;
    fb1a             = (1+AlphaN)*fb1;
    fb2             = (0.5-Beta)*dt*dt;
    GlobData['fb1a'] = fb1a
    
    RefMeshPart['WeightVector'] = MP_WeightVector[MP_LocDof_eff]
    RefMeshPart['LumpedMassVector'] = MP_DiagMVector
    RefMeshPart['InvDiagPreCondVector1'] = []
    
    if Tol < eps:        
        raise Warning('PCG : Too Small Tolerance')
        GlobData['Tol'] = eps        
    elif Tol > 1:        
        raise Warning('PCG : Too Big Tolerance')
        GlobData['Tol'] = 1 - eps
    
    MP_Un = (1e-200)*np.random.rand(MP_NDOF)
    MP_Vn = (1e-200)*np.random.rand(MP_NDOF)
    MP_An = (1e-200)*np.random.rand(MP_NDOF)
    
    DeltaLambda0 = DeltaLambdaList[0]
    if DeltaLambda0 > 1e-20:
        MP_Fext     = DeltaLambda0*MP_RefLoadVector
        InvDiagPreConditioner = 1.0/MP_DPDiagMVector
        RefMeshPart['InvDiagPreCondVector0'] = InvDiagPreConditioner[MP_LocDof_eff]
        MP_An, Flag, RelRes, Iter = PCG(MP_An, MP_Fext, RefMeshPart, MP_TimeRecData, A0=1.0, B0 = 0.0)
        if Rank==0:
            TimeList_Flag[0] = Flag; TimeList_RelRes[0] = RelRes; TimeList_Iter[0] = Iter
    
    InvDiagPreConditioner = 1.0/(MP_DPDiagMVector + fb1a*MP_DPDiagKVector)
    RefMeshPart['InvDiagPreCondVector0'] = InvDiagPreConditioner[MP_LocDof_eff]
    
    
    
    TimeList = [i*dt for i in range(RefMaxTimeStepCount)]
    ExportCount = 1
    
    if PlotFlag == 1:
        MP_PlotLocalDofVec  = MP_RefPlotData['LocalDofVec']
        MP_PlotNDofs       = len(MP_PlotLocalDofVec)
        RefPlotDofVec       = MP_RefPlotData['RefPlotDofVec']
        qpoint              = MP_RefPlotData['qpoint']
        TestPlotFlag        = MP_RefPlotData['TestPlotFlag']
        
        if MP_PlotNDofs > 0:
            MP_PlotDispVector               = np.zeros([MP_PlotNDofs, RefMaxTimeStepCount])
            MP_PlotDispVector[:,0]          = MP_Un[MP_PlotLocalDofVec]
            MP_PlotVelVector               = np.zeros([MP_PlotNDofs, RefMaxTimeStepCount])
            MP_PlotVelVector[:,0]          = MP_Vn[MP_PlotLocalDofVec]
            MP_PlotAccVector               = np.zeros([MP_PlotNDofs, RefMaxTimeStepCount])
            MP_PlotAccVector[:,0]          = MP_An[MP_PlotLocalDofVec]
            
            
        else:   
            MP_PlotDispVector               = []
            MP_PlotVelVector               = []
            MP_PlotAccVector               = []
        
    
    if ExportFlag == 1:  

        ExportNow = False
        if ExportKeyFrm>0 or 0 in ExportFrms:
            ExportNow = True
        
        if ExportNow:
            Time_dT = 0*dt
            
            GlobUnVector = getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData)
            exportDispVecData(OutputFileName, ExportCount, Time_dT, GlobUnVector)
            
            ExportCount += 1
                
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')
    tref = time()
    for TimeStepCount in range(1, RefMaxTimeStepCount):
    
        if Rank==0: print(TimeStepCount, time()-tref)
        
        MP_Un_1     = MP_Un
        MP_Vn_1     = MP_Vn
        MP_An_1     = MP_An
        
        
        
        MP_Unp_1    = MP_Un_1 + dt*MP_Vn_1 + fb2*MP_An_1;
        MP_Vnp_1    = MP_Vn_1 + fg2*MP_An_1;
        
        MP_Fext     = DeltaLambdaList[TimeStepCount]*MP_RefLoadVector
        MP_Fint_1   = calcMPFint((1+AlphaN)*MP_Unp_1 - AlphaN*MP_Un_1, RefMeshPart, MP_TimeRecData)
        
        MP_Fc       = MP_Fext - MP_Fint_1
        
        MP_An, Flag, RelRes, Iter = PCG(MP_An_1, MP_Fc, RefMeshPart, MP_TimeRecData)
        
        MP_Un       = MP_Unp_1 + fb1*MP_An;
        MP_Vn       = MP_Vnp_1 + fg1*MP_An;
        
        
        if Rank==0:
            TimeList_Flag[TimeStepCount] = Flag
            TimeList_RelRes[TimeStepCount] = RelRes
            TimeList_Iter[TimeStepCount] = Iter
            
            
        if PlotFlag == 1:
            if MP_PlotNDofs>0:
                MP_PlotDispVector[:,TimeStepCount] = MP_Un[MP_PlotLocalDofVec]
                MP_PlotVelVector[:,TimeStepCount] = MP_Vn[MP_PlotLocalDofVec]
                MP_PlotAccVector[:,TimeStepCount] = MP_An[MP_PlotLocalDofVec]
                
            if (TimeStepCount)%10==0:
                if TestPlotFlag:
                    plotDispVecData(PlotFileName+'_'+str(TimeStepCount), TimeList, MP_PlotDispVector)
            
            
        if ExportFlag == 1:    
            
            ExportNow = False
            if ExportKeyFrm>0:
                if (TimeStepCount)%ExportKeyFrm==0: ExportNow = True
                
            if TimeStepCount in ExportFrms:
                ExportNow = True
            
            if ExportNow:
                Time_dT = TimeStepCount*dt
                
                GlobUnVector = getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData)
                exportDispVecData(OutputFileName, ExportCount, Time_dT, GlobUnVector)
                
                ExportCount += 1
                
        
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('TotalTimeStepCount', RefMaxTimeStepCount-1) 
    
    
    #Saving CPU Time
    MP_TimeRecData['dT_Total_Verify'] = t0_End - t0_Start 
    MP_TimeRecData['t0_Start'] = t0_Start
    MP_TimeRecData['t0_End'] = t0_End
    MPList_TimeRecData = Comm.gather(MP_TimeRecData, root=0)
    
    if PlotFlag == 1:   
        MPList_PlotDispVector = Comm.gather(MP_PlotDispVector, root=0)
        MPList_PlotVelVector = Comm.gather(MP_PlotVelVector, root=0)
        MPList_PlotAccVector = Comm.gather(MP_PlotAccVector, root=0)
        
    if Rank == 0 :
        TimeData = configTimeRecData(MPList_TimeRecData)
        TimeData['PBS_JobId'] = PBS_JobId
        TimeData['Flag'] = TimeList_Flag
        TimeData['Iter'] = TimeList_Iter
        TimeData['RelRes'] = TimeList_RelRes
        
        #print('Iter', TimeList_Iter)       
        
        TimeDataFileName = OutputFileName + '_MP' +  str(N_MshPrt) + '_' + FintCalcMode + '_TimeData'
        np.savez_compressed(TimeDataFileName, TimeData = TimeData)
        savemat(TimeDataFileName +'.mat', TimeData)
        
        
        
        #Exporting Plots        
        if PlotFlag == 1:
            N_TotalPlotDofs     = len(RefPlotDofVec)
            TimeList_PlotDispVector = np.zeros([N_TotalPlotDofs, RefMaxTimeStepCount])
            TimeList_PlotVelVector = np.zeros([N_TotalPlotDofs, RefMaxTimeStepCount])
            TimeList_PlotAccVector = np.zeros([N_TotalPlotDofs, RefMaxTimeStepCount])
            
            for i in range(N_Workers):
                MP_PlotDispVector_i = MPList_PlotDispVector[i]
                MP_PlotVelVector_i = MPList_PlotVelVector[i]
                MP_PlotAccVector_i = MPList_PlotAccVector[i]
                
                RefPlotDofIndices_i = MPList_RefPlotDofIndicesList[i]
                N_PlotDofs_i = len(MP_PlotDispVector_i)
                if N_PlotDofs_i>0:
                    for j in range(N_PlotDofs_i):
                        TimeList_PlotDispVector[RefPlotDofIndices_i[j],:] = MP_PlotDispVector_i[j]
                        TimeList_PlotVelVector[RefPlotDofIndices_i[j],:] = MP_PlotVelVector_i[j]
                        TimeList_PlotAccVector[RefPlotDofIndices_i[j],:] = MP_PlotAccVector_i[j]
                
                
            #Saving Data File
            PlotTimeData = {'Plot_T': TimeList, 
                            'Plot_U': TimeList_PlotDispVector, 
                            'Plot_V': TimeList_PlotVelVector, 
                            'Plot_A': TimeList_PlotAccVector, 
                            'Plot_Dof': RefPlotDofVec+1, 
                            'qpoint': qpoint}
            np.savez_compressed(PlotFileName+'_PlotData', PlotData = PlotTimeData)
            savemat(PlotFileName+'_PlotData.mat', PlotTimeData)
        
        
        np.set_printoptions(precision=12)
        
        
    
    #Printing Results
    GlobUnVector = getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData, RecTime = False)
    if Rank == 0:
        np.set_printoptions(precision=12)
        print('\n\n\n')
        I = np.argmax(GlobUnVector)
        print('I', I)
        for i in range(10): print(GlobUnVector[I+i-5])
    
    