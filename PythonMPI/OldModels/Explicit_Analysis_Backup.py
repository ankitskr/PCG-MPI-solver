# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:10:24 2020

@author: z5166762
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
gc.collect()
gc.set_threshold(5600, 20, 20)
#gc.disable()

import matplotlib.pyplot as plt
from datetime import datetime
import sys
from time import time, sleep
import numpy as np
import os.path
import shutil

import pickle
import zlib

import mpi4py
from mpi4py import MPI

#mpi4py.rc.threads = False

#import logging
#from os.path import abspath
#from Cython_Array.Array_cy import updateLocFint, apply_sum
from scipy.io import savemat
from GeneralFunc import configTimeRecData


def calcMPFint(MP_UnVector_1, FintCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpFintVecList, MP_TimeRecData):
    
    #Calculating Local Fint Vector
    MP_NDOF = len(MP_UnVector_1)    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList)    
    if FintCalcMode in ['infor', 'inbin']:
        MP_LocFintVec = np.zeros(MP_NDOF, dtype=float)    
    elif FintCalcMode == 'outbin':            
        Flat_ElemLocDof = np.zeros(NCount, dtype=int)
        Flat_ElemFint = np.zeros(NCount)
        I=0        
        
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]        
        Ke = RefTypeGroup['ElemStiffMat']
        ElemList_LocDofVector = RefTypeGroup['ElemList_LocDofVector']
        ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
        ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
        ElemList_Level = RefTypeGroup['ElemList_Level']
        
        ElemList_Un_1 =   MP_UnVector_1[ElemList_LocDofVector]
        ElemList_Un_1[ElemList_SignVector] *= -1.0        
        ElemList_Fint =   np.dot(Ke, ElemList_Level*ElemList_Un_1)
        ElemList_Fint[ElemList_SignVector] *= -1.0
        
        if FintCalcMode == 'inbin':            
            MP_LocFintVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_Fint.ravel(), minlength=MP_NDOF)        
        elif FintCalcMode == 'infor':
            apply_sum(ElemList_LocDofVector, MP_LocFintVec, ElemList_Fint)            
        elif FintCalcMode == 'outbin':
            N = len(ElemList_LocDofVector_Flat)
            Flat_ElemLocDof[I:I+N]=ElemList_LocDofVector_Flat
            Flat_ElemFint[I:I+N]=ElemList_Fint.ravel()
            I += N
            
    if FintCalcMode == 'outbin':    MP_LocFintVec = np.bincount(Flat_ElemLocDof, weights=Flat_ElemFint, minlength=MP_NDOF)
         
    
    #Calculating Overlapping Fint Vectors
    MP_OvrlpFintVecList = []
    N_NbrMP = len(MP_OvrlpLocalDofVecList)    
    for j in range(N_NbrMP):    MP_OvrlpFintVecList.append(MP_LocFintVec[MP_OvrlpLocalDofVecList[j]])    
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
     
    #Calculating Q
    MP_FintVec = MP_LocFintVec
    for j in range(N_NbrMP):        
        MP_FintVec[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpFintVecList[j]    
    updateTime(MP_TimeRecData, 'dT_Calc')
     
    return MP_FintVec


    
    

def updateTime(MP_TimeRecData, Ref):
    
    t1 = time()
    MP_TimeRecData[Ref] += t1 - MP_TimeRecData['t0']
    MP_TimeRecData['t0'] = t1
    
    
    
    
    

def calcMPDispVec(MP_FintVector, MP_RefLoadVector, MP_InvLumpedMassVector, MP_FixedLocalDofVector, MP_UnVector_2, MP_UnVector_1, dt, DeltaLambda_1, Damping_Alpha):
   
    MP_FextVector = DeltaLambda_1*MP_RefLoadVector
    DampTerm = 0.5*Damping_Alpha*dt
    MP_UnVector = (1.0/(1.0+DampTerm))*(2.0*MP_UnVector_1 - (1-DampTerm)*MP_UnVector_2 + dt*dt*MP_InvLumpedMassVector*(MP_FextVector - MP_FintVector))
    MP_UnVector[MP_FixedLocalDofVector] = 0.0
    
    return MP_UnVector




def getGlobDispVec(GathUnVector, GathDofVector, GlobNDOF):
    
    GlobDispVector = np.zeros(GlobNDOF, dtype=float)
    GlobDispVector[GathDofVector] = GathUnVector
        
    return GlobDispVector




def plotDispVecData(PlotFileName, TimeList_PlotdT, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList_PlotdT, TimeList_PlotDispVector)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    
            



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
    
    p =                 int(np.log2(N_Workers)) #N_MP = 2**p
    ModelName =         sys.argv[1]
    ScratchPath =       sys.argv[2]
    R0 =                sys.argv[3]
    
    
    #Creating directories
    ResultPath          = ScratchPath + 'Results_Run' + str(R0) + '/'
    PlotPath = ResultPath + 'PlotData/'
    DispVecPath = ResultPath + 'DispVecData/'
    if Rank==0:    
        if not os.path.exists(ResultPath):            
            os.makedirs(PlotPath)        
            os.makedirs(DispVecPath)
    
 
    OutputFileName = DispVecPath + ModelName
    PlotFileName = PlotPath + ModelName
    
    #Reading Model Data Files
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(p)  + '/'
    RefMeshPart_FileName = PyDataPath + str(Rank) + '.zpkl'
    Cmpr_RefMeshPart = open(RefMeshPart_FileName, 'rb').read()
    RefMeshPart = pickle.loads(zlib.decompress(Cmpr_RefMeshPart))
    
    MP_SubDomainData =                   RefMeshPart['SubDomainData']
    MP_NbrMPIdVector =                     RefMeshPart['NbrIdVector']
    MP_OvrlpLocalDofVecList =            RefMeshPart['NbrOvrlpLocalDOFIdVectorList']
    MP_NDOF =                            RefMeshPart['NDOF']
    MP_RefLoadVector =          RefMeshPart['RefTransientLoadVector']
    MP_InvLumpedMassVector =             RefMeshPart['InvLumpedMassVector']
    MP_FixedLocalDofVector =     RefMeshPart['ConstrainedLocalDOFIdVector']
    GathDofVector =                    RefMeshPart['GathDOFIdVector']
    MPList_NDofVec =                   RefMeshPart['MPList_NDOFIdVec']
    GlobData =                           RefMeshPart['GlobData']
        
    GlobNDOF =                           GlobData['GlobNDOF']
    RefDOFIdList =                       GlobData['RefDOFIdList']
    
    MaxTime =                            GlobData['MaxTime']
    Damping_Alpha =                      GlobData['Damping_Alpha']
    dt =                                 GlobData['TimeStepSize']
    DeltaLambdaList =                    GlobData['DeltaLambdaList']
    dT_Plot =                            GlobData['dT_Plot']
    dT_Export =                          GlobData['dT_Export']
    PlotFlag =                           GlobData['PlotFlag']
    ExportFlag =                         GlobData['ExportFlag']
    FintCalcMode =                       GlobData['FintCalcMode']
    
    if Rank == 0:    print(PlotFlag, ExportFlag, FintCalcMode)
    
    if not FintCalcMode in ['inbin', 'infor', 'outbin']:  raise ValueError("FintCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    
    if PlotFlag == 0 and ExportFlag == 0:      dT = MaxTime; UpdateKFrm = None;
    elif PlotFlag == 1 and ExportFlag == 0:    dT = dT_Plot; UpdateKFrm = None;
    elif PlotFlag == 0 and ExportFlag == 1:    dT = dT_Export; UpdateKFrm = 1;
    elif PlotFlag == 1 and ExportFlag == 1:    dT = dT_Plot;
    
    RefMaxTimeStepCount =                int(np.ceil(MaxTime/dt))
    TimeChunkSize =                      int(np.ceil(dT/dt))
    dT =                                 dt*TimeChunkSize
    N_TimeChunk =                        max([int(np.ceil(RefMaxTimeStepCount/TimeChunkSize)) - 1, 1])
    
    if PlotFlag == 1 and ExportFlag == 1:    UpdateKFrm = round(dT_Export/dT);
    
    
    if Rank==0:    
        
        print('dt', dt)
        print('RefMaxTimeStepCount', RefMaxTimeStepCount)
        print('TimeChunkSize', TimeChunkSize)
        print('dT', dT)
        print('N_TimeChunk', N_TimeChunk)
        #print('RefDOFIdList', RefDOFIdList+1)
        
    
    if Rank==0:
        N_MeshParts = len(MPList_NDofVec)
        MPList_UnVector_1 = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]
        
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    
    #Initializing Variables
    N_NbrMP = len(MP_NbrMPIdVector)
    N_OvrlpLocalDofVecList = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    MP_InvOvrlpFintVecList = [np.zeros(N_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    N_NbrDof               = np.sum(N_OvrlpLocalDofVecList)
    
    NCount = 0
    N_Type = len(MP_SubDomainData['StrucDataList'])
    for j in range(N_Type):    NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
    
    MP_UnVector_2 = np.zeros(MP_NDOF, dtype=float)
    MP_UnVector_1 = np.zeros(MP_NDOF, dtype=float)
    
    GlobDispVector_2 = np.zeros(GlobNDOF, dtype=float)
    GlobDispVector_1 = np.zeros(GlobNDOF, dtype=float)
    
    
    
    TimeList_dT =                 [0.0]
    #TimeList_GlobDispVector =     [[GlobDispVector_2, GlobDispVector_1]]
    TimeList_GlobDispVector =       [GlobDispVector_1]
    
    TimeList_PlotdT =             [0.0]
    PlotDispVector_1 =            GlobDispVector_1[RefDOFIdList]
    TimeList_PlotDispVector =     [PlotDispVector_1]
    
    
    TimeChunkCount0 =             0
    TimeStepCount =               1
    
    
    
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')
    
    for TimeChunkCount in range(TimeChunkCount0, N_TimeChunk):
        
        for tp in range(TimeChunkSize):
            
            #Calculating Fint
            MP_FintVec       = calcMPFint(MP_UnVector_1, FintCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, NCount, MP_NbrMPIdVector, MP_InvOvrlpFintVecList, MP_TimeRecData)
    
            #Calculating Displacement Vector
            DeltaLambda_1 = DeltaLambdaList[TimeStepCount-1]
            MP_UnVector = calcMPDispVec(MP_FintVec, MP_RefLoadVector, MP_InvLumpedMassVector, MP_FixedLocalDofVector, MP_UnVector_2, MP_UnVector_1, dt, DeltaLambda_1, Damping_Alpha)
                          
            MP_UnVector_2 = MP_UnVector_1
            MP_UnVector_1 = MP_UnVector
            
            TimeStepCount += 1
            
           
        #Gathering Data at each time chunk
        if PlotFlag == 1 or ExportFlag == 1:
        
            updateTime(MP_TimeRecData, 'dT_Calc')
            SendReq = Comm.Isend(MP_UnVector_1, dest=0, tag=Rank)            
            if Rank == 0:                
                for j in range(N_Workers):    Comm.Recv(MPList_UnVector_1[j], source=j, tag=j)                
            SendReq.Wait()
            updateTime(MP_TimeRecData, 'dT_CommWait')
            
            if Rank == 0:
                GathUnVector = np.hstack(MPList_UnVector_1)
                Glob_RefDispVector_1 = getGlobDispVec(GathUnVector, GathDofVector, GlobNDOF)
            
                if PlotFlag == 1:
                    PlotDispVector = Glob_RefDispVector_1[RefDOFIdList]
                    TimeList_PlotDispVector.append(PlotDispVector)
                    TimeList_PlotdT.append((TimeChunkCount+1)*dT)
                    
                    if (TimeChunkCount+1)%100==0:
                        plotDispVecData(PlotFileName+'_'+str(TimeChunkCount+1), TimeList_PlotdT, TimeList_PlotDispVector)
                    
                if ExportFlag == 1:
                    if (TimeChunkCount+1)%UpdateKFrm==0:
                        TimeList_GlobDispVector.append(Glob_RefDispVector_1)
                        TimeList_dT.append((TimeChunkCount+1)*dT)
                        
            
    
    
    TotalTimeStepCount = TimeStepCount-1
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('TotalTimeStepCount', TotalTimeStepCount) 
    
    
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
        np.savez_compressed(OutputFileName + '_MP' +  str(p) + '_' + FintCalcMode, TimeData = TimeData)
    
        #Exporting Results        
        if PlotFlag == 1:            
            #Saving Data File
            PlotTimeData = {'Plot_T': TimeList_PlotdT, 'Plot_U': TimeList_PlotDispVector, 'Plot_Dof': RefDOFIdList+1}
            np.savez_compressed(PlotFileName+'_PlotData', PlotData = PlotTimeData)
            savemat(PlotFileName+'_PlotData.mat', PlotTimeData)
        
        if ExportFlag == 1:            
            NDispVec = len(TimeList_dT)
            for i in range(NDispVec):
                DispData = {'T': TimeList_dT[i], 'U': TimeList_GlobDispVector[i]}
                savemat(OutputFileName+'_'+str(i+1)+'.mat', DispData)
            
    
    #Printing Results
    SendReq = Comm.Isend(MP_UnVector_1, dest=0, tag=Rank)
    if Rank == 0:
        np.set_printoptions(precision=12)
        
        for j in range(N_Workers):    Comm.Recv(MPList_UnVector_1[j], source=j, tag=j)
        GathUnVector = np.hstack(MPList_UnVector_1)
        
        print('\n\n\n')
        Glob_RefDispVector_1 = getGlobDispVec(GathUnVector, GathDofVector, GlobNDOF)
        MaxUn = np.max(Glob_RefDispVector_1)
        I = np.where(Glob_RefDispVector_1==MaxUn)[0][0]
        print('Result', I, Glob_RefDispVector_1[I-5:I+5])
    SendReq.Wait()
    
    
    

