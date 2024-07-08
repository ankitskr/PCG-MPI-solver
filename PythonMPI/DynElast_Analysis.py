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


import matplotlib.pyplot as plt
from datetime import datetime
import sys
from time import time, sleep
import numpy as np
import os.path
import shutil
import scipy.io

import pickle
import zlib

import mpi4py
from mpi4py import MPI

#mpi4py.rc.threads = False

#import logging
#from os.path import abspath
#from Cython_Array.Array_cy import updateLocFint, apply_sum
from scipy.io import savemat
from GeneralFunc import configTimeRecData, getPrincipalStrain, getPrincipalStress, GaussLobattoIntegrationTable, GaussIntegrationTable, readMPIFile, readMPIFile_parallel, writeMPIFile_parallel
from scipy.interpolate import interp1d, interp2d


def initGlobData():
    
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    #Reading inputs
    ModelName =         sys.argv[1]
    ScratchPath =       sys.argv[2]
    R0 =                sys.argv[3]
    SpeedTestFlag =     int(sys.argv[4])
    PBS_JobId =         sys.argv[5]
    
    GlobData                   = {}
    GlobData['MP_TimeRecData'] = {'dT_FileRead':        0.0,
                                  'dT_Calc':            0.0,
                                  'dT_CommWait':        0.0,
                                  'dT_CalcList':        [],
                                  'dT_CommWaitList':    [],
                                  'TimeStepCountList':  [],
                                  't0':                 time()}
    
    
    #Creating result directories
    ResultPath          = ScratchPath + 'Results_Run' + str(R0)
    if SpeedTestFlag == 1:  ResultPath += '_SpeedTest/'
    else:                   ResultPath += '/'
    
    PlotPath = ResultPath + 'PlotData/'
    ResVecPath = ResultPath + 'ResVecData/'
    if Rank==0:
        if os.path.exists(ResultPath):
            try:        os.rename(ResultPath, ResultPath[:-1]+'_'+ datetime.now().strftime('%d%m%Y_%H%M%S'))
            except:     raise Exception('Result Path in use!')
        os.makedirs(PlotPath)        
        os.makedirs(ResVecPath)
    Comm.barrier()
    
    
    GlobData['ScratchPath']     = ScratchPath
    GlobData['PlotPath']        = PlotPath
    GlobData['ResVecPath']      = ResVecPath
    GlobData['ModelName']       = ModelName
    GlobData['PlotFileName']    = PlotPath + ModelName
    GlobData['SpeedTestFlag']   = SpeedTestFlag
    GlobData['PBS_JobId']       = PBS_JobId
    
    return GlobData



def readModelData(GlobData):

    t1_             = time()
    N_MshPrt        = N_Workers
    PyDataPath      = GlobData['ScratchPath'] + 'ModelData/MPI/'
    Data_FileName   = PyDataPath + str(N_MshPrt)
    
    """
    Data_Buffer = readMPIFile_parallel(Data_FileName, Comm)
    """
    
    #"""
    metadat = np.load(Data_FileName+'_metadat.npy', allow_pickle=True).item()
    Nf = metadat['NfData'][Rank]; DType = metadat['DTypeData'][Rank]
    MetaData = [Nf, DType]
    Data_Buffer = readMPIFile(Data_FileName+ '_' + str(Rank), MetaData)
    #"""
    
    RefMeshPart = pickle.loads(zlib.decompress(Data_Buffer.tobytes()))
    GlobData.update(RefMeshPart['GlobData']) 
    RefMeshPart['GlobData'] = GlobData
    
    return RefMeshPart
    


def readGlobalSettings(GlobData):

    #Reading Global data file
    MatDataPath         = GlobData['ScratchPath'] + 'ModelData/Mat/'
    GlobSettingsFile    = MatDataPath + 'GlobSettings.mat'
    GlobSettings        = scipy.io.loadmat(GlobSettingsFile)
    
    GlobData['FintCalcMode']            = GlobSettings['FintCalcMode'][0]
    GlobData['ExportVars']              = GlobSettings['ExportVars'][0]
    GlobData['PlotFlag']                = GlobSettings['PlotFlag'][0][0]   
    GlobData['ExportFlag']              = GlobSettings['ExportFlag'][0][0]
    GlobData['ExportKeyFrm']            = int(GlobSettings['ExportFrmRate'][0][0])
    GlobData['ExportFrms']              = GlobSettings['ExportFrms']
    GlobData['RefMaxTimeStepCount']     = GlobSettings['RefMaxTimeStepCount'][0][0]
    GlobData['TimeStepDelta']           = GlobSettings['TimeStepDelta'][0]
    GlobData['EnergyFlag']              = 0
    
    if not GlobData['FintCalcMode'] in ['inbin', 'infor', 'outbin']:  
        raise ValueError("FintCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    if GlobData['SpeedTestFlag']==1:  
        GlobData['PlotFlag'] = 0; GlobData['ExportFlag'] = 0; GlobData['FintCalcMode'] = 'outbin'; GlobData['EnergyFlag'] = 0;
    


def initExportData(RefMeshPart):
    
    MP_RefPlotData                      = RefMeshPart['RefPlotData']
    MP_DofWeightVector                  = RefMeshPart['DofWeightVector']
    MP_NodeWeightVector                 = RefMeshPart['NodeWeightVector']
    MP_DofVector                        = RefMeshPart['DofVector']
    MP_NodeIdVector                     = RefMeshPart['NodeIdVector']
    GlobData                            = RefMeshPart['GlobData']
    
    EnergyFlag              = GlobData['EnergyFlag']
    PlotFlag                = GlobData['PlotFlag']
    ExportFlag              = GlobData['ExportFlag']
    ExportKeyFrm            = GlobData['ExportKeyFrm']
    RefMaxTimeStepCount     = GlobData['RefMaxTimeStepCount']
    
    ExportFrms                          = np.array(GlobData['ExportFrms'], dtype=int) 
    if len(ExportFrms)>0:    
        ExportFrms                      = ExportFrms[0] - 1
    GlobData['ExportFrms']              = ExportFrms
    
    
    dt = GlobData['dt']
    GlobData['TimeList'] = [i*dt for i in range(RefMaxTimeStepCount)]
    GlobData['TimeStepCount'] = 0
    GlobData['TimeList_T'] = []
    
    RefMeshPart['PEList'] = []
    RefMeshPart['DE'] = 0.0
    RefMeshPart['DEList'] = []
    
    
    if PlotFlag == 1:
        MP_PlotLocalDofVec  = MP_RefPlotData['LocalDofVec']
        MP_PlotNDofs        = len(MP_PlotLocalDofVec)
        RefPlotDofVec       = MP_RefPlotData['RefPlotDofVec']
        qpoint              = MP_RefPlotData['qpoint']
        TestPlotFlag        = MP_RefPlotData['TestPlotFlag']
        
        if MP_PlotNDofs > 0:
            
            MP_PlotDispVector               = np.zeros([MP_PlotNDofs, RefMaxTimeStepCount])
            MP_PlotDispVector[:,0]          = RefMeshPart['MP_Un'][MP_PlotLocalDofVec]
            MP_PlotLoadVector               = np.zeros(RefMaxTimeStepCount)
            
        else:   
            MP_PlotDispVector               = []
            MP_PlotLoadVector               = []
        
        RefMeshPart['PlotNDofs']        = MP_PlotNDofs
        RefMeshPart['qpoint']           = qpoint
        RefMeshPart['PlotLocalDofVec']  = MP_PlotLocalDofVec
        RefMeshPart['PlotDispVector']   = MP_PlotDispVector
        RefMeshPart['PlotLoadVector']   = MP_PlotLoadVector
        
    
    if ExportFlag == 1:
        
        #Computing vectors for MPI-based export 
        MP_DofWeightVector_Export = MP_DofWeightVector.astype(bool)
        MP_NodeWeightVector_Export = MP_NodeWeightVector.astype(bool)
        
        writeMPIFile_parallel(GlobData['ResVecPath']+'Dof', MP_DofVector[MP_DofWeightVector_Export], Comm)
        writeMPIFile_parallel(GlobData['ResVecPath']+'NodeId', MP_NodeIdVector[MP_NodeWeightVector_Export], Comm)

        #Exporting data
        RefMeshPart['ExportCount'] = 0
        RefMeshPart['DofWeightVector_Export'] = MP_DofWeightVector_Export
        RefMeshPart['NodeWeightVector_Export'] = MP_NodeWeightVector_Export
        
        exportContourData(RefMeshPart)
     

     
def updateTimeStep(RefMeshPart, TimeStepCount):
    
    GlobData = RefMeshPart['GlobData']
    
    GlobData['TimeStepCount'] = TimeStepCount
    DeltaLambda_1 = GlobData['TimeStepDelta'][TimeStepCount-1]
    RefMeshPart['MP_Fext_1'] = RefMeshPart['RefLoadVector']*DeltaLambda_1
    RefMeshPart['MP_Un_2'] = RefMeshPart['MP_Un_1']
    RefMeshPart['MP_Un_1'] = RefMeshPart['MP_Un']
    
    if Rank==0:
        if TimeStepCount%500==0:    
            print('TimeStepCount', TimeStepCount)
            print('Time', np.round([GlobData['MP_TimeRecData']['dT_Calc'], GlobData['MP_TimeRecData']['dT_CommWait']],1))
    
          
            
def calcMatVecProd(RefMeshPart, ComputeReference='Strain', MP_Xn = []):
    
    #Reading Variables
    MP_SubDomainData            = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector            = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList     = RefMeshPart['OvrlpLocalDofVecList']
    Flat_ElemLocDof             = RefMeshPart['Flat_ElemLocDof']
    MP_NCountDof                = RefMeshPart['NCountDof']
    MP_NDOF                     = RefMeshPart['NDOF']
    
    GlobData                    = RefMeshPart['GlobData']
    RefVecCalcMode              = GlobData['FintCalcMode']
    
    #Calculating Local RefVec Vector for Octree cells
    if RefVecCalcMode == 'outbin':
        Flat_ElemRefVec = np.zeros(MP_NCountDof, dtype=float)
        I=0     
    else:
        MP_LocRefVec = np.zeros(MP_NDOF, dtype=float)   
    
    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemList_LocDofVector = RefTypeGroup['ElemList_LocDofVector']
        ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
        ElemList_LocNodeIdVector    = RefTypeGroup['ElemList_LocNodeIdVector']
        
        if ComputeReference == 'Strain':
            
            Kd                  = RefTypeGroup['ElemStiffMat_d']
            Kv                  = RefTypeGroup['ElemStiffMat_v']
            ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
            ElemList_Ckd      = RefTypeGroup['ElemList_Ckd']
            ElemList_Ckv      = RefTypeGroup['ElemList_Ckv']
            
            ElemList_Un         = MP_Xn[ElemList_LocDofVector]
            ElemList_Un[ElemList_SignVector] *= -1.0      
            ElemList_RefVec     = np.dot(Kd, ElemList_Ckd*ElemList_Un) + np.dot(Kv, ElemList_Ckv*ElemList_Un)
            ElemList_RefVec[ElemList_SignVector] *= -1.0
        
           
        if RefVecCalcMode == 'inbin':            
            MP_LocRefVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_RefVec.ravel(), minlength=MP_NDOF)        
        #elif RefVecCalcMode == 'infor':
        #    apply_sum(ElemList_LocDofVector, MP_LocRefVec, ElemList_RefVec)            
        elif RefVecCalcMode == 'outbin':
            N = len(ElemList_LocDofVector_Flat)
            Flat_ElemRefVec[I:I+N]=ElemList_RefVec.ravel()
            I += N

    if RefVecCalcMode == 'outbin':
        MP_LocRefVec = np.bincount(Flat_ElemLocDof, weights=Flat_ElemRefVec, minlength=MP_NDOF)
    
    
    #Calculating Overlapping RefVec Vectors
    MP_OvrlpRefVecList = []
    MP_InvOvrlpRefVecList = []
    N_NbrMP = len(MP_OvrlpLocalDofVecList)    
    for j in range(N_NbrMP):
        MP_OvrlpRefVec = MP_LocRefVec[MP_OvrlpLocalDofVecList[j]]
        MP_OvrlpRefVecList.append(MP_OvrlpRefVec)   
        
        N_NbrDof_j = len(MP_OvrlpLocalDofVecList[j]);
        MP_InvOvrlpRefVecList.append(np.zeros(N_NbrDof_j))
    
    updateTime(GlobData, 'dT_Calc')
    
    
    #Communicating Overlapping RefVec
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        SendReq     = Comm.Isend(MP_OvrlpRefVecList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpRefVecList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(GlobData, 'dT_CommWait')
     
    #Calculating RefVec
    MP_RefVec = MP_LocRefVec
    for j in range(N_NbrMP):
        MP_RefVec[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpRefVecList[j]
     
    return MP_RefVec



def calcMPFint(RefMeshPart):
    
    MP_Un_1 = RefMeshPart['MP_Un_1']
    RefMeshPart['MP_Fint_1'] = calcMatVecProd(RefMeshPart, MP_Xn = MP_Un_1)
    
    
"""
def updateStrainVec(MP_Un, RefMeshPart, MP_TimeRecData, MP_DmgTimeRecData):
    
    MP_SubDomainData    = RefMeshPart['SubDomainData']
    MP_TypeGroupList    = MP_SubDomainData['StrucDataList']
    N_Type              = len(MP_TypeGroupList) 
    
    for j in range(N_Type):        
        RefTypeGroup            = MP_TypeGroupList[j]   
        
        ElemList_LocDofVector   = RefTypeGroup['ElemList_LocDofVector']
        StrainMode              = RefTypeGroup['ElemStrainModeMat']
        ElemList_SignVector     = RefTypeGroup['ElemList_SignVector']
        ElemList_Ce             = RefTypeGroup['ElemList_Ce']
        
        ElemList_Un         =   MP_Un[ElemList_LocDofVector]
        ElemList_Un[ElemList_SignVector] *= -1.0        
        ElemList_LocStrainVec  =   np.dot(StrainMode, ElemList_Ce*ElemList_Un) #Strains have been computed in the local coordinate system of each pattern
        RefTypeGroup['ElemList_LocStrainVec'] = ElemList_LocStrainVec #Saving to calculate (damaged) element stress later, if required
"""
    


def calcMPDispVec(RefMeshPart):
    
    Damping_Alpha           = 0.0
    dt                      = RefMeshPart['GlobData']['dt']
    MP_LocFixedDof          = RefMeshPart['LocFixedDof']
    MP_InvLumpedMassVector  = RefMeshPart['InvDiagM']
    MP_Vd                   = RefMeshPart['Vd']
    MP_Un_2                 = RefMeshPart['MP_Un_2']
    MP_Un_1                 = RefMeshPart['MP_Un_1']
    MP_Fint_1               = RefMeshPart['MP_Fint_1']
    MP_Fext_1               = RefMeshPart['MP_Fext_1']
    
    DampTerm = 0.5*Damping_Alpha*dt
    MP_Un = (1.0/(1.0+DampTerm))*(2.0*MP_Un_1 - (1-DampTerm)*MP_Un_2 + dt*dt*MP_InvLumpedMassVector*(MP_Fext_1 - MP_Fint_1))
    MP_Un[MP_LocFixedDof] = MP_Un_2[MP_LocFixedDof] +2*dt*MP_Vd[MP_LocFixedDof]
    
    RefMeshPart['MP_Un'] = MP_Un




def calcEnergy(RefMeshPart):
    
    GlobData    = RefMeshPart['GlobData']
    MP_Un_1     = RefMeshPart['MP_Un_1']
    MP_Fint_1   = RefMeshPart['MP_Fint_1']
    
    if GlobData['EnergyFlag'] == 1:
        #Elastic Strain Energy
        MP_PE_1 = 0.5*np.dot(MP_Un_1, MP_Fint_1*RefMeshPart['DofWeightVector'])
        PE_1 = MPI_SUM(MP_PE_1, MP_TimeRecData)
        RefMeshPart['PEList'].append(PE_1)
        
        #Dissipation 
        MP_FintVec_1p = calcMPFint(MP_Un_1, RefMeshPart, MP_TimeRecData, DmgCkRef=False)
        MP_PE_1p = 0.5*np.dot(MP_Un_1, MP_FintVec_1p*RefMeshPart['DofWeightVector'])
        PE_1p = MPI_SUM(MP_PE_1p, MP_TimeRecData)
        RefMeshPart['DE'] += (PE_1p - PE_1) 
        RefMeshPart['DEList'].append(RefMeshPart['DE']) 
    


def MPI_SUM(MP_RefVar, MP_TimeRecData):
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    Glob_RefVar = Comm.allreduce(MP_RefVar, op=MPI.SUM)
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    return Glob_RefVar
   

def updateTime(GlobData, RefKey, TimeStepCount=None):
    
    MP_TimeRecData = GlobData['MP_TimeRecData']
    if RefKey == 'UpdateList':
        MP_TimeRecData['TimeStepCountList'].append(TimeStepCount)
        MP_TimeRecData['dT_CalcList'].append(MP_TimeRecData['dT_Calc'])
        MP_TimeRecData['dT_CommWaitList'].append(MP_TimeRecData['dT_CommWait'])
    else:  
        t1 = time()
        MP_TimeRecData[RefKey] += t1 - MP_TimeRecData['t0']
        MP_TimeRecData['t0'] = t1
        
        
        
def plotDispVecData(PlotFileName, TimeList, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList, TimeList_PlotDispVector.T)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    
  
def getNodalScalarVar(RefMeshPart, MP_TimeRecData, Ref):
    
    MP_SubDomainData            = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector            = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalNodeIdVecList  = RefMeshPart['OvrlpLocalNodeIdVecList']
    MP_NNode                    = RefMeshPart['NNode']
    
    MP_TypeGroupList            = MP_SubDomainData['StrucDataList']
    N_Type                      = len(MP_TypeGroupList) 
    
    MP_VarSum                = np.zeros(MP_NNode)
    MP_VarCount              = np.zeros(MP_NNode)
    
    for j in range(N_Type):        
        RefTypeGroup                = MP_TypeGroupList[j]    
        NNodes_ElemType             = RefTypeGroup['NNodes']
        ElemList_LocNodeIdVector    = RefTypeGroup['ElemList_LocNodeIdVector']
        ElemList_Omega              = RefTypeGroup['ElemList_Omega'] #Dmg
        N_Elem                      = len(ElemList_Omega)
        
        if Ref=='Dmg':
            ElemList_RefVar = ElemList_Omega
        elif Ref=='EqvStrain':
            ElemList_RefVar = RefTypeGroup['ElemList_EqvStrain']
        
        ElemList_VarVec     = ElemList_RefVar*np.ones(NNodes_ElemType)[np.newaxis].T
        MP_VarSum += np.bincount(ElemList_LocNodeIdVector.ravel(), weights=ElemList_VarVec.ravel(), minlength=MP_NNode)
        
        FlatElemList_VarCount     = np.ones(NNodes_ElemType*N_Elem)
        MP_VarCount += np.bincount(ElemList_LocNodeIdVector.ravel(), weights=FlatElemList_VarCount, minlength=MP_NNode)
        
    
    #Calculating Overlapping Fint Vectors
    N_NbrMP = len(MP_NbrMPIdVector)
            
    MP_OvrlpVarList = []
    MP_InvOvrlpVarList = []
    for j in range(N_NbrMP):
        MP_OvrlpVar = np.hstack([MP_VarSum[MP_OvrlpLocalNodeIdVecList[j]], MP_VarCount[MP_OvrlpLocalNodeIdVecList[j]]])
        MP_OvrlpVarList.append(MP_OvrlpVar)   
        
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_InvOvrlpVarList.append(np.zeros(2*N_NbrNode_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping Var
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id        = MP_NbrMPIdVector[j]
        SendReq         = Comm.Isend(MP_OvrlpVarList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id        = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpVarList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    #Updating Var at the Meshpart boundary
    for j in range(N_NbrMP): 
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_VarSum[MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpVarList[j][:N_NbrNode_j]
        MP_VarCount[MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpVarList[j][N_NbrNode_j:] 
    
    MP_Var = MP_VarSum/(MP_VarCount+1e-15) #1e-15 is added for VarCount = 0
    
    
    return MP_Var
    

  
def getNodalPS(RefMeshPart, MP_TimeRecData, Ref):

    MP_SubDomainData            = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector            = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalNodeIdVecList  = RefMeshPart['OvrlpLocalNodeIdVecList']
    MP_NNode                    = RefMeshPart['NNode']
    
    MP_TypeGroupList            = MP_SubDomainData['StrucDataList']
    N_Type                      = len(MP_TypeGroupList) 
    
    MP_PSSum                   = np.zeros([3,MP_NNode])
    MP_PSCount                 = np.zeros(MP_NNode)
    
    for j in range(N_Type):        
        RefTypeGroup                = MP_TypeGroupList[j]    
        NNodes_ElemType             = RefTypeGroup['NNodes']
        ElemList_LocNodeIdVector    = RefTypeGroup['ElemList_LocNodeIdVector']
        ElemList_Omega              = RefTypeGroup['ElemList_Omega'] #Dmg
        ElasticityMat               = RefTypeGroup['ElasticityMat']
        ElemList_E                  = RefTypeGroup['ElemList_E']
        ElemList_LocStrainVec       = RefTypeGroup['ElemList_LocStrainVec']
        
        if Ref=='Stress':       
            ElemList_LocStressVec   = (1.0-ElemList_Omega)*ElemList_E*np.dot(ElasticityMat, ElemList_LocStrainVec)
            ElemList_PSVec          = getPrincipalStress(ElemList_LocStressVec) 
        
        elif Ref=='Strain':     
            ElemList_PSVec          = getPrincipalStrain(ElemList_LocStrainVec) 
        
        else:   raise Exception
        
        #Calculating Nodal PS
        ElemList_LocNodeIdVector_Flat  = ElemList_LocNodeIdVector.ravel()
        for i in range(3):
            ElemList_PS_i      = ElemList_PSVec[i,:]*np.ones(NNodes_ElemType)[np.newaxis].T
            MP_PSSum[i,:] += np.bincount(ElemList_LocNodeIdVector_Flat, weights=ElemList_PS_i.ravel(), minlength=MP_NNode)
            
        N_Elem                   = len(ElemList_Omega)
        FlatElemList_PSCount     = np.ones(NNodes_ElemType*N_Elem)
        MP_PSCount += np.bincount(ElemList_LocNodeIdVector_Flat, weights=FlatElemList_PSCount, minlength=MP_NNode)
        
    
    #Calculating Overlapping Fint Vectors
    N_NbrMP = len(MP_NbrMPIdVector)
            
    MP_OvrlpPSList = []
    MP_InvOvrlpPSList = []
    for j in range(N_NbrMP):
        MP_OvrlpPS = np.hstack([MP_PSSum[:, MP_OvrlpLocalNodeIdVecList[j]].ravel(), MP_PSCount[MP_OvrlpLocalNodeIdVecList[j]]])
        MP_OvrlpPSList.append(MP_OvrlpPS)
        
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_InvOvrlpPSList.append(np.zeros(4*N_NbrNode_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping PS
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id        = MP_NbrMPIdVector[j]
        SendReq         = Comm.Isend(MP_OvrlpPSList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id        = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpPSList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    #Updating PS at the Meshpart boundary
    for j in range(N_NbrMP): 
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_PSSum[0, MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][:N_NbrNode_j]
        MP_PSSum[1, MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][N_NbrNode_j:2*N_NbrNode_j]
        MP_PSSum[2, MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][2*N_NbrNode_j:3*N_NbrNode_j]
        MP_PSCount[MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][3*N_NbrNode_j:] 
    
    MP_PS = MP_PSSum/(MP_PSCount+1e-15) #1e-15 is added for PSCount = 0
    
    
    return MP_PS
    


def updateHistoryPlotData(RefMeshPart):
    
    GlobData     = RefMeshPart['GlobData']
    
    if GlobData['PlotFlag'] == 1:
        MP_PlotNDofs        = RefMeshPart['PlotNDofs']
        MP_PlotLocalDofVec  = RefMeshPart['PlotLocalDofVec']
        MP_PlotDispVector   = RefMeshPart['PlotDispVector']
        MP_PlotLoadVector   = RefMeshPart['PlotLoadVector']
        MP_Un               = RefMeshPart['MP_Un']
        
        #if Rank==0: print(TimeStepCount)
        
        # MP_Rc   = calcMPFint(MP_Un, RefMeshPart, MP_TimeRecData) - MP_Fi
        # MP_Rc   = MP_Rc*MP_WeightVector
        # MP_Load =  np.sum(MP_Rc[MP_PlotLocalDofVec])
        # Load    = MPI_SUM(MP_Load, MP_TimeRecData)
        # EStress = Load/(1.0*1.0)
        EStress = 0.0
        
        if MP_PlotNDofs>0:
            MP_PlotLoadVector[TimeStepCount] = EStress
            MP_PlotDispVector[:,TimeStepCount] = MP_Un[MP_PlotLocalDofVec]
            
    
def exportContourData(RefMeshPart):
    
    GlobData                    = RefMeshPart['GlobData']
    
    ExportKeyFrm    = GlobData['ExportKeyFrm']
    TimeStepCount   = GlobData['TimeStepCount']
    ExportFrms      = GlobData['ExportFrms']
    ResVecPath      = GlobData['ResVecPath']
    TimeList        = GlobData['TimeList']
    TimeList_T      = GlobData['TimeList_T']
    
    if GlobData['ExportFlag'] == 1:
        
        ExportNow = False
        if ExportKeyFrm>0:
            if (TimeStepCount)%ExportKeyFrm==0: ExportNow = True
        
        if TimeStepCount in ExportFrms:
            ExportNow = True
        
        if ExportNow:
            
            ExportCount                 = RefMeshPart['ExportCount']
            MP_DofWeightVector_Export   = RefMeshPart['DofWeightVector_Export']
            MP_NodeWeightVector_Export  = RefMeshPart['NodeWeightVector_Export']
            
            if 'U' in GlobData['ExportVars']:
                MP_Un = RefMeshPart['MP_Un']
                writeMPIFile_parallel(ResVecPath + 'U_' + str(ExportCount), MP_Un[MP_DofWeightVector_Export], Comm)
            
            if 'D' in GlobData['ExportVars']:
                MP_Dmg = getNodalScalarVar(RefMeshPart, 'Dmg')
                writeMPIFile_parallel(ResVecPath + 'D_' + str(ExportCount), MP_Dmg[MP_NodeWeightVector_Export], Comm)
            
            if 'ES' in GlobData['ExportVars']:
                MP_ES = getNodalScalarVar(RefMeshPart, 'EqvStrain')
                writeMPIFile_parallel(ResVecPath + 'ES_' + str(ExportCount), MP_ES[MP_NodeWeightVector_Export], Comm)
            
            if 'PE' in GlobData['ExportVars']:
                MP_PE = getNodalPS(RefMeshPart, 'Strain')
                if 'PE1' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PE1_' + str(ExportCount), MP_PE[0, MP_NodeWeightVector_Export], Comm)
                if 'PE2' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PE2_' + str(ExportCount), MP_PE[1, MP_NodeWeightVector_Export], Comm)
                if 'PE3' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PE3_' + str(ExportCount), MP_PE[2, MP_NodeWeightVector_Export], Comm)
            
            if 'PS' in GlobData['ExportVars']:
                MP_PS = getNodalPS(RefMeshPart, 'Stress')
                if 'PS1' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PS1_' + str(ExportCount), MP_PS[0, MP_NodeWeightVector_Export], Comm)
                if 'PS2' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PS2_' + str(ExportCount), MP_PS[1, MP_NodeWeightVector_Export], Comm)
                if 'PS3' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PS3_' + str(ExportCount), MP_PS[2, MP_NodeWeightVector_Export], Comm)
            
            if Rank==0: 
                T_i = TimeList[TimeStepCount]
                TimeList_T.append(T_i)
                np.save(ResVecPath + 'Time_T', TimeList_T)
             
            RefMeshPart['ExportCount'] += 1


def exportHistoryPlotData(RefMeshPart):
    
    MP_RefPlotData                      = RefMeshPart['RefPlotData']
    MPList_RefPlotDofIndicesList        = RefMeshPart['MPList_RefPlotDofIndicesList']
    qpoint                              = RefMeshPart['qpoint']
        
    GlobData                = RefMeshPart['GlobData']
    TimeList                = GlobData['TimeList']
    RefMaxTimeStepCount     = GlobData['RefMaxTimeStepCount']
    PlotFileName            = GlobData['PlotFileName']
    
    if GlobData['PlotFlag'] == 1:   
        
        MPList_PlotDispVector = Comm.gather(RefMeshPart['PlotDispVector'], root=0)
        MPList_PlotLoadVector = Comm.gather(RefMeshPart['PlotLoadVector'], root=0)
        
        if Rank == 0:
            RefPlotDofVec       = MP_RefPlotData['RefPlotDofVec']
            N_TotalPlotDofs     = len(RefPlotDofVec)
            TimeList_PlotDispVector = np.zeros([N_TotalPlotDofs, RefMaxTimeStepCount])
            TimeList_PlotLoadVector = np.zeros(RefMaxTimeStepCount)
            
            for i in range(N_Workers):
                MP_PlotDispVector_i = MPList_PlotDispVector[i]
                MP_PlotLoadVector_i = MPList_PlotLoadVector[i]
                
                RefPlotDofIndices_i = MPList_RefPlotDofIndicesList[i]
                N_PlotDofs_i = len(MP_PlotDispVector_i)
                if N_PlotDofs_i>0:
                    for j in range(N_PlotDofs_i):
                        TimeList_PlotDispVector[RefPlotDofIndices_i[j],:] = MP_PlotDispVector_i[j]
                
                if len(MP_PlotLoadVector_i)>0:
                    TimeList_PlotLoadVector = MP_PlotLoadVector_i
                        
                
            #Saving Data File
            PlotTimeData = {'Plot_T':   TimeList, 
                            'Plot_U':   TimeList_PlotDispVector, 
                            'Plot_L':   TimeList_PlotLoadVector,
                            'Plot_Dof': RefPlotDofVec+1, 
                            'qpoint':   qpoint,
                            'PEList':   RefMeshPart['PEList']}
            np.savez_compressed(PlotFileName+'_PlotData', PlotData = PlotTimeData)
            savemat(PlotFileName+'_PlotData.mat', PlotTimeData)
            
            if GlobData['EnergyFlag'] == 1:
                EnergyData = {'Plot_T':   TimeList[:-1], 
                              'PEList':   RefMeshPart['PEList'], 
                              'DEList':   RefMeshPart['DEList']}
                np.savez_compressed(PlotFileName+'_EnergyData', EnergyData = EnergyData)
                savemat(PlotFileName+'_EnergyData.mat', EnergyData)
            
        
 
def exportTimeData(RefMeshPart):
    
    GlobData        = RefMeshPart['GlobData']
    PlotFileName    = GlobData['PlotFileName']
    N_MshPrt        = N_Workers
    
    MPList_TimeRecData = Comm.gather(GlobData['MP_TimeRecData'], root=0)
    
    if Rank == 0:        
        TimeData = configTimeRecData(MPList_TimeRecData, ExportLoadUnbalanceData=True, Dmg=True)
        TimeData['PBS_JobId'] = GlobData['PBS_JobId']
        
        TimeDataFileName = PlotFileName + '_MP' +  str(N_MshPrt) + '_TimeData'
        np.savez_compressed(TimeDataFileName, TimeData = TimeData)
        savemat(TimeDataFileName +'.mat', TimeData)
            
    
    
if __name__ == "__main__":
    
    #Initializing----------------------------------------
    Comm        = MPI.COMM_WORLD
    Rank        = Comm.Get_rank()
    N_Workers   = Comm.Get_size()
    GlobData    = initGlobData()
    eps         = np.finfo(float).eps
    
    sleep(Rank*1e-4)    #A small sleep to avoid hanging
    
    #Reading Model Data Files-----------------------------
    t1_ = time()
    RefMeshPart = readModelData(GlobData)
    readGlobalSettings(GlobData)
    
    MP_NDOF                 = RefMeshPart['NDOF']
    MP_Ud                   = RefMeshPart['Ud']
    MP_NNode                = RefMeshPart['NNode']
    MP_RefLoadVector        = RefMeshPart['RefLoadVector']
    RefMaxTimeStepCount     = GlobData['RefMaxTimeStepCount']
    TimeStepDelta           = GlobData['TimeStepDelta']
    
    Comm.barrier() #barrier so that all processes start at same time
    if Rank==0: print('Time (sec) taken to read files: ', np.round(time()-t1_,2))
    
    updateTime(GlobData, 'dT_FileRead')
    t0_Start = time()
    
    #Initialising variables---------------------------------    
    RefMeshPart['MP_Un_2'] = (1e-200)*np.random.rand(MP_NDOF)
    RefMeshPart['MP_Un_1'] = (1e-200)*np.random.rand(MP_NDOF)
    RefMeshPart['MP_Un'] = (1e-200)*np.random.rand(MP_NDOF)
    
    initExportData(RefMeshPart)
    
    for TimeStepCount in range(1, RefMaxTimeStepCount):
        
        updateTimeStep(RefMeshPart, TimeStepCount)
        calcMPFint(RefMeshPart)
        calcMPDispVec(RefMeshPart)
        
        calcEnergy(RefMeshPart)
        updateHistoryPlotData(RefMeshPart)
        exportContourData(RefMeshPart)
        
        
    updateTime(GlobData, 'dT_Calc')      
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('TotalTimeStepCount', RefMaxTimeStepCount) 
    
    
    #Saving CPU Time for verification
    GlobData['MP_TimeRecData']['dT_Total_Verify'] = t0_End - t0_Start 
    GlobData['MP_TimeRecData']['t0_Start'] = t0_Start
    GlobData['MP_TimeRecData']['t0_End'] = t0_End
    GlobData['MP_TimeRecData']['MP_NDOF'] = MP_NDOF
    
    exportHistoryPlotData(RefMeshPart)
    #exportTimeData(RefMeshPart)
    