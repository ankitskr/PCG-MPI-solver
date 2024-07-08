# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import numpy as np
import scipy.io
#from tqdm import tqdm
import os, sys
import os.path
import shutil
from numpy.linalg import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools
from GeneralFunc import splitSerialData

import pickle
import zlib

import mpi4py
from mpi4py import MPI


from os import listdir
from os.path import isfile, join
    
import gc



def getIndices(A,B, CheckIntersection = False):
    
    if CheckIntersection:
        
        if not len(np.intersect1d(A,B)) == len(B):   raise Exception
    
    A_orig_indices = A.argsort()
    I = A_orig_indices[np.searchsorted(A[A_orig_indices], B)]

    return I



def extract_GlobalSettings(MPGData, PyDataPath, MatDataPath):
    
    MeshData_Glob_FileName = PyDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob = open(MeshData_Glob_FileName, 'rb').read()
    MeshData_Glob = pickle.loads(zlib.decompress(MeshData_Glob))
    
    
    #Reading Global data file
    GlobSettingsFile = MatDataPath + 'GlobSettings.mat'
    GlobSettings = scipy.io.loadmat(GlobSettingsFile)
    
    MeshData_Glob['MaxIter']             = int(GlobSettings['MaxIter'][0][0])
    MeshData_Glob['FintCalcMode']        = GlobSettings['FintCalcMode'][0] 
    MeshData_Glob['Tol']                 = GlobSettings['Tol'][0][0]    
    MeshData_Glob['Alpha']               = GlobSettings['Alpha'][0][0]    
    MeshData_Glob['Beta']                = GlobSettings['Beta'][0][0]    
    MeshData_Glob['Gamma']               = GlobSettings['Gamma'][0][0]    
    MeshData_Glob['MaxTime']             = GlobSettings['Tmax'][0][0] 
    MeshData_Glob['DeltaLambdaList']     = np.array(GlobSettings['ft'][0], dtype=float)
    MeshData_Glob['TimeStepSize']        = GlobSettings['dt'][0][0] 
    MeshData_Glob['dT_Export']           = GlobSettings['dT_Export'][0][0] 
    MeshData_Glob['PlotFlag']            = GlobSettings['PlotFlag'][0][0]   
    MeshData_Glob['ExportFlag']          = GlobSettings['ExportFlag'][0][0]  
    MeshData_Glob['UseLumpedMass']       = bool(GlobSettings['UseLumpedMass'][0][0])
    MeshData_Glob['qpoint']              = GlobSettings['qpoint']
    MeshData_Glob['RefPlotDofVec']       = GlobSettings['RefPlotDofVec'].T[0] - 1
    
    MPGData['Glob'] = MeshData_Glob
    
    





def extract_Elepart(PyDataPath, N_TotalMshPrt, N_MPGs, Rank):

    N_TotalMshPrt_Ref = N_TotalMshPrt
    if N_TotalMshPrt == 1:   
        N_TotalMshPrt_Ref = 2
    
    MeshPartFile = PyDataPath + 'MeshPart_' + str(N_TotalMshPrt_Ref) + '.npz'
    ElePart = np.load(MeshPartFile, allow_pickle=True)['Data']
    
    
    if N_TotalMshPrt == 1:
        N_Elem = len(ElePart)
        ElePart = np.zeros(N_Elem, dtype = int)
    
    if not N_TotalMshPrt%N_MPGs == 0:   raise Exception
    
    MPGSize = int(N_TotalMshPrt/N_MPGs)
    N0 = Rank*MPGSize
    N1 = (Rank+1)*MPGSize
    MPIdList = list(range(N0, N1))
    
    RefElemIdVecList = []
    for MP_Id in MPIdList:
        RefElemIdVec = np.where(ElePart==MP_Id)[0]
        RefElemIdVecList.append(RefElemIdVec)
        
    del ElePart
        
    return RefElemIdVecList, MPIdList




def extract_ElemMeshData(MPGData, PyDataPath, RefElemIdVecList, MPIdList):

    MeshData_Elem_FileName = PyDataPath + 'MeshData_Elem.zpkl'
    MeshData_Elem = open(MeshData_Elem_FileName, 'rb').read()
    MeshData_Elem = pickle.loads(zlib.decompress(MeshData_Elem))
    
    [DofGlb, Sign, Type, Level, Cm, Ck] = MeshData_Elem

    MPGData['MeshPartList'] = []
    MPGSize = len(MPIdList)
    MPGData['MPGSize'] = MPGSize
            
    for i in range(MPGSize):
        MP_Id = MPIdList[i]
        RefElemIdList = RefElemIdVecList[i]
        
        MeshPart = {}
        MeshPart['Id']      = MP_Id
        MeshPart['DofGlb']  = DofGlb[RefElemIdList]
        MeshPart['Sign']    = Sign[RefElemIdList]
        MeshPart['Type']    = Type[RefElemIdList]
        MeshPart['Level']   = Level[RefElemIdList]
        MeshPart['Cm']      = Cm[RefElemIdList]
        MeshPart['Ck']      = Ck[RefElemIdList]
        MPGData['MeshPartList'].append(MeshPart)
    
    del DofGlb, Sign, Type, Level, Cm, Ck, MeshData_Elem
    



def config_ElemVectors(MPGData):
    
    #print('Computing element vectors..')
    
    MeshPartList = MPGData['MeshPartList']
    N_MeshPart = len(MeshPartList)
    for i in range(N_MeshPart):
        MeshPart =  MeshPartList[i]
        DofGlb =    MeshPart['DofGlb']
        Sign =      MeshPart['Sign']
        N_Elem =    len(DofGlb)
        
        #Calc DofVector for each Mesh Part
        MP_NDOF = []
        MP_SignVector = []
        MP_CumDofVector = []
        
        for io in range(N_Elem):
            Elem_DofVector = DofGlb[io]
            MP_NDOF.append(len(Elem_DofVector))
            MP_CumDofVector.append(Elem_DofVector)
            
            Elem_SignVector = np.array(Sign[io], dtype=bool)
            MP_SignVector.append(Elem_SignVector)
        
        MP_CumDofVector = np.hstack(MP_CumDofVector)
        MP_UniqueDofVector = np.unique(MP_CumDofVector)
        MP_UniqueNodeIdVecotr = np.int64(MP_UniqueDofVector[0::3]/3)
        
        #Calc LocDofVector for each element
        MP_LocDofVector = []
        Elem_CumLocDofVector = getIndices(MP_UniqueDofVector, MP_CumDofVector)
        
        I0 = 0
        for io in range(N_Elem):
            Elem_NDOF = MP_NDOF[io]
            I1 = I0+Elem_NDOF
            Elem_LocDofVector = Elem_CumLocDofVector[I0:I1]
            I0 = I1
            MP_LocDofVector.append(Elem_LocDofVector)
        
        MeshPart['DofVector'] = MP_UniqueDofVector
        MeshPart['LocDofVector'] = np.array(MP_LocDofVector, dtype=object)
        MeshPart['NodeIdVector'] = MP_UniqueNodeIdVecotr
        MeshPart['SignVector'] = np.array(MP_SignVector, dtype=object)
        








def extract_NodalVectors(MPGData, PyDataPath, Rank):
        
    #print('Computing Nodal vectors..')
    
    #Reading Nodal Data file
    MeshData_Nodal_FileName = PyDataPath + 'MeshData_Nodal.zpkl'
    MeshData_Nodal = open(MeshData_Nodal_FileName, 'rb').read()
    MeshData_Nodal = pickle.loads(zlib.decompress(MeshData_Nodal))
    
    [DiagM, DP_DiagK, DP_DiagM, X0, Dof_eff, F, NodeCoordVec] = MeshData_Nodal
    MeshData_Glob                                                  = MPGData['Glob']
    MeshPartList                                              = MPGData['MeshPartList']
    N_MeshPart                                                = len(MeshPartList)
    MPGData['NodeCoordVec'] = NodeCoordVec
    
    qpoint          = MeshData_Glob['qpoint']
    RefPlotDofVec   = MeshData_Glob['RefPlotDofVec']
    TestPlotFlag = True
    MPList_RefPlotDofIndicesList = []
    MPList_DofVector                      = []
    MPList_NDofVec                        = []
    
    for i in range(N_MeshPart):
        MeshPart                                =  MeshPartList[i]
        MP_Id                                   = MeshPart['Id']
        
        MP_DofVector                            = MeshPart['DofVector']
        MP_NDOF                                 = len(MP_DofVector)
        MPList_DofVector.append(MP_DofVector)
        MPList_NDofVec.append(MP_NDOF)
        
        MP_Dof_eff                                  = np.intersect1d(MP_DofVector, Dof_eff)
        MP_LocDof_eff                               = getIndices(MP_DofVector, MP_Dof_eff)
        MP_NDof_eff                                 = len(MP_Dof_eff)
        
        MP_DiagMVector                              = DiagM[MP_DofVector]
        MP_DPDiagKVector                            = DP_DiagK[MP_DofVector]
        MP_DPDiagMVector                            = DP_DiagM[MP_DofVector]
        MP_RefLoadVector                            = F[MP_DofVector]
        
        if len(X0)>0:  MP_X0                        = X0[MP_DofVector]
        else:          MP_X0                        = np.zeros(MP_NDOF)
        
        
        MP_RefPlotData                  = {}
        MP_RefPlotData['TestPlotFlag']  = False
        MP_RefPlotData['J']             = []
        MP_RefPlotData['LocalDofVec']   = []
        MP_RefPlotData['DofVec']        = np.intersect1d(MP_DofVector, RefPlotDofVec)
        if len(MP_RefPlotData['DofVec']) > 0:
            MP_RefPlotData['LocalDofVec'] = getIndices(MP_DofVector, MP_RefPlotData['DofVec'])
            MP_RefPlotData['J'] = getIndices(RefPlotDofVec, MP_RefPlotData['DofVec'])
            if TestPlotFlag:    
                MP_RefPlotData['TestPlotFlag'] = True
                TestPlotFlag = False
        
        if MP_Id == 0:
            MP_RefPlotData['RefPlotDofVec'] = RefPlotDofVec
            MP_RefPlotData['qpoint']        = qpoint
        else:
            MP_RefPlotData['RefPlotDofVec'] = []
            MP_RefPlotData['qpoint']        = []
            
        MPList_RefPlotDofIndicesList.append(MP_RefPlotData['J'])
        
        
        MeshPart['RefLoadVector']           = MP_RefLoadVector
        MeshPart['DiagMVector']             = MP_DiagMVector
        MeshPart['DPDiagKVector']           = MP_DPDiagKVector
        MeshPart['DPDiagMVector']           = MP_DPDiagMVector
        MeshPart['X0']                      = MP_X0
        MeshPart['Dof_eff']                 = MP_Dof_eff
        MeshPart['LocDof_eff']              = MP_LocDof_eff
        MeshPart['NDOF']                    = MP_NDOF
        MeshPart['MP_NDof_eff']             = MP_NDof_eff
        MeshPart['RefPlotData']             = MP_RefPlotData
        
        MeshPart['GathDofVector']    = []
        MeshPart['MPList_NDofVec']   = []
        MeshPart['MPList_RefPlotDofIndicesList'] = []
    
    
    Gath_MPList_DofVector = Comm.gather(MPList_DofVector,root=0)
    Gath_MPList_NDofVec = Comm.gather(MPList_NDofVec,root=0)
    Gath_MPList_RefPlotDofIndicesList = Comm.gather(MPList_RefPlotDofIndicesList,root=0)
    
    if Rank == 0:
        GathDofVector = np.hstack(np.hstack(Gath_MPList_DofVector))
        MPList_NDofVec = np.hstack(Gath_MPList_NDofVec)
        MPList_RefPlotDofIndicesList = list(itertools.chain.from_iterable(Gath_MPList_RefPlotDofIndicesList))
        
        for i in range(N_MeshPart):
            MeshPart =  MeshPartList[i]
            MP_Id = MeshPart['Id']
            if MP_Id == 0 :    
                MeshPart['GathDofVector']    = GathDofVector
                MeshPart['MPList_NDofVec']   = MPList_NDofVec
                MeshPart['MPList_RefPlotDofIndicesList'] = MPList_RefPlotDofIndicesList
                break
        
        else:   raise Exception
            

    del DiagM, DP_DiagK, DP_DiagM, X0, Dof_eff, F, MeshData_Nodal





def config_TypeGroupList(MPGData):

    #print('Grouping Octrees based on their types..')
    
    MeshPartList  = MPGData['MeshPartList']
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        
        MP_LocDofVector     = MeshPart['LocDofVector']
        MP_SignVector       = MeshPart['SignVector']
        MP_Type             = MeshPart['Type']
        MP_Level            = MeshPart['Level']
        MP_Cm               = MeshPart['Cm']
        MP_Ck               = MeshPart['Ck']
        
        UniqueElemTypeList = np.unique(MP_Type)
        N_Type = len(UniqueElemTypeList)
        
        MP_TypeGroupList = []
        
        for j in range(N_Type):
            
            RefElemTypeId = UniqueElemTypeList[j]
            I = np.where(MP_Type==RefElemTypeId)[0]
            
            RefElemList_LocDofVector = np.array(tuple(MP_LocDofVector[I]), dtype=int).T
            RefElemList_SignVector = np.array(tuple(MP_SignVector[I]), dtype=bool).T
            RefElemList_Level = MP_Level[I]
            RefElemList_Cm = MP_Cm[I]
            RefElemList_Ck = MP_Ck[I]
        
            MP_TypeGroup = {}
            MP_TypeGroup['ElemTypeId'] = RefElemTypeId
            MP_TypeGroup['ElemList_LocDofVector'] = RefElemList_LocDofVector
            MP_TypeGroup['ElemList_LocDofVector_Flat'] = RefElemList_LocDofVector.flatten()
            MP_TypeGroup['ElemList_SignVector'] = RefElemList_SignVector
            MP_TypeGroup['ElemList_Level'] = RefElemList_Level
            MP_TypeGroup['ElemList_Cm'] = RefElemList_Cm
            MP_TypeGroup['ElemList_Ck'] = RefElemList_Ck
            
            MP_TypeGroupList.append(MP_TypeGroup)
            
        MeshPart['TypeGroupList'] = MP_TypeGroupList
          
        
    

def config_ElemStiffMat(MPGData, PyDataPath):

    MeshData_Lib_FileName = PyDataPath + 'MeshData_Lib.zpkl'
    MeshData_Lib = open(MeshData_Lib_FileName, 'rb').read()
    MeshData_Lib = pickle.loads(zlib.decompress(MeshData_Lib))
    
    #print('Computing element stiffness matrices..')
    
    [Me, Ke]        = MeshData_Lib
    MeshPartList    = MPGData['MeshPartList']
    N_MeshPart      = len(MeshPartList)
    
    
    #Reading Stiffness Matrices of Structured cells
    DB_TypeList = range(len(Ke))
    
    #Calculating SubdomainData
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_TypeGroupList    = MeshPart['TypeGroupList']
        N_Type              = len(MP_TypeGroupList)
        
        MP_SubDomainData = {'MixedDataList': {},
                            'StrucDataList': MP_TypeGroupList}
        
        for j in range(N_Type):
            RefTypeGroup = MP_TypeGroupList[j]
            RefElemTypeId = RefTypeGroup['ElemTypeId']
            j0 = DB_TypeList.index(RefElemTypeId)
            
            RefTypeGroup['ElemStiffMat'] = np.array(Ke[j0], dtype=float)
            RefTypeGroup['ElemMassMat'] = np.array(Me[j0], dtype=float)
        
        MeshPart['SubDomainData'] = MP_SubDomainData





def identify_PotentialNeighbours(MPGData):
    
    NodeCoordVec        = MPGData['NodeCoordVec']
    MeshPartList        = MPGData['MeshPartList']
    N_TotalMeshPart     = MPGData['TotalMeshPartCount']
    N_MeshPart          = len(MeshPartList)
    
    MPG_SendBfr = []
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        
        MP_DofVector = MeshPart['DofVector']
        XCoordList = NodeCoordVec[MP_DofVector[0::3]]
        YCoordList = NodeCoordVec[MP_DofVector[1::3]]
        ZCoordList = NodeCoordVec[MP_DofVector[2::3]]
        MeshPart['SendBfr'] = np.array([np.min(XCoordList), np.max(XCoordList), np.min(YCoordList), np.max(YCoordList), np.min(ZCoordList), np.max(ZCoordList), len(MeshPart['NodeIdVector'])], dtype=float)
        
        MPG_SendBfr.append(MeshPart['SendBfr'])        
    
    MPList_SendBfr = np.zeros([N_TotalMeshPart, 7],dtype=float)
    Comm.Allgather(np.array(MPG_SendBfr, dtype=float), MPList_SendBfr)
        
        
    Tol = 1e-6
    for i in range(N_MeshPart):
        MeshPart_i            = MeshPartList[i]
        MeshPart_i['PotentialNbrMPIdVector'] = []
        MeshPart_i['PotentialNbrMPId_NNodeList'] = []
    
        MP_Id_i                 = MeshPart_i['Id']
        
        MP_CoordLmt_i = MeshPart_i['SendBfr'][:-1]
        XMin_i = MP_CoordLmt_i[0] - Tol
        XMax_i = MP_CoordLmt_i[1] + Tol
        YMin_i = MP_CoordLmt_i[2] - Tol
        YMax_i = MP_CoordLmt_i[3] + Tol
        ZMin_i = MP_CoordLmt_i[4] - Tol
        ZMax_i = MP_CoordLmt_i[5] + Tol
        
        for MP_Id_j in range(N_TotalMeshPart):
            if not MP_Id_i == MP_Id_j:
                MP_CoordLmt_j = MPList_SendBfr[MP_Id_j][:-1]
                NNode_j = int(MPList_SendBfr[MP_Id_j][-1])
                
                XMin_j = MP_CoordLmt_j[0] - Tol
                XMax_j = MP_CoordLmt_j[1] + Tol
                YMin_j = MP_CoordLmt_j[2] - Tol
                YMax_j = MP_CoordLmt_j[3] + Tol
                ZMin_j = MP_CoordLmt_j[4] - Tol
                ZMax_j = MP_CoordLmt_j[5] + Tol
            
                if not (XMin_j > XMax_i or XMin_i > XMax_j or \
                        YMin_j > YMax_i or YMin_i > YMax_j or \
                        ZMin_j > ZMax_i or ZMin_i > ZMax_j):
                        
                    MeshPart_i['PotentialNbrMPIdVector'].append(MP_Id_j)
                    MeshPart_i['PotentialNbrMPId_NNodeList'].append(NNode_j)
                   
                    


def config_Neighbours(MPGData, Rank):
    
    #print('Computing neighbours..')
    MeshPartList        = MPGData['MeshPartList']
    MPGSize            = MPGData['MPGSize']
    N_TotalMeshPart     = MPGData['TotalMeshPartCount']
    N_MeshPart          = len(MeshPartList)
    
    #Sending NodeIds
    SendReqList = []
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id                 = MeshPart['Id']
        
        MeshPart['NbrNodeIdVecList'] = []
        
        N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
        for j in range(N_PotNbrMP):        
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if not NbrWorkerId == Rank:
                SendReq = Comm.Isend(MeshPart['NodeIdVector'], dest=NbrWorkerId, tag=MP_Id)
                SendReqList.append(SendReq)
    
    
    #Receiving NodeIds
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id                 = MeshPart['Id']
        
        N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
        for j in range(N_PotNbrMP):          
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]
            NbrMP_NNode = MeshPart['PotentialNbrMPId_NNodeList'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if NbrWorkerId == Rank:
                J = NbrMP_Id%MPGSize
                NbrMP_NodeIdVector = MeshPartList[J]['NodeIdVector']
            else:
                NbrMP_NodeIdVector = np.zeros(NbrMP_NNode, dtype=np.int64)
                Comm.Recv(NbrMP_NodeIdVector, source=NbrWorkerId, tag=NbrMP_Id)
                
            MeshPart['NbrNodeIdVecList'].append(NbrMP_NodeIdVector)
        """
        if MP_Id ==5:
            N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
            for j in range(N_PotNbrMP):
                NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]
                NbrMP_NodeIdVector = MeshPart['NbrNodeIdVecList'][j]
                NbrWorkerId = int(NbrMP_Id/MPGSize)
                J = NbrMP_Id%MPGSize
                print(NbrWorkerId, Rank, J)
                print(NbrMP_Id, len(NbrMP_NodeIdVector), NbrMP_NodeIdVector)
        """
         
                
    MPI.Request.Waitall(SendReqList)    
     
        
    #Finding OverLapping Nodes/Dofs
    RefDirVec = np.array([[0],[1],[2]], dtype=int)
    for i in range(N_MeshPart):
        MeshPart                    = MeshPartList[i]
        MP_NodeIdVector             = MeshPart['NodeIdVector']
        N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
        
        MeshPart['OvrlpLocalDofVecList'] = []
        MeshPart['NbrMPIdVector'] = []
        
        for j in range(N_PotNbrMP):         
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]        
            NbrMP_NodeIdVector = MeshPart['NbrNodeIdVecList'][j]
            
            OvrlpNodeIdVector = np.intersect1d(MP_NodeIdVector,NbrMP_NodeIdVector, assume_unique=True)
            
            if len(OvrlpNodeIdVector) > 0:
                MP_OvrlpLocalNodeIdVector = getIndices(MP_NodeIdVector, OvrlpNodeIdVector)
                MP_OvrlpLocalDofVec = (3*MP_OvrlpLocalNodeIdVector+RefDirVec).T.ravel()
                MeshPart['OvrlpLocalDofVecList'].append(MP_OvrlpLocalDofVec)
                MeshPart['NbrMPIdVector'].append(NbrMP_Id)
    
    
    #Calculating Variables for Fint Calculation
    for i in range(N_MeshPart): 
        MeshPart            = MeshPartList[i]
        
        MP_OvrlpLocalDofVecList = MeshPart['OvrlpLocalDofVecList']
        MP_SubDomainData = MeshPart['SubDomainData']
        
        N_NbrMP = len(MeshPart['NbrMPIdVector'])
        N_OvrlpLocalDofVecList = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
        N_NbrDof               = np.sum(N_OvrlpLocalDofVecList)
        
        NCount = 0
        N_Type = len(MP_SubDomainData['StrucDataList'])
        for j in range(N_Type):    NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
        
        Flat_ElemLocDof = np.zeros(NCount, dtype=int)
        I=0
        MP_TypeGroupList = MP_SubDomainData['StrucDataList']
        N_Type = len(MP_TypeGroupList) 
        for j in range(N_Type):        
            RefTypeGroup = MP_TypeGroupList[j]  
            ElemTypeId = RefTypeGroup['ElemTypeId']
            if ElemTypeId >= 0:
                ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
                N = len(ElemList_LocDofVector_Flat)
                Flat_ElemLocDof[I:I+N]=ElemList_LocDofVector_Flat
                I += N
        
        MeshPart['Flat_ElemLocDof'] = Flat_ElemLocDof
        MeshPart['NCount'] = NCount
        MeshPart['N_NbrDof'] = N_NbrDof
    
    
    #Calculating the Weight Vector    
    for i in range(N_MeshPart): 
        MeshPartList[i]['WeightVector'] = np.ones(MeshPartList[i]['NDOF'])     
        
    if N_TotalMeshPart > 1:
        for i in range(N_MeshPart): 
            MeshPart            = MeshPartList[i]        
            MP_WeightVec = MeshPart['WeightVector']
            MP_Id = MeshPart['Id']
            N_Nbr = len(MeshPart['NbrMPIdVector'])
            for jo in range(N_Nbr):
                NbrMP_Id = MeshPart['NbrMPIdVector'][jo]
                NbrMP_OvrlpLocalDofVec = MeshPart['OvrlpLocalDofVecList'][jo]
                if MP_Id > NbrMP_Id:
                    MP_WeightVec[NbrMP_OvrlpLocalDofVec] = 0
            
            """            
            if MP_Id ==5:
                for jo in range(N_Nbr):
                    NbrMP_Id = MeshPart['NbrMPIdVector'][jo]
                    NbrMP_OvrlpLocalDofVec = MeshPart['OvrlpLocalDofVecList'][jo]
                    print(NbrMP_Id, len(NbrMP_OvrlpLocalDofVec), NbrMP_OvrlpLocalDofVec)
            """
        """
        OvrlpDofVecList = []
        for i in range(N_MeshPart): 
            MeshPart            = MeshPartList[i]        
            MP_DofVector = MeshPart['DofVector']
            OvrlpLocalDofVector_Flat = np.hstack(MeshPart['OvrlpLocalDofVecList'])
            OvrlpDofVecList.append(MP_DofVector[OvrlpLocalDofVector_Flat])
        
        Unique_OvrlpDofVec = np.unique(np.hstack(OvrlpDofVecList))
        
        Unique_OvrlpDofVec_Flagged = np.array([],dtype=int)
        for i in range(N_MeshPart):
            MeshPart            = MeshPartList[i]
            MP_DofVector = MeshPart['DofVector']
            MP_WeightVector = MeshPart['WeightVector']
            
            MP_OvrlpDofVec = np.intersect1d(MP_DofVector, Unique_OvrlpDofVec, assume_unique=True)
            MP_OvrlpDofVec_Flagged = np.intersect1d(MP_OvrlpDofVec, Unique_OvrlpDofVec_Flagged)
            if len(MP_OvrlpDofVec_Flagged)>0:
                MP_OvrlpLocalDofVector_Flagged = getIndices(MP_DofVector, MP_OvrlpDofVec_Flagged)
                MP_WeightVector[MP_OvrlpLocalDofVector_Flagged] = 0.0
                
            MP_OvrlpDofVec_UnFlagged = np.setdiff1d(MP_OvrlpDofVec, MP_OvrlpDofVec_Flagged)
            if len(MP_OvrlpDofVec_UnFlagged)>0:
                Unique_OvrlpDofVec_Flagged = np.append(Unique_OvrlpDofVec_Flagged, MP_OvrlpDofVec_UnFlagged)
        """




def exportMP(MPGData, PyDataPath_Part):
    
    #print('Computing neighbours..')
    MeshPartList = MPGData['MeshPartList']
    MeshData_Glob = MPGData['Glob']
    N_MeshPart = len(MeshPartList)
    
    RefKeyList = ['Id', 'SubDomainData', 'NDOF', 'DofVector', 'DiagMVector', \
                  'DPDiagKVector', 'DPDiagMVector', 'RefLoadVector', 'NbrMPIdVector', \
                  'OvrlpLocalDofVecList', 'RefPlotData', 'MPList_RefPlotDofIndicesList', \
                  'GathDofVector', 'LocDof_eff', 'MPList_NDofVec', 'WeightVector', \
                  'X0', 'Flat_ElemLocDof', 'NCount']
    
    for i in range(N_MeshPart):
        MeshPart = MeshPartList[i]
        MP_Id = MeshPart['Id']
        RefMeshPart_FileName = PyDataPath_Part + str(MP_Id) + '.zpkl'
        
        RefMeshPart = {'GlobData': MeshData_Glob}
        for RefKey in RefKeyList:   
            RefMeshPart[RefKey] = MeshPart[RefKey]
            
        Cmpr_RefMeshPart = zlib.compress(pickle.dumps(RefMeshPart))
            
        f = open(RefMeshPart_FileName, 'wb')
        f.write(Cmpr_RefMeshPart)
        f.close()
   
   


if __name__ == "__main__":

    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    if Rank==0:    print('N_Workers', N_Workers)
    
    N_TotalMshPrt =      int(sys.argv[1])
    dN =                float(sys.argv[2])
    ScratchPath =       sys.argv[3]
    
    N_MPGs = N_Workers #No. of MeshPartGroups
    
    if not N_TotalMshPrt%N_MPGs == 0:   
        raise Exception
    
    #Checking directories
    PyDataPath = ScratchPath + 'ModelData/Py/'
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    if not (os.path.exists(PyDataPath) or os.path.exists(MatDataPath)):
        raise Exception('DataPath does not exists!')
    
    PyDataPath_Part = ScratchPath + 'ModelData/' + 'MP' +  str(N_TotalMshPrt)  + '/'
    if Rank==0:
        if os.path.exists(PyDataPath_Part):
            try:    shutil.rmtree(PyDataPath_Part)
            except:    raise Exception('PyDataPath_Part in use!')
        os.makedirs(PyDataPath_Part)
        
    Comm.barrier()
    
    MPGData = {}
    MPGData['TotalMeshPartCount'] = N_TotalMshPrt
    MPGData['N_MPGs'] = N_MPGs
    
    extract_GlobalSettings(MPGData, PyDataPath, MatDataPath)
    RefElemIdVecList, MPIdList = extract_Elepart(PyDataPath, N_TotalMshPrt, N_MPGs, Rank)
    gc.collect()
    
    
    """
    --> Map MPI processes by NUMA nodes as explained below:
    https://opus.nci.org.au/display/Help/nci-parallel
    
    
    --> UPDATE "if Rank%RefN>0:" with "Rank<RefN:" etc below
    """
    
    RefN = int(dN*N_Workers)
    if not N_MPGs%4==0:  raise Exception('N_Split must be a multiple of 4')
    if not N_MPGs%RefN==0:  raise Exception('Keep the value of dN as 1, 0.5 or 0.25')
    #if Rank >= RefN:
    if Rank%RefN>0:
        Comm.recv(source=Rank-1)
    
    if Rank in range(RefN): print(Rank, 'A')
    
    extract_ElemMeshData(MPGData, PyDataPath, RefElemIdVecList, MPIdList)
    gc.collect()
    
    if Rank+1 < N_MPGs:
        #Comm.isend(1, dest=Rank+RefN)
        Comm.isend(1, dest=Rank+1)
    
    if Rank in range(RefN): print(Rank, 'B')
    
    config_ElemVectors(MPGData)
    extract_NodalVectors(MPGData, PyDataPath, Rank)
    gc.collect()
    if Rank in range(RefN): print(Rank, 'C')
    config_TypeGroupList(MPGData)
    config_ElemStiffMat(MPGData, PyDataPath)
    identify_PotentialNeighbours(MPGData)
    config_Neighbours(MPGData, Rank)
    exportMP(MPGData, PyDataPath_Part)
    if Rank in range(RefN): print(Rank, 'D')
    
    

