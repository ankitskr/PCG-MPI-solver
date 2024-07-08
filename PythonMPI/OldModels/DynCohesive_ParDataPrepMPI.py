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

    
    


def extract_PlotSettings(MPGData, PyDataPath, MatDataPath):
    
    MeshData_Glob_FileName = PyDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob = open(MeshData_Glob_FileName, 'rb').read()
    MeshData_Glob = pickle.loads(zlib.decompress(MeshData_Glob))
    
    
    #Reading Global data file
    PlotPnts = MatDataPath + 'PlotPnts.mat'
    PlotPnts = scipy.io.loadmat(PlotPnts)
    
    MeshData_Glob['qpoint']              = PlotPnts['qpoint']
    MeshData_Glob['RefPlotDofVec']       = PlotPnts['RefPlotDofVec'].T[0] - 1
    
    MPGData['Glob'] = MeshData_Glob
    
    





def extract_Elepart(PyDataPath, N_TotalMshPrt, N_MPGs, Rank):

    MeshPartFile = PyDataPath + 'MeshPart_' + str(N_TotalMshPrt) + '.npz'
    ElePart = np.load(MeshPartFile, allow_pickle=True)['Data']
    
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
    gc.collect()
    
    return RefElemIdVecList, MPIdList




def extract_ElemMeshData(MPGData, PyDataPath, RefElemIdVecList, MPIdList):
    
    MPGData['MeshPartList'] = []
    MPGSize = len(MPIdList)
    MPGData['MPGSize'] = MPGSize
    for i in range(MPGSize):
        MP_Id = MPIdList[i]
            
        MeshPart = {}
        MeshPart['Id']          = MP_Id
         
        MPGData['MeshPartList'].append(MeshPart)
        
    DataNameList = ['DofGlb', 'Sign', 'Type', 'Level', 'Ck', 'Cm', 'IntfcElem']
    N_Data = len(DataNameList)
    for j in range(N_Data):
        DataName = DataNameList[j]
        Data_FileName = PyDataPath + 'MeshData_'+ DataName + '.zpkl'
        
        RefElemData = open(Data_FileName, 'rb').read()
        RefElemData = pickle.loads(zlib.decompress(RefElemData))
    
        for i in range(MPGSize):
            RefElemIdList = RefElemIdVecList[i]
            
            if len(RefElemData)>0:  MPGData['MeshPartList'][i][DataName]      = RefElemData[RefElemIdList]
            else:                   MPGData['MeshPartList'][i][DataName]      = []
            
        del RefElemData
        gc.collect()
    



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
        
        if not len(MP_UniqueDofVector)%3==0:    raise Exception
    
        MP_UniqueNodeIdVector = np.int64(MP_UniqueDofVector[0::3]/3)
        
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
        
        MeshPart['DofVector'] = np.array(MP_UniqueDofVector, dtype=int)
        MeshPart['LocDofVector'] = np.array(MP_LocDofVector, dtype=object)
        MeshPart['NodeIdVector'] = MP_UniqueNodeIdVector
        MeshPart['SignVector'] = np.array(MP_SignVector, dtype=object)
        MeshPart['NNode'] = len(MP_UniqueNodeIdVector)
        








def extract_NodalVectors(MPGData, PyDataPath, Rank, PyDataPath_Part):
        
    #print('Computing Nodal vectors..')
    
    #Reading Nodal Data file
    MeshData_Nodal_FileName = PyDataPath + 'MeshData_Nodal.zpkl'
    MeshData_Nodal = open(MeshData_Nodal_FileName, 'rb').read()
    MeshData_Nodal = pickle.loads(zlib.decompress(MeshData_Nodal))
    
    [DiagM, F, Fixed, NodeCoordVec] = MeshData_Nodal
    MeshData_Glob                                             = MPGData['Glob']
    MeshPartList                                              = MPGData['MeshPartList']
    N_MeshPart                                                = len(MeshPartList)
    MPGData['NodeCoordVec'] = NodeCoordVec
    
    InvDiagM = np.array(1.0/DiagM, dtype=float)
    
    qpoint          = MeshData_Glob['qpoint']
    RefPlotDofVec   = MeshData_Glob['RefPlotDofVec']
    TestPlotFlag = True
    MPList_RefPlotDofIndicesList = []
    MPList_DofVector                      = []
    MPList_NodeIdVector                      = []
    MPList_NDofVec                        = []
    MPList_NNode                        = []
    
    for i in range(N_MeshPart):
        MeshPart                                =  MeshPartList[i]
        MP_Id                                   = MeshPart['Id']
        
        MP_DofVector                            = MeshPart['DofVector']
        MP_NDOF                                 = len(MP_DofVector)
        MP_FixedLocDofVector                     = getIndices(MP_DofVector, np.intersect1d(MP_DofVector, Fixed))
        
        MP_InvDiagM                             = InvDiagM[MP_DofVector]
        MP_NodeCoordVec                         = NodeCoordVec[MP_DofVector]
        
        MP_NodeIdVector                         = MeshPart['NodeIdVector']
        MP_NNode                                = MeshPart['NNode']
        
        MPList_DofVector.append(MP_DofVector)
        MPList_NodeIdVector.append(MP_NodeIdVector)
        MPList_NDofVec.append(MP_NDOF)
        MPList_NNode.append(MP_NNode)
        
        MP_DiagMVector                              = DiagM[MP_DofVector]
        MP_RefLoadVector                            = F[MP_DofVector]
        
        
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
        MeshPart['NDOF']                    = MP_NDOF
        MeshPart['RefPlotData']             = MP_RefPlotData
        MeshPart['NodeCoordVec']             = MP_NodeCoordVec
        MeshPart['InvDiagM']             = MP_InvDiagM
        MeshPart['FixedLocDofVector']         = MP_FixedLocDofVector
        
        MeshPart['GathDofVector']    = []
        MeshPart['GathNodeIdVector']    = []
        MeshPart['MPList_NDofVec']   = []
        MeshPart['MPList_NNode']   = []
        MeshPart['MPList_RefPlotDofIndicesList'] = []
    
    
    #Gathering Data from all the workers
    FlatDofVector = np.hstack(MPList_DofVector)
    FlatNodeIdVector = np.hstack(MPList_NodeIdVector)
    
    #Check if the MPI communication works for large models here (use the latest version of openmpi) 
    """
    NFlatDofVec = len(FlatDofVector)
    NFlatNodeVec = len(FlatNodeIdVector)
    Comm.send([NFlatDofVec, NFlatNodeVec], dest=0, tag=Rank)
    FlatNDofVecList = []
    FlatNNodeVecList = []
    if Rank == 0:                
        for j in range(N_Workers):    
            RcvdData = Comm.recv(source=j, tag=j)
            FlatNDofVecList.append(RcvdData[0])
            FlatNNodeVecList.append(RcvdData[1])
        
        GathDofVector = [np.zeros(FlatNDofVecList[j], dtype=int) for j in range(N_Workers)]
        GathNodeIdVector = [np.zeros(FlatNNodeVecList[j], dtype=int) for j in range(N_Workers)]
    
    print(0, Rank)
    Comm.barrier()
    
    SendReq0 = Comm.Isend(FlatDofVector, dest=0, tag=Rank)            
    if Rank == 0:                
        for j in range(N_Workers):    Comm.Recv(GathDofVector[j], source=j, tag=j)                
    SendReq0.Wait()
    
    SendReq1 = Comm.Isend(FlatNodeIdVector, dest=0, tag=Rank)            
    if Rank == 0:                
        for j in range(N_Workers):    Comm.Recv(GathNodeIdVector[j], source=j, tag=j)                
    SendReq1.Wait()
    """
    FlatDofFileName = PyDataPath_Part + 'FlatDofVector_' + str(Rank) + '.zpkl'
    Cmpr_FlatDofVector = zlib.compress(pickle.dumps(FlatDofVector, pickle.HIGHEST_PROTOCOL))
    f = open(FlatDofFileName, 'wb')
    f.write(Cmpr_FlatDofVector)
    f.close()
    
    FlatNodeFileName = PyDataPath_Part + 'FlatNodeIdVector_' + str(Rank) + '.zpkl'
    Cmpr_FlatNodeIdVector = zlib.compress(pickle.dumps(FlatNodeIdVector, pickle.HIGHEST_PROTOCOL))
    f = open(FlatNodeFileName, 'wb')
    f.write(Cmpr_FlatNodeIdVector)
    f.close()
    
    Gath_MPList_NDofVec = Comm.gather(MPList_NDofVec,root=0)
    Gath_MPList_NNode = Comm.gather(MPList_NNode,root=0)
    Gath_MPList_RefPlotDofIndicesList = Comm.gather(MPList_RefPlotDofIndicesList,root=0)
    
    Comm.barrier()
    
    if Rank == 0:
        
        GathDofVector = []
        GathNodeIdVector = []
        for j in range(N_Workers):
            FlatDofFileName = PyDataPath_Part + 'FlatDofVector_' + str(j) + '.zpkl'
            FlatDofData = open(FlatDofFileName, 'rb').read()
            FlatDofData = pickle.loads(zlib.decompress(FlatDofData))
            GathDofVector.append(FlatDofData)
            
            FlatNodeFileName = PyDataPath_Part + 'FlatNodeIdVector_' + str(j) + '.zpkl'
            FlatNodeIdData = open(FlatNodeFileName, 'rb').read()
            FlatNodeIdData = pickle.loads(zlib.decompress(FlatNodeIdData))
            GathNodeIdVector.append(FlatNodeIdData)
        
        GathDofVector = np.hstack(GathDofVector)
        GathNodeIdVector = np.hstack(GathNodeIdVector)
        MPList_NDofVec = np.hstack(Gath_MPList_NDofVec)
        MPList_NNode = np.hstack(Gath_MPList_NNode)
        MPList_RefPlotDofIndicesList = list(itertools.chain.from_iterable(Gath_MPList_RefPlotDofIndicesList))
        
        MeshPart =  MeshPartList[0]
        MP_Id = MeshPart['Id']
        if not MP_Id == 0 :    raise Exception
        MeshPart['GathDofVector'] = GathDofVector
        MeshPart['GathNodeIdVector'] = GathNodeIdVector
        MeshPart['MPList_NDofVec']   = MPList_NDofVec
        MeshPart['MPList_NNode']   = MPList_NNode
        MeshPart['MPList_RefPlotDofIndicesList'] = MPList_RefPlotDofIndicesList
        

    del DiagM, F, Fixed, MeshData_Nodal
    gc.collect()
    




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
        MP_Ck               = MeshPart['Ck']
        MP_Cm               = MeshPart['Cm']
        MP_IntfcElem         = MeshPart['IntfcElem']
        
        UniqueElemTypeList = np.unique(MP_Type)
        N_Type = len(UniqueElemTypeList)
        
        MP_TypeGroupList = []
        
        for j in range(N_Type):
            
            RefElemTypeId = UniqueElemTypeList[j]
            I = np.where(MP_Type==RefElemTypeId)[0]
            
            RefElemList_LocDofVector = np.array(tuple(MP_LocDofVector[I]), dtype=int).T
            RefElemList_SignVector = np.array(tuple(MP_SignVector[I]), dtype=bool).T
            RefElemList_Level = MP_Level[I]
            RefElemList_Ck = MP_Ck[I]
            RefElemList_Cm = MP_Cm[I]
            
            if len(MP_IntfcElem)>0: RefElemList_IntfcElem = MP_IntfcElem[I]
            else:                   RefElemList_IntfcElem = []
            
            MP_TypeGroup = {}
            MP_TypeGroup['ElemTypeId'] = RefElemTypeId
            MP_TypeGroup['ElemList_LocDofVector'] = RefElemList_LocDofVector
            MP_TypeGroup['ElemList_LocDofVector_Flat'] = RefElemList_LocDofVector.flatten()
            MP_TypeGroup['ElemList_SignVector'] = RefElemList_SignVector
            MP_TypeGroup['ElemList_Level'] = RefElemList_Level
            MP_TypeGroup['ElemList_Ck'] = RefElemList_Ck
            MP_TypeGroup['ElemList_Cm'] = RefElemList_Cm
            MP_TypeGroup['ElemList_IntfcElem'] = RefElemList_IntfcElem
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
            if RefElemTypeId in [-2, -1]: #Interface Elements
                pass
            else: #Octree Elements
                j0 = DB_TypeList.index(RefElemTypeId)
                RefTypeGroup['ElemStiffMat'] = np.array(Ke[j0], dtype=float)
                RefTypeGroup['ElemMassMat'] = np.array(Me[j0], dtype=float)
            
        MeshPart['SubDomainData'] = MP_SubDomainData



        
        
def config_IntfcElem(MPGData):
    
    MeshPartList        = MPGData['MeshPartList']
    N_MeshPart          = len(MeshPartList)
    
    MPList_IntfcNodeIdVector                      = []
    MPList_IntfcNNode                             = []
    
    #Calculating Interfacial node Id List for each Meshpart
    for i in range(N_MeshPart): 
        MeshPart                        = MeshPartList[i]
        MP_SubDomainData                = MeshPart['SubDomainData']
        MP_TypeGroupList                = MP_SubDomainData['StrucDataList']
        N_Type                          = len(MP_TypeGroupList) 
        MeshPart['IntfcNodeIdList']    = []
        MeshPart['IntfcLocalNodeIdList'] = []
        for tp in range(N_Type):        
            RefTypeGroup = MP_TypeGroupList[tp]    
            ElemTypeId = RefTypeGroup['ElemTypeId']
            if ElemTypeId in [-2, -1]: #Interface Elements
                IntfcElemList = RefTypeGroup['ElemList_IntfcElem']
                N_IntfcElem = len(IntfcElemList)
                MeshPart['IntfcNodeIdList'] += [IntfcElemList[io]['NodeIdList'] for io in range(N_IntfcElem)]
            
                #Calculating Local node Id for each interface element
                for io in range(N_IntfcElem):
                    IntfcElem = IntfcElemList[io]
                    IntfcElem['LocNodeIdList'] = getIndices(MeshPart['NodeIdVector'], IntfcElem['NodeIdList'])
                
        if len(MeshPart['IntfcNodeIdList'])>0:
            MeshPart['IntfcNodeIdList'] = np.array(np.unique(np.hstack(MeshPart['IntfcNodeIdList'])), dtype=int)
            MeshPart['IntfcLocalNodeIdList'] = getIndices(MeshPart['NodeIdVector'], MeshPart['IntfcNodeIdList'])

        
        MPList_IntfcNodeIdVector.append(MeshPart['IntfcNodeIdList'])
        MPList_IntfcNNode.append(len(MeshPart['IntfcNodeIdList']))
        
        MeshPart['MPList_IntfcNodeIdVector']    = []
        MeshPart['MPList_IntfcNNode']    = []
        
    
    Gath_MPList_IntfcNodeIdVector = Comm.gather(MPList_IntfcNodeIdVector,root=0)
    Gath_MPList_IntfcNNode = Comm.gather(MPList_IntfcNNode)
    if Rank == 0:
        N_Gath = len(Gath_MPList_IntfcNodeIdVector)
        for i in range(N_Gath):
            MeshPartList[0]['MPList_IntfcNodeIdVector'] += Gath_MPList_IntfcNodeIdVector[i]
            MeshPartList[0]['MPList_IntfcNNode'] += Gath_MPList_IntfcNNode[i]

              

                
                

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
        MeshPart['SendBfr'] = np.array([np.min(XCoordList), np.max(XCoordList), np.min(YCoordList), np.max(YCoordList), np.min(ZCoordList), np.max(ZCoordList), len(MeshPart['NodeIdVector']), len(MeshPart['IntfcNodeIdList'])], dtype=float)
        
        MPG_SendBfr.append(MeshPart['SendBfr'])        
    
    MPList_SendBfr = np.zeros([N_TotalMeshPart, 8],dtype=float)
    Comm.Allgather(np.array(MPG_SendBfr, dtype=float), MPList_SendBfr)
        
        
    Tol = 1e-6
    for i in range(N_MeshPart):
        MeshPart_i            = MeshPartList[i]
        MeshPart_i['PotentialNbrMPIdVector'] = []
        MeshPart_i['PotentialNbrMPId_NNodeList'] = []
        MeshPart_i['PotentialNbrMPId_IntfcNNodeList'] = []
    
        MP_Id_i                 = MeshPart_i['Id']
        
        MP_CoordLmt_i = MeshPart_i['SendBfr'][:-2]
        XMin_i = MP_CoordLmt_i[0] - Tol
        XMax_i = MP_CoordLmt_i[1] + Tol
        YMin_i = MP_CoordLmt_i[2] - Tol
        YMax_i = MP_CoordLmt_i[3] + Tol
        ZMin_i = MP_CoordLmt_i[4] - Tol
        ZMax_i = MP_CoordLmt_i[5] + Tol
        
        for MP_Id_j in range(N_TotalMeshPart):
            if not MP_Id_i == MP_Id_j:
                MP_CoordLmt_j = MPList_SendBfr[MP_Id_j][:-2]
                NNode_j = int(MPList_SendBfr[MP_Id_j][-2])
                IntfcNNode_j = int(MPList_SendBfr[MP_Id_j][-1])
                
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
                    MeshPart_i['PotentialNbrMPId_IntfcNNodeList'].append(IntfcNNode_j)
                   
                    


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
        
        MeshPart['PotentialNbrNodeIdVecList'] = []
        
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
                
            MeshPart['PotentialNbrNodeIdVecList'].append(NbrMP_NodeIdVector)
        """
        if MP_Id ==5:
            N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
            for j in range(N_PotNbrMP):
                NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]
                NbrMP_NodeIdVector = MeshPart['PotentialNbrNodeIdVecList'][j]
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
        MeshPart['OvrlpNodeIdVecList'] = []
        MeshPart['NbrMPIdVector'] = []
        
        for j in range(N_PotNbrMP):         
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]        
            NbrMP_NodeIdVector = MeshPart['PotentialNbrNodeIdVecList'][j]
            
            OvrlpNodeIdVector = np.intersect1d(MP_NodeIdVector,NbrMP_NodeIdVector, assume_unique=True)
            
            if len(OvrlpNodeIdVector) > 0:
                MP_OvrlpLocalNodeIdVector = getIndices(MP_NodeIdVector, OvrlpNodeIdVector)
                MP_OvrlpLocalDofVec = (3*MP_OvrlpLocalNodeIdVector+RefDirVec).T.ravel()
                MeshPart['OvrlpLocalDofVecList'].append(MP_OvrlpLocalDofVec)
                MeshPart['NbrMPIdVector'].append(NbrMP_Id)
                MeshPart['OvrlpNodeIdVecList'].append(OvrlpNodeIdVector)
            
    
    
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


def config_IntfcNeighbours(MPGData, Rank):
    
    #print('Computing neighbours..')
    MeshPartList        = MPGData['MeshPartList']
    MPGSize             = MPGData['MPGSize']
    N_MeshPart          = len(MeshPartList)
    
    #Sending NodeIds
    SendReqList = []
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id               = MeshPart['Id']
        
        MeshPart['PotentialNbrIntfcNodeIdVecList'] = []
        N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
        for j in range(N_PotNbrMP):        
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if not NbrWorkerId == Rank:
                if len(MeshPart['IntfcNodeIdList'])>0:
                    SendReq = Comm.Isend(MeshPart['IntfcNodeIdList'], dest=NbrWorkerId, tag=MP_Id)
                    SendReqList.append(SendReq)
                
                
    
    #Receiving NodeIds
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id               = MeshPart['Id']
        
        N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
        for j in range(N_PotNbrMP):          
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]
            NbrMP_IntfcNNode = MeshPart['PotentialNbrMPId_IntfcNNodeList'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if NbrWorkerId == Rank:
                J = NbrMP_Id%MPGSize
                NbrMP_IntfcNodeIdVector = MeshPartList[J]['IntfcNodeIdList']
            else:
                NbrMP_IntfcNodeIdVector = np.zeros(NbrMP_IntfcNNode, dtype=np.int64)
                if len(NbrMP_IntfcNodeIdVector)>0:
                    Comm.Recv(NbrMP_IntfcNodeIdVector, source=NbrWorkerId, tag=NbrMP_Id)
                
            MeshPart['PotentialNbrIntfcNodeIdVecList'].append(NbrMP_IntfcNodeIdVector)
        
    MPI.Request.Waitall(SendReqList)    
     
        
    #Finding OverLapping Nodes/Dofs
    for i in range(N_MeshPart):
        MeshPart                    = MeshPartList[i]
        MP_NodeIdVector             = MeshPart['NodeIdVector']
        MP_IntfcNodeIdVector        = MeshPart['IntfcNodeIdList']
        N_PotNbrMP = len(MeshPart['PotentialNbrMPIdVector'])
        
        MeshPart['IntfcOvrlpLocalNodeIdVecList'] = []
        MeshPart['IntfcNbrMPIdVector'] = []
        
        for j in range(N_PotNbrMP):         
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]        
            NbrMP_IntfcNodeIdVector = MeshPart['PotentialNbrIntfcNodeIdVecList'][j]
            
            IntfcOvrlpNodeIdVector = np.intersect1d(MP_IntfcNodeIdVector, NbrMP_IntfcNodeIdVector, assume_unique=True)
            
            if len(IntfcOvrlpNodeIdVector) > 0:
                IntfcOvrlpLocalNodeIdVector = getIndices(MP_NodeIdVector, IntfcOvrlpNodeIdVector)
                
                MeshPart['IntfcNbrMPIdVector'].append(NbrMP_Id)
                MeshPart['IntfcOvrlpLocalNodeIdVecList'].append(IntfcOvrlpLocalNodeIdVector)
            
    
    
                    
                    

def exportMP(MPGData, PyDataPath_Part):
    
    #print('Computing neighbours..')
    MeshPartList = MPGData['MeshPartList']
    MeshData_Glob = MPGData['Glob']
    N_MeshPart = len(MeshPartList)
    
    RefKeyList = ['Id', 'SubDomainData', 'NDOF', 'NNode', 'DofVector', 'InvDiagM', \
                  'FixedLocDofVector', 'RefLoadVector', 'NbrMPIdVector', \
                  'OvrlpLocalDofVecList', 'RefPlotData', 'MPList_RefPlotDofIndicesList', \
                  'GathDofVector', 'GathNodeIdVector', 'IntfcLocalNodeIdList', 'MPList_IntfcNodeIdVector', 'MPList_IntfcNNode', 'MPList_NDofVec', 'MPList_NNode', 'WeightVector', \
                  'Flat_ElemLocDof', 'NCount', 'N_NbrDof', \
                  'IntfcNbrMPIdVector', 'IntfcOvrlpLocalNodeIdVecList']
    
    for i in range(N_MeshPart):
        MeshPart = MeshPartList[i]
        MP_Id = MeshPart['Id']
        RefMeshPart_FileName = PyDataPath_Part + str(MP_Id) + '.zpkl'
        
        RefMeshPart = {'GlobData': MeshData_Glob}
        for RefKey in RefKeyList:   
            RefMeshPart[RefKey] = MeshPart[RefKey]
            
        Cmpr_RefMeshPart = zlib.compress(pickle.dumps(RefMeshPart, pickle.HIGHEST_PROTOCOL))
            
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
        raise Exception('No. of workers must be a factor of TotalMeshPart')
    
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
    
    extract_PlotSettings(MPGData, PyDataPath, MatDataPath)
    RefElemIdVecList, MPIdList = extract_Elepart(PyDataPath, N_TotalMshPrt, N_MPGs, Rank)
    
    print('*****************')
    print('Use shared memory functions in MPI! Check one-sided communication')
    print('https://groups.google.com/g/mpi4py/c/Fme1n9niNwQ/m/lk3VJ54WAQAJ?pli=1')
    print('https://mpi4py.readthedocs.io/en/stable/overview.html')
    print('https://stackoverflow.com/questions/32485122/shared-memory-in-mpi4py')
    print('*****************')
    
    if N_Workers == 1:
        extract_ElemMeshData(MPGData, PyDataPath, RefElemIdVecList, MPIdList)
        
    else:
        if not N_MPGs%4==0:  raise Exception('N_Workers must be a multiple of 4')
        
        """
        NCoresPerCompNode = 48
        if N_Workers <= NCoresPerCompNode:
        """
        
        #To parallelize/optimize RAM allocation across single compute nodes
        RefN = int(dN*N_Workers)
        if not N_MPGs%RefN==0:  raise Exception('Keep the value of dN as 1, 0.5 or 0.25')
        if Rank >= RefN:
            Comm.recv(source=Rank-RefN)
        if Rank==0: print(Rank, 'A')
        extract_ElemMeshData(MPGData, PyDataPath, RefElemIdVecList, MPIdList)
        if Rank+RefN < N_MPGs:
            Comm.isend(1, dest=Rank+RefN)
        
        """
        else:
            #To parallelize/optimize RAM allocation across multiple compute nodes
            if not N_Workers%NCoresPerCompNode == 0:    raise Exception('N_Workers must be a multiple of 48')
            RefN = int(dN*NCoresPerCompNode)
            if not (NCoresPerCompNode%RefN==0 and N_MPGs%RefN==0):  raise Exception('Keep the value of dN as 1, 0.5 or 0.25')
            LocalRank = Rank%NCoresPerCompNode
            if LocalRank >= RefN:
                Comm.recv(source=Rank-RefN)
            if Rank==0: print(Rank, 'A')
            extract_ElemMeshData(MPGData, PyDataPath, RefElemIdVecList, MPIdList)
            if LocalRank+RefN < NCoresPerCompNode:
                Comm.isend(1, dest=Rank+RefN)
        """
        
        Comm.barrier()
    

    if Rank==0: print(Rank, 'B')
    
    config_ElemVectors(MPGData)
    extract_NodalVectors(MPGData, PyDataPath, Rank, PyDataPath_Part)
    if Rank==0: print(Rank, 'C')
    config_TypeGroupList(MPGData)
    if Rank==0: print(Rank, 'D')
    config_ElemStiffMat(MPGData, PyDataPath)
    if Rank==0: print(Rank, 'E')
    config_IntfcElem(MPGData)
    if Rank==0: print(Rank, 'F')
    identify_PotentialNeighbours(MPGData)
    if Rank==0: print(Rank, 'G')
    config_Neighbours(MPGData, Rank)
    if Rank==0: print(Rank, 'H')
    config_IntfcNeighbours(MPGData, Rank)
    if Rank==0: print(Rank, 'I')
    exportMP(MPGData, PyDataPath_Part)
    if Rank==0: print(Rank, 'J')
    
    

