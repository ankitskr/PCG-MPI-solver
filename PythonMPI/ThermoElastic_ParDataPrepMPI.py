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
#import matplotlib.pyplot as plt
import itertools
import pickle
import zlib
import mpi4py
from mpi4py import MPI
from os import listdir
from os.path import isfile, join
import gc
from scipy.sparse import csr_matrix
from GeneralFunc import loadBinDataInSharedMem
from time import time, sleep



def initGlobDataPrep():
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    #Reading inputs
    N_TotalMshPrt =     int(sys.argv[1])
    ScratchPath =       sys.argv[2]
    
    N_MPGs = N_Workers #No. of MeshPartGroups
    
    if not N_TotalMshPrt%N_MPGs == 0:   
        raise Exception('No. of workers must be a factor of TotalMeshPart')
    
    #Checking directories
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    if not os.path.exists(MatDataPath):
        raise Exception('DataPath does not exists!')
    
    PyDataPath_Part = ScratchPath + 'ModelData/MPI/'
    if Rank==0:
        if not os.path.exists(PyDataPath_Part):
            os.makedirs(PyDataPath_Part)
    
    GlobData               = {'ScratchPath':        ScratchPath,
                              'N_TotalMshPrt':      N_TotalMshPrt,
                              'N_MPGs':             N_MPGs,
                              'MatDataPath':        MatDataPath}
    
    return GlobData


def getIndices(A,B, CheckIntersection = False):
    
    if CheckIntersection:
        if not len(np.intersect1d(A,B)) == len(B):   raise Exception
    
    A_orig_indices = A.argsort()
    I = A_orig_indices[np.searchsorted(A[A_orig_indices], B)]

    return I



def extract_Elepart(MPGData):
    
    GlobData        = MPGData['GlobData']
    MatDataPath     = GlobData['MatDataPath']
    N_TotalMshPrt   = GlobData['N_TotalMshPrt']
    N_MPGs          = GlobData['N_MPGs']
    
    MeshData_Glob_FileName = MatDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
    GlobData.update(MeshData_Glob) <--- check if this works
    
    GlobNElem = GlobData['GlobNElem']
    
    #Creating share comm to handle mulitple nodes
    SharedComm = Comm.Split_type(MPI.COMM_TYPE_SHARED)
    LocalRank = SharedComm.Get_rank() #local rank on a compute-node
    
    #Loading Mesh-partitioning file into Rank 0 
    ItemSize = MPI.LONG.Get_size()
    if LocalRank == 0:    NBytes = GlobNElem*ItemSize
    else:            NBytes = 0
    
    Win = MPI.Win.Allocate_shared(NBytes, ItemSize, comm=SharedComm)

    Buf, ItemSize = Win.Shared_query(0)
    assert ItemSize == MPI.LONG.Get_size()
    ElePart = np.ndarray(buffer=Buf, dtype=int, shape=(GlobNElem,))

    if LocalRank == 0:
        MeshPartFile = MatDataPath + 'MeshPart_' + str(N_TotalMshPrt) + '.npy'
        ElePart[:] = np.load(MeshPartFile)
    Comm.barrier()
    
    assert N_TotalMshPrt%N_MPGs == 0
    
    #Loading mesh-partitioning data into individual cores
    MPGSize = int(N_TotalMshPrt/N_MPGs)
    N0 = Rank*MPGSize
    N1 = (Rank+1)*MPGSize
    MPIdList = list(range(N0, N1))
    
    RefElemIdVecList = []
    for MP_Id in MPIdList:
        RefElemIdVec = np.where(ElePart==MP_Id)[0]
        RefElemIdVecList.append(RefElemIdVec)
    
    Comm.barrier()
    del ElePart
    gc.collect()
    
    MPGData['RefElemIdVecList'] = RefElemIdVecList
    MPGData['MPIdList']         = MPIdList
    


def extract_PlotSettings(MPGData):
    
    #Reading Global data file
    GlobData    = MPGData['GlobData']
    MatDataPath = GlobData['MatDataPath']
    
    qpoint = np.fromfile(MatDataPath+'qpoint.bin', dtype=np.float64).astype(float)
    Nqpts = int(len(qpoint)/3)
    GlobData['qpoint'] = qpoint.reshape((Nqpts,3),order='F')
    
    GlobData['RefPlotDofVec'] = np.fromfile(MatDataPath+'RefPlotDofVec.bin', dtype=np.int64).astype(int)
    


    
def extract_ElemMeshData(MPGData):
    
    GlobData            = MPGData['GlobData']
    RefElemIdVecList    = MPGData['RefElemIdVecList']
    
    MPGData['MeshPartList'] = []
    MPIdList = MPGData['MPIdList']
    MPGSize = len(MPIdList)
    MPGData['MPGSize'] = MPGSize
    
    GlobNElem       = GlobData['GlobNElem']
    MatDataPath     = GlobData['MatDataPath']
    
    #A small sleep to avoid hanging
    Comm.barrier()
    sleep(Rank*1e-3)
    
    for i in range(MPGSize):
        MP_Id = MPIdList[i]
        MeshPart = {}
        MeshPart['Id']          = MP_Id
        MeshPart['IntfcElem']   = []
        MPGData['MeshPartList'].append(MeshPart)
    
    #TODO: Use appropriate DataTypes to reduce memory requirements. Change corresponding data types in ExportMatData.m
    DataNameList =    [ 'NodeGlbOffset',    'DofGlbOffset',       'SignOffset',       'Type',       'Level',          'Ck',          'Cm',          'Ce',    'PolyMat',        'sctrs']
    BinDataTypeList = [       np.int64,          np.int64 ,           np.int64,     np.int32,    np.float64,    np.float64,    np.float64,    np.float64,     np.int32,     np.float64]
    PyDataTypeList =  [            int,               int ,                int,          int,         float,         float,         float,         float,          int,          float]
    DataShapeList =   [  (GlobNElem,2),     (GlobNElem,2) ,      (GlobNElem,2), (GlobNElem,),  (GlobNElem,),  (GlobNElem,),  (GlobNElem,),  (GlobNElem,), (GlobNElem,),  (GlobNElem,3)]
    N_Data = len(DataNameList)
    
    
    for j in range(N_Data):
        DataName        = DataNameList[j]
        BinDataType     = BinDataTypeList[j]
        PyDataType      = PyDataTypeList[j]
        DataShape       = DataShapeList[j]
        
        Data_FileName = MatDataPath + DataName + '.bin'
        RefElemData = loadBinDataInSharedMem(Data_FileName, BinDataType, PyDataType, DataShape, Comm, LoadingRank=j%4)
        
        #Loading Mesh-part data into individual cores
        for i in range(MPGSize):
            RefElemIdVec = RefElemIdVecList[i]
            MeshPart =  MPGData['MeshPartList'][i]
            
            MeshPart['ElemIdVector'] = RefElemIdVec
            MeshPart['NElem'] = len(RefElemIdVec)
        
            if len(RefElemData)>0:  MeshPart[DataName]      = RefElemData[RefElemIdVec]
            else:                   MeshPart[DataName]      = []
    
        if DataName in ['Type', 'Level', 'sctrs']:
            MPGData['Glob'+DataName] = RefElemData
        
    


def config_ElemVectors(MPGData):
    
    #print('Computing element vectors..<<>>..')
    #A small sleep to avoid hanging
    sleep(Rank*1e-3)
    GlobData = MPGData['GlobData']
    GlobNNodeGlbFlat = GlobData['GlobNNodeGlbFlat']
    GlobNDofGlbFlat = GlobData['GlobNDofGlbFlat']
    SignFlat    = loadBinDataInSharedMem(MatDataPath + 'SignFlat.bin', np.int8, bool, (GlobNDofGlbFlat,), Comm, LoadingRank=2, Wait=False)
    NodeGlbFlat  = loadBinDataInSharedMem(MatDataPath + 'NodeGlbFlat.bin', np.int32, int, (GlobNNodeGlbFlat,), Comm, LoadingRank=3, Wait=False)
    DofGlbFlat  = loadBinDataInSharedMem(MatDataPath + 'DofGlbFlat.bin', np.int32, int, (GlobNDofGlbFlat,), Comm, LoadingRank=0)
    
    MeshPartList = MPGData['MeshPartList']
    N_MeshPart = len(MeshPartList)
    for i in range(N_MeshPart):
        MeshPart =  MeshPartList[i]
        
        #Calc DofVector for each Mesh Part
        MP_NElemNode         = []
        MP_SignVector   = []
        MP_CumNodeIdVector = []
        MP_NDOF         = []
        MP_CumDofVector = []
        
        MP_DofGlbOffset = MeshPart['DofGlbOffset']
        MP_NodeGlbOffset = MeshPart['NodeGlbOffset']
        MP_SignOffset   = MeshPart['SignOffset']
        MP_NElem        = MeshPart['NElem']
        
        #TODO: Perform the element loop in Cython
        for io in range(MP_NElem):
            Elem_NodeIdVector = NodeGlbFlat[MP_NodeGlbOffset[io,0]:MP_NodeGlbOffset[io,1]+1]
            MP_NElemNode.append(len(Elem_NodeIdVector))
            MP_CumNodeIdVector.append(Elem_NodeIdVector)
            
            Elem_DofVector = DofGlbFlat[MP_DofGlbOffset[io,0]:MP_DofGlbOffset[io,1]+1]
            MP_NDOF.append(len(Elem_DofVector))
            MP_CumDofVector.append(Elem_DofVector)
            
            Elem_SignVector = np.array(SignFlat[MP_SignOffset[io,0]:MP_SignOffset[io,1]+1], dtype=bool)
            MP_SignVector.append(Elem_SignVector)
        
        MP_CumNodeIdVector = np.hstack(MP_CumNodeIdVector)
        MP_UniqueNodeIdVector = np.unique(MP_CumNodeIdVector)
        MP_NNode = len(MP_UniqueNodeIdVector)
        
        MP_CumDofVector = np.hstack(MP_CumDofVector)
        MP_UniqueDofVector = np.unique(MP_CumDofVector)
        
        #Calc LocDofVector for each element
        MP_LocDofVector = []
        MP_LocNodeIdVector = []
        Elem_CumLocNodeIdVector = getIndices(MP_UniqueNodeIdVector, MP_CumNodeIdVector)
        Elem_CumLocDofVector = getIndices(MP_UniqueDofVector, MP_CumDofVector)
        
        I0 = 0
        #TODO: Perform the element loop in Cython
        for io in range(MP_NElem):
            Elem_NNode = MP_NElemNode[io]
            I1 = I0+Elem_NNode
            Elem_LocNodeIdVector = Elem_CumLocNodeIdVector[I0:I1]
            I0 = I1
            MP_LocNodeIdVector.append(Elem_LocNodeIdVector)
            
        I0 = 0
        #TODO: Perform the element loop in Cython
        for io in range(MP_NElem):
            Elem_NDOF = MP_NDOF[io]
            I1 = I0+Elem_NDOF
            Elem_LocDofVector = Elem_CumLocDofVector[I0:I1]
            I0 = I1
            MP_LocDofVector.append(Elem_LocDofVector)
            
        
        MeshPart['DofVector'] = np.array(MP_UniqueDofVector, dtype=int)
        MeshPart['LocDofVector'] = np.array(MP_LocDofVector, dtype=object)
        MeshPart['NodeIdVector'] = MP_UniqueNodeIdVector
        MeshPart['LocNodeIdVector'] = np.array(MP_LocNodeIdVector, dtype=object)
        MeshPart['SignVector'] = np.array(MP_SignVector, dtype=object)
        MeshPart['NNode'] = MP_NNode
        
    del NodeGlbFlat, SignFlat
    gc.collect()
    


def extract_NodalVectors(MPGData):
        
    #print('Computing Nodal vectors..')
    GlobData        = MPGData['GlobData']
    
    MatDataPath     = GlobData['MatDataPath']
    GlobNDof        = GlobData['GlobNDof']
    GlobNFixedDof   = GlobData['GlobNFixedDof']
    GlobNDofEff     = GlobData['GlobNDofEff']
    
    #A small sleep to avoid hanging
    Comm.barrier()
    sleep(Rank*1e-3)
    
    if Rank==0: t1 = time()
    #Reading Nodal Data file
    DiagM           = loadBinDataInSharedMem(MatDataPath + 'DiagM.bin', np.float64, float, (GlobNDof,), Comm, LoadingRank=1, Wait=False)
    F               = loadBinDataInSharedMem(MatDataPath + 'F.bin', np.float64, float, (GlobNDof,), Comm, LoadingRank=2, Wait=False)
    DofEff          = loadBinDataInSharedMem(MatDataPath + 'DofEff.bin', np.int32, int, (GlobNDofEff,), Comm, LoadingRank=3, Wait=False)
    FixedDof        = loadBinDataInSharedMem(MatDataPath + 'FixedDof.bin', np.int32, int, (GlobNFixedDof,), Comm, LoadingRank=0, Wait=False)
    Ud              = loadBinDataInSharedMem(MatDataPath + 'Ud.bin', np.float64, float, (GlobNDof,), Comm, LoadingRank=1, Wait=False)
    Vd              = loadBinDataInSharedMem(MatDataPath + 'Vd.bin', np.float64, float, (GlobNDof,), Comm, LoadingRank=2, Wait=False)
    NodeCoordVec    = loadBinDataInSharedMem(MatDataPath + 'NodeCoordVec.bin', np.float64, float, (GlobNDof,), Comm, LoadingRank=3)
    
    GlobData           = MPGData['GlobData']
    MeshPartList            = MPGData['MeshPartList']
    N_MeshPart              = len(MeshPartList)
    MPGData['NodeCoordVec'] = NodeCoordVec
    
    qpoint                          = GlobData['qpoint']
    RefPlotDofVec                   = GlobData['RefPlotDofVec']
    TestPlotFlag                    = True
    MPList_RefPlotDofIndicesList    = []
    
    for i in range(N_MeshPart):
        
        MeshPart                                =  MeshPartList[i]
        MP_Id                                   = MeshPart['Id']
        
        MP_DofVector                            = MeshPart['DofVector']
        MP_NDOF                                 = len(MP_DofVector)
        
        MP_DofEff                               = np.intersect1d(MP_DofVector, DofEff)
        MP_LocDofEff                            = getIndices(MP_DofVector, MP_DofEff)
        
        MP_LocFixedDof                             = getIndices(MP_DofVector, np.intersect1d(MP_DofVector, FixedDof))
        
        
        MP_InvDiagM                             = np.array(1.0/DiagM[MP_DofVector], dtype=float)
        MP_NodeCoordVec                         = NodeCoordVec[MP_DofVector]
        
        MP_NodeIdVector                         = MeshPart['NodeIdVector']
        MP_NNode                                = MeshPart['NNode']
        
        MP_DiagMVector                              = DiagM[MP_DofVector]
        MP_RefLoadVector                            = F[MP_DofVector]
        MP_Ud                                       = Ud[MP_DofVector]
        MP_Vd                                       = Vd[MP_DofVector]
        
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
        MeshPart['Ud']                      = MP_Ud
        MeshPart['Vd']                      = MP_Vd
        MeshPart['NDOF']                    = MP_NDOF
        MeshPart['RefPlotData']             = MP_RefPlotData
        MeshPart['NodeCoordVec']            = MP_NodeCoordVec
        MeshPart['InvDiagM']                = MP_InvDiagM
        MeshPart['DofEff']                  = MP_DofEff
        MeshPart['LocDofEff']               = MP_LocDofEff
        MeshPart['LocFixedDof']                = MP_LocFixedDof
        
        MeshPart['MPList_RefPlotDofIndicesList'] = []
    
    Gath_MPList_RefPlotDofIndicesList = Comm.gather(MPList_RefPlotDofIndicesList,root=0)
    
    Comm.barrier()
    
    if Rank == 0:
        
        MPList_RefPlotDofIndicesList = list(itertools.chain.from_iterable(Gath_MPList_RefPlotDofIndicesList))
        
        MeshPart =  MeshPartList[0]
        MP_Id = MeshPart['Id']
        if not MP_Id == 0 :    raise Exception
        MeshPart['MPList_RefPlotDofIndicesList'] = MPList_RefPlotDofIndicesList
        
    del DiagM, F, Ud, Vd, DofEff
    gc.collect()
    


def config_TypeGroupList(MPGData):

    #print('Grouping Octrees based on their types..')
    
    MeshPartList  = MPGData['MeshPartList']
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        
        MP_LocDofVector     = MeshPart['LocDofVector']
        MP_LocNodeIdVector  = MeshPart['LocNodeIdVector']
        MP_SignVector       = MeshPart['SignVector']
        MP_Type             = MeshPart['Type']
        MP_Level            = MeshPart['Level']
        MP_Ck               = MeshPart['Ck']
        MP_Cm               = MeshPart['Cm']
        MP_Ce               = MeshPart['Ce']
        MP_PolyMat          = MeshPart['PolyMat']
        MP_IntfcElem        = MeshPart['IntfcElem']
        
        UniqueElemTypeList = np.unique(MP_Type)
        N_Type = len(UniqueElemTypeList)
        
        MP_TypeGroupList = []
        
        for j in range(N_Type):
            
            RefElemTypeId = UniqueElemTypeList[j]
            I = np.where(MP_Type==RefElemTypeId)[0]
            
            RefElemList_LocDofVector    = np.array(tuple(MP_LocDofVector[I]), dtype=int).T
            RefElemList_NodeIdVector    = np.array(tuple(MP_LocNodeIdVector[I]), dtype=int).T
            RefElemList_SignVector      = np.array(tuple(MP_SignVector[I]), dtype=bool).T
            RefElemList_Level           = MP_Level[I]
            RefElemList_Ck              = MP_Ck[I]
            RefElemList_Cm              = MP_Cm[I]
            RefElemList_Ce              = MP_Ce[I]
            RefElemList_PolyMat         = MP_PolyMat[I]
            N_Elem                      = len(RefElemList_Level)
            if len(MP_IntfcElem)>0: RefElemList_IntfcElem = MP_IntfcElem[I]
            else:                   RefElemList_IntfcElem = []
            
            MP_TypeGroup = {}
            MP_TypeGroup['ElemTypeId']                  = RefElemTypeId
            MP_TypeGroup['ElemList_LocDofVector']       = RefElemList_LocDofVector
            MP_TypeGroup['ElemList_LocDofVector_Flat']  = RefElemList_LocDofVector.flatten()
            MP_TypeGroup['ElemList_LocNodeIdVector']    = RefElemList_NodeIdVector
            MP_TypeGroup['ElemList_SignVector']         = RefElemList_SignVector
            MP_TypeGroup['ElemList_Level']              = RefElemList_Level
            MP_TypeGroup['ElemList_Ck']                 = RefElemList_Ck
            MP_TypeGroup['ElemList_DmgCk_1']            = RefElemList_Ck
            MP_TypeGroup['ElemList_DmgCk']              = RefElemList_Ck
            MP_TypeGroup['ElemList_Omega']              = np.zeros(N_Elem, dtype=float)
            MP_TypeGroup['ElemList_Cm']                 = RefElemList_Cm
            MP_TypeGroup['ElemList_Ce']                 = RefElemList_Ce
            MP_TypeGroup['ElemList_E']                  = RefElemList_Ce*RefElemList_Ck
            MP_TypeGroup['ElemList_PolyMat']            = RefElemList_PolyMat
            MP_TypeGroup['ElemList_IntfcElem']          = RefElemList_IntfcElem
            MP_TypeGroup['ElemList_LocElemId']          = I
            MP_TypeGroup['N_Elem']                      = N_Elem
            
            MP_TypeGroupList.append(MP_TypeGroup)
            
        MeshPart['TypeGroupList'] = MP_TypeGroupList
          
        
    
def config_ElemMaterial(MPGData):
    
    """
    Update the code for multiple poisson's ratio
    """
    
    GlobData        = MPGData['GlobData']
    MatDataPath     = GlobData['MatDataPath']
    
    #Reading Material and Damage Properties
    MatPropFile         = MatDataPath + 'MatProp.mat'
    MatProp_Raw         = scipy.io.loadmat(MatPropFile, struct_as_record=False)['Data'][0]
    N_Mat = len(MatProp_Raw)
    MatProp = []
    for i in range(N_Mat):
        RefMatProp = MatProp_Raw[i].__dict__
        MatProp_i = {}
        MatProp_i['E']                  = RefMatProp['E'][0][0]
        MatProp_i['v']                  = RefMatProp['v'][0][0]
        MatProp_i['Rho']                = RefMatProp['Rho'][0][0]
        RefSofteningParam               = RefMatProp['SofteningParam'][0]
        
        SofteningParam  = {}
        N_SoftParam     = int(len(RefSofteningParam)/2)
        for io in range(N_SoftParam):
            Key                 = str(RefSofteningParam[2*io][0])
            Val                 = float(RefSofteningParam[2*io+1][0][0])
            SofteningParam[Key] = Val
        
        MatProp_i['SofteningParam'] = SofteningParam
        MatProp.append(MatProp_i)
    
    DmgPropFile         = MatDataPath + 'DmgProp.mat'
    DmgProp_Raw         = scipy.io.loadmat(DmgPropFile, struct_as_record=False)['Data'][0]
    RefDmgProp          = DmgProp_Raw[0].__dict__
    DmgProp             = {'Type' :          RefDmgProp['Type'][0],
                           'SofteningModel': RefDmgProp['SofteningModel'][0]}
    
    RefEqvStrainModel   = RefDmgProp['EqvStrainModel'][0]
    EqvStrainModel      = {}
    N_Param             = int(len(RefEqvStrainModel)/2)
    for i in range(N_Param):
        Key = str(RefEqvStrainModel[2*i][0])
        if Key == 'Name':   Val  = str(RefEqvStrainModel[2*i+1][0])
        else:               Val  = float(RefEqvStrainModel[2*i+1][0][0])
        EqvStrainModel[Key] = Val
    DmgProp['EqvStrainModel'] = EqvStrainModel 
    
    MPGData['MatProp']      = MatProp
    MPGData['DmgProp']      = DmgProp
    
    #Updating SubdomainData
    MeshPartList    = MPGData['MeshPartList']
    N_MeshPart      = len(MeshPartList)
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MeshPart['MatProp'] = MatProp
        MeshPart['DmgProp'] = DmgProp
        
        if not DmgProp['Type'] == 'None':
            MP_TypeGroupList    = MeshPart['TypeGroupList']
            N_Type              = len(MP_TypeGroupList)
            
            for j in range(N_Type):
                RefTypeGroup        = MP_TypeGroupList[j]
                ElemList_PolyMat    = RefTypeGroup['ElemList_PolyMat']
                ElemList_Level      = RefTypeGroup['ElemList_Level']
                N_Elem              = RefTypeGroup['N_Elem']
                Unilateral          = DmgProp['EqvStrainModel']['Unilateral']
                
                RefTypeGroup['ElemList_Omega']  = np.zeros(N_Elem)
                
                if Unilateral == 1:
                    RefTypeGroup['ElemList_Kappa_c']    = (1e-200)*np.random.rand(N_Elem)
                    RefTypeGroup['ElemList_Kappa_t']    = (1e-200)*np.random.rand(N_Elem)
                    RefTypeGroup['ElemList_ep0_c']      = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_ep0_t']      = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_A_c']        = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_A_t']        = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_B_c']        = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_B_t']        = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_k']          = np.zeros(N_Elem)
                    
                    for ki in range(N_Elem):
                        RefMatProp  = MatProp[ElemList_PolyMat[ki]]
                        ep0_c        = RefMatProp['SofteningParam']['ep0_c']
                        ep0_t        = RefMatProp['SofteningParam']['ep0_t']
                        A_c          = RefMatProp['SofteningParam']['A_c']
                        A_t          = RefMatProp['SofteningParam']['A_t']
                        k           = RefMatProp['SofteningParam']['k']
                        
                        if DmgProp['Type'] == 'PartReg':
                            raise NotImplementedError
                        
                        elif DmgProp['Type'] in ['Local', 'NonLocal']:
                            B_c = RefMatProp['SofteningParam']['B_c']
                            B_t = RefMatProp['SofteningParam']['B_t']
                            
                        else:   raise Exception
                        
                        RefTypeGroup['ElemList_ep0_c'][ki]   = ep0_c
                        RefTypeGroup['ElemList_ep0_t'][ki]   = ep0_t
                        RefTypeGroup['ElemList_A_c'][ki]     = A_c
                        RefTypeGroup['ElemList_A_t'][ki]     = A_t
                        RefTypeGroup['ElemList_B_c'][ki]     = B_c
                        RefTypeGroup['ElemList_B_t'][ki]     = B_t
                        RefTypeGroup['ElemList_k'][ki]       = k
                    
                else:
                    
                    RefTypeGroup['ElemList_Kappa']   = (1e-200)*np.random.rand(N_Elem)
                    RefTypeGroup['ElemList_ep0']    = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_epf']    = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_Alpha']  = np.zeros(N_Elem)
                    RefTypeGroup['ElemList_Beta']   = np.zeros(N_Elem)
                    
                    for ki in range(N_Elem):
                        RefMatProp = MatProp[ElemList_PolyMat[ki]]
                        ep0 = RefMatProp['SofteningParam']['ep0']
                        Alpha = RefMatProp['SofteningParam']['Alpha']
                        
                        if DmgProp['Type'] == 'PartReg':
                            E   = RefMatProp['E']
                            Gf  = RefMatProp['SofteningParam']['Gf']
                            lc  = 1.0*ElemList_Level[ki] #use 1.24? --> Diameter of a sphere with same volume as the cubic cell
                            
                            #l_lim = 0.5*(2*Gf/(E*ep0*ep0)) #Limiting characteristic length: Crack band theory for fracture of concrete 
                            #if lc>l_lim:    lc = l_lim
                            
                            if 'Linear' == DmgProp['SofteningModel']:
                                epf = 2*Gf/(E*lc*ep0)
                            elif 'Exponential' in DmgProp['SofteningModel']:
                                epf = Gf/(E*lc*ep0) + ep0/2.0
                            else:   raise Exception
                            
                            if epf < ep0:   raise Exception
                            
                            Beta = 1.0/(epf-ep0)
                        
                        elif DmgProp['Type'] in ['Local', 'NonLocal']:
                            Beta = RefMatProp['SofteningParam']['Beta']
                            epf     = 1.0/Beta + ep0;
                        
                        else:   raise Exception
                        
                        RefTypeGroup['ElemList_ep0'][ki] = ep0
                        RefTypeGroup['ElemList_epf'][ki] = epf
                        RefTypeGroup['ElemList_Alpha'][ki] = Alpha
                        RefTypeGroup['ElemList_Beta'][ki] = Beta
                    
                                
        
def config_ElemLib(MPGData):
    
    
    GlobData        = MPGData['GlobData']
    MatDataPath     = GlobData['MatDataPath']
    
    KeDataFile          = MatDataPath + 'Ke.mat'
    MeDataFile          = MatDataPath + 'Me.mat'
    SeDataFile          = MatDataPath + 'Se.mat'
    Ke                  = scipy.io.loadmat(KeDataFile)['Data'][0]
    Me                  = scipy.io.loadmat(MeDataFile)['Data'][0]
    Se                  = scipy.io.loadmat(SeDataFile)['Data'][0]
    
    MeshPartList    = MPGData['MeshPartList']
    N_MeshPart      = len(MeshPartList)
    
    
    #Reading Stiffness Matrices of Structured cells
    N_UnqTypes = len(Ke)
    DB_UnqTypeList = range(N_UnqTypes)
    
    #Calculating SubdomainData
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_TypeGroupList    = MeshPart['TypeGroupList']
        N_Type              = len(MP_TypeGroupList)
        
        MP_SubDomainData = {'MixedDataList': {},
                            'StrucDataList': MP_TypeGroupList}
        
        MatProp = MeshPart['MatProp']
        
        for j in range(N_Type):
            RefTypeGroup = MP_TypeGroupList[j]
            
            RefElemTypeId = RefTypeGroup['ElemTypeId']
            if RefElemTypeId in [-2, -1]: #Interface Elements
                pass
            else: #Octree Elements
                j0 = DB_UnqTypeList.index(RefElemTypeId)
                RefTypeGroup['ElemStiffMat']        = np.array(Ke[j0], dtype=float)
                RefTypeGroup['ElemDiagStiffMat']    = np.array(np.diag(Ke[j0]), dtype=float)
                RefTypeGroup['ElemMassMat']         = np.array(Me[j0], dtype=float)
                RefTypeGroup['ElemStrainModeMat']   = np.array(Se[j0], dtype=float)
                RefTypeGroup['NNodes']              = int(len(Ke[j0])/3)
                
                """
                Update the code for multiple poisson's ratio
                """
                v = MatProp[0]['v']
                E = 1.0
                pos = v
                D = (E/((1+pos)*(1-2*pos)))*np.array([[ (1-pos),    pos,        pos,        0,              0,              0],
                                                     [   pos,       (1-pos),    pos,        0,              0,              0],
                                                     [   pos,       pos,        (1-pos),    0,              0,              0],
                                                     [   0,         0,          0,          (1-2*pos)/2.0,  0,              0],
                                                     [   0,         0,          0,          0,              (1-2*pos)/2.0,  0],
                                                     [   0,         0,          0,          0,              0,              (1-2*pos)/2]], dtype=float)

                RefTypeGroup['PoissonRatio'] = v
                RefTypeGroup['ElasticityMat'] = D

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



def checkBoxIntersection(BB0, BB1, Tol):
                
    XMin_i = BB0[0] - Tol;     XMax_i = BB0[1] + Tol
    YMin_i = BB0[2] - Tol;     YMax_i = BB0[3] + Tol
    ZMin_i = BB0[4] - Tol;     ZMax_i = BB0[5] + Tol

    XMin_j = BB1[0] - Tol;     XMax_j = BB1[1] + Tol
    YMin_j = BB1[2] - Tol;     YMax_j = BB1[3] + Tol
    ZMin_j = BB1[4] - Tol;     ZMax_j = BB1[5] + Tol

    if not (XMin_j > XMax_i or XMin_i > XMax_j or \
            YMin_j > YMax_i or YMin_i > YMax_j or \
            ZMin_j > ZMax_i or ZMin_i > ZMax_j):
        
            return True
            
    else:   return False
    
    
              
def identify_PotentialNeighbours(MPGData, Tol=1e-6, Mode='ImmediateNbr'):
    
    def initData(MPGData):
            
        NodeCoordVec                    = MPGData['NodeCoordVec']
        MeshPartList                    = MPGData['MeshPartList']
        N_TotalMeshPart                 = MPGData['GlobData']['N_TotalMshPrt']
        N_MeshPart                      = len(MeshPartList)
        MPGData['PotentialNbrDataFlag'] = True
        MPG_SendBfr                     = []
        for i in range(N_MeshPart):
            MeshPart            = MeshPartList[i]
            
            MP_DofVector        = MeshPart['DofVector']
            XCoordList          = NodeCoordVec[MP_DofVector[0::3]]
            YCoordList          = NodeCoordVec[MP_DofVector[1::3]]
            ZCoordList          = NodeCoordVec[MP_DofVector[2::3]]
            MeshPart['PotentialNbr_SendBfr'] = np.array([np.min(XCoordList), np.max(XCoordList), np.min(YCoordList), np.max(YCoordList), np.min(ZCoordList), np.max(ZCoordList), \
                                                         len(MeshPart['NodeIdVector']), len(MeshPart['IntfcNodeIdList']), len(MeshPart['ElemIdVector'])], dtype=float)
            MPG_SendBfr.append(MeshPart['PotentialNbr_SendBfr'])
        
        MPList_SendBfr = np.zeros([N_TotalMeshPart, 9],dtype=float)
        MPGData['MPList_PotNbrSendBfr'] = MPList_SendBfr
        Comm.Allgather(np.array(MPG_SendBfr, dtype=float), MPList_SendBfr)
        
        
        
    if MPGData['PotentialNbrDataFlag'] == False:    initData(MPGData)
        
    MeshPartList        = MPGData['MeshPartList']
    N_MeshPart          = len(MeshPartList)
    MPList_SendBfr      = MPGData['MPList_PotNbrSendBfr']
    N_TotalMeshPart     = MPGData['GlobData']['N_TotalMshPrt']
        
    for i in range(N_MeshPart):
        MeshPart_i            = MeshPartList[i]
        if Mode=='ImmediateNbr':
            MeshPart_i['PotentialNbrMPIdVector'] = []
            MeshPart_i['PotentialNbrMPId_NNodeList'] = []
            MeshPart_i['PotentialNbrMPId_IntfcNNodeList'] = []
            
        elif Mode=='NonLocalNbr':
            MeshPart_i['PotNonLocNbrMP_IdVector'] = []
            MeshPart_i['PotNonLocNbrMP_NElemList'] = []
            MeshPart_i['PotNonLocNbrMP_BoundingBoxList'] = []
        
        MP_Id_i                 = MeshPart_i['Id']
        
        MP_CoordLmt_i = MeshPart_i['PotentialNbr_SendBfr'][:6]
        
        for MP_Id_j in range(N_TotalMeshPart):
            if not MP_Id_i == MP_Id_j:
                MP_CoordLmt_j = MPList_SendBfr[MP_Id_j][:6]
                NNode_j = int(MPList_SendBfr[MP_Id_j][6])
                IntfcNNode_j = int(MPList_SendBfr[MP_Id_j][7])
                NElem_j = int(MPList_SendBfr[MP_Id_j][8])
                
                if checkBoxIntersection(MP_CoordLmt_i, MP_CoordLmt_j, Tol):
                
                    if Mode=='ImmediateNbr':
                        MeshPart_i['PotentialNbrMPIdVector'].append(MP_Id_j)
                        MeshPart_i['PotentialNbrMPId_NNodeList'].append(NNode_j)
                        MeshPart_i['PotentialNbrMPId_IntfcNNodeList'].append(IntfcNNode_j)
                    
                    elif Mode=='NonLocalNbr':
                        MeshPart_i['PotNonLocNbrMP_IdVector'].append(MP_Id_j)
                        MeshPart_i['PotNonLocNbrMP_NElemList'].append(NElem_j)
                        MeshPart_i['PotNonLocNbrMP_BoundingBoxList'].append(MP_CoordLmt_j)
                        

   
def config_Neighbours(MPGData):
    
    #print('Computing neighbours..')
    MeshPartList        = MPGData['MeshPartList']
    MPGSize            = MPGData['MPGSize']
    N_TotalMeshPart     = MPGData['GlobData']['N_TotalMshPrt']
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
        MeshPart['OvrlpLocalNodeIdVecList'] = []
        MeshPart['OvrlpNodeIdVecList'] = []
        MeshPart['NbrMPIdVector'] = []
        
        for j in range(N_PotNbrMP):         
            NbrMP_Id    = MeshPart['PotentialNbrMPIdVector'][j]        
            NbrMP_NodeIdVector = MeshPart['PotentialNbrNodeIdVecList'][j]
            
            #TODO: Find MP_BoundaryNodeIdVector (instead of MP_NodeIdVector) by using Matlab code where unique boundary faces are identified
            OvrlpNodeIdVector = np.intersect1d(MP_NodeIdVector,NbrMP_NodeIdVector, assume_unique=True)
            
            if len(OvrlpNodeIdVector) > 0:
                MP_OvrlpLocalNodeIdVector = getIndices(MP_NodeIdVector, OvrlpNodeIdVector)
                MP_OvrlpLocalDofVec = (3*MP_OvrlpLocalNodeIdVector+RefDirVec).T.ravel()
                MeshPart['OvrlpLocalDofVecList'].append(MP_OvrlpLocalDofVec)
                MeshPart['NbrMPIdVector'].append(NbrMP_Id)
                MeshPart['OvrlpNodeIdVecList'].append(OvrlpNodeIdVector)
                MeshPart['OvrlpLocalNodeIdVecList'].append(MP_OvrlpLocalNodeIdVector)
                
    
    
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
        MeshPart                        = MeshPartList[i]
        MeshPart['DofWeightVector']     = np.ones(MeshPart['NDOF'])     
        MeshPart['NodeWeightVector']    = np.ones(MeshPart['NNode'])     
        
    if N_TotalMeshPart > 1:
        for i in range(N_MeshPart): 
            MeshPart        = MeshPartList[i]        
            MP_DofWeightVec = MeshPart['DofWeightVector']
            MP_NodeWeightVec = MeshPart['NodeWeightVector']
            
            MP_Id           = MeshPart['Id']
            N_Nbr           = len(MeshPart['NbrMPIdVector'])
            for jo in range(N_Nbr):
                NbrMP_Id                    = MeshPart['NbrMPIdVector'][jo]
                NbrMP_OvrlpLocalDofVec      = MeshPart['OvrlpLocalDofVecList'][jo]
                NbrMP_OvrlpLocalNodeIdVec   = MeshPart['OvrlpLocalNodeIdVecList'][jo]
                if MP_Id > NbrMP_Id:
                    MP_DofWeightVec[NbrMP_OvrlpLocalDofVec]     = 0
                    MP_NodeWeightVec[NbrMP_OvrlpLocalNodeIdVec] = 0
            
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
            MP_DofWeightVector = MeshPart['DofWeightVector']
            
            MP_OvrlpDofVec = np.intersect1d(MP_DofVector, Unique_OvrlpDofVec, assume_unique=True)
            MP_OvrlpDofVec_Flagged = np.intersect1d(MP_OvrlpDofVec, Unique_OvrlpDofVec_Flagged)
            if len(MP_OvrlpDofVec_Flagged)>0:
                MP_OvrlpLocalDofVector_Flagged = getIndices(MP_DofVector, MP_OvrlpDofVec_Flagged)
                MP_DofWeightVector[MP_OvrlpLocalDofVector_Flagged] = 0.0
                
            MP_OvrlpDofVec_UnFlagged = np.setdiff1d(MP_OvrlpDofVec, MP_OvrlpDofVec_Flagged)
            if len(MP_OvrlpDofVec_UnFlagged)>0:
                Unique_OvrlpDofVec_Flagged = np.append(Unique_OvrlpDofVec_Flagged, MP_OvrlpDofVec_UnFlagged)
        """


       
def config_NonlocalNeighbours(MPGData):
        
    """
    Create a separate python program file to compute Nonlocal element neighbours
        - partition the mesh into 10000 or more parts. Ensure each partition has less than 100k elements.
    """
    
    t1 = time()
    
    #Calculating Reference characteristic length (so as to efficiently calculate neighbour elements within the area of influence of Gaussian distribution
    DmgProp                     = MPGData['DmgProp']
    
    if not DmgProp['Type'] == 'NonLocal':    return
    
    MatProp                     = MPGData['MatProp']
    N_Mat                       = len(MatProp)
    Ko                          = 3.2
    RefLc                       = Ko*np.max([MatProp[i]['SofteningParam']['Lc'] for i in range(N_Mat)])
    
    # Identify potential non-local neighbour Mesh Parts
    identify_PotentialNeighbours(MPGData, Tol=RefLc, Mode='NonLocalNbr')
    
    # Identify potential non-local neighbour Elements
    MeshPartList        = MPGData['MeshPartList']
    MPGSize             = MPGData['MPGSize']
    N_TotalMeshPart     = MPGData['GlobData']['N_TotalMshPrt']
    N_MeshPart          = len(MeshPartList)
    
    #Sending ElemIds
    SendReqList = []
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id                = MeshPart['Id']
        
        MeshPart['PotNonLocNbrMP_ElemIdVecList'] = []
        
        N_PotNbrMP = len(MeshPart['PotNonLocNbrMP_IdVector'])
        for j in range(N_PotNbrMP):        
            NbrMP_Id    = MeshPart['PotNonLocNbrMP_IdVector'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if not NbrWorkerId == Rank:
                SendReq = Comm.Isend(MeshPart['ElemIdVector'], dest=NbrWorkerId, tag=MP_Id)
                SendReqList.append(SendReq)
    
    #Receiving ElemIds
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id                 = MeshPart['Id']
        
        N_PotNbrMP = len(MeshPart['PotNonLocNbrMP_IdVector'])
        for j in range(N_PotNbrMP):          
            NbrMP_Id    = MeshPart['PotNonLocNbrMP_IdVector'][j]
            NbrMP_NElem = MeshPart['PotNonLocNbrMP_NElemList'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if NbrWorkerId == Rank:
                J = NbrMP_Id%MPGSize
                NbrMP_ElemIdVec = MeshPartList[J]['ElemIdVector']
            else:
                NbrMP_ElemIdVec = np.zeros(NbrMP_NElem, dtype=np.int64)
                Comm.Recv(NbrMP_ElemIdVec, source=NbrWorkerId, tag=NbrMP_Id)
                
            MeshPart['PotNonLocNbrMP_ElemIdVecList'].append(NbrMP_ElemIdVec)
                
    MPI.Request.Waitall(SendReqList)    
    
    
    
    #Calculate non-local neighbour Elements
    GlobSctrs                   = MPGData['Globsctrs']
    GlobType                    = MPGData['GlobType']
    GlobLevel                   = MPGData['GlobLevel']
    if not np.all((0<=GlobType) & (GlobType<=143)): 
        raise NotImplementedError
    
    
    for i in range(N_MeshPart):
        MeshPart                    = MeshPartList[i]
        # NL_MPBBList                 = MeshPart['PotNonLocNbrMP_BoundingBoxList']
        PotNL_ElemIdVecList            = MeshPart['PotNonLocNbrMP_ElemIdVecList']
        MP_PolyMat                  = MeshPart['PolyMat']
        MP_Sctrs                    = MeshPart['sctrs']
        MP_NElem                    = MeshPart['NElem']
        
        MP_ElemList_NLElemIds       = []
        MeshPart['ElemList_NLElemIds'] = MP_ElemList_NLElemIds
        
        PotNL_ElemIdVec = np.hstack([MeshPart['ElemIdVector']] + PotNL_ElemIdVecList)
        PotNL_ElemSctrList = GlobSctrs[PotNL_ElemIdVec, :]
        PotNL_NElem = len(PotNL_ElemIdVec)
        if PotNL_NElem <=1 :   raise Exception
        Io = np.arange(PotNL_NElem)
        
        for j in range(MP_NElem):
        
            #Elem Bounding box
            ElemSctr = MP_Sctrs[j];     dR = RefLc*np.ones(3)
            ElemBB_L = ElemSctr - dR;   ElemBB_U = ElemSctr + dR
            
            #NbrMP Bounding box
            """
            PotNL_ElemIdVec = [MeshPart['ElemIdVector']]
            BB0 = np.array([ElemSctr[0], ElemSctr[0], ElemSctr[1], ElemSctr[1], ElemSctr[2], ElemSctr[2]], dtype=float)
            for k in range(N_PotNbrMP):
                BB1 = NL_MPBBList[k]
                if checkBoxIntersection(BB0, BB1, RefLc):
                    PotNL_ElemIdVec.append(NL_ElemIdVecList[k])
            PotNL_ElemIdVec = np.hstack(PotNL_ElemIdVec)
            NbrElem_Sctrs = GlobSctrs[PotNL_ElemIdVec]
            PotNL_NElem = len(PotNL_ElemIdVec)
            if PotNL_NElem <=1 :   raise Exception
            Io = np.arange(PotNL_NElem)
            """
            
            #Searching for NonLoc Nbr Elements
            Ii = Io[np.all((ElemBB_L <= PotNL_ElemSctrList) & (PotNL_ElemSctrList <= ElemBB_U), axis=1)]
            
            """
            W = NbrElem_Sctrs[:,0]
            Ii = Io[(ElemBB_L[0] <= W) & (W <= ElemBB_U[0])]
            for i in [1,2]:
                W = NbrElem_Sctrs[Ii,i]
                Ii = Ii[(ElemBB_L[i] <= W) & (W <= ElemBB_U[i])]
            """
            
            MP_ElemList_NLElemIds.append(PotNL_ElemIdVec[Ii])
      
    """
    Create a separate python program file for the above part
    """
    if Rank==0: print('I0', time()-t1)
    t1 = time()
    Comm.barrier()
    
    for i in range(N_MeshPart):
        
        MeshPart                    = MeshPartList[i]
        MP_ElemList_NLElemIds       = MeshPart['ElemList_NLElemIds']
        MP_ElemIdVector             = MeshPart['ElemIdVector']
        MP_PolyMat                  = MeshPart['PolyMat']
        MP_Sctrs                    = MeshPart['sctrs']
        MP_NElem                    = MeshPart['NElem']
        
        NL_ElemIdVec               = np.unique(np.hstack(MP_ElemList_NLElemIds))
        NL_BoundaryElemIdVec       = np.setdiff1d(NL_ElemIdVec, MP_ElemIdVector, assume_unique=True)
        
        NL_NElem                   = len(NL_ElemIdVec)
        MeshPart['NL_ElemIdVec']   = NL_ElemIdVec
        MeshPart['NL_BoundaryElemIdVec']   = NL_BoundaryElemIdVec
        MeshPart['NL_NElem']        = NL_NElem
        
        MP_NLElemLocIdVec           = getIndices(NL_ElemIdVec, MP_ElemIdVector)
        MeshPart['NL_ElemLocIdVec'] = MP_NLElemLocIdVec
        
        #MP_NLSpWeightMatrix         = csr_matrix((MP_NElem, NL_NElem), dtype=float)
        
        NL_ElemSctrList             = GlobSctrs[NL_ElemIdVec, :]
        NL_ElemLevelList            = GlobLevel[NL_ElemIdVec]
        NL_CellVolList              = np.power(NL_ElemLevelList, 3)  
        
        #TODO: Perform the element loop in Cython
        ElemList_WeightVec = []
        ElemList_Ii = []
        for j in range(MP_NElem):
            #if Rank in [0, 1000, 2000, 3000] and j%1000==0: print(Rank, np.round(j*100/MP_NElem))
            
            ElemSctr                = MP_Sctrs[j]
            
            Elem_NLElemIds          = MP_ElemList_NLElemIds[j]
            Ii                      = getIndices(NL_ElemIdVec, Elem_NLElemIds)
            ElemList_Ii.append(Ii)
            
            
            Elem_NLElemSctrList     = NL_ElemSctrList[Ii, :]
            re                      = np.linalg.norm(Elem_NLElemSctrList - ElemSctr, axis=1);
            Lc                      = MatProp[MP_PolyMat[j]]['SofteningParam']['Lc']
            Elem_AlphaList          = np.exp(-0.5*re*re/(Lc*Lc))
            Elem_CellVolList        = NL_CellVolList[Ii]
            
            Elem_WeightVec          = Elem_AlphaList*Elem_CellVolList
            Elem_WeightVec          /= np.sum(Elem_WeightVec) #To remove the effect of the boundary
            ElemList_WeightVec.append(Elem_WeightVec)
            
            #Todo: Update CSR Matrix only once (outside this loop) to speed-up
            #To achieve this, save Elem_WeightVec and its location in a separate variable -- done
            #MP_NLSpWeightMatrix     += csr_matrix((Elem_WeightVec, (j*np.ones(len(Ii),dtype=int), Ii)), shape=(MP_NElem, NL_NElem), dtype=float)
        
        MP_WeightVec = np.hstack(ElemList_WeightVec)
        MP_WeightVec_RowInd = np.hstack([j*np.ones(len(ElemList_Ii[j]),dtype=int) for j in range(MP_NElem)])
        MP_WeightVec_ColInd = np.hstack(ElemList_Ii)
        
        MP_NLSpWeightMatrix     = csr_matrix((MP_WeightVec, (MP_WeightVec_RowInd, MP_WeightVec_ColInd)), shape=(MP_NElem, NL_NElem), dtype=float)
        MeshPart['NLSpWeightMatrix'] = MP_NLSpWeightMatrix
        
        
    if Rank==0: print('I1', time()-t1)
    t1 = time()
    Comm.barrier()
    
    
    #Send/Recv NLElemIds
    SendReqList = []
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id                = MeshPart['Id']
        
        MeshPart['PotNonLocNbrMP_BoundaryElemIdVecList'] = []
        
        N_PotNbrMP = len(MeshPart['PotNonLocNbrMP_IdVector'])
        for j in range(N_PotNbrMP):        
            NbrMP_Id    = MeshPart['PotNonLocNbrMP_IdVector'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if not NbrWorkerId == Rank:
                SendReq = Comm.isend(MeshPart['NL_BoundaryElemIdVec'], dest=NbrWorkerId, tag=MP_Id)
                SendReqList.append(SendReq)
    
    for i in range(N_MeshPart):
        MeshPart            = MeshPartList[i]
        MP_Id                 = MeshPart['Id']
        
        N_PotNbrMP = len(MeshPart['PotNonLocNbrMP_IdVector'])
        for j in range(N_PotNbrMP):          
            NbrMP_Id    = MeshPart['PotNonLocNbrMP_IdVector'][j]
            NbrWorkerId = int(NbrMP_Id/MPGSize)
            if NbrWorkerId == Rank:
                J = NbrMP_Id%MPGSize
                NbrMP_BoundaryElemIdVec = MeshPartList[J]['NL_BoundaryElemIdVec']
            else:
                NbrMP_BoundaryElemIdVec = Comm.recv(source=NbrWorkerId, tag=NbrMP_Id)
                
            MeshPart['PotNonLocNbrMP_BoundaryElemIdVecList'].append(NbrMP_BoundaryElemIdVec)
                
    MPI.Request.Waitall(SendReqList)    
    
    if Rank==0: print('I2', time()-t1)
    t1 = time()
    Comm.barrier()
    
    
    
    #Finding Non-local OverLapping elments
    for i in range(N_MeshPart):
        MeshPart                    = MeshPartList[i]
        MP_Id                       = MeshPart['Id']
        NL_ElemIdVec                = MeshPart['NL_ElemIdVec']
        NL_BoundaryElemIdVec        = MeshPart['NL_BoundaryElemIdVec']
        MP_ElemIdVector             = MeshPart['ElemIdVector']
        N_PotNbrMP                  = len(MeshPart['PotNonLocNbrMP_IdVector'])
        
        MeshPart['OvrlpLocElemIdVecList']       = []
        MeshPart['NL_InvOvrlpLocElemIdVecList'] = []
        MeshPart['NL_NbrMPIdVector']            = []
        
        for j in range(N_PotNbrMP):         
            NbrMP_Id            = MeshPart['PotNonLocNbrMP_IdVector'][j]        
            
            NbrMP_BoundaryElemIdVector  = MeshPart['PotNonLocNbrMP_BoundaryElemIdVecList'][j]
            OvrlpElemIdVector   = np.intersect1d(MP_ElemIdVector,NbrMP_BoundaryElemIdVector, assume_unique=True)
            Flag = False
            if len(OvrlpElemIdVector) > 0:
                Flag = True
                MP_OvrlpLocElemIdVec = getIndices(MP_ElemIdVector, OvrlpElemIdVector)
                MeshPart['NL_NbrMPIdVector'].append(NbrMP_Id)
                MeshPart['OvrlpLocElemIdVecList'].append(MP_OvrlpLocElemIdVec)
            
            NbrMP_ElemIdVector  = MeshPart['PotNonLocNbrMP_ElemIdVecList'][j]
            InvOvrlpElemIdVector   = np.intersect1d(NL_BoundaryElemIdVec,NbrMP_ElemIdVector, assume_unique=True)
            InvFlag = False
            if len(InvOvrlpElemIdVector) > 0:
                InvFlag = True
                NL_InvOvrlpLocElemIdVec = getIndices(NL_ElemIdVec, InvOvrlpElemIdVector)
                MeshPart['NL_InvOvrlpLocElemIdVecList'].append(NL_InvOvrlpLocElemIdVec)
            
            if not Flag == InvFlag: 
                
                print('Warning: Non-local inconsistency detected for meshparts:', [MP_Id, NbrMP_Id])
                
                if Flag:  
                    if len(MeshPart['NL_NbrMPIdVector'])>0:
                        MeshPart['NL_NbrMPIdVector'].pop(-1)
                        MeshPart['OvrlpLocElemIdVecList'].pop(-1)
                    
                elif InvFlag:
                    if len(MeshPart['NL_InvOvrlpLocElemIdVecList'])>0:
                        MeshPart['NL_InvOvrlpLocElemIdVecList'].pop(-1)
            
    if Rank==0: print('I3', time()-t1)
    t1 = time()
    

    
def exportMP(MPGData, PyDataPath_Part):
    
    MeshPartList = MPGData['MeshPartList']
    GlobData = MPGData['GlobData']
    N_MeshPart = len(MeshPartList)
    
    RefKeyList = ['Id', 'SubDomainData', 'NDOF', 'NNode', 'DofVector', 'NodeIdVector', 'InvDiagM', 'NodeWeightVector', \
                  'RefLoadVector', 'NbrMPIdVector','ElemIdVector', 'OvrlpLocalNodeIdVecList', \
                  'OvrlpLocalDofVecList', 'RefPlotData', 'MPList_RefPlotDofIndicesList', \
                  'IntfcLocalNodeIdList', 'MPList_IntfcNodeIdVector',\
                  'MPList_IntfcNNode', 'DofWeightVector', 'LocFixedDof', \
                  'Flat_ElemLocDof', 'NCount', 'N_NbrDof', 'Ud', 'Vd', 'DofEff', 'LocDofEff', 'NElem',\
                  'IntfcNbrMPIdVector', 'IntfcOvrlpLocalNodeIdVecList', 'DmgProp', 'MatProp', 'NodeCoordVec']
                  
    DmgProp = MPGData['DmgProp']
    if DmgProp['Type'] == 'NonLocal':
        RefKeyList += ['OvrlpLocElemIdVecList', 'NL_InvOvrlpLocElemIdVecList', 'NL_NbrMPIdVector', \
                        'NLSpWeightMatrix', 'NL_ElemLocIdVec', 'NL_NElem']
    
    MP_BufferDataList = []
    MP_MetaDataList = []
    for i in range(N_MeshPart):
        MeshPart = MeshPartList[i]
        
        RefMeshPart = {'GlobData': GlobData}
        for RefKey in RefKeyList:   
            RefMeshPart[RefKey] = MeshPart[RefKey]
        
        Buffer_RefMeshPart = np.frombuffer(zlib.compress(pickle.dumps(RefMeshPart, pickle.HIGHEST_PROTOCOL)), 'b')
        MP_BufferDataList.append(Buffer_RefMeshPart)
        
        Mem = Buffer_RefMeshPart.nbytes
        Nf = len(Buffer_RefMeshPart)
        Dtype = Buffer_RefMeshPart.dtype
        MetaData = [Mem, Nf, Dtype]
        MP_MetaDataList.append(MetaData)
    
    MetaDataList = Comm.gather(MP_MetaDataList,root=0)
    
    N_TotalMshPrt = MPGData['GlobData']['N_TotalMshPrt']
    Data_FileName = PyDataPath_Part + str(N_TotalMshPrt)
    if Rank == 0:
        MetaDataList = np.vstack(MetaDataList)
        OffsetData = np.cumsum(np.hstack([[0],MetaDataList[:-1,0]]))
        metadat = np.array({'NfData': MetaDataList[:,1],
                           'DTypeData': MetaDataList[:,2],
                           'OffsetData': OffsetData}, dtype=object)
        np.save(Data_FileName+'_metadat', metadat)
    else:
        OffsetData = None
    
    OffsetData = Comm.bcast(OffsetData, root=0)
    Comm.barrier()
    
    #A small sleep to avoid hanging
    sleep(Rank*1e-3)
    
    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
    
    for i in range(N_MeshPart):
        MP_Id = MPGData['MPIdList'][i]
        fh = MPI.File.Open(MPI.COMM_SELF, Data_FileName+'_'+str(MP_Id)+'.mpidat', amode)
        Data_Buffer = MP_BufferDataList[i]
        fh.Write(Data_Buffer)
        #print('W0', MP_Id)
        fh.Close()
    
    """
    
    #A small sleep to avoid hanging
    Comm.barrier()
    sleep(Rank*1e-3)
    
    fh = MPI.File.Open(Comm, Data_FileName+'.mpidat', amode)
    for i in range(N_MeshPart):
        MP_Id = MPGData['MPIdList'][i]
        Offset = OffsetData[MP_Id]
        Data_Buffer = MP_BufferDataList[i]
        fh.Write_at(Offset, Data_Buffer)
        print('W1', MP_Id)
    fh.Close()
    """
    


if __name__ == "__main__":

    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm            = MPI.COMM_WORLD
    Rank            = Comm.Get_rank()
    N_Workers       = Comm.Get_size()
    GlobData        = initGlobDataPrep()
    Comm.barrier()
    
    MPGData = {'GlobData':              GlobData,
               'PotentialNbrDataFlag':  False}
    
    extract_Elepart(MPGData)
    extract_PlotSettings(MPGData)
    
    if Rank==0: print(Rank, 'A')
    
    if N_Workers > 1 and not GlobData['N_MPGs']%4==0:  raise Exception('N_Workers must be a multiple of 4')
    extract_ElemMeshData(MPGData)
    Comm.barrier()
    
    if Rank==0: print(Rank, 'B0')
    config_ElemVectors(MPGData)
    if Rank==0: print(Rank, 'B1')
    extract_NodalVectors(MPGData)
    if Rank==0: print(Rank, 'C')
    config_TypeGroupList(MPGData)
    if Rank==0: print(Rank, 'D')
    config_ElemMaterial(MPGData)
    config_ElemLib(MPGData)
    if Rank==0: print(Rank, 'E')
    config_IntfcElem(MPGData)
    if Rank==0: print(Rank, 'F')
    identify_PotentialNeighbours(MPGData)
    if Rank==0: print(Rank, 'G')
    config_Neighbours(MPGData)
    if Rank==0: print(Rank, 'H')
    config_IntfcNeighbours(MPGData)
    if Rank==0: print(Rank, 'I')
    config_NonlocalNeighbours(MPGData)
    if Rank==0: print(Rank, 'J')
    exportMP(MPGData, PyDataPath_Part)
    if Rank==0: print(Rank, 'K')
    