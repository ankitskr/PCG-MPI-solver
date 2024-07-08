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


import pickle
import zlib

from os import listdir
from os.path import isfile, join
    


def getIndices(A,B, CheckIntersection = False):
    
    if CheckIntersection:
        
        if not len(np.intersect1d(A,B)) == len(B):   raise Exception
    
    A_orig_indices = A.argsort()
    I = A_orig_indices[np.searchsorted(A[A_orig_indices], B)]

    return I




def readInputFiles(MatPath, GlobData):
    
    #print('Reading Matlab files..')
    
    N_Splits = len([f for f in listdir(MatPath) if isfile(join(MatPath, f)) and f.split('_')[0]=='DofGlb'])
    
    MeshFile_Level =              MatPath + 'Level.mat'
    MeshFile_Cm =                 MatPath + 'Cm.mat'
    MeshFile_Ck =                 MatPath + 'Ck.mat'
    MeshFile_Type =               MatPath + 'Type.mat'
    
    MeshFile_DofGlbFileList =     [MatPath + 'DofGlb_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_SignFileList =       [MatPath + 'Sign_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DiagMFileList =      [MatPath + 'DiagM_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DPDiagKFileList =    [MatPath + 'DPDiagK_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DPDiagMFileList =    [MatPath + 'DPDiagM_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_X0FileList =         [MatPath + 'X0_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_FFileList =          [MatPath + 'F_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_Dof_effFileList =    [MatPath + 'Dof_eff_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_NodeCoordVecFileList = [MatPath + 'NodeCoordVec_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    ExistX0 = False
    if isfile(MeshFile_X0FileList[0]):     ExistX0 = True
    
    KeDataFile =                   MatPath + 'Ke.mat'
    MeDataFile =                   MatPath + 'Me.mat'
    
    DofGlb = []
    Sign = []
    DiagM = []
    DP_DiagK = []
    DP_DiagM = []
    F = []
    Dof_eff = []
    X0 = []
    NodeCoordVec = []
    
    for i in range(N_Splits):
        
        DofGlb_Data_i = scipy.io.loadmat(MeshFile_DofGlbFileList[i])
        DofGlb_i = [a.T[0]-1 for a in DofGlb_Data_i['RefDofGlb'][0]]
        DofGlb += DofGlb_i
        
        Sign_Data_i = scipy.io.loadmat(MeshFile_SignFileList[i])
        Sign_i = [a.T[0] for a in Sign_Data_i['RefSign'][0]]
        Sign += Sign_i
        
        DiagM_i = scipy.io.loadmat(MeshFile_DiagMFileList[i])['RefDiagM'].T[0]
        DiagM.append(DiagM_i)
        
        DP_DiagK_i = scipy.io.loadmat(MeshFile_DPDiagKFileList[i])['RefDPDiagK'].T[0]
        DP_DiagK.append(DP_DiagK_i)
    
        DP_DiagM_i = scipy.io.loadmat(MeshFile_DPDiagMFileList[i])['RefDPDiagM'].T[0]
        DP_DiagM.append(DP_DiagM_i)
        
        F_i = scipy.io.loadmat(MeshFile_FFileList[i])['RefF'].T[0]
        F.append(F_i)
        
        Dof_eff_i = scipy.io.loadmat(MeshFile_Dof_effFileList[i])['RefDof_eff'].T[0] - 1
        Dof_eff.append(Dof_eff_i)
        
        NodeCoordVec_i = scipy.io.loadmat(MeshFile_NodeCoordVecFileList[i])['RefNodeCoordVec'][0]
        NodeCoordVec.append(NodeCoordVec_i)
        
    
        
        if ExistX0:
            X0_i = scipy.io.loadmat(MeshFile_X0FileList[i])['RefX0'].T[0]
            X0.append(X0_i)
            
    
    Level =     scipy.io.loadmat(MeshFile_Level)['Level'].T[0]
    Cm =        scipy.io.loadmat(MeshFile_Cm)['Cm'].T[0]
    Ck =        scipy.io.loadmat(MeshFile_Ck)['Ck'].T[0]
    Type =      scipy.io.loadmat(MeshFile_Type)['Type'][0] - 1
    Ke =        scipy.io.loadmat(KeDataFile)['Ke'][0]
    Me =        scipy.io.loadmat(MeDataFile)['Me'][0]
    
    DiagM = np.hstack(DiagM)
    DP_DiagK = np.hstack(DP_DiagK)
    DP_DiagM = np.hstack(DP_DiagM)
    F = np.hstack(F)
    Dof_eff = np.hstack(Dof_eff)
    NodeCoordVec = np.hstack(NodeCoordVec)
    
    X0 = np.array(X0)
    if ExistX0:     X0  = np.hstack(X0)
    
    GlobSettingsFile = MatPath + 'GlobSettings.mat'
    
    GlobSettings = scipy.io.loadmat(GlobSettingsFile)
    
    RefPlotDofVec =  GlobSettings['RefPlotDofVec'].T[0] - 1
    Tol =           GlobSettings['Tol'][0][0]
    MaxIter =       int(GlobSettings['MaxIter'][0][0])
    Alpha =          GlobSettings['Alpha'][0][0]
    Gamma =         GlobSettings['Gamma'][0][0]
    Beta =          GlobSettings['Beta'][0][0]
    MaxTime =       GlobSettings['Tmax'][0][0] 
    ft =            GlobSettings['ft'][0]                           
    dt =            GlobSettings['dt'][0][0]                        #time increment
    FintCalcMode =     GlobSettings['FintCalcMode'][0] 
    dT_Export =     GlobSettings['dT_Export'][0][0] 
    PlotFlag =      GlobSettings['PlotFlag'][0][0]              
    ExportFlag =    GlobSettings['ExportFlag'][0][0]   
    UseLumpedMass = GlobSettings['UseLumpedMass'][0][0]
    qpoint =        GlobSettings['qpoint']
    
    
    GlobData['GlobNDof_eff'] = len(Dof_eff)
    GlobData['GlobNDof'] = len(F)
    GlobData['RefPlotDofVec'] = RefPlotDofVec
    GlobData['MaxIter'] = MaxIter
    GlobData['FintCalcMode'] = FintCalcMode
    GlobData['Tol'] = Tol    
    GlobData['Alpha'] = Alpha    
    GlobData['Beta'] = Beta    
    GlobData['Gamma'] = Gamma    
    GlobData['MaxTime'] = MaxTime
    GlobData['DeltaLambdaList'] = np.array(ft, dtype=float)
    GlobData['TimeStepSize'] = dt
    GlobData['dT_Export'] = dT_Export
    GlobData['PlotFlag'] = PlotFlag
    GlobData['ExportFlag'] = ExportFlag
    GlobData['UseLumpedMass'] = bool(UseLumpedMass)
    
    GlobData['ExistDP0'] = True
    GlobData['ExistDP1'] = False
    
    
    return DofGlb, Sign, Type, Level, Cm, Ck, Me, Ke, DiagM, DP_DiagK, DP_DiagM, X0, Dof_eff, F, NodeCoordVec, RefPlotDofVec, qpoint







def getElePart(MatPath, N_MshPrt):
   
    if N_MshPrt == 1:
        
        MeshFile_MeshPartFile = MatPath + 'MeshPart_' + str(2) + '.mat'
        
        MeshPartData = scipy.io.loadmat(MeshFile_MeshPartFile)
        ElePart_List = MeshPartData['RefPart'][0] 
        if len(ElePart_List) == 1:  ElePart_List = ElePart_List[0][0]
        
        N_Elem = len(ElePart_List)
        ElePart = np.zeros(N_Elem, dtype = int)
    
    else:
        
        MeshFile_MeshPartFile = MatPath + 'MeshPart_' + str(N_MshPrt) + '.mat'
        
        MeshPartData = scipy.io.loadmat(MeshFile_MeshPartFile)
        ElePart_List = MeshPartData['RefPart'][0]    
        if len(ElePart_List) == 1:  ElePart_List = ElePart_List[0][0]
        
        ElePart =   ElePart_List - 1           #the index of part each element belongs to, {1 1 1 2 3 1 3 2 2 ...}
    
    
    return ElePart




def configMP_ElemVectors(MeshPartList, ElePart, DofGlb, Type, Level, Cm, Ck, Sign): 
    
    MPList_Id = np.unique(ElePart)
    N_MeshPart = len(MPList_Id)
    
    print('N_MeshParts: ', len(MPList_Id))
    
    #print('Computing element vectors..')
    
    for i in range(N_MeshPart):
        
        MP_Id = MPList_Id[i]
        RefElemIdList = np.where(ElePart==MP_Id)[0]
        
        #Calc DofVector for each Mesh Part
        MP_ElemList_NDOF = []
        MP_ElemList_SignVector = []
        MP_CumDofVector = []
        
        for RefElemId in RefElemIdList:
            
            Elem_DofVector = DofGlb[RefElemId]
            MP_ElemList_NDOF.append(len(Elem_DofVector))
            MP_CumDofVector += list(Elem_DofVector)
            
            Elem_SignVector = np.array(Sign[RefElemId], dtype=bool)
            MP_ElemList_SignVector.append(Elem_SignVector)
        
        
        MP_UniqueDofVector = np.unique(MP_CumDofVector)
        
        
        #Calc LocDofVector for each element
        N_Elem = len(RefElemIdList)
        MP_ElemList_LocDofVector = []
        
        Elem_CumLocDofVector = getIndices(MP_UniqueDofVector, MP_CumDofVector)
        
        I0 = 0
        for io in range(N_Elem):
            
            Elem_NDOF = MP_ElemList_NDOF[io]
            
            I1 = I0+Elem_NDOF
            Elem_LocDofVector = Elem_CumLocDofVector[I0:I1]
            I0 = I1
            
            MP_ElemList_LocDofVector.append(Elem_LocDofVector)
        
        
        #Calc Type/Level for each element
        ElemList_Type0 = Type[RefElemIdList]
        ElemList_Level0 = Level[RefElemIdList]
        ElemList_Cm0 = Cm[RefElemIdList]
        ElemList_Ck0 = Ck[RefElemIdList]
        
        
        MeshPart = {}
        MeshPart['Id'] = MP_Id
        MeshPart['DofVector'] = MP_UniqueDofVector
        MeshPart['ElemList_LocDofVector'] = MP_ElemList_LocDofVector
        MeshPart['ElemList_SignVector'] = MP_ElemList_SignVector
        MeshPart['ElemList_Type'] = ElemList_Type0
        MeshPart['ElemList_Level'] = ElemList_Level0
        MeshPart['ElemList_Cm'] = ElemList_Cm0
        MeshPart['ElemList_Ck'] = ElemList_Ck0
        
        MeshPartList.append(MeshPart)
        
    






def configMP_GlobVectors(MeshPartList, F, DiagM, DP_DiagK, DP_DiagM, Dof_eff, X0, RefPlotDofVec, qpoint):
                         
    #print('Computing global vectors..')
    
    N_MeshPart                              = len(MeshPartList)
    MPList_DofVector                      = []
    MPList_NDofVec                        = []
    
    GlobalRefLoadVector                     = np.array(F, dtype=float)
    GlobalDiagMVector                       = np.array(DiagM, dtype=float)
    GlobalDPDiagKVector                     = np.array(DP_DiagK, dtype=float)
    GlobalDPDiagMVector                     = np.array(DP_DiagM, dtype=float)
    X0                                       = np.array(X0, dtype=float)
    GlobDof_eff                             = np.array(Dof_eff,dtype=int)
    TestPlotFlag = True
    MPList_RefPlotDofIndicesList = []
    
    for i in range(N_MeshPart):
        
        MP_DofVector                            = MeshPartList[i]['DofVector']
        MP_NDOF                                 = len(MP_DofVector)
        MPList_DofVector.append(MP_DofVector)
        MPList_NDofVec.append(MP_NDOF)
        
        MP_Dof_eff                                = np.intersect1d(MP_DofVector, Dof_eff)
        MP_LocDof_eff                           = getIndices(MP_DofVector, MP_Dof_eff)
        MP_NDof_eff                                 = len(MP_Dof_eff)
        
        MP_DiagMVector                      = GlobalDiagMVector[MP_DofVector]
        MP_DPDiagKVector                      = GlobalDPDiagKVector[MP_DofVector]
        MP_DPDiagMVector                      = GlobalDPDiagMVector[MP_DofVector]
        MP_RefLoadVector                        = GlobalRefLoadVector[MP_DofVector]
        
        if len(X0)>0:  MP_X0                    = X0[MP_DofVector]
        else:          MP_X0                    = np.zeros(MP_NDOF)
        
        
        MP_RefPlotData                  = {}
        MP_RefPlotData['TestPlotFlag']  = False
        MP_RefPlotData['J']             = []
        MP_RefPlotData['LocalDofVec']   = []
        MP_RefPlotData['DofVec']        = np.intersect1d(MP_DofVector, RefPlotDofVec)
        if len(MP_RefPlotData['DofVec']) > 0:
            MP_RefPlotData['LocalDofVec'] = getIndices(MP_DofVector, MP_RefPlotData['DofVec'])
            MP_RefPlotData['J'] = getIndices(RefPlotDofVec, np.intersect1d(RefPlotDofVec, MP_RefPlotData['DofVec']))
            if TestPlotFlag:    
                MP_RefPlotData['TestPlotFlag'] = True
                TestPlotFlag = False
        
        if i == 0:
            MP_RefPlotData['RefPlotDofVec'] = RefPlotDofVec
            MP_RefPlotData['qpoint']        = qpoint
        else:
            MP_RefPlotData['RefPlotDofVec'] = []
            MP_RefPlotData['qpoint']        = []
            
        MPList_RefPlotDofIndicesList.append(MP_RefPlotData['J'])
        
        
        
        MeshPartList[i]['RefLoadVector']        = MP_RefLoadVector
        MeshPartList[i]['DiagMVector']          = MP_DiagMVector
        MeshPartList[i]['DPDiagKVector']          = MP_DPDiagKVector
        MeshPartList[i]['DPDiagMVector']          = MP_DPDiagMVector
        MeshPartList[i]['X0']                   = MP_X0
        MeshPartList[i]['Dof_eff']              = MP_Dof_eff
        MeshPartList[i]['LocDof_eff']              = MP_LocDof_eff
        MeshPartList[i]['NDOF']                 = MP_NDOF
        MeshPartList[i]['MP_NDof_eff']          = MP_NDof_eff
        MeshPartList[i]['RefPlotData']          = MP_RefPlotData
        
        
    
    GathDofVector = np.hstack(MPList_DofVector)
    
    for i in range(N_MeshPart):
        
        if i == 0 :    
            
            MeshPartList[i]['GathDofVector']    = GathDofVector
            MeshPartList[i]['MPList_NDofVec']   = MPList_NDofVec
            MeshPartList[i]['MPList_RefPlotDofIndicesList'] = MPList_RefPlotDofIndicesList
        
        else:        
            
            MeshPartList[i]['GathDofVector']    = []
            MeshPartList[i]['MPList_NDofVec']   = []
            MeshPartList[i]['MPList_RefPlotDofIndicesList'] = []
        





def configMP_TypeGroupList(MeshPartList):

    #print('Grouping Octrees based on their types..')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        
        MP_ElemList_LocDofVector = np.array(MeshPartList[i]['ElemList_LocDofVector'])
        MP_ElemList_SignVector = np.array(MeshPartList[i]['ElemList_SignVector'])
        MP_ElemList_Type = np.array(MeshPartList[i]['ElemList_Type'])
        MP_ElemList_Level = np.array(MeshPartList[i]['ElemList_Level'])
        MP_ElemList_Cm = np.array(MeshPartList[i]['ElemList_Cm'])
        MP_ElemList_Ck = np.array(MeshPartList[i]['ElemList_Ck'])
        
        UniqueElemTypeList = np.unique(MP_ElemList_Type)
        N_Type = len(UniqueElemTypeList)
        
        MP_TypeGroupList = []
        
        for j in range(N_Type):
            
            RefElemTypeId = UniqueElemTypeList[j]
            I = np.where(MP_ElemList_Type==RefElemTypeId)[0]
            
            RefElemList_LocDofVector = np.array(tuple(MP_ElemList_LocDofVector[I]), dtype=int).T
            RefElemList_SignVector = np.array(tuple(MP_ElemList_SignVector[I]), dtype=bool).T
            RefElemList_Level = np.array(tuple(MP_ElemList_Level[I]), dtype=float)
            RefElemList_Cm = np.array(tuple(MP_ElemList_Cm[I]), dtype=float)
            RefElemList_Ck = np.array(tuple(MP_ElemList_Ck[I]), dtype=float)
        
            MP_TypeGroup = {}
            MP_TypeGroup['ElemTypeId'] = RefElemTypeId
            MP_TypeGroup['ElemList_LocDofVector'] = RefElemList_LocDofVector
            MP_TypeGroup['ElemList_LocDofVector_Flat'] = RefElemList_LocDofVector.flatten()
            MP_TypeGroup['ElemList_SignVector'] = RefElemList_SignVector
            MP_TypeGroup['ElemList_Level'] = RefElemList_Level
            MP_TypeGroup['ElemList_LevelCubed'] = RefElemList_Level**3
            MP_TypeGroup['ElemList_Cm'] = RefElemList_Cm
            MP_TypeGroup['ElemList_Ck'] = RefElemList_Ck
            
            MP_TypeGroupList.append(MP_TypeGroup)
            
        MeshPartList[i]['TypeGroupList'] = MP_TypeGroupList
          
        
    

def configMP_ElemStiffMat(MeshPartList, Ke, Me):
    
    #print('Computing element stiffness matrices..')
    
    N_MeshPart = len(MeshPartList)
    
    #Reading Stiffness Matrices of Structured cells
    StrucSDList_StiffMatDB = Ke
    StrucSDList_MassMatDB = Me
    StrucSDList_TypeListDB = range(len(Ke))
    
    #Calculating SubdomainData
    for i in range(N_MeshPart):
        
        MP_TypeGroupList = MeshPartList[i]['TypeGroupList']
        N_Type = len(MP_TypeGroupList)
        
        MP_SubDomainData = {'MixedDataList': {},
                            'StrucDataList': MP_TypeGroupList}
        
        for j in range(N_Type):
            
            RefTypeGroup = MP_TypeGroupList[j]
            
            RefElemTypeId = RefTypeGroup['ElemTypeId']
            j0 = StrucSDList_TypeListDB.index(RefElemTypeId)
            
            Ke_j = StrucSDList_StiffMatDB[j0]
            RefTypeGroup['ElemStiffMat'] = np.array(Ke_j, dtype=float)
            
            Me_j = StrucSDList_MassMatDB[j0]
            RefTypeGroup['ElemMassMat'] = np.array(Me_j, dtype=float)
        
        
        MeshPartList[i]['SubDomainData'] = MP_SubDomainData
        
    



def identify_PotentialNeighbourMP(MeshPartList, NodeCoordVec):
    
    print('WARNING: Verify Potential Neighbours!!')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        
        MP_DofVector = MeshPartList[i]['DofVector']
        
        XCoordList = NodeCoordVec[MP_DofVector[0::3]]
        YCoordList = NodeCoordVec[MP_DofVector[1::3]]
        ZCoordList = NodeCoordVec[MP_DofVector[2::3]]
        
        MeshPartList[i]['CoordLimits'] = [np.min(XCoordList), np.max(XCoordList), np.min(YCoordList), np.max(YCoordList), np.min(ZCoordList), np.max(ZCoordList)]
        
        MeshPartList[i]['PotentialNbrMPIdVector'] = []
    
    
    Tol = 1e-6
    
    for i in range(N_MeshPart):
    
        XMin_i = MeshPartList[i]['CoordLimits'][0] - Tol
        XMax_i = MeshPartList[i]['CoordLimits'][1] + Tol
        YMin_i = MeshPartList[i]['CoordLimits'][2] - Tol
        YMax_i = MeshPartList[i]['CoordLimits'][3] + Tol
        ZMin_i = MeshPartList[i]['CoordLimits'][4] - Tol
        ZMax_i = MeshPartList[i]['CoordLimits'][5] + Tol
    
        for j in range(i, N_MeshPart):
            
            if not i == j:
            
                XMin_j = MeshPartList[j]['CoordLimits'][0] - Tol
                XMax_j = MeshPartList[j]['CoordLimits'][1] + Tol
                YMin_j = MeshPartList[j]['CoordLimits'][2] - Tol
                YMax_j = MeshPartList[j]['CoordLimits'][3] + Tol
                ZMin_j = MeshPartList[j]['CoordLimits'][4] - Tol
                ZMax_j = MeshPartList[j]['CoordLimits'][5] + Tol
            
                if not (XMin_j > XMax_i or XMin_i > XMax_j or \
                        YMin_j > YMax_i or YMin_i > YMax_j or \
                        ZMin_j > ZMax_i or ZMin_i > ZMax_j):
                        
                    MeshPartList[i]['PotentialNbrMPIdVector'].append(j)
                    MeshPartList[j]['PotentialNbrMPIdVector'].append(i)
                   
                    
                    
                
                
        




def configMP_Neighbours(MeshPartList, Mode='Slow', NodeCoordVec = None):
    
    #print('Computing neighbours..')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        
        MeshPartList[i]['NbrMPIdVector'] = []
        MeshPartList[i]['OvrlpLocalDofVecList'] = []
        
        
    #Finding Neighbour MeshParts and OverLapping DOFs
    if Mode == 'Slow':
        for i in range(N_MeshPart):
            MP_DofVector_i = MeshPartList[i]['DofVector']
            MP_OvrlpLocalDofVecList_i = MeshPartList[i]['OvrlpLocalDofVecList']
            
            for j in range(i, N_MeshPart):
                if not i == j:
                    MP_DofVector_j = MeshPartList[j]['DofVector']
                    MP_OvrlpLocalDofVecList_j = MeshPartList[j]['OvrlpLocalDofVecList']
                    OvrlpDofVector = np.intersect1d(MP_DofVector_i,MP_DofVector_j, assume_unique=True)
                    
                    if len(OvrlpDofVector) > 0:
                        MP_OvrlpLocalDofVector_i = getIndices(MP_DofVector_i, OvrlpDofVector)
                        MP_OvrlpLocalDofVecList_i.append(MP_OvrlpLocalDofVector_i)
                        MeshPartList[i]['NbrMPIdVector'].append(j)
                        
                        MP_OvrlpLocalDofVector_j = getIndices(MP_DofVector_j, OvrlpDofVector)
                        MP_OvrlpLocalDofVecList_j.append(MP_OvrlpLocalDofVector_j)
                        MeshPartList[j]['NbrMPIdVector'].append(i)
    
    
    elif Mode == 'Fast':
        print('WARNING: Fast mode is NOT fully tested!!')
        identify_PotentialNeighbourMP(MeshPartList, NodeCoordVec)
        
        for i in range(N_MeshPart):
            MP_DofVector_i = MeshPartList[i]['DofVector']
            MP_OvrlpLocalDofVecList_i = MeshPartList[i]['OvrlpLocalDofVecList']
            
            for j in range(i, N_MeshPart):
                if j in MeshPartList[i]['PotentialNbrMPIdVector']:
                    MP_DofVector_j = MeshPartList[j]['DofVector']
                    MP_OvrlpLocalDofVecList_j = MeshPartList[j]['OvrlpLocalDofVecList']
                    OvrlpDofVector = np.intersect1d(MP_DofVector_i,MP_DofVector_j, assume_unique=True)
                    
                    if len(OvrlpDofVector) > 0:
                        MP_OvrlpLocalDofVector_i = getIndices(MP_DofVector_i, OvrlpDofVector)
                        MP_OvrlpLocalDofVecList_i.append(MP_OvrlpLocalDofVector_i)
                        MeshPartList[i]['NbrMPIdVector'].append(j)
                        
                        MP_OvrlpLocalDofVector_j = getIndices(MP_DofVector_j, OvrlpDofVector)
                        MP_OvrlpLocalDofVecList_j.append(MP_OvrlpLocalDofVector_j)
                        MeshPartList[j]['NbrMPIdVector'].append(i)
        
    else:   raise Exception
    
    
    
    #Calculating Variables for Fint Calculation
    for i in range(N_MeshPart): 
    
        MP_OvrlpLocalDofVecList = MeshPartList[i]['OvrlpLocalDofVecList']
        MP_SubDomainData = MeshPartList[i]['SubDomainData']
        
        N_NbrMP = len(MeshPartList[i]['NbrMPIdVector'])
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
        
        MeshPartList[i]['Flat_ElemLocDof'] = Flat_ElemLocDof
        MeshPartList[i]['NCount'] = NCount
        MeshPartList[i]['N_NbrDof'] = N_NbrDof
    
    
    
    #Calculating the Weight Vector    
    for i in range(N_MeshPart):        
        MP_DofVector = MeshPartList[i]['DofVector']
        MP_WeightVector = np.ones(len(MP_DofVector))        
        MeshPartList[i]['WeightVector'] = MP_WeightVector    
    
    if N_MeshPart > 1:    
        OvrlpDofVecList = []
        for i in range(N_MeshPart):            
            MP_DofVector = MeshPartList[i]['DofVector']
            OvrlpLocalDofVector_Flat = np.hstack(MeshPartList[i]['OvrlpLocalDofVecList'])
            Unique_OvrlpDofVec0 = MP_DofVector[np.unique(OvrlpLocalDofVector_Flat)]
            OvrlpDofVecList.append(Unique_OvrlpDofVec0)
        
        Unique_OvrlpDofVec = np.unique(np.hstack(OvrlpDofVecList))
        Unique_OvrlpDofVec_Flagged = np.array([])
        for i in range(N_MeshPart):
            MP_DofVector = MeshPartList[i]['DofVector']
            MP_WeightVector = MeshPartList[i]['WeightVector']
            
            MP_OvrlpDofVec = np.intersect1d(MP_DofVector, Unique_OvrlpDofVec)
            MP_OvrlpDofVec_Flagged = np.intersect1d(MP_OvrlpDofVec, Unique_OvrlpDofVec_Flagged)
            
            if len(MP_OvrlpDofVec_Flagged)>0:
                MP_OvrlpLocalDofVector_Flagged = getIndices(MP_DofVector, MP_OvrlpDofVec_Flagged)
                MP_WeightVector[MP_OvrlpLocalDofVector_Flagged] = 0.0
                
            MP_OvrlpDofVec_UnFlagged = np.setdiff1d(MP_OvrlpDofVec, MP_OvrlpDofVec_Flagged)
            if len(MP_OvrlpDofVec_UnFlagged)>0:
                Unique_OvrlpDofVec_Flagged = np.hstack([Unique_OvrlpDofVec_Flagged, MP_OvrlpDofVec_UnFlagged])
                
    
    """    
    #Calculating the Weight Vector
    for i in range(N_MeshPart):
        
        MP_DofVector = MeshPartList[i]['DofVector']
        MP_WeightVector = np.ones(len(MP_DofVector))
        
        MeshPartList[i]['WeightVector'] = MP_WeightVector
    
    
    if N_MeshPart > 1:
    
        OvrlpDofVecList = []
        for i in range(N_MeshPart):
            
            MP_DofVector = MeshPartList[i]['DofVector']
            OvrlpLocalDofVector_Flat = np.hstack(MeshPartList[i]['OvrlpLocalDofVecList'])
            Unique_OvrlpDofVec0 = MP_DofVector[np.unique(OvrlpLocalDofVector_Flat)]
            OvrlpDofVecList.append(Unique_OvrlpDofVec0)
        
        Unique_OvrlpDofVec, Freq_OvrlpDofVec = np.unique(np.hstack(OvrlpDofVecList), return_counts=True)
    
        for i in range(N_MeshPart):
            
            MP_DofVector = MeshPartList[i]['DofVector']
            MP_WeightVector = MeshPartList[i]['WeightVector']
            
            MP_OvrlpDofVector = np.intersect1d(MP_DofVector, Unique_OvrlpDofVec)
            MP_OvrlpLocalDofVector = getIndices(MP_DofVector, MP_OvrlpDofVector)
            MP_FreqOvrlpLocalDofVector = Freq_OvrlpDofVec[getIndices(Unique_OvrlpDofVec, MP_OvrlpDofVector)]
            MP_WeightVector[MP_OvrlpLocalDofVector] = 1.0/MP_FreqOvrlpLocalDofVector
    """
    





def exportMP(MeshPartList, GlobData, PyDataPath):

    #print('Exporting Mesh Parts..')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
    
        RefMeshPart_FileName = PyDataPath + str(i) + '.zpkl'
        
        RefMeshPart = {}
        RefMeshPart['Id']                          = MeshPartList[i]['Id']
        RefMeshPart['SubDomainData']               = MeshPartList[i]['SubDomainData']
        RefMeshPart['NDOF']                        = MeshPartList[i]['NDOF']
        RefMeshPart['DofVector']                   = MeshPartList[i]['DofVector']
        RefMeshPart['DiagMVector']                  = MeshPartList[i]['DiagMVector']
        RefMeshPart['DPDiagKVector']                  = MeshPartList[i]['DPDiagKVector']
        RefMeshPart['DPDiagMVector']                  = MeshPartList[i]['DPDiagMVector']
        RefMeshPart['RefLoadVector']               = MeshPartList[i]['RefLoadVector']
        RefMeshPart['NbrMPIdVector']               = MeshPartList[i]['NbrMPIdVector']
        RefMeshPart['OvrlpLocalDofVecList']        = MeshPartList[i]['OvrlpLocalDofVecList']
        RefMeshPart['RefPlotData']                 = MeshPartList[i]['RefPlotData']
        RefMeshPart['MPList_RefPlotDofIndicesList']= MeshPartList[i]['MPList_RefPlotDofIndicesList']
        RefMeshPart['GathDofVector']               = MeshPartList[i]['GathDofVector']
        RefMeshPart['LocDof_eff']               = MeshPartList[i]['LocDof_eff']
        RefMeshPart['MPList_NDofVec']              = MeshPartList[i]['MPList_NDofVec']
        RefMeshPart['WeightVector']                = MeshPartList[i]['WeightVector']
        RefMeshPart['X0']                          = MeshPartList[i]['X0']
        RefMeshPart['Flat_ElemLocDof']         = MeshPartList[i]['Flat_ElemLocDof']
        RefMeshPart['NCount']                  = MeshPartList[i]['NCount']
        RefMeshPart['GlobData']                    = GlobData
        
        Cmpr_RefMeshPart                           = zlib.compress(pickle.dumps(RefMeshPart))
        
            
        f = open(RefMeshPart_FileName, 'wb')
        f.write(Cmpr_RefMeshPart)
        f.close()
   
   




def calcMPQualityData(MeshPartList, Path, ModelName, p):
    
    N_MeshPart = len(MeshPartList)
    
    N_MPDOFList = []
    N_MPOvrlpDOFList = []
    
    for i in range(N_MeshPart):
    
        MP_DofVector = MeshPartList[i]['DofVector']
        MPOvrlpDOFList = np.hstack(MeshPartList[i]['OvrlpLocalDofVecList'])
        
        N_MPDOFList.append(len(MP_DofVector))
        N_MPOvrlpDOFList.append(len(MPOvrlpDOFList))
    
    
    
    Mean_N_MPDOF = np.round(np.mean(N_MPDOFList), 2)
    Min_N_MPDOF = np.min(N_MPDOFList)
    Max_N_MPDOF = np.max(N_MPDOFList)
    
    MinPercent_MPDOF = np.round(100.0*Min_N_MPDOF/Mean_N_MPDOF, 2)
    MaxPercent_MPDOF = np.round(100.0*Max_N_MPDOF/Mean_N_MPDOF, 2)
    
    
    Mean_N_MPOvrlpDOF = np.round(np.mean(N_MPOvrlpDOFList), 2)
    Min_N_MPOvrlpDOF = np.min(N_MPOvrlpDOFList)
    Max_N_MPOvrlpDOF = np.max(N_MPOvrlpDOFList)
    
    MinPercent_MPOvrlpDOF = np.round(100.0*Min_N_MPOvrlpDOF/Mean_N_MPOvrlpDOF, 2)
    MaxPercent_MPOvrlpDOF = np.round(100.0*Max_N_MPOvrlpDOF/Mean_N_MPOvrlpDOF, 2)
    
    
    
    MP_QData = {}
    MP_QData['N_MPDOFList'] = N_MPDOFList
    MP_QData['N_MPDOFStats'] = [Mean_N_MPDOF, Min_N_MPDOF, Max_N_MPDOF]
    
    MP_QData['N_MPOvrlpDOFList'] = N_MPOvrlpDOFList
    MP_QData['N_MPOvrlpDOFStats'] = [Mean_N_MPOvrlpDOF, Min_N_MPOvrlpDOF, Max_N_MPOvrlpDOF]
    
    
    MP_QDataFile = Path + 'Py_Results/QData_MP' + str(p) +'.npz' 
    np.savez_compressed(MP_QDataFile, MP_QData = MP_QData)
    
    
    
    """
    dash = '-' * 40
    
    print('\n\n')
    print(dash)
    print('{:<16s}{:<16s}{:<16s}{:<16s}'.format(' ', 'Mean', 'Min %', 'Max %'))
    print(dash)
    print('{:<16s}{:<16.2f}{:<16.2f}{:<16.2f}'.format('N_MPDOF', Mean_N_MPDOF, MinPercent_MPDOF, MaxPercent_MPDOF))
    print('{:<16s}{:<16.2f}{:<16.2f}{:<16.2f}'.format('N_OvrlpDOF', Mean_N_MPOvrlpDOF, MinPercent_MPOvrlpDOF, MaxPercent_MPOvrlpDOF))
    print('')
    print('{:<16s}{:<16.2f}{:<16.2f}{:<16.2f}'.format('N_MPDOF', Mean_N_MPDOF, Min_N_MPDOF, Max_N_MPDOF))
    print('{:<16s}{:<16.2f}{:<16.1f}{:<16.1f}'.format('N_OvrlpDOF', Mean_N_MPOvrlpDOF, Min_N_MPOvrlpDOF, Max_N_MPOvrlpDOF))
    """
    
    
        
    
    N_MPDOFList = np.array(N_MPDOFList)/1000.0
    N_MPOvrlpDOFList = np.array(N_MPOvrlpDOFList)/1000.0
    #Bins = np.max([10, int(0.25*N_MeshPart)])


    fig = plt.figure(figsize=(12,3.5))
    fig.suptitle(ModelName + ' (N_MeshParts = ' + str(N_MeshPart) + ')')
    
    ax0 = plt.subplot(121)
    ax0.yaxis.grid(linestyle='--')
    ax0.xaxis.grid(linestyle='--')
    ax0.set_axisbelow(True)

    ax0.hist(N_MPDOFList, bins=30, label=ModelName)
    ax0.set_ylabel('No. of Mesh Parts')
    ax0.set_xlabel('Total N_DOF per Mesh Part  (x1000)')
    
    MinLim = np.round(np.min(N_MPDOFList),1)
    MaxLim = np.round(np.max(N_MPDOFList),1)    
    #plt.xticks(np.arange(MinLim, MaxLim+0.1, 0.5))
    
    plt.title('(a)')
    plt.legend()



    ax1 = plt.subplot(122)
    ax1.yaxis.grid(linestyle='--')
    ax1.xaxis.grid(linestyle='--')
    ax1.set_axisbelow(True)
    ax1.set_ylabel('No. of Mesh Parts')
    ax1.set_xlabel('Interfacial N_DOF per Mesh Part  (x1000)')

    ax1.hist(N_MPOvrlpDOFList, bins=30, label=ModelName)
    
    MinLim = np.round(np.min(N_MPOvrlpDOFList),1)
    MaxLim = np.round(np.max(N_MPOvrlpDOFList),1)    
    #plt.xticks(np.arange(MinLim, MaxLim+0.2, 1.0))
    
    plt.title('(b)')
    plt.legend()


    FileName = Path + 'Py_Results/QData_MP' + str(p) +'.png' 
    fig.savefig(FileName, dpi = 360, bbox_inches='tight')

    
    
    
    
            
def main(N_MshPrt, ScratchPath):
    
    #Creating directories
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(N_MshPrt)  + '/'
    if os.path.exists(PyDataPath):
        
        try:    shutil.rmtree(PyDataPath)
        except:    raise Exception('PyDataPath in use!')
    
    os.makedirs(PyDataPath)
        
    
    #Reading Matlab Input Files
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    GlobData = {}
    DofGlb, Sign, Type, Level, Cm, Ck, Me, Ke, DiagM, DP_DiagK, DP_DiagM, X0, Dof_eff, F, NodeCoordVec, RefPlotDofVec, qpoint = readInputFiles(MatDataPath, GlobData)
    ElePart = getElePart(MatDataPath, N_MshPrt)
    
    
    #Converting Data
    MeshPartList = []
    configMP_ElemVectors(MeshPartList, ElePart, DofGlb, Type, Level, Cm, Ck, Sign)
    configMP_GlobVectors(MeshPartList, F, DiagM, DP_DiagK, DP_DiagM, Dof_eff, X0, RefPlotDofVec, qpoint)
    configMP_TypeGroupList(MeshPartList)
    configMP_ElemStiffMat(MeshPartList, Ke, Me)
    configMP_Neighbours(MeshPartList)
    #configMP_Neighbours(MeshPartList, Mode='Fast', NodeCoordVec = NodeCoordVec)
    
    exportMP(MeshPartList, GlobData, PyDataPath)
    
    #calcMPQualityData(MeshPartList, PyDataPath, ModelName)

    
    
    

if __name__ == "__main__":

    N_MshPrt =      int(sys.argv[1])
    ScratchPath =   sys.argv[2]
    
    main(N_MshPrt, ScratchPath)
    


