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
    
    MeshFile_Level =               MatPath + 'Level.mat'
    MeshFile_Cm =                  MatPath + 'Cm.mat'
    MeshFile_Ck =                  MatPath + 'Ck.mat'
    MeshFile_Type =                MatPath + 'Type.mat'
    
    MeshFile_DofGlbFileList =     [MatPath + 'DofGlb_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_SignFileList =       [MatPath + 'Sign_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DP0FileList =        [MatPath + 'DPDiagK_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DP1FileList =        [MatPath + 'DP1_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_X0FileList =         [MatPath + 'X0_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    ExistDP0 = False; ExistDP1 = False; ExistX0 = False
    if isfile(MeshFile_DP0FileList[0]):    ExistDP0 = True
    if isfile(MeshFile_DP1FileList[0]):    ExistDP1 = True
    if isfile(MeshFile_X0FileList[0]):     ExistX0 = True
    
    KeDataFile =                   MatPath + 'Ke.mat'
    
    DofGlb = []
    Sign = []
    M = []
    DP0 = []
    DP1 = []
    X0 = []
    for i in range(N_Splits):
        
        DofGlb_Data_i = scipy.io.loadmat(MeshFile_DofGlbFileList[i])
        DofGlb_i = [a.T[0]-1 for a in DofGlb_Data_i['RefDofGlb'][0]]
        DofGlb += DofGlb_i
        
        Sign_Data_i = scipy.io.loadmat(MeshFile_SignFileList[i])
        Sign_i = [a.T[0] for a in Sign_Data_i['RefSign'][0]]
        Sign += Sign_i
        
        if ExistDP0:    
            DP0_i = scipy.io.loadmat(MeshFile_DP0FileList[i])['RefDPDiagK'].T[0]
            DP0.append(DP0_i)
        
        if ExistDP1:
            DP1_i = scipy.io.loadmat(MeshFile_DP1FileList[i])['RefDP1'].T[0]
            DP1.append(DP1_i)
        
        if ExistX0:
            X0_i = scipy.io.loadmat(MeshFile_X0FileList[i])['RefX0'].T[0]
            X0.append(X0_i)
            
    
    Level =     scipy.io.loadmat(MeshFile_Level)['Level'].T[0]
    Cm =        scipy.io.loadmat(MeshFile_Cm)['Cm'].T[0]
    Ck =        scipy.io.loadmat(MeshFile_Ck)['Ck'].T[0]
    Type =      scipy.io.loadmat(MeshFile_Type)['Type'][0] - 1
    Ke =        scipy.io.loadmat(KeDataFile)['Ke'][0]
    
    
    DP0 = np.array(DP0); DP1 = np.array(DP1); X0 = np.array(X0)
    if ExistDP0:    DP0 = np.hstack(DP0)
    if ExistDP1:    DP1 = np.hstack(DP1)
    if ExistX0:     X0  = np.hstack(X0)
    
    GlobData['ExistDP0'] = ExistDP0
    GlobData['ExistDP1'] = ExistDP1
    
    return DofGlb, Sign, Type, Level, Cm, Ck, Ke, DP0, DP1, X0





    
def readLoadConfig(MatPath, GlobData):
    
    DofEffDataFile = MatPath + 'Dof_eff.mat'
    GlobSettingsFile = MatPath + 'GlobSettings.mat'
    FDataFile = MatPath + 'F.mat'
    
    FData = scipy.io.loadmat(FDataFile)
    GlobSettings = scipy.io.loadmat(GlobSettingsFile)
    DofEffData = scipy.io.loadmat(DofEffDataFile)
    
    Dof_eff =      DofEffData['Dof_eff'].flatten() - 1                  #DOFs which are fixed
    
    F =             FData['F'].T[0]                             #force vector, nsize by 1
    ExportFlag =    GlobSettings['ExportFlag'][0][0]            
    PlotFlag =      GlobSettings['PlotFlag'][0][0]            
    if PlotFlag:    RefDofIdList =  GlobSettings['RefDofIdList'].T[0] - 1
    else:            RefDofIdList = np.array([])
    MaxIter =       int(GlobSettings['MaxIter'][0][0])
    QCalcMode =     GlobSettings['QCalcMode'][0]
    Tol =           GlobSettings['Tol'][0][0]
    
    GlobData['GlobNDof_eff'] = len(Dof_eff)
    GlobData['GlobNDof'] = len(F)
    GlobData['ExportFlag'] = ExportFlag
    GlobData['PlotFlag'] = PlotFlag
    GlobData['RefDofIdList'] = RefDofIdList
    GlobData['MaxIter'] = MaxIter
    GlobData['QCalcMode'] = QCalcMode
    GlobData['Tol'] = Tol
    
    
    return Dof_eff, F






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
        
    






def configMP_GlobVectors(MeshPartList, F, DP0, DP1, Dof_eff, X0):
    
    #print('Computing global vectors..')
    
    N_MeshPart                              = len(MeshPartList)
    MPList_Dof_eff                        = []
    MPList_NDof_eff                          = []
    
    GlobalRefLoadVector                     = np.array(F, dtype=float)
    InvDiagPreCondVector0                   = np.array(1.0/DP0, dtype=float)
    InvDiagPreCondVector1                   = np.array(1.0/DP1, dtype=float)
    X0                                       = np.array(X0, dtype=float)
    GlobDof_eff                             = np.array(Dof_eff,dtype=int)
    
    for i in range(N_MeshPart):
        
        MP_DofVector                            = MeshPartList[i]['DofVector']
        MP_NDOF                                 = len(MP_DofVector)
        
        MP_Dof_eff                                = np.intersect1d(MP_DofVector, Dof_eff)
        MP_LocDof_eff                           = getIndices(MP_DofVector, MP_Dof_eff)
        MP_NDof_eff                                 = len(MP_Dof_eff)
        
        MPList_Dof_eff.append(MP_Dof_eff)
        MPList_NDof_eff.append(MP_NDof_eff)
        
        MP_RefLoadVector                        = GlobalRefLoadVector[MP_DofVector]
        
        if len(DP0)>0: MP_InvDiagPreCondVector0 = InvDiagPreCondVector0[MP_DofVector]
        else:          MP_InvDiagPreCondVector0 = InvDiagPreCondVector0
        if len(DP1)>0: MP_InvDiagPreCondVector1 = InvDiagPreCondVector1[MP_DofVector]
        else:          MP_InvDiagPreCondVector1 = InvDiagPreCondVector1
        if len(X0)>0:  MP_X0                    = X0[MP_DofVector]
        else:          MP_X0                    = np.zeros(MP_NDOF)
        
        MeshPartList[i]['RefLoadVector']        = MP_RefLoadVector
        MeshPartList[i]['InvDiagPreCondVector0']= MP_InvDiagPreCondVector0
        MeshPartList[i]['InvDiagPreCondVector1']= MP_InvDiagPreCondVector1
        MeshPartList[i]['X0']                   = MP_X0
        MeshPartList[i]['Dof_eff']              = MP_Dof_eff
        MeshPartList[i]['LocDof_eff']              = MP_LocDof_eff
        MeshPartList[i]['NDOF']                 = MP_NDOF
        MeshPartList[i]['MP_NDof_eff']          = MP_NDof_eff
        
        
    
    GathDof_eff = np.hstack(MPList_Dof_eff)
    
    for i in range(N_MeshPart):
        
        if i == 0 :    
            
            MeshPartList[i]['GathDof_eff']    = GathDof_eff
            MeshPartList[i]['MPList_NDof_eff']   = MPList_NDof_eff
        
        else:        
            
            MeshPartList[i]['GathDof_eff']    = []
            MeshPartList[i]['MPList_NDof_eff']   = []
        





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
            MP_TypeGroup['ElemList_Cm'] = RefElemList_Cm
            MP_TypeGroup['ElemList_Ck'] = RefElemList_Ck
            
            MP_TypeGroupList.append(MP_TypeGroup)
            
        MeshPartList[i]['TypeGroupList'] = MP_TypeGroupList
          
        
    

def configMP_ElemStiffMat(MeshPartList, Ke):
    
    #print('Computing element stiffness matrices..')
    
    N_MeshPart = len(MeshPartList)
    
    #Reading Stiffness Matrices of Structured cells
    StrucSDList_StiffMatDB = Ke
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
        
        
        MeshPartList[i]['SubDomainData'] = MP_SubDomainData
        
    




def configMP_Neighbours(MeshPartList):
    
    #print('Computing neighbours..')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        
        MeshPartList[i]['NbrMPIdVector'] = []
        MeshPartList[i]['OvrlpLocalDofVecList'] = []
        
        
    #Finding Neighbour MeshParts and OverLapping DOFs
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
        RefMeshPart['RefLoadVector']               = MeshPartList[i]['RefLoadVector']
        RefMeshPart['NbrMPIdVector']               = MeshPartList[i]['NbrMPIdVector']
        RefMeshPart['OvrlpLocalDofVecList']        = MeshPartList[i]['OvrlpLocalDofVecList']
        RefMeshPart['GathDof_eff']               = MeshPartList[i]['GathDof_eff']
        RefMeshPart['LocDof_eff']               = MeshPartList[i]['LocDof_eff']
        RefMeshPart['MPList_NDof_eff']              = MeshPartList[i]['MPList_NDof_eff']
        RefMeshPart['WeightVector']                = MeshPartList[i]['WeightVector']
        RefMeshPart['InvDiagPreCondVector0']       = MeshPartList[i]['InvDiagPreCondVector0']
        RefMeshPart['InvDiagPreCondVector1']       = MeshPartList[i]['InvDiagPreCondVector1']
        RefMeshPart['X0']                          = MeshPartList[i]['X0']
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
    DofGlb, Sign, Type, Level, Cm, Ck, Ke, DP0, DP1, X0 = readInputFiles(MatDataPath, GlobData)
    Dof_eff, F = readLoadConfig(MatDataPath, GlobData)
    ElePart = getElePart(MatDataPath, N_MshPrt)

    
    #Converting Data
    MeshPartList = []
    configMP_ElemVectors(MeshPartList, ElePart, DofGlb, Type, Level, Cm, Ck, Sign)
    configMP_GlobVectors(MeshPartList, F, DP0, DP1, Dof_eff, X0)
    configMP_TypeGroupList(MeshPartList)
    configMP_ElemStiffMat(MeshPartList, Ke)
    configMP_Neighbours(MeshPartList)
    
    exportMP(MeshPartList, GlobData, PyDataPath)
    
    #calcMPQualityData(MeshPartList, PyDataPath, ModelName)

    
    
    

if __name__ == "__main__":

    N_MshPrt =      int(sys.argv[1])
    ScratchPath =   sys.argv[2]
    
    main(N_MshPrt, ScratchPath)
    


