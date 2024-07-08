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




def readInputFiles(MatPath):
    
    #print('Reading Matlab files..')
    
    N_Splits = len([f for f in listdir(MatPath) if isfile(join(MatPath, f)) and f.split('_')[0]=='M'])
    
    MeshFile_Level =              MatPath + 'Level.mat'
    MeshFile_Type =               MatPath + 'Type.mat'
    
    MeshFile_DofGlbFileList =     [MatPath + 'DofGlb_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_SignFileList =       [MatPath + 'Sign_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_MFileList =           [MatPath + 'M_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    KeDataFile =                   MatPath + 'Ke.mat'
    
    DofGlb = []
    Sign = []
    M = []
    for i in range(N_Splits):
        
        DofGlb_Data_i = scipy.io.loadmat(MeshFile_DofGlbFileList[i])
        DofGlb_i = [a.T[0]-1 for a in DofGlb_Data_i['RefDofGlb'][0]]
        DofGlb += DofGlb_i
        
        Sign_Data_i = scipy.io.loadmat(MeshFile_SignFileList[i])
        Sign_i = [a.T[0] for a in Sign_Data_i['RefSign'][0]]
        Sign += Sign_i
        
        M_i = scipy.io.loadmat(MeshFile_MFileList[i])['RefM'].T[0]
        M.append(M_i)
        
    M = np.hstack(M)
            
    
    Level =     scipy.io.loadmat(MeshFile_Level)['Level'].T[0]
    Type =      scipy.io.loadmat(MeshFile_Type)['Type'][0] - 1
    Ke =        scipy.io.loadmat(KeDataFile)['Ke'][0]
    
    
    return DofGlb, Sign, Type, Level, M, Ke





    
def readLoadConfig(MatPath):
    
    BCDataFile = MatPath + 'Fixed.mat'
    TimeDataFile = MatPath + 'TimeData.mat'
    FDataFile = MatPath + 'F.mat'
    DampingDataFile = MatPath + 'Damping.mat'
    
    BCData = scipy.io.loadmat(BCDataFile)
    TimeData = scipy.io.loadmat(TimeDataFile)
    FData = scipy.io.loadmat(FDataFile)
    DampingData = scipy.io.loadmat(DampingDataFile) 
    print('DampingData', DampingData)
    
    Fixed =         BCData['Fixed'] - 1                  #DOFs which are fixed
    F =             FData['F'].toarray().T[0]                 #force vector, nsize by 1
    Damping_Alpha = DampingData['Alpha'][0][0]
    
    RefPlotDofVec =  TimeData['RefPlotDofVec'].T[0] - 1
    MaxTime =       TimeData['Tmax'][0][0] 
    ft =            TimeData['ft'][0]                           
    dt =            TimeData['dt'][0][0]                        #time increment
    dT_Plot =       TimeData['dT_Plot'][0][0] 
    dT_Export =     TimeData['dT_Export'][0][0] 
    PlotFlag =      TimeData['PlotFlag'][0][0]              
    ExportFlag =    TimeData['ExportFlag'][0][0]            
    FintCalcMode =  TimeData['FintCalcMode'][0]
    qpoint =        TimeData['qpoint']

    GlobData = {}
    GlobData['GlobNDOF'] = len(F)
    #GlobData['RefPlotDofVec'] = RefPlotDofVec
    GlobData['Damping_Alpha'] = Damping_Alpha
    GlobData['MaxTime'] = MaxTime
    GlobData['DeltaLambdaList'] = np.array(ft, dtype=float)
    GlobData['TimeStepSize'] = dt
    GlobData['dT_Plot'] = dT_Plot
    GlobData['dT_Export'] = dT_Export
    GlobData['PlotFlag'] = PlotFlag
    GlobData['ExportFlag'] = ExportFlag
    GlobData['FintCalcMode'] = FintCalcMode
    
    print('GlobNDOF:', GlobData['GlobNDOF'])
    
    return Fixed, F, RefPlotDofVec, qpoint, GlobData







def getElePart(MatPath, p):
   
    #N_MP = 2**p
    
    if p == 0:
        
        MeshFile_MeshPartFile = MatPath + 'MeshPart_' + str(1) + '.mat'
        
        MeshPartData = scipy.io.loadmat(MeshFile_MeshPartFile)
        ElePart_List = MeshPartData['RefPart'][0][0]    
        
        N_Elem = len(ElePart_List[0])
        ElePart = np.zeros(N_Elem, dtype = int)
    
    else:
        
        MeshFile_MeshPartFile = MatPath + 'MeshPart_' + str(p) + '.mat'
        
        MeshPartData = scipy.io.loadmat(MeshFile_MeshPartFile)
        ElePart_List = MeshPartData['RefPart'][0][0]    
        
        ElePart =   ElePart_List[0] - 1           #the index of part each element belongs to, {1 1 1 2 3 1 3 2 2 ...}
    
    
    return ElePart
    







def configMP_ElemVectors(MeshPartList, ElePart, DofGlb, Type, Level, Sign): 
    
    MPList_Id = np.unique(ElePart)
    N_MeshPart = len(MPList_Id)
    
    print('N_Elem:', len(ElePart))
    print('N_MeshParts: ', len(MPList_Id))
    
    #print('Computing element vectors..')
    
    for i in range(N_MeshPart):
        
        MP_Id = MPList_Id[i]
        RefElemIdList = np.where(ElePart==MP_Id)[0]
        
        #Calc DOFIdVector for each Mesh Part
        MP_ElemList_NDOF = []
        MP_ElemList_SignVector = []
        MP_CumDOFIdVector = []
        
        for RefElemId in RefElemIdList:
            
            Elem_DOFIdVector = DofGlb[RefElemId]
            MP_ElemList_NDOF.append(len(Elem_DOFIdVector))
            MP_CumDOFIdVector += list(Elem_DOFIdVector)
            
            Elem_SignVector = np.array(Sign[RefElemId], dtype=bool)
            MP_ElemList_SignVector.append(Elem_SignVector)
        
        
        MP_UniqueDOFIdVector = np.unique(MP_CumDOFIdVector)
        
        
        #Calc LocDOFIdVector for each element
        N_Elem = len(RefElemIdList)
        MP_ElemList_LocDOFIdVector = []
        
        Elem_CumLocDOFIdVector = getIndices(MP_UniqueDOFIdVector, MP_CumDOFIdVector)
        
        I0 = 0
        for io in range(N_Elem):
            
            Elem_NDOF = MP_ElemList_NDOF[io]
            
            I1 = I0+Elem_NDOF
            Elem_LocDOFIdVector = Elem_CumLocDOFIdVector[I0:I1]
            I0 = I1
            
            MP_ElemList_LocDOFIdVector.append(Elem_LocDOFIdVector)
        
        
        #Calc Type/Level for each element
        ElemList_Type0 = Type[RefElemIdList]
        ElemList_Level0 = Level[RefElemIdList]
        
        
        MeshPart = {}
        MeshPart['Id'] = MP_Id
        MeshPart['DOFIdVector'] = MP_UniqueDOFIdVector
        MeshPart['ElemList_LocDofVector'] = MP_ElemList_LocDOFIdVector
        MeshPart['ElemList_SignVector'] = MP_ElemList_SignVector
        MeshPart['ElemList_Type'] = ElemList_Type0
        MeshPart['ElemList_Level'] = ElemList_Level0
        
        MeshPartList.append(MeshPart)
        
    






def configMP_GlobVectors(MeshPartList, F, M, Fixed, qpoint, RefPlotDofVec):
    
    #print('Computing global vectors..')
    
    N_MeshPart = len(MeshPartList)
    MPList_DOFIdVector = []
    MPList_NDOFIdVec = []
    
    GlobalRefTransientLoadVector = np.array(F, dtype=float)
    InvGlobalDiagMassVector = np.array(1.0/M, dtype=float)
    TestPlotFlag = True
    MPList_RefPlotDofIndicesList = []
    
    for i in range(N_MeshPart):
        
        MP_DOFIdVector = MeshPartList[i]['DOFIdVector']
        MP_NDOF = len(MP_DOFIdVector)
        MPList_DOFIdVector.append(MP_DOFIdVector)
        MPList_NDOFIdVec.append(MP_NDOF)
        
        MP_ConstrainedLocalDOFIdVector = getIndices(MP_DOFIdVector, np.intersect1d(MP_DOFIdVector, Fixed))
        MP_InvLumpedMassVector = InvGlobalDiagMassVector[MP_DOFIdVector]
        MP_RefTransientLoadVector = GlobalRefTransientLoadVector[MP_DOFIdVector]
        
        MP_RefPlotData                  = {}
        MP_RefPlotData['TestPlotFlag']  = False
        MP_RefPlotData['J']             = []
        MP_RefPlotData['LocalDofVec']   = []
        MP_RefPlotData['DofVec']        = np.intersect1d(MP_DOFIdVector, RefPlotDofVec)
        if len(MP_RefPlotData['DofVec']) > 0:
            MP_RefPlotData['LocalDofVec'] = getIndices(MP_DOFIdVector, MP_RefPlotData['DofVec'])
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
        
        MeshPartList[i]['RefTransientLoadVector'] = MP_RefTransientLoadVector
        MeshPartList[i]['InvLumpedMassVector'] = MP_InvLumpedMassVector
        MeshPartList[i]['ConstrainedLocalDOFIdVector'] = MP_ConstrainedLocalDOFIdVector
        MeshPartList[i]['RefPlotData'] = MP_RefPlotData
        MeshPartList[i]['NDOF'] = MP_NDOF
    
    
    GathDOFIdVector = np.hstack(MPList_DOFIdVector)
    
    for i in range(N_MeshPart):
        
        if i == 0 :    
            
            MeshPartList[i]['GathDOFIdVector'] = GathDOFIdVector
            MeshPartList[i]['MPList_NDOFIdVec'] = MPList_NDOFIdVec
            MeshPartList[i]['MPList_RefPlotDofIndicesList'] = MPList_RefPlotDofIndicesList
        
        else:        
            
            MeshPartList[i]['GathDOFIdVector'] = []
            MeshPartList[i]['MPList_NDOFIdVec'] = []
            MeshPartList[i]['MPList_RefPlotDofIndicesList'] = []
        





def configMP_TypeGroupList(MeshPartList):

    #print('Grouping Octrees based on their types..')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
        
        MP_ElemList_LocDOFIdVector = np.array(MeshPartList[i]['ElemList_LocDofVector'])
        MP_ElemList_SignVector = np.array(MeshPartList[i]['ElemList_SignVector'])
        MP_ElemList_Type = np.array(MeshPartList[i]['ElemList_Type'])
        MP_ElemList_Level = np.array(MeshPartList[i]['ElemList_Level'])
        
        UniqueElemTypeList = np.unique(MP_ElemList_Type)
        N_Type = len(UniqueElemTypeList)
        
        MP_TypeGroupList = []
        
        for j in range(N_Type):
            
            RefElemTypeId = UniqueElemTypeList[j]
            I = np.where(MP_ElemList_Type==RefElemTypeId)[0]
            
            RefElemList_LocDOFIdVector = np.array(tuple(MP_ElemList_LocDOFIdVector[I]), dtype=int).T
            RefElemList_SignVector = np.array(tuple(MP_ElemList_SignVector[I]), dtype=bool).T
            RefElemList_Level = np.array(tuple(MP_ElemList_Level[I]), dtype=float)
        
            MP_TypeGroup = {}
            MP_TypeGroup['ElemTypeId'] = RefElemTypeId
            MP_TypeGroup['ElemList_LocDofVector'] = RefElemList_LocDOFIdVector
            MP_TypeGroup['ElemList_LocDofVector_Flat'] = RefElemList_LocDOFIdVector.flatten()
            MP_TypeGroup['ElemList_SignVector'] = RefElemList_SignVector
            MP_TypeGroup['ElemList_Level'] = RefElemList_Level
            
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
        
        MeshPartList[i]['NbrIdVector'] = []
        MeshPartList[i]['NbrOvrlpLocalDOFIdVectorList'] = []
        
        
    #Finding Neighbour MeshParts and OverLapping DOFs
    for i in range(N_MeshPart):
    
        MP_DOFIdVector_i = MeshPartList[i]['DOFIdVector']
        MP_NbrOvrlpLocalDOFIdVectorList_i = MeshPartList[i]['NbrOvrlpLocalDOFIdVectorList']
        
        for j in range(i, N_MeshPart):
            
            if not i == j:
                
                MP_DOFIdVector_j = MeshPartList[j]['DOFIdVector']
                MP_NbrOvrlpLocalDOFIdVectorList_j = MeshPartList[j]['NbrOvrlpLocalDOFIdVectorList']
        
                OvrlpDOFIdVector = np.intersect1d(MP_DOFIdVector_i,MP_DOFIdVector_j, assume_unique=True)
                
                if len(OvrlpDOFIdVector) > 0:
                    
                    MP_NbrOvrlpLocalDOFIdVector_i = getIndices(MP_DOFIdVector_i, OvrlpDOFIdVector)
                    MP_NbrOvrlpLocalDOFIdVectorList_i.append(MP_NbrOvrlpLocalDOFIdVector_i)
                    MeshPartList[i]['NbrIdVector'].append(j)
                    
                    MP_NbrOvrlpLocalDOFIdVector_j = getIndices(MP_DOFIdVector_j, OvrlpDOFIdVector)
                    MP_NbrOvrlpLocalDOFIdVectorList_j.append(MP_NbrOvrlpLocalDOFIdVector_j)
                    MeshPartList[j]['NbrIdVector'].append(i)
                    





def exportMP(MeshPartList, GlobData, PyDataPath):

    #print('Exporting Mesh Parts..')
    
    N_MeshPart = len(MeshPartList)
    
    for i in range(N_MeshPart):
    
        RefMeshPart_FileName = PyDataPath + str(i) + '.zpkl'
        
        RefMeshPart = {}
        RefMeshPart['Id'] = MeshPartList[i]['Id']
        RefMeshPart['SubDomainData'] = MeshPartList[i]['SubDomainData']
        RefMeshPart['NDOF'] = MeshPartList[i]['NDOF']
        RefMeshPart['DOFIdVector'] = MeshPartList[i]['DOFIdVector']
        RefMeshPart['InvLumpedMassVector'] = MeshPartList[i]['InvLumpedMassVector']
        RefMeshPart['ConstrainedLocalDOFIdVector'] = MeshPartList[i]['ConstrainedLocalDOFIdVector']
        RefMeshPart['RefTransientLoadVector'] = MeshPartList[i]['RefTransientLoadVector']
        RefMeshPart['NbrIdVector'] = MeshPartList[i]['NbrIdVector']
        RefMeshPart['NbrOvrlpLocalDOFIdVectorList'] = MeshPartList[i]['NbrOvrlpLocalDOFIdVectorList']
        RefMeshPart['GathDOFIdVector'] = MeshPartList[i]['GathDOFIdVector']
        RefMeshPart['MPList_NDOFIdVec'] = MeshPartList[i]['MPList_NDOFIdVec']
        RefMeshPart['MPList_RefPlotDofIndicesList'] = MeshPartList[i]['MPList_RefPlotDofIndicesList']
        RefMeshPart['RefPlotData'] = MeshPartList[i]['RefPlotData']
        
        RefMeshPart['GlobData'] = GlobData
        
        Cmpr_RefMeshPart = zlib.compress(pickle.dumps(RefMeshPart, pickle.HIGHEST_PROTOCOL))
        
            
        f = open(RefMeshPart_FileName, 'wb')
        f.write(Cmpr_RefMeshPart)
        f.close()
   
   

   
   
   
   
def saveCheckPoint(ChkPnt_Data, ChkPnt_Num, PyDataPath):
    
    ChkPnt_FileName = PyDataPath + 'ChkPnt_' + str(ChkPnt_Num) + '.zpkl'
        
    Cmpr_ChkPnt_Data = zlib.compress(pickle.dumps(ChkPnt_Data, pickle.HIGHEST_PROTOCOL))
        
            
    f = open(ChkPnt_FileName, 'wb')
    f.write(Cmpr_ChkPnt_Data)
    f.close()
    
    print('CheckPoint:', ChkPnt_Num)
   
   
   
   
   
def loadCheckPoint(ChkPnt_Num, PyDataPath):
    
    ChkPnt_FileName = PyDataPath + 'ChkPnt_' + str(ChkPnt_Num) + '.zpkl'
        
    Cmpr_ChkPnt_Data = open(ChkPnt_FileName, 'rb').read()
    ChkPnt_Data = pickle.loads(zlib.decompress(Cmpr_ChkPnt_Data))
    
    print('CheckPoint Loaded:', ChkPnt_Num)
    
    return ChkPnt_Data




def calcMPQualityData(MeshPartList, Path, ModelName, p):
    
    N_MeshPart = len(MeshPartList)
    
    N_MPDOFList = []
    N_MPOvrlpDOFList = []
    
    for i in range(N_MeshPart):
    
        MP_DOFIdVector = MeshPartList[i]['DOFIdVector']
        MPOvrlpDOFList = np.hstack(MeshPartList[i]['NbrOvrlpLocalDOFIdVectorList'])
        
        N_MPDOFList.append(len(MP_DOFIdVector))
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

    
    

    
        
def main(p, ScratchPath):
    
    #Creating directories
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(p)  + '/'
    if not os.path.exists(PyDataPath):
        os.makedirs(PyDataPath)
    
    
    """
    #Reading Matlab Input Files
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    DofGlb, Sign, Type, Level, M, Ke = readInputFiles(MatDataPath)
    Fixed, F, RefPlotDofVec, qpoint, GlobData = readLoadConfig(MatDataPath)
    ElePart = getElePart(MatDataPath, p)
    """
    
    #ChkPnt_Data = [DofGlb, Sign, Type, Level, M, Ke, Fixed, F, RefPlotDofVec, qpoint, GlobData, ElePart]
    #saveCheckPoint(ChkPnt_Data, 1, PyDataPath)
    ChkPnt_Data = loadCheckPoint(1, PyDataPath)
    [DofGlb, Sign, Type, Level, M, Ke, Fixed, F, RefPlotDofVec, qpoint, GlobData, ElePart] = ChkPnt_Data
    
    
    #Converting Data
    #MeshPartList = []
    #configMP_ElemVectors(MeshPartList, ElePart, DofGlb, Type, Level, Sign)
    #configMP_GlobVectors(MeshPartList, F, M, Fixed, qpoint, RefPlotDofVec)
    #saveCheckPoint(MeshPartList, 2, PyDataPath)
    #MeshPartList = loadCheckPoint(2, PyDataPath)
    
    
    #configMP_TypeGroupList(MeshPartList)
    #saveCheckPoint(MeshPartList, 3, PyDataPath)
    MeshPartList = loadCheckPoint(3, PyDataPath)
    
    
    configMP_ElemStiffMat(MeshPartList, Ke)
    configMP_Neighbours(MeshPartList)    
    exportMP(MeshPartList, GlobData, PyDataPath)
    
    #calcMPQualityData(MeshPartList, PyDataPath, ModelName)


    
    
    
        
def main_Backup(p, ScratchPath):
    
    #Creating directories
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(p)  + '/'
    if os.path.exists(PyDataPath):
        
        try:    shutil.rmtree(PyDataPath)
        except:    raise Exception('PyDataPath in use!')
    
    os.makedirs(PyDataPath)
        
    
    #Reading Matlab Input Files
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    DofGlb, Sign, Type, Level, M, Ke = readInputFiles(MatDataPath)
    Fixed, F, RefPlotDofVec, qpoint, GlobData = readLoadConfig(MatDataPath)
    ElePart = getElePart(MatDataPath, p)

    
    #Converting Data
    MeshPartList = []
    configMP_ElemVectors(MeshPartList, ElePart, DofGlb, Type, Level, Sign)
    configMP_GlobVectors(MeshPartList, F, M, Fixed, qpoint, RefPlotDofVec)
    configMP_TypeGroupList(MeshPartList)
    configMP_ElemStiffMat(MeshPartList, Ke)
    configMP_Neighbours(MeshPartList)
    
    exportMP(MeshPartList, GlobData, PyDataPath)
    
    #calcMPQualityData(MeshPartList, PyDataPath, ModelName)

    
    
    

if __name__ == "__main__":

    p =           int(sys.argv[1])
    ScratchPath =   sys.argv[2]
    
    main(p, ScratchPath)
    


