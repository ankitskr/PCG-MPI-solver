# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import numpy as np
import scipy.io
import os, sys
import os.path
import shutil
from numpy.linalg import norm
from GeneralFunc import splitSerialData
import pickle
import zlib

from os import listdir
from os.path import isfile, join
    



def readInputFiles(MatDataPath):
    
    #print('Reading Matlab files..')
    
    N_Splits = len([f for f in listdir(MatDataPath) if isfile(join(MatDataPath, f)) and f.split('_')[0]=='DofGlb'])
    
    #Reading Element data files (Partitioned)
    MeshFile_DofGlbFileList =     [MatDataPath + 'DofGlb_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_SignFileList =       [MatDataPath + 'Sign_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_LevelFileList =      [MatDataPath + 'Level_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_TypeFileList =       [MatDataPath + 'Type_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_CmFileList =         [MatDataPath + 'Cm_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_CkFileList =         [MatDataPath + 'Ck_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_IntfcElemDataFile =  [MatDataPath + 'IntfcElem_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    KeDataFile =                MatDataPath + 'Ke.mat'
    MeDataFile =                MatDataPath + 'Me.mat'
    
    
    DofGlb = []
    Sign = []
    Level = []
    Type = []
    Cm = []
    Ck = []
    IntfcElem0 = []
    
    for i in range(N_Splits):
        
        DofGlb_Data_i = scipy.io.loadmat(MeshFile_DofGlbFileList[i])
        DofGlb_i = [a.T[0]-1 for a in DofGlb_Data_i['Data'][0]]
        DofGlb += DofGlb_i
        
        Sign_Data_i = scipy.io.loadmat(MeshFile_SignFileList[i])
        Sign_i = [a.T[0] if len(a.T)>0 else [0] for a in Sign_Data_i['Data'][0]]
        Sign += Sign_i
            
        Level_i =     scipy.io.loadmat(MeshFile_LevelFileList[i])['Data'].T[0]
        Level.append(Level_i)
        
        Type_i =      scipy.io.loadmat(MeshFile_TypeFileList[i])['Data'][0] - 1
        Type.append(Type_i)
        
        if os.path.exists(MeshFile_IntfcElemDataFile[i]):
            IntfcElem_Data_i = scipy.io.loadmat(MeshFile_IntfcElemDataFile[i], mat_dtype=True)
            IntfcElem_i = [a[0] for a in IntfcElem_Data_i['Data'][0]]
            IntfcElem0 += IntfcElem_i
        
        if os.path.exists(MeshFile_CmFileList[i]):
            Cm_i =     scipy.io.loadmat(MeshFile_CmFileList[i])['Data'].T[0]
            Cm.append(Cm_i)
            
            Ck_i =     scipy.io.loadmat(MeshFile_CkFileList[i])['Data'].T[0]
            Ck.append(Ck_i)
            
            
    print("*******************************************************")
    print("TODO: Save DofGlb and Sign in flatten form (in Pythor or Matlab). Use matlab: horzcat/cell2mat, Python: ravel, unravel_index functions")
    print("*******************************************************")
    
    DofGlb = np.array(DofGlb, dtype=object)
    Sign = np.array(Sign, dtype=object)
    Level = np.hstack(Level)
    Cm = np.hstack(Cm)
    Ck = np.hstack(Ck)
    Type = np.hstack(Type)
    
        
    Ke =        scipy.io.loadmat(KeDataFile)['Ke'][0]
    Me =        scipy.io.loadmat(MeDataFile)['Me'][0]
    
    N_Elem = len(Level)
    IntfcElem = [{} for i in range(N_Elem)]
    N_IntfcElem = len(IntfcElem0)
    for i in range(N_IntfcElem):
        RefElem={}
        RefElem['IntfcArea']            = IntfcElem0[i][0][0][0]
        RefElem['RefPerpVector']        = IntfcElem0[i][1][0]
        RefElem['N_Node']               = int(IntfcElem0[i][2][0][0])
        RefElem['CoordList']            = IntfcElem0[i][3]
        RefElem['Nodal_NormalGapList']  = IntfcElem0[i][4][0]
        RefElem['NDOF']                 = int(IntfcElem0[i][5][0][0])
        RefElem['TSL_Param']            = {}
        RefElem['TSL_Param']['Kn0']     = IntfcElem0[i][6][0][0]
        RefElem['TSL_Param']['ft']      = IntfcElem0[i][6][0][1]
        RefElem['TSL_Param']['GIc']     = IntfcElem0[i][6][0][2]
        RefElem['TSL_Param']['Ks0']     = IntfcElem0[i][6][0][3]
        RefElem['TSL_Param']['fs']      = IntfcElem0[i][6][0][4]
        RefElem['TSL_Param']['GIIc']    = IntfcElem0[i][6][0][5]
        RefElem['Damage']               = IntfcElem0[i][7][0][0]
        RefElem['NodeIdList']           = IntfcElem0[i][8][0] - 1
        RefElem['DofIdList']            = IntfcElem0[i][9] - 1
        RefElem['ElemId']               = int(IntfcElem0[i][10][0][0] - 1)
        RefElem['InitCoordVector']      = RefElem['CoordList'].ravel()
        
        IntfcElem[RefElem['ElemId']]    = RefElem
        
    IntfcElem = np.array(IntfcElem, dtype=object)
    
    
    #Reading Global nodal data files
    N_Splits =                    scipy.io.loadmat(MatDataPath + 'J.mat')['J'][0][0]
    MeshFile_DiagMFileList =      [MatDataPath + 'DiagM_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_FFileList =          [MatDataPath + 'F_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_NodeCoordVecFileList = [MatDataPath + 'NodeCoordVec_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    
    DiagM = []
    F = []
    NodeCoordVec = []
    
    for i in range(N_Splits):
        
        DiagM_i = scipy.io.loadmat(MeshFile_DiagMFileList[i])['Data'].T[0]
        DiagM.append(DiagM_i)
        
        F_i = scipy.io.loadmat(MeshFile_FFileList[i])['Data'].T[0]
        F.append(F_i)
        
        NodeCoordVec_i = scipy.io.loadmat(MeshFile_NodeCoordVecFileList[i])['Data'][0]
        NodeCoordVec.append(NodeCoordVec_i)
        
    
    DiagM             	= np.hstack(DiagM)
    F                 	= np.hstack(F)
    NodeCoordVec     	= np.hstack(NodeCoordVec)
    
    BCDataFile         	= MatDataPath + 'Fixed.mat'
    BCData             	= scipy.io.loadmat(BCDataFile)
    DampingDataFile 	= MatDataPath + 'Damping.mat'
    DampingData     	= scipy.io.loadmat(DampingDataFile)
    
    Fixed             	= BCData['Fixed'] - 1                  #DOFs which are fixed
    Damping_Alpha     	= DampingData['Alpha'][0][0]
    
    MeshData_Glob                        = {}
    MeshData_Glob['GlobNDof']            = len(F)
    MeshData_Glob['Damping_Alpha']       = Damping_Alpha
        
    MeshData_Lib     = [Me, Ke]
    MeshData_Nodal     = [DiagM, F, Fixed, NodeCoordVec]
    
    MeshData = {'DofGlb':       DofGlb,
                'Sign':         Sign,
                'Type':         Type,
                'Level':        Level,
                'Ck':           Ck,
                'Cm':           Cm,
                'IntfcElem':    IntfcElem, 
                'Lib':          MeshData_Lib, 
                'Nodal':        MeshData_Nodal, 
                'Glob':         MeshData_Glob}
    
    return MeshData





def exportMeshData(PyDataPath, MeshData):
    
    DataNameList = list(MeshData.keys())
    N_Data = len(DataNameList)
    for i in range(N_Data):
        DataName = DataNameList[i]
        ExportData = zlib.compress(pickle.dumps(MeshData[DataName], pickle.HIGHEST_PROTOCOL))
        Data_FileName = PyDataPath + 'MeshData_'+ DataName + '.zpkl'
        f = open(Data_FileName, 'wb')
        f.write(ExportData)
        f.close()
    
    



if __name__ == "__main__":

    ScratchPath =   sys.argv[1]
    
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    PyDataPath = ScratchPath + 'ModelData/Py/'
    
    #Creating directories
    if os.path.exists(PyDataPath):
        try:    shutil.rmtree(PyDataPath)
        except:    raise Exception('PyDataPath in use!')
    
    os.makedirs(PyDataPath)
    
    #Reading Matlab Input Files
    MeshData = readInputFiles(MatDataPath)
    exportMeshData(PyDataPath, MeshData)
    

