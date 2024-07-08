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
    
from mgmetis import metis



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
    
    KeDataFile =                MatDataPath + 'Ke.mat'
    MeDataFile =                MatDataPath + 'Me.mat'
    
    
    DofGlb = []
    Sign = []
    Level = []
    Type = []
    Cm = []
    Ck = []
    
    for i in range(N_Splits):
        
        DofGlb_Data_i = scipy.io.loadmat(MeshFile_DofGlbFileList[i])
        DofGlb_i = [a.T[0]-1 for a in DofGlb_Data_i['RefDofGlb'][0]]
        DofGlb += DofGlb_i
        
        Sign_Data_i = scipy.io.loadmat(MeshFile_SignFileList[i])
        Sign_i = [a.T[0] for a in Sign_Data_i['RefSign'][0]]
        Sign += Sign_i
            
        Level_i =     scipy.io.loadmat(MeshFile_LevelFileList[i])['RefLevel'].T[0]
        Level.append(Level_i)
        
        Type_i =      scipy.io.loadmat(MeshFile_TypeFileList[i])['RefType'][0] - 1
        Type.append(Type_i)
        
        Cm_i =        scipy.io.loadmat(MeshFile_CmFileList[i])['RefCm'].T[0]
        Cm.append(Cm_i)
        
        Ck_i =        scipy.io.loadmat(MeshFile_CkFileList[i])['RefCk'].T[0]
        Ck.append(Ck_i)
    
    DofGlb = np.array(DofGlb)
    Sign = np.array(Sign)
    Level = np.hstack(Level)
    Type = np.hstack(Type)
    Cm = np.hstack(Cm)
    Ck = np.hstack(Ck)
    
        
    Ke =        scipy.io.loadmat(KeDataFile)['Ke'][0]
    Me =        scipy.io.loadmat(MeDataFile)['Me'][0]
    
    
    #Reading Global nodal data files
    N_Splits =                    scipy.io.loadmat(MatDataPath + 'J.mat')['J'][0][0]
    MeshFile_DiagMFileList =      [MatDataPath + 'DiagM_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DPDiagKFileList =    [MatDataPath + 'DPDiagK_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_DPDiagMFileList =    [MatDataPath + 'DPDiagM_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_X0FileList =         [MatDataPath + 'X0_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_FFileList =          [MatDataPath + 'F_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_Dof_effFileList =    [MatDataPath + 'Dof_eff_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_NodeCoordVecFileList = [MatDataPath + 'NodeCoordVec_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    
    if isfile(MeshFile_X0FileList[0]):      ExistX0 = True
    else:                                   ExistX0 = False
    
    DiagM = []
    DP_DiagK = []
    DP_DiagM = []
    F = []
    Dof_eff = []
    X0 = []
    NodeCoordVec = []
    
    for i in range(N_Splits):
        
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
            
    
    DiagM = np.hstack(DiagM)
    DP_DiagK = np.hstack(DP_DiagK)
    DP_DiagM = np.hstack(DP_DiagM)
    F = np.hstack(F)
    Dof_eff = np.hstack(Dof_eff)
    NodeCoordVec = np.hstack(NodeCoordVec)
    
    X0 = np.array(X0)
    if ExistX0:     X0  = np.hstack(X0)
    
    
    MeshData_Glob                       = {}
    MeshData_Glob['GlobNDof_eff']        = len(Dof_eff)
    MeshData_Glob['GlobNDof']            = len(F)
    MeshData_Glob['ExistDP0']            = True
    MeshData_Glob['ExistDP1']            = False
        
    MeshData_Elem = [DofGlb, Sign, Type, Level, Cm, Ck]
    MeshData_Lib = [Me, Ke]
    MeshData_Nodal = [DiagM, DP_DiagK, DP_DiagM, X0, Dof_eff, F, NodeCoordVec]
    
    MeshData = {'Elem':     MeshData_Elem, 
                'Lib':      MeshData_Lib, 
                'Nodal':    MeshData_Nodal, 
                'Glob':     MeshData_Glob}
    
    return MeshData





def partitionMeshData(MatDataPath, PyDataPath, MeshData):
    MeshFile_MshPrtsNum = MatDataPath + 'MshPrtsNum.mat'
    MeshFile_WtGlb      = MatDataPath + 'WtGlb.mat'
    
    MshPrtsNum =      np.array(scipy.io.loadmat(MeshFile_MshPrtsNum)['MshPrtsNum'][0], dtype=int)
    WtGlb =           np.array(scipy.io.loadmat(MeshFile_WtGlb)['WtGlb'][0], dtype=float)
    
    DofGlb = MeshData['Elem'][0]
    N_Elem = len(DofGlb)
    NodeGlb = [np.array(DofGlb[i][::3]/3,dtype=int) for i in range(N_Elem)]
    
    for N_TotalMshPrt in MshPrtsNum:
        print('N_TotalMshPrt', N_TotalMshPrt)
        objval, epart, npart = metis.part_mesh_dual(N_TotalMshPrt, NodeGlb, vwgt=WtGlb)
        OutputFileName = PyDataPath + 'MeshPart_' + str(N_TotalMshPrt) +'.npz'
        np.savez_compressed(OutputFileName, Data = epart)
          



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
    partitionMeshData(MatDataPath, PyDataPath, MeshData)
    exportMeshData(PyDataPath, MeshData)
    


