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
    

    
    
        
def main(ScratchPath):
    
    #Reading Matlab Input Files
    MatPath = ScratchPath + 'ModelData/Mat/'
    
    #print('Reading Matlab files..')
    
    N_Splits = len([f for f in listdir(MatPath) if isfile(join(MatPath, f)) and f.split('_')[0]=='IntfcElem'])
    MeshFile_IntfcElemDataFile =    [MatPath + 'IntfcElem_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_FFileList =            [MatPath + 'F_' + str(i+1) + '.mat' for i in range(N_Splits)]
    MeshFile_NodeCoordVecFileList = [MatPath + 'NodeCoordVec_' + str(i+1) + '.mat' for i in range(N_Splits)]
    
    
    IntfcElemList = []
    F = []
    NodeCoordVec = []
    
    for i in range(N_Splits):
        
        IntfcElem_Data_i = scipy.io.loadmat(MeshFile_IntfcElemDataFile[i], mat_dtype=True)
        IntfcElem_i = [a[0] for a in IntfcElem_Data_i['RefIntfcElem'][0]]
        IntfcElemList += IntfcElem_i
        
        F_i = scipy.io.loadmat(MeshFile_FFileList[i])['RefF'].T[0]
        F.append(F_i)
        
        NodeCoordVec_i = scipy.io.loadmat(MeshFile_NodeCoordVecFileList[i])['RefNodeCoordVec'][0]
        NodeCoordVec.append(NodeCoordVec_i)
        
        
        
    F = np.hstack(F)
    NodeCoordVec = np.hstack(NodeCoordVec)
    
    
    N_IntfcElem = len(IntfcElemList)
    RefIntfcElemList = []
    for i in range(N_IntfcElem):
        N_Items = len(IntfcElemList[i])
        RefElem={}
        if N_Items>0:
            RefElem['IntfcArea']            = IntfcElemList[i][0][0][0]
            RefElem['RefPerpVector']        = IntfcElemList[i][1][0]
            RefElem['N_Node']               = int(IntfcElemList[i][2][0][0])
            RefElem['CoordList']            = IntfcElemList[i][3]
            RefElem['Nodal_NormalGapList']  = IntfcElemList[i][4][0]
            RefElem['NDOF']                 = int(IntfcElemList[i][5][0][0])
            RefElem['Knn']                  = IntfcElemList[i][6][0][0]
            RefElem['Kss']                  = IntfcElemList[i][6][0][1]
            RefElem['Damage']               = IntfcElemList[i][7][0][0]
            RefElem['NodeIdList']           = np.array(IntfcElemList[i][8][0] - 1, dtype=int)
            RefElem['DofIdList']            = np.array(IntfcElemList[i][9][0] - 1, dtype=int)
            RefElem['ElemId']               = IntfcElemList[i][10][0][0] - 1
            RefElem['InitCoordVector']      = RefElem['CoordList'].ravel()
            RefElem['RefPenaltyWeight']     = RefElem['Knn']*RefElem['IntfcArea']
            RefIntfcElemList.append(RefElem)
            
    IntfcElemList = RefIntfcElemList
    
    
    
    GlobSettingsFile = MatPath + 'GlobSettings.mat'
    GlobSettings = scipy.io.loadmat(GlobSettingsFile)
    
    GlobData                        = {}
    GlobData['GlobNDOF']            = len(F)
    GlobData['del_lambda_ini']      = GlobSettings['del_lambda_ini'][0][0]
    GlobData['RelTol']              = GlobSettings['RelTol'][0][0] 
    GlobData['m']                   = GlobSettings['m'][0][0] 
    GlobData['Nd0']                 = GlobSettings['Nd0'][0][0] 
    GlobData['n']                   = GlobSettings['n'][0][0] 
    GlobData['MaxIterCount']        = GlobSettings['MaxIterCount'][0][0] 
    GlobData['ExportFlag']          = GlobSettings['ExportFlag'][0][0]    
    GlobData['PlotFlag']          = GlobSettings['PlotFlag'][0][0]    
    GlobData['N_IntegrationPoints'] = GlobSettings['N_IntegrationPoints'][0][0]
    GlobData['MaxTimeStepCount']        = GlobSettings['MaxTimeStepCount'][0][0]
    GlobData['MaxFailedConvergenceCount']        = GlobSettings['MaxFailedConvergenceCount'][0][0]
    
    if GlobData['PlotFlag'] == 1:
    
        RefPlotData = {}
        RefPlotData['RefPlotDofVec'] = GlobSettings['RefPlotDofVec'].T[0] - 1
        RefPlotData['qpoint'] = GlobSettings['qpoint']  
        GlobData['RefPlotData'] = RefPlotData
    
    
    print('GlobNDOF:', GlobData['GlobNDOF'])
    
    BCDataFile = MatPath + 'Fixed.mat'
    BCData = scipy.io.loadmat(BCDataFile)
    Fixed =         BCData['Fixed'][0] - 1                  #DOFs which are fixed
    #Fixed = np.array([])
    
    GlobKDataFile = MatPath + 'GlobK.mat'
    GlobK = scipy.io.loadmat(GlobKDataFile)['GlobK'].todense()
    
    
    RefMeshData = {}
    RefMeshData['ConstrainedDOFIdVector'] = Fixed
    RefMeshData['RefTransientLoadVector'] = F
    RefMeshData['IntfcElemList'] = IntfcElemList
    RefMeshData['GlobK'] = GlobK
    RefMeshData['GlobData'] = GlobData
    RefMeshData['NodeCoordVec'] = NodeCoordVec
    
    
    Cmpr_RefMeshData = zlib.compress(pickle.dumps(RefMeshData, pickle.HIGHEST_PROTOCOL))
    
    RefMeshData_FileName = ScratchPath + 'ModelData/PyData.zpkl'
    f = open(RefMeshData_FileName, 'wb')
    f.write(Cmpr_RefMeshData)
    f.close()

   
    
    

if __name__ == "__main__":

    ScratchPath =   sys.argv[1]
    main(ScratchPath)
    


