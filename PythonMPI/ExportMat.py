# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import os
import numpy as np

from scipy.io import savemat

import pickle
import zlib
import sys
import shutil
import mpi4py
from mpi4py import MPI
from GeneralFunc import loadBinDataInSharedMem, readMPIBinFile, getIndices
import glob


#Initializing MPI
Comm = MPI.COMM_WORLD
N_Workers = Comm.Get_size()
Rank = Comm.Get_rank()

#Reading input arguments
Model           = sys.argv[1]
ScratchPath     = sys.argv[2]
Run             = sys.argv[3]
ExportVars      = sys.argv[4].split(' ')

MatResPath = ScratchPath + 'Results_Run' + Run + '/MatRes/'
if Rank==0:
    if os.path.exists(MatResPath):
        try:    shutil.rmtree(MatResPath)
        except:    raise Exception('VTK Path in use!')
    os.makedirs(MatResPath)
Comm.barrier()

#Loading Mesh data
MatDataPath             = ScratchPath + 'ModelData/Mat/'
MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
MeshData_Glob   = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))

GlobNNode       = MeshData_Glob['GlobNNode']
GlobNDof        = MeshData_Glob['GlobNDof']

#Loading results
ResVecDataPath = ScratchPath + 'Results_Run' + Run + '/ResVecData/'
NFiles = len(glob.glob(ResVecDataPath +  ExportVars[0] + '_*.mpidat'))

#TODO: Read the following in shared memory
RefDof = readMPIBinFile(ResVecDataPath+'Dof')
RefNodeId = readMPIBinFile(ResVecDataPath+'NodeId')
#Time_T = np.load(ResVecDataPath+'Time_T.npy')

"""
for i in range(NFiles):
    if i%N_Workers==Rank:
        print(i)
        Data = {}
        for Var in ExportVars:
            InpData         = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
            
            if Var in ['D', 'ES', 'PS1', 'PS2', 'PS3', 'PE1', 'PE2', 'PE3']:      
                A = np.zeros(GlobNNode, dtype=InpData.dtype)
                A[RefNodeId] = InpData
            elif Var == 'U':
                A = np.zeros(GlobNDof, dtype=InpData.dtype)
                A[RefDof] = InpData
        
            Data[Var] = A
        
        MatFile = MatResPath + Model + '_' + str(i)
        savemat(MatFile+'.mat', Data)
"""


RefVars = ['U', 'D', 'GS', 'GE', 'PS', 'PE']
if not len(RefNodeId) == GlobNNode: raise Exception
for i in range(NFiles):
    if i%N_Workers==Rank:
        print(i)
        Data = {}
        for Var in RefVars:
            if Var in ExportVars:
                if Var in ['D', 'GS', 'GE', 'PS', 'PE']:
                    if Var in ['GS', 'GE']:      VarComps = ['XX', 'YY', 'ZZ', 'YZ', 'XZ', 'XY']
                    elif Var in ['PS', 'PE']:    VarComps = ['1', '2', '3']
                    elif Var in ['D']:           VarComps = ['']
                    NComp = len(VarComps)
                    InpData = readMPIBinFile(ResVecDataPath + Var + '_' + str(i)).reshape([GlobNNode, NComp]).T
                    A = np.zeros(GlobNNode, dtype=InpData.dtype)
                    for k in range(NComp):
                        VarComp = Var+VarComps[k]
                        A[RefNodeId] = InpData[k,:]
                        Data[VarComp] = np.array(A,order='C')
                        
                elif Var == 'U':
                    InpData = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
                    A = np.zeros(GlobNDof, dtype=InpData.dtype)
                    A[RefDof] = InpData
                    Data[Var] = A
        
        MatFile = MatResPath + Model + '_' + str(i)
        savemat(MatFile+'.mat', Data)
