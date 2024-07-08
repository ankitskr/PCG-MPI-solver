# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import os
import numpy as np

import scipy.io

import pickle
import zlib
import sys
import shutil
import mpi4py
from mpi4py import MPI
from GeneralFunc import loadBinDataInSharedMem, readMPIFile, getIndices
import glob


ScratchPath     = '/g/data/ud04/Ankit/PCG_Concrete_Image/'
MatDataPath     = ScratchPath + 'ModelData/Mat/'
MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
MeshData_Glob   = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
GlobNDof        = MeshData_Glob['GlobNDof']


#A = np.load("/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1/ResVecData/Dof_metadat.npy", allow_pickle=True)

#print(A.item()['OffsetData'])


#Loading results
print('loading 0')
RefDof0 = readMPIFile('/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1/ResVecData/Dof')
#InpData0 = readMPIFile('/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1/ResVecData/U_1')
#A0 = np.zeros(GlobNDof, dtype=InpData0.dtype)
#A0[RefDof0] = InpData0


#A = open("/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1/ResVecData/Dof.mpidat", "rb")

dtype = np.dtype('B')
f = open("/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1/ResVecData/Dof.mpidat", "rb")
numpy_data = np.fromfile(f,dtype=np.int64)
print(numpy_data)
print(RefDof0)


exit()

#print(np.count_nonzero(A0==0), np.count_nonzero(InpData0==0))
print(np.where(RefDof0[268434944:]>0))
print(len(RefDof0), len(np.unique(RefDof0)))
#print(len(A0), len(np.unique(A0)))
#print(len(InpData0), len(np.unique(InpData0)))

exit()

print('loading 1')
RefDof1 = readMPIFile('/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1_01072022_191351/ResVecData/Dof')
InpData1 = readMPIFile('/g/data/ud04/Ankit/PCG_Concrete_Image/Results_Run1_01072022_191351/ResVecData/U_1')
A1 = np.zeros(GlobNDof, dtype=InpData1.dtype)
A1[RefDof1] = InpData1

#print(np.linalg.norm(np.sort(InpData0)-np.sort(InpData1))/np.linalg.norm(InpData1))
print(np.count_nonzero(np.array([0, 1,4,0,1,2,5.6, 0,0,0,0,0.1,5])==0))
print(np.count_nonzero(RefDof0==0), np.count_nonzero(RefDof1==0))




