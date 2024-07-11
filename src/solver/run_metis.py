# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import numpy as np
import scipy.io
import os, sys
from os import listdir
from os.path import isfile, join

import pickle
import zlib
from time import time
from file_operations import loadBinDataInSharedMem 

def config_GlobData(MDF_Path, MeshData_Glob_FileName):
    
    GlobNFile           = MDF_Path + 'GlobN.mat'
    GlobN               = scipy.io.loadmat(GlobNFile)['Data'][0]
    
    MeshData_Glob                        = {}
    MeshData_Glob['GlobNElem']           = int(GlobN[0])
    MeshData_Glob['GlobNDof']            = int(GlobN[1])
    MeshData_Glob['GlobNNode']           = int(GlobN[1]/3)
    MeshData_Glob['GlobNDofGlbFlat']     = int(GlobN[2])
    MeshData_Glob['GlobNNodeGlbFlat']    = int(GlobN[3])
    MeshData_Glob['GlobNDofEff']         = int(GlobN[4])
    MeshData_Glob['GlobNFacesFlat']      = int(GlobN[5])
    MeshData_Glob['GlobNFaces']          = int(GlobN[6])
    MeshData_Glob['GlobNPolysFlat']      = int(GlobN[7])
    MeshData_Glob['GlobNFixedDof']       = int(GlobN[8])
    
    dtFile                               = MDF_Path + 'dt.mat'
    dt                                   = scipy.io.loadmat(dtFile)['Data'][0][0]
    MeshData_Glob['dt']                  = float(dt)
    
    ExportData = zlib.compress(pickle.dumps(MeshData_Glob, pickle.HIGHEST_PROTOCOL))
    f = open(MeshData_Glob_FileName, 'wb')
    f.write(ExportData)
    f.close()

    
if __name__ == "__main__":
    
    print(">loading model data..")
    
    t1_ = time()
    
    #N_TotalMshPrt = int(os.environ.get('NPrt'))
    N_TotalMshPrt = int(sys.argv[1])
    
    ModelDataPath_FileName = '__pycache__/ModelDataPaths.zpkl'
    ModelDataPaths = pickle.loads(zlib.decompress(open(ModelDataPath_FileName, 'rb').read()))
    MDF_Path = ModelDataPaths['MDF_Path']
    
    NodeGlbFlat_FileName        = MDF_Path + 'NodeGlbFlat.bin'
    NodeGlbOffset_FileName      = MDF_Path + 'NodeGlbOffset.bin'
    MeshData_Glob_FileName      = MDF_Path + 'MeshData_Glob.zpkl'
    
    
    config_GlobData(MDF_Path, MeshData_Glob_FileName)
    
    MeshData_Glob = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
    GlobNElem = MeshData_Glob['GlobNElem']
    GlobNNodeGlbFlat = MeshData_Glob['GlobNNodeGlbFlat']
    
    NodeGlbFlat = np.fromfile(NodeGlbFlat_FileName, dtype=np.int32).astype(int)
    NodeGlbOffset = np.fromfile(NodeGlbOffset_FileName, dtype=np.int64).astype(int).reshape((GlobNElem,2),order='F')
    
    N_Elem = GlobNElem
    WtGlb = np.ones(N_Elem)
    
    #TODO: Perform the element loop in Cython
    NodeGlb = [NodeGlbFlat[NodeGlbOffset[i,0]:NodeGlbOffset[i,1]+1] for i in range(N_Elem)]
    del NodeGlbFlat, NodeGlbOffset
    
    #generating Mesh partition indices
    print(f">generating indices for {N_TotalMshPrt} mesh parts..")
    
    RefMeshPrts = N_TotalMshPrt
    if RefMeshPrts == 1:
        ElePart = np.zeros(N_Elem, dtype=int)
    else:
        from mgmetis import metis
        objval, ElePart, npart = metis.part_mesh_dual(RefMeshPrts, NodeGlb, vwgt=WtGlb)
    
    
    OutputFileName = MDF_Path + 'MeshPart_' + str(RefMeshPrts)
    np.save(OutputFileName+'.npy', ElePart)
    scipy.io.savemat(OutputFileName+'.mat', {'RefPart': ElePart+1})
    
    print(">success!")
    
    print(f">total runtime: {np.round(time()-t1_,2)} sec")
    
    