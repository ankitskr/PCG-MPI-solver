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

import mpi4py
from mpi4py import MPI
import gc
import pickle
import zlib
from time import time
from GeneralFunc import loadBinDataInSharedMem 

def config_GlobData(MatDataPath, m, MeshData_Glob_FileName):
    
    GlobNFile           = MatDataPath + 'GlobN_L' + str(m+1) + '.mat'
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
    
    dtFile                               = MatDataPath + 'dt.mat'
    dt                                   = scipy.io.loadmat(dtFile)['Data'][0][0]
    MeshData_Glob['dt']                  = float(dt)
    
    ExportData = zlib.compress(pickle.dumps(MeshData_Glob, pickle.HIGHEST_PROTOCOL))
    f = open(MeshData_Glob_FileName, 'wb')
    f.write(ExportData)
    f.close()

    
if __name__ == "__main__":
    
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    
    ScratchPath     = sys.argv[1]
    ModelDataPath   = ScratchPath + 'ModelData/'
    MatDataPath     = ModelDataPath + 'Mat/'
    
    MaxMGLevel_FileName         = ModelDataPath + 'MaxMGLevel.mat'
    MaxMGLevel = np.array(scipy.io.loadmat(MaxMGLevel_FileName)['MaxMGLevel'][0][0], dtype=int)
    MetisSettings_FileName      = MatDataPath + 'MetisSettings.mat'
        
    for m in range(MaxMGLevel):
        
        NodeGlbFlat_FileName        = MatDataPath + 'NodeGlbFlat_L' + str(m+1) + '.bin'
        NodeGlbOffset_FileName      = MatDataPath + 'NodeGlbOffset_L' + str(m+1) + '.bin'
        Type_FileName               = MatDataPath + 'Type_L' + str(m+1) + '.bin'
        Level_FileName              = MatDataPath + 'Level_L' + str(m+1) + '.bin'
        MeshData_Glob_FileName      = MatDataPath + 'MeshData_Glob_L' + str(m+1) + '.zpkl'
        
        t1 = time()
        
        if Rank==0: config_GlobData(MatDataPath, m, MeshData_Glob_FileName)
        Comm.barrier()
        
        MeshData_Glob = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
        GlobNElem = MeshData_Glob['GlobNElem']
        GlobNNodeGlbFlat = MeshData_Glob['GlobNNodeGlbFlat']
        
        NodeGlbFlat  = loadBinDataInSharedMem(NodeGlbFlat_FileName, np.int32, int, (GlobNNodeGlbFlat,), Comm)
        NodeGlbOffset  = loadBinDataInSharedMem(NodeGlbOffset_FileName, np.int64, int, (GlobNElem,2), Comm)
        Type  = loadBinDataInSharedMem(Type_FileName, np.int32, int, (GlobNElem,), Comm)
        Level  = loadBinDataInSharedMem(Level_FileName, np.float64, float, (GlobNElem,), Comm)
        
        
        #IntfcWt = np.array(scipy.io.loadmat(MetisSettings_FileName)['IntfcWt'][0], dtype=float)
        MshPrtsNum = np.array(scipy.io.loadmat(MetisSettings_FileName)['MshPrtsNum'][0], dtype=int)
        
        if Rank==0: print('Am', time()-t1)
        t1 = time()
        
        #Assigning Weight to elements
        N_Elem = len(Type)
        WtGlb = np.ones(N_Elem)
        #for i in range(N_Elem):
        #    if Type[i]==-2:     WtGlb[i] = IntfcWt[0]
        #    elif Type[i]==-1:   WtGlb[i] = IntfcWt[1]
        
        
        
        assert N_Elem==np.shape(NodeGlbOffset)[0]
        
        if Rank==0: print('Bm', time()-t1)
        t1 = time()
        
        #TODO: Perform the element loop in Cython
        NodeGlb = [NodeGlbFlat[NodeGlbOffset[i,0]:NodeGlbOffset[i,1]+1] for i in range(N_Elem)]
        del NodeGlbFlat, NodeGlbOffset, Type
        
        if Rank==0: print('Cm', time()-t1)
        t1 = time()
        
        #Partitioning Mesh
        RefMeshPrts = MshPrtsNum[Rank]
        if RefMeshPrts == 1:
            ElePart = np.zeros(N_Elem, dtype=int)
        else:
            from mgmetis import metis
            objval, ElePart, npart = metis.part_mesh_dual(RefMeshPrts, NodeGlb, vwgt=WtGlb, ncommon=4) #<--- Check argument ncommon=4
        
        if Rank==0: print('Dm', time()-t1)
        t1 = time()
        
        print('N_Parts: ', RefMeshPrts)
        
        OutputFileName = MatDataPath + 'MeshPart_' + str(RefMeshPrts) + '_L' + str(m+1)
        np.save(OutputFileName+'.npy', ElePart)
        #scipy.io.savemat(OutputFileName+'.mat', {'RefPart': ElePart+1})
        
        if Rank==0: print('Em', time()-t1)
        t1 = time()
        
