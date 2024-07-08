# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import os
from evtk.hl import unstructuredGridToVTK
from evtk.vtk import VtkPolygon, VtkTetra
import numpy as np

import scipy.io

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
Mode            = sys.argv[5]

VTKPath = ScratchPath + 'Results_Run' + Run + '/VTKs/'
if Rank==0:
    if os.path.exists(VTKPath):
        try:    shutil.rmtree(VTKPath)
        except:    raise Exception('VTK Path in use!')
    os.makedirs(VTKPath)
Comm.barrier()

#Loading Mesh data
MatDataPath             = ScratchPath + 'ModelData/Mat/'
FacesFlat_FileName      = MatDataPath + 'FacesFlat.bin'
FacesOffset_FileName    = MatDataPath + 'FacesOffset.bin'
MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
Nodes_FileName          = MatDataPath + 'nodes.bin'

MeshData_Glob   = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))

GlobNFacesFlat  = MeshData_Glob['GlobNFacesFlat']
GlobNFaces      = MeshData_Glob['GlobNFaces']
GlobNNode       = MeshData_Glob['GlobNNode']
GlobNDof        = MeshData_Glob['GlobNDof']

#TODO: Load the folllowing data on different NUMA nodes in case of memory overflow.
FacesFlat   = loadBinDataInSharedMem(FacesFlat_FileName, np.int32, int, (GlobNFacesFlat,), Comm)
FacesOffset = loadBinDataInSharedMem(FacesOffset_FileName, np.int64, int, (GlobNFaces,2), Comm)
Nodes       = loadBinDataInSharedMem(Nodes_FileName, np.float64, float, (GlobNNode,3), Comm)


#Loading results
ResVecDataPath = ScratchPath + 'Results_Run' + Run + '/ResVecData/'
NFiles = len(glob.glob(ResVecDataPath +  ExportVars[0] + '_*.mpidat'))

#TODO: Read the following in shared memory
RefDof = readMPIBinFile(ResVecDataPath+'Dof')
RefNodeId = readMPIBinFile(ResVecDataPath+'NodeId')
Time_T = np.load(ResVecDataPath+'Time_T.npy')

if Mode in ['MidSlices', 'Boundary']:
    
    if Mode =='MidSlices': 
        #RefX = [0.0128, 0.0064, 0.0128]
        #RefX = [0.00992, 0.01016, 0.00992]
        #Identifying mesh corresponding to mid-slices
        SlcFaceIds = []
        FacesIds = np.arange(GlobNFaces)
        Lch = np.max(Nodes)-np.min(Nodes)
        for io in range(3):
            if Rank==0: print('io', io)
            X = Nodes[:,io]
            MaxX = np.max(X); MinX = np.min(X)
            MidX = 0.5*(MinX+MaxX)
            #MidX = RefX[io]
            TempX = Nodes[FacesFlat[FacesOffset[:,0]],io]
            TempFaceIds = FacesIds[(np.abs(TempX-MidX)/Lch<1e-8)]
            I = (np.abs(X-MidX)/Lch<1e-8)
            Io = TempFaceIds[np.array([np.all(I[FacesFlat[FacesOffset[fid,0]:FacesOffset[fid,1]+1]]) for fid in TempFaceIds])]
            SlcFaceIds.append(Io)
        SlcFaceIds = np.hstack(SlcFaceIds) #face ids corresponding to mid-slices
        
    elif Mode == 'Boundary':
    
        #Identifying mesh corresponding to boundary-slices
        PolysFlat_FileName      = MatDataPath + 'PolysFlat.bin'
        GlobNPolysFlat          = MeshData_Glob['GlobNPolysFlat']
        PolysFlat               = loadBinDataInSharedMem(PolysFlat_FileName, np.int32, int, (GlobNPolysFlat,), Comm)
        
        SlcFaceIds              = np.where(np.bincount(np.abs(PolysFlat), minlength=GlobNFaces)==1)[0]  #face ids corresponding to boundary-slices
        
        #Removing repeat cut cells
        Lch = np.max(Nodes)-np.min(Nodes)
        Nodes_Norm = np.round(Nodes/Lch,14)
        FaceMidNodes = [np.mean(Nodes_Norm[FacesFlat[FacesOffset[fid,0]:FacesOffset[fid,1]+1],:], axis=0) for fid in SlcFaceIds]
        u, io, c = np.unique(FaceMidNodes, return_counts=True, return_index=True, axis=0)
        I = io[np.where(c==1)[0]]
        SlcFaceIds = SlcFaceIds[I]
        
    
    SlcFaces = [FacesFlat[FacesOffset[fid,0]:FacesOffset[fid,1]+1] for fid in SlcFaceIds]
    NSlcFaces = len(SlcFaceIds)
    SlcFacesOffset1 = np.cumsum(np.array([len(SlcFaces[f]) for f in range(NSlcFaces)]))
    SlcFacesOffset0 = np.hstack([[0], SlcFacesOffset1[:-1]])+1
    SlcFacesOffset = np.vstack([SlcFacesOffset0, SlcFacesOffset1]).T -1
    SlcFacesFlat = np.hstack(SlcFaces)
    SlcNodeIds = np.unique(SlcFacesFlat)
    SlcNodes = Nodes[SlcNodeIds,:]
    NSlcNodes = len(SlcNodeIds)
    Old2New = np.zeros(GlobNNode,dtype=int)
    Old2New[SlcNodeIds] = np.arange(NSlcNodes)
    SlcFacesFlat = Old2New[SlcFacesFlat]
    
    #Defining vertices
    Slc_x = np.array(SlcNodes[:,0],order='C')
    Slc_y = np.array(SlcNodes[:,1],order='C')
    Slc_z = np.array(SlcNodes[:,2],order='C')
    
    #Defining cell types
    ctype   = VtkPolygon.tid*np.ones(NSlcFaces)        

    
    for i in range(NFiles):
        if i%N_Workers==Rank:
            print(i)
            pointData = {}
            for Var in ExportVars:
                InpData         = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
                
                if Var in ['D', 'ES', 'PS1', 'PS2', 'PS3', 'PE1', 'PE2', 'PE3']:      
                    A = np.zeros(GlobNNode, dtype=InpData.dtype)
                    A[RefNodeId] = InpData
                    pointData[Var] = A[SlcNodeIds]
                elif Var == 'U':
                    A = np.zeros(GlobNDof, dtype=InpData.dtype)
                    A[RefDof] = InpData
                    Ux = np.array(A[::3],order='C')
                    Uy = np.array(A[1::3],order='C')
                    Uz = np.array(A[2::3],order='C')
                    pointData[Var] = (Ux[SlcNodeIds], Uy[SlcNodeIds], Uz[SlcNodeIds])
            
            VTKFile = VTKPath + Model + '_' + str(i)
            unstructuredGridToVTK(VTKFile, Slc_x, Slc_y, Slc_z, connectivity = SlcFacesFlat, offsets = SlcFacesOffset[:,1]+1, cell_types = ctype, cellData = {}, pointData = pointData)


    if Rank==0:
        VTKInfoFile = open( VTKPath + 'VTKInfo.txt', 'w')
        VTKInfoFile.write('%15s  %12s\n'%('VTKFileCount','Time (s)'))
        for i in range(NFiles):
            VTKInfoFile.write('%15d  %12.2e\n'%(i,Time_T[i]))
        VTKInfoFile.close()


            
elif Mode=='Delaunay':

    from scipy.spatial import Delaunay
    Polys           = Delaunay(Nodes).simplices
    N_Elem          = len(Polys)
    PolysOffset1    = np.cumsum(np.array([len(Polys[p]) for p in range(N_Elem)]))
    PolysOffset0    = np.hstack([[0], PolysOffset1[:-1]])+1
    PolysOffset     = np.vstack([PolysOffset0, PolysOffset1]).T -1
    PolysFlat       = np.hstack(Polys)
    
    #Defining vertices
    x = np.array(Nodes[:,0],order='C')
    y = np.array(Nodes[:,1],order='C')
    z = np.array(Nodes[:,2],order='C')
    
    #Defining cell types
    ctype   = VtkTetra.tid*np.ones(N_Elem)        
    
    for i in range(NFiles):
        if i%N_Workers==Rank:
            print(i)
            pointData = {}
            for Var in ExportVars:
                InpData         = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
                
                if Var in ['D', 'ES', 'PS1', 'PS2', 'PS3', 'PE1', 'PE2', 'PE3']:      
                    A = np.zeros(GlobNNode, dtype=InpData.dtype)
                    A[RefNodeId] = InpData
                    pointData[Var] = A
                elif Var == 'U':
                    A = np.zeros(GlobNDof, dtype=InpData.dtype)
                    A[RefDof] = InpData
                    Ux = np.array(A[::3],order='C')
                    Uy = np.array(A[1::3],order='C')
                    Uz = np.array(A[2::3],order='C')
                    pointData[Var] = (Ux, Uy, Uz)
            
            VTKFile = VTKPath + Model + '_' + str(i)
            unstructuredGridToVTK(VTKFile, x, y, z, connectivity = PolysFlat, offsets = PolysOffset[:,1]+1, cell_types = ctype, cellData = {}, pointData = pointData)
            
            
    
elif Mode=='Full':
    #Defining vertices
    x = np.array(Nodes[:,0],order='C')
    y = np.array(Nodes[:,1],order='C')
    z = np.array(Nodes[:,2],order='C')
    
    #Defining cell types
    ctype   = VtkPolygon.tid*np.ones(GlobNFaces)        

    for i in range(NFiles):
        if i%N_Workers==Rank:
            print(i)
            pointData = {}
            for Var in ExportVars:
                InpData         = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
                
                if Var in ['D', 'ES', 'PS1', 'PS2', 'PS3', 'PE1', 'PE2', 'PE3']:      
                    A = np.zeros(GlobNNode, dtype=InpData.dtype)
                    A[RefNodeId] = InpData
                    pointData[Var] = A
                elif Var == 'U':
                    A = np.zeros(GlobNDof, dtype=InpData.dtype)
                    A[RefDof] = InpData
                    Ux = np.array(A[::3],order='C')
                    Uy = np.array(A[1::3],order='C')
                    Uz = np.array(A[2::3],order='C')
                    pointData[Var] = (Ux, Uy, Uz)
            
            VTKFile = VTKPath + Model + '_' + str(i)
            unstructuredGridToVTK(VTKFile, x, y, z, connectivity = FacesFlat, offsets = FacesOffset[:,1]+1, cell_types = ctype, cellData = {}, pointData = pointData)
            
    
