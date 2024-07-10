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
from file_operations import loadBinDataInSharedMem, readMPIBinFile, getIndices
import glob
from time import time


#Initializing MPI
Comm = MPI.COMM_WORLD
N_Workers = Comm.Get_size()
Rank = Comm.Get_rank()

t1_ = time()
    
#Reading input arguments
Run             = sys.argv[1]
ExportVars      = sys.argv[2].split(' ')
Mode            = sys.argv[3]

if Rank==0: print('>loading model and result data..')
    
ModelDataPath_FileName = '__pycache__/ModelDataPaths.zpkl'
ModelDataPaths = pickle.loads(zlib.decompress(open(ModelDataPath_FileName, 'rb').read()))
ModelName = ModelDataPaths['ModelName']
ScratchPath = ModelDataPaths['ScratchPath']
MDF_Path = ModelDataPaths['MDF_Path']

VTKPath = ScratchPath + '/Results_Run' + Run + '/VTKs/'
if Rank==0:
    if os.path.exists(VTKPath):
        try:    shutil.rmtree(VTKPath)
        except:    raise Exception('VTK Path in use!')
    os.makedirs(VTKPath)
Comm.barrier()

#Loading Mesh data
FacesFlat_FileName      = MDF_Path + 'FacesFlat.bin'
FacesOffset_FileName    = MDF_Path + 'FacesOffset.bin'
MeshData_Glob_FileName  = MDF_Path + 'MeshData_Glob.zpkl'
Nodes_FileName          = MDF_Path + 'nodes.bin'

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
ResVecDataPath = ScratchPath + '/Results_Run' + Run + '/ResVecData/'
NFiles = len(glob.glob(ResVecDataPath +  ExportVars[0] + '_*.mpidat'))

#TODO: Read the following in shared memory
RefDof = readMPIBinFile(ResVecDataPath+'Dof')
RefNodeId = readMPIBinFile(ResVecDataPath+'NodeId')
Time_T = np.load(ResVecDataPath+'Time_T.npy')

if Rank==0: print('>exporting vtk files..')

if Mode in ['MidSlices', 'Boundary']:
    
    if Mode =='MidSlices': 
        #RefX = [0.0128, 0.0064, 0.0128]
        #RefX = [0.00992, 0.01016, 0.00992]
        #Identifying mesh corresponding to mid-slices
        SlcFaceIds = []
        FacesIds = np.arange(GlobNFaces)
        Lch = np.max(Nodes)-np.min(Nodes)
        for io in range(3):
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
        PolysFlat_FileName      = MDF_Path + 'PolysFlat.bin'
        GlobNPolysFlat          = MeshData_Glob['GlobNPolysFlat']
        PolysFlat               = loadBinDataInSharedMem(PolysFlat_FileName, np.int32, int, (GlobNPolysFlat,), Comm)
        
        SlcFaceIds              = np.where(np.bincount(np.abs(PolysFlat), minlength=GlobNFaces)==1)[0]  #face ids corresponding to boundary-slices
        
    
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

    RefVars = ['U', 'D', 'GS', 'GE', 'PS', 'PE']
    if not len(RefNodeId) == GlobNNode: raise Exception
    for i in range(NFiles):
        if i%N_Workers==Rank:
            pointData = {}
            
            for Var in RefVars:
                if Var in ExportVars:
                    if Var in ['D', 'GS', 'GE', 'PS', 'PE']:
                        if Var in ['GS', 'GE']:      VarComps = ['-XX', '-YY', '-ZZ', '-YZ', '-XZ', '-XY']
                        elif Var in ['PS', 'PE']:    VarComps = ['-1', '-2', '-3']
                        elif Var in ['D']:           VarComps = ['']
                        NComp = len(VarComps)
                        InpData = readMPIBinFile(ResVecDataPath + Var + '_' + str(i)).reshape([GlobNNode, NComp]).T
                        A = np.zeros(GlobNNode, dtype=InpData.dtype)
                        for k in range(NComp):
                            VarComp = Var+VarComps[k]
                            A[RefNodeId] = InpData[k,:]
                            pointData[VarComp] = A[SlcNodeIds]
                    
                    elif Var == 'U':
                        InpData = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
                        A = np.zeros(GlobNDof, dtype=InpData.dtype)
                        A[RefDof] = InpData
                        Ux = np.array(A[::3],order='C')
                        Uy = np.array(A[1::3],order='C')
                        Uz = np.array(A[2::3],order='C')
                        pointData[Var] = (Ux[SlcNodeIds], Uy[SlcNodeIds], Uz[SlcNodeIds])
            
            VTKFile = VTKPath + ModelName + '_' + str(i)
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
            
            VTKFile = VTKPath + ModelName + '_' + str(i)
            unstructuredGridToVTK(VTKFile, x, y, z, connectivity = PolysFlat, offsets = PolysOffset[:,1]+1, cell_types = ctype, cellData = {}, pointData = pointData)
            
            
    
elif Mode=='Full':
    #Defining vertices
    x = np.array(Nodes[:,0],order='C')
    y = np.array(Nodes[:,1],order='C')
    z = np.array(Nodes[:,2],order='C')
    
    #Defining cell types
    ctype   = VtkPolygon.tid*np.ones(GlobNFaces)        

    RefVars = ['U', 'D', 'GS', 'GE', 'PS', 'PE']
    if not len(RefNodeId) == GlobNNode: raise Exception
    for i in range(NFiles):
        if i%N_Workers==Rank:
            pointData = {}
            
            for Var in RefVars:
                if Var in ExportVars:
                    if Var in ['D', 'GS', 'GE', 'PS', 'PE']:
                        if Var in ['GS', 'GE']:      VarComps = ['-XX', '-YY', '-ZZ', '-YZ', '-XZ', '-XY']
                        elif Var in ['PS', 'PE']:    VarComps = ['-1', '-2', '-3']
                        elif Var in ['D']:           VarComps = ['']
                        NComp = len(VarComps)
                        InpData = readMPIBinFile(ResVecDataPath + Var + '_' + str(i)).reshape([GlobNNode, NComp]).T
                        A = np.zeros(GlobNNode, dtype=InpData.dtype)
                        for k in range(NComp):
                            VarComp = Var+VarComps[k]
                            A[RefNodeId] = InpData[k,:]
                            pointData[VarComp] = np.array(A,order='C')
                    
                    elif Var == 'U':
                        InpData = readMPIBinFile(ResVecDataPath + Var + '_' + str(i))
                        A = np.zeros(GlobNDof, dtype=InpData.dtype)
                        A[RefDof] = InpData
                        Ux = np.array(A[::3],order='C')
                        Uy = np.array(A[1::3],order='C')
                        Uz = np.array(A[2::3],order='C')
                        pointData[Var] = (Ux, Uy, Uz)
            
            VTKFile = VTKPath + ModelName + '_' + str(i)
            unstructuredGridToVTK(VTKFile, x, y, z, connectivity = FacesFlat, offsets = FacesOffset[:,1]+1, cell_types = ctype, cellData = {}, pointData = pointData)

if Rank==0: 
    print('>success!')
    print(f">total runtime: {np.round(time()-t1_,2)} sec")
    