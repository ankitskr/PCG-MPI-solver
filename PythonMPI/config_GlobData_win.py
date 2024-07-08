# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import numpy as np
import scipy.io
import sys
import pickle
import zlib

    
if __name__ == "__main__":
    
    ScratchPath     = sys.argv[1]
    MatDataPath     = ScratchPath + 'ModelData/Mat/'
    MeshData_Glob_FileName = MatDataPath + 'MeshData_Glob.zpkl'
    
    GlobNFile           = MatDataPath + 'GlobN.mat'
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

    