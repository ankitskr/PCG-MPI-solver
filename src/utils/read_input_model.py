# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import os, pickle, zlib, sys, shutil
from file_operations import exportz
from time import time
import numpy as np
import scipy.io

t1_ = time()
    
#Reading input arguments
WorkDir         = sys.argv[1]
ModelName       = sys.argv[2]
ScratchPath     = sys.argv[3]
InputModelFile  = sys.argv[4]


print(">creating directories..")
#Checking directories
temp_folder = WorkDir + '/__pycache__'
MDF_Path = ScratchPath + '/ModelData/MDF/' #Model Definition Files Path
PyDataPath_Part = ScratchPath + '/ModelData/MPI/'

if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
if not os.path.exists(MDF_Path):
    os.makedirs(MDF_Path)
    
if not os.path.exists(PyDataPath_Part):
    os.makedirs(PyDataPath_Part)

print(f">extracting files from {ModelName}")
shutil.unpack_archive(InputModelFile, MDF_Path) #Importing Model Definition Files
    
ModelDataPaths         = {'ScratchPath':            ScratchPath,
                          'MDF_Path':               MDF_Path,
                          'PyDataPath_Part':        PyDataPath_Part,
                          'ModelName':              ModelName}

#Exporting Directories data
ModelDataPath_FileName = temp_folder + '/ModelDataPaths.zpkl'
exportz(ModelDataPath_FileName, ModelDataPaths)

GlobNFile           = MDF_Path + 'GlobN.mat'
GlobN               = scipy.io.loadmat(GlobNFile)['Data'][0]
GlobNElem           = int(GlobN[0])
GlobNDof            = int(GlobN[1])
GlobNNode           = int(GlobN[1]/3)


msg = f"""
>elements:  {GlobNElem}
>nodes:     {GlobNNode} 
>dofs:      {GlobNDof}
"""
print(">success!")
print(msg)
print(f">total runtime: {np.round(time()-t1_,2)} sec")
