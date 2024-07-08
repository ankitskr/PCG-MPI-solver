# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:33:01 2020

@author: z5166762
"""

from mpi4py import MPI
import numpy as np




def writeMPIFile_parallel(Data_FileName, Data_Buffer, Comm):
    
    Rank = Comm.Get_rank()

    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
    fh = MPI.File.Open(Comm, Data_FileName+'.mpidat', amode)

    Mem = Data_Buffer.nbytes
    Nf = len(Data_Buffer)
    MetaDataList = Comm.gather([Mem, Nf],root=0)
    if Rank == 0:
        MetaDataList = np.array(MetaDataList)
        OffsetData = np.cumsum(np.hstack([[0],MetaDataList[:-1,0]]))
        metadat = {'Nf': np.sum(MetaDataList[:,1]),
                   'DType': Data_Buffer.dtype}
        np.save(Data_FileName+'_metadat', metadat)
    else:
        OffsetData = None
    Offset = Comm.scatter(OffsetData, root=0)
    
    fh.Write_at(Offset, Data_Buffer)
    fh.Close()
    


def readMPIFile(Data_FileName, Comm):
    
    metadat = np.load(Data_FileName+'_metadat.npy', allow_pickle=True).item()
    Nf = metadat['Nf']
    DType = metadat['DType']
    
    amode = MPI.MODE_RDONLY
    fh = MPI.File.Open(Comm, Data_FileName+'.mpidat', amode)
    buffer = np.empty(Nf, dtype=DType)

    fh.Read(buffer)
    print(Rank, buffer)
    fh.Close()


Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()
Data_Buffer = np.ones((Rank+5),dtype=np.int64)*Rank
writeMPIFile_parallel("./datafile", Data_Buffer, Comm)
print(Rank, Data_Buffer)

readMPIFile("./datafile", Comm)