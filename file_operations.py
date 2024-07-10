# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:10:24 2020

@author: z5166762
"""


import numpy as np
from numpy.linalg import norm
import mpi4py
from mpi4py import MPI
import pickle, zlib
import glob
import shutil
from scipy.io import savemat
from time import time, sleep


def getIndices(A,B, CheckIntersection = False):
    
    if CheckIntersection:
        
        if not len(np.intersect1d(A,B)) == len(B):   raise Exception
    
    A_orig_indices = A.argsort()
    I = A_orig_indices[np.searchsorted(A[A_orig_indices], B)]

    return I


def exportz(FileName, Data):
    
    ExportData = zlib.compress(pickle.dumps(Data, pickle.HIGHEST_PROTOCOL))
    f = open(FileName, 'wb')
    f.write(ExportData)
    f.close()

def importz(FileName):
    
    Data = pickle.loads(zlib.decompress(open(FileName, 'rb').read()))
    return Data


def splitSerialData(Data, N_Splits, FileName=None):
    
    N = len(Data)
    Nj = int(N/N_Splits)
    
    DataChunkList = []
    for j in range(N_Splits):
        if j==0:
            N1 = 0; N2 = Nj;
        elif j == N_Splits-1:
            N1 = N2; N2 = N;
        else:
            N1 = N2; N2 = (j+1)*Nj;
        
        DataChunk_i = Data[N1:N2]
        DataChunkList.append(DataChunk_i)
        
        if FileName:
            FileName_i = FileName + str(j) + '.zpkl'
            f = open(FileName_i, 'wb')
            f.write(DataChunk_i)
            f.close()
    
    return DataChunkList
    


def configTimeRecData(MPList_TimeRecData, ExportLoadUnbalanceData=False, Dmg=False):
    
    
    MPList_dT_FileRead     = np.array([MP_TimeRecData['dT_FileRead'] for MP_TimeRecData in MPList_TimeRecData])
    MPList_dT_Calc         = np.array([MP_TimeRecData['dT_Calc'] for MP_TimeRecData in MPList_TimeRecData])
    MPList_dT_CommWait     = np.array([MP_TimeRecData['dT_CommWait'] for MP_TimeRecData in MPList_TimeRecData])
    MPList_dT_Total_Verify     = np.array([MP_TimeRecData['dT_Total_Verify'] for MP_TimeRecData in MPList_TimeRecData])
    MPList_t0_Start     = np.array([MP_TimeRecData['t0_Start'] for MP_TimeRecData in MPList_TimeRecData])
    MPList_t0_End         = np.array([MP_TimeRecData['t0_End'] for MP_TimeRecData in MPList_TimeRecData])
    
    if Dmg==True:
        MPList_dT_Elast         = np.array([MP_TimeRecData['dT_Elast'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_dT_Dmg           = np.array([MP_TimeRecData['dT_Dmg'] for MP_TimeRecData in MPList_TimeRecData])
    
    
    if 'TimeStepCountList' in MPList_TimeRecData[0].keys():
        MPList_dT_CalcList         = np.array([MP_TimeRecData['dT_CalcList'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_dT_CommWaitList     = np.array([MP_TimeRecData['dT_CommWaitList'] for MP_TimeRecData in MPList_TimeRecData])
        
        dT_CalcList                = np.mean(MPList_dT_CalcList, axis=0)
        dT_CommWaitList            = np.mean(MPList_dT_CommWaitList, axis=0)
        TimeStepCountList          = np.array(MPList_TimeRecData[0]['TimeStepCountList'])
        dT_CPUTimeList             = [TimeStepCountList, dT_CalcList, dT_CommWaitList]
        
    
    
    
    N_Workers = len(MPList_dT_FileRead)

    ts0 = np.min(MPList_t0_Start)
    te1 = np.max(MPList_t0_End)      
    TotalTime = te1 - ts0
    
    MPList_dT_Wait0 = MPList_t0_Start - ts0
    MPList_dT_Wait1 = te1 - MPList_t0_End
    MPList_dT_Wait = MPList_dT_Wait0 + MPList_dT_Wait1
    
    MPList_dT_CommWait += MPList_dT_Wait
    
    if Dmg==True:
        MPList_dT_Elast += MPList_dT_Wait
    
    
    MaxCommWaitTime = np.max(MPList_dT_CommWait)
    MinCommWaitTime = np.min(MPList_dT_CommWait)
    
    if ExportLoadUnbalanceData:
        MPList_NCount    = np.array([MP_TimeRecData['MP_NCount'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_NDOF     = np.array([MP_TimeRecData['MP_NDOF'] for MP_TimeRecData in MPList_TimeRecData])
        MPList_N_NbrDof = np.array([MP_TimeRecData['N_NbrDof'] for MP_TimeRecData in MPList_TimeRecData])
        LoadUnbalanceData = [MPList_NCount, MPList_N_NbrDof, MPList_dT_Calc, MPList_dT_CommWait]
        
        #MPList_dT_CalcTest         = np.array([MP_TimeRecDataTest['dT_CalcTest'] for MP_TimeRecDataTest in MPList_TimeRecDataTest])
        #MPList_dT_CommWaitTest     = np.array([MP_TimeRecDataTest['dT_CommWaitTest'] for MP_TimeRecDataTest in MPList_TimeRecDataTest])
        #I = np.argsort(MPList_NDOF)
        #LoadUnbalanceData = [MPList_NDOF[I], MPList_dT_Calc[I], MPList_dT_CommWait[I], MPList_N_NbrDof[I], MPList_dT_CalcTest[I], MPList_dT_CommWaitTest[I]]
        #LoadUnbalanceData = [MPList_NDOF, MPList_dT_Calc, MPList_dT_CommWait, MPList_N_NbrDof, MPList_dT_CalcTest, MPList_dT_CommWaitTest]
    
    else:
        LoadUnbalanceData = []
    
    MPList_TotalTime = np.zeros(N_Workers)
    for i in range(N_Workers):            
        MPList_TotalTime[i] = MPList_dT_Calc[i] + MPList_dT_CommWait[i]
    
    Mean_FileReadTime     = np.mean(MPList_dT_FileRead)
    Mean_CalcTime         = np.mean(MPList_dT_Calc)
    Mean_CommWaitTime     = np.mean(MPList_dT_CommWait)
    TotalTime_Analysis     = Mean_CalcTime + Mean_CommWaitTime
    
    Mean_TotalTime         = np.mean(MPList_dT_Total_Verify)
    Max_TotalTime         = np.max(MPList_dT_Total_Verify)
    
    #Saving Data
    TimeData = {'TotalTime'                : TotalTime, 
                'MaxCommWaitTime'        : MaxCommWaitTime, 
                'MinCommWaitTime'        : MinCommWaitTime, 
                'Mean_FileReadTime'        : Mean_FileReadTime,
                'Mean_CalcTime'            : Mean_CalcTime,
                'Mean_CommWaitTime'        : Mean_CommWaitTime,
                'Max_TotalTime_i'          : Max_TotalTime,
                'LoadUnbalanceData'            :LoadUnbalanceData}
    
    if 'TimeStepCountList' in MPList_TimeRecData[0].keys():
        TimeData['dT_CPUTimeList'] =  dT_CPUTimeList 
    
    if Dmg==True:
        Mean_ElastTime          = np.mean(MPList_dT_Elast)
        Mean_DmgTime            = np.mean(MPList_dT_Dmg)
        TimeData['Mean_ElastTime']   = Mean_ElastTime
        TimeData['Mean_DmgTime']     = Mean_DmgTime
    
    """
    print('\n')
    print('-----------------')
    print('Verify Total Time:', norm(TotalTime - MPList_TotalTime)/N_Workers, np.abs(TotalTime-TotalTime_Analysis))
    print('Total Time:', TotalTime, ['Mean', Mean_TotalTime], ['Max_i', Max_TotalTime])
    print('\n')
    """
                
    return TimeData



    
def GaussIntegrationTable(N_Points):
    
    if N_Points == 1:
        
        ni = [0.0]
        wi = [2.0]
        
    elif N_Points == 2:
        
#        ni = [-0.577350269189626,   0.577350269189626]
#        wi = [ 1.0,                 1.0]
        
        ni = [-1.0/3**0.5,          1.0/3**0.5]
        wi = [ 1.0,                 1.0]
        
        
    elif N_Points == 3:
        
#        ni = [-0.774596669241483,   0.0,                    0.774596669241483]
#        wi = [ 0.555555555555556,   0.888888888888889,      0.555555555555556]
        
        ni = [-(3.0/5.0)**0.5, 0.0, (3.0/5.0)**0.5]
        wi = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        
        
    elif N_Points == 4:
        
#        ni = [-0.861136311594053,  -0.339981043584856,      0.339981043584856,      0.861136311594053]
#        wi = [ 0.347854845137454,   0.652145154862546,      0.652145154862546,      0.347854845137454]
        
        ni = [-(3.0/7.0 + (2.0/7.0)*(6.0/5.0)**0.5)**0.5, -(3.0/7.0 - (2.0/7.0)*(6.0/5.0)**0.5)**0.5, (3.0/7.0 - (2.0/7.0)*(6.0/5.0)**0.5)**0.5, (3.0/7.0 + (2.0/7.0)*(6.0/5.0)**0.5)**0.5]
        wi = [(18-30**0.5)/36.0, (18+30**0.5)/36.0, (18+30**0.5)/36.0, (18-30**0.5)/36.0]
        
        
        
    ni = np.array(ni, dtype = float)
    wi = np.array(wi, dtype = float)
    
    return ni, wi
    
    
    


def GaussLobattoIntegrationTable(N_Points):
    
    if N_Points == 2:
        
        ni = [-1.0,                 1.0]
        wi = [ 1.0,                 1.0]
        
    elif N_Points == 3:
        
        ni = [-1.0,                 0.0,                    1.0]
        wi = [ 1.0/3,               4.0/3,                  1.0/3]
        
    elif N_Points == 4:
        
        ni = [-1.0,                -1.0/5**0.5,             1.0/5**0.5,               1.0]
        wi = [ 1.0/6,               5.0/6,                  5.0/6,                    1.0/6]
    
    elif N_Points == 5:
        
        ni = [-1.0,                -(3.0/7)**0.5,           0.0,                     (3.0/7)**0.5,           1.0]
        wi = [ 1.0/10,              49.0/90,                32.0/45,                  49.0/90,               1.0/10]
    
    
    ni = np.array(ni, dtype = float)
    wi = np.array(wi, dtype = float)
    
    return ni, wi
    


def getPrincipalStrain(ElemList_LocStrainVec):
    return getPrincipalStress(ElemList_LocStrainVec)




def getPrincipalStress(ElemList_LocStressVec):

    s11 = ElemList_LocStressVec[0, :]
    s22 = ElemList_LocStressVec[1, :]
    s33 = ElemList_LocStressVec[2, :]
    s23 = ElemList_LocStressVec[3, :]
    s13 = ElemList_LocStressVec[4, :]
    s12 = ElemList_LocStressVec[5, :]
    
    s12_2 = s12*s12
    s23_2 = s23*s23
    s13_2 = s13*s13
    
    I1 = s11 + s22 + s33
    I2 = s11*s22 + s22*s33 + s33*s11 - s12_2 - s23_2 - s13_2;
    I3 = s11*s22*s33 - s11*s23_2 - s22*s13_2 - s33*s12_2 + 2*s12*s23*s13
    
    I1_2 = I1*I1
    I1_3 = I1_2*I1
    
    J2 = I1_2 - 3*I2 + 1e-24*np.max(np.abs(ElemList_LocStressVec))
    a1 = 1/3
    
    PhiBracket = 0.5*(2*I1_3 - 9*I1*I2 + 27*I3)/np.power(J2,1.5)
    PhiBracket[PhiBracket>1.0] = 1.0
    PhiBracket[PhiBracket<-1.0] = -1.0
    Phi    = a1*(np.arccos(PhiBracket));
    
    P_tmp = np.zeros([3, len(I1)])
    f_tmp = 2*a1*np.sqrt(J2)
    P_tmp[0,:] = a1*I1 + f_tmp*np.cos(Phi)
    P_tmp[1,:] = a1*I1 + f_tmp*np.cos(Phi + 2*a1*np.pi)
    P_tmp[2,:] = a1*I1 + f_tmp*np.cos(Phi + 4*a1*np.pi)
    P_tmp = np.real(P_tmp)
    
    #Principal Stresses
    PSig1 = np.max(P_tmp,0)
    PSig3 = np.min(P_tmp,0)
    PSig2 = I1 - PSig1 - PSig3
    
    P_tmp[0,:] = PSig1  
    P_tmp[1,:] = PSig2
    P_tmp[2,:] = PSig3
    
    return P_tmp
    



def loadBinDataInSharedMem(Data_FileName, BinDataType, PyDataType, DataShape, Comm, LoadingRank=0, Wait=True):

    SharedComm = Comm.Split_type(MPI.COMM_TYPE_SHARED)
    LocalRank = SharedComm.Get_rank() #local rank on a compute-node
    
    if PyDataType == int:   ItemSize = MPI.LONG.Get_size()
    elif PyDataType == float:   ItemSize = MPI.DOUBLE.Get_size()
    elif PyDataType == bool:   ItemSize = MPI.BOOL.Get_size()
    else:   raise NotImplementedError
    
    #Loading Mesh-partitioning file into the LoadingRank
    if LocalRank == LoadingRank:     NBytes = np.prod(DataShape)*ItemSize
    else:                            NBytes = 0
    
    Win = MPI.Win.Allocate_shared(NBytes, ItemSize, comm=SharedComm)

    Buf, ItemSize = Win.Shared_query(LoadingRank)
    if PyDataType == int:       assert ItemSize == MPI.LONG.Get_size()
    elif PyDataType == float:   assert ItemSize == MPI.DOUBLE.Get_size()
    elif PyDataType == bool:   assert ItemSize == MPI.BOOL.Get_size()
    
    RefData = np.ndarray(buffer=Buf, dtype=PyDataType, shape=DataShape)
    
    if LocalRank == LoadingRank:
        BinData = np.fromfile(Data_FileName, dtype=BinDataType).astype(PyDataType)
        if len(BinData)>0:
            if len(DataShape) == 1:
                RefData[:] = BinData
            elif len(DataShape) == 2:
                RefData[:,:] = BinData.reshape(DataShape,order='F')
    
    if Wait:    Comm.barrier()
        
    return RefData


def zip_files(zip_filename, dir_name):
    shutil.make_archive(zip_filename, 'zip', dir_name)




def writeMPIFile_parallel(Data_FileName, Data_Buffer, Comm):
    
    Rank = Comm.Get_rank()
    N_Workers = Comm.Get_size()
    
    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
    fh = MPI.File.Open(Comm, Data_FileName+'.mpidat', amode)

    Mem = Data_Buffer.nbytes
    Nf = len(Data_Buffer)
    Dtype = Data_Buffer.dtype
    MetaDataList = Comm.gather([Mem, Nf, Dtype],root=0)
    if Rank == 0:
        MetaDataList = np.array(MetaDataList, dtype=object)
        OffsetData = np.cumsum(np.hstack([[0],MetaDataList[:-1,0]]))
        metadat = np.array({'NfData': MetaDataList[:,1],
                           'DTypeData': MetaDataList[:,2],
                           'OffsetData': OffsetData}, dtype=object)
        np.save(Data_FileName+'_metadat', metadat)
        #TODO: save metadat_bytes at Offset=0. Then, compute the size of metadat_bytes (=metadat_bytes_size).
        #Then, bcast to all workers. Finally, do: fh.Write_at(metadat_bytes_size+Offset, Data_Buffer)
    else:
        OffsetData = None
    Offset = Comm.scatter(OffsetData, root=0)
    Comm.barrier()
    
    fh.Write_at(Offset, Data_Buffer)
    fh.Close()
    


def readMPIFile_parallel(Data_FileName, Comm):
    
    Rank = Comm.Get_rank()
    
    metadat = np.load(Data_FileName+'_metadat.npy', allow_pickle=True).item()
    Nf = metadat['NfData'][Rank]
    DType = metadat['DTypeData'][Rank]
    Offset = metadat['OffsetData'][Rank]
    
    #Reading Buffer
    amode = MPI.MODE_RDONLY
    fh = MPI.File.Open(Comm, Data_FileName+'.mpidat', amode)
    Data_Buffer = np.empty(Nf, dtype=DType)
    fh.Read_at(Offset, Data_Buffer)
    fh.Close()
    
    return Data_Buffer


"""
class Film:

    title = ""
    direction = ""
    music = ""
    caste = ""
    year  = -1
    video = None

    def __init__(self, title, direction, music, cast, video):
        self.title      = title
        self.direction  = direction
        self.music      = music
        self.cast       = None
        self.video      = video  

    def identify(self):
        print("Movie title:%s"%(self.title))
        print("Directed by:%s"%(self.direction))
        print("Music:%s"%(self.music))
        print("Cast:%s"%(self.cast))
        print("Video Buffer:%s"%(self.video))

import pickle
import zlib
Comm = MPI.COMM_WORLD

film = Film("Sound Of Music",
            "Robert Wise",
            ["Julie Andrews", "Christopher Plummer"],
            1965,
            Comm.Get_rank())

"""

"""            
Data_Buffer = np.frombuffer(zlib.compress(pickle.dumps(film, pickle.HIGHEST_PROTOCOL)), 'b')

writeMPIFile_parallel('testmpi', Data_Buffer, Comm)

A = readMPIFile_parallel('testmpi', Comm)

B = pickle.loads(zlib.decompress(A.tobytes()))
print(B.title,',', B.direction,',', B.music,',', B.cast,',', B.video)
"""


"""
Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()

A = []
for i in range(2):
    a = [Rank, i, 'a']
    A.append(a)

A_List = Comm.gather(A,root=0)

if Rank==0:
    print(Rank, type(np.vstack(A_List)))

[[[0, 0, 'a'], [0, 1, 'a']], 
 [[1, 0, 'a'], [1, 1, 'a']], 
 [[2, 0, 'a'], [2, 1, 'a']], 
 [[3, 0, 'a'], [3, 1, 'a']]]
"""

"""
InpData = np.array([np.load('testmpi_metadat.npy', allow_pickle=True).item(), film])

Data_Buffer = np.frombuffer(zlib.compress(pickle.dumps(InpData, pickle.HIGHEST_PROTOCOL)), 'b')

writeMPIFile_parallel('testmpi', Data_Buffer, Comm)

A = readMPIFile_parallel('testmpi', Comm)

B = pickle.loads(zlib.decompress(A.tobytes()))


print(Rank, B[0],  B[1].music,',', B[1].cast,',', B[1].video)
"""

"""
amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
fh = MPI.File.Open(Comm, 'TestMPI.mpidat', amode)
fh.Write(Data_Buffer)
fh.Close()

amode = MPI.MODE_RDONLY
fh = MPI.File.Open(Comm, 'TestMPI.mpidat', amode)
Data_Buffer0 = memoryview(np.str_(""))
print(Data_Buffer0.format)
fh.Read(Data_Buffer0)
print(Data_Buffer0)
fh.Close()
"""



def readMPIFile(Data_FileName, MetaData = None):
    
    if MetaData == None:
        metadat = np.load(Data_FileName+'_metadat.npy', allow_pickle=True).item()
        Nf = np.sum(metadat['NfData'])
        DType = metadat['DTypeData'][0]
    else:
        Nf = MetaData[0]
        DType = MetaData[1]
    
    fh = MPI.File.Open(MPI.COMM_SELF, Data_FileName+'.mpidat', MPI.MODE_RDONLY)
    Data_Buffer = np.empty(Nf, dtype=DType)
    fh.Read(Data_Buffer)
    fh.Close()
    
    return Data_Buffer



def readMPIBinFile(Data_FileName, MetaData = None):
    
    if MetaData == None:
        metadat = np.load(Data_FileName+'_metadat.npy', allow_pickle=True).item()
        Nf = np.sum(metadat['NfData'])
        DType = metadat['DTypeData'][0]
    else:
        Nf = MetaData[0]
        DType = MetaData[1]
    
    fh = open(Data_FileName+'.mpidat', "rb")
    Data_Buffer = np.fromfile(fh,dtype=DType)
    fh.close()
    
    return Data_Buffer










def calcCrackTipVelocity_TensileBranching(ScratchPath, Run):
    
    MatDataPath             = ScratchPath + 'ModelData/Mat/'
    MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob           = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
    GlobNNode               = MeshData_Glob['GlobNNode']
    
    print('A')
    MatDataPath         = ScratchPath + 'ModelData/Mat/'
    Nodes_FileName      = MatDataPath + 'nodes.bin'
    Nodes               = np.fromfile(Nodes_FileName, dtype=np.float64).astype(float).reshape((GlobNNode,3),order='F')
    NNodes = np.shape(Nodes)[0]
    
    ResVecDataPath = ScratchPath + 'Results_Run' + str(Run) + '/ResVecData/'
    ResNodeId = readMPIFile(ResVecDataPath+'NodeId')
    Time_T = np.load(ResVecDataPath+'Time_T.npy')
    NSteps = len(Time_T)
    NSteps-= 10
    
    print('B')
    
    DmgNodeCoord = np.zeros([NSteps,2])
    ArrInd_2d = np.array([0,1])
    dR = 0.5e-3
    for i in range(NSteps):
        if i%100==0:    print(np.round(i*100/NSteps))
        InpData = readMPIFile(ResVecDataPath + 'D_' + str(i))
        D = np.zeros(GlobNNode, dtype=InpData.dtype)
        D[ResNodeId] = InpData
        
        RefNodes = Nodes[np.logical_and(D>=0.9, Nodes[:,1]<0.02),:]
        if len(RefNodes)>0:
            I = np.argmax(RefNodes[:,0])
            DmgNodeCoord[i,:] = RefNodes[I, ArrInd_2d];
    
    print('C')
    
    #Applying smoothing
    DmgNodeCoord_Smooth = np.zeros([NSteps,2]);
    so = 25
    for q in range(so,NSteps-so):
        DmgNodeCoord_Smooth[q,:] = np.mean(DmgNodeCoord[q-so:q+so+1,:],0)
    DmgNodeCoord = DmgNodeCoord_Smooth;

    DmgNodeCoord_Smooth = np.zeros([NSteps,2]);
    for q in range(so,NSteps-so):
        DmgNodeCoord_Smooth[q,:] = np.mean(DmgNodeCoord[q-so:q+so+1,:],0)
    DmgNodeCoord = DmgNodeCoord_Smooth;
    
    print('D')
    
    #Ref: Jian-Ying Wu et al. 2019. Phase-field modelling of fracture.
    CrkLen = np.zeros(NSteps)
    CTVel = np.zeros(NSteps)
    for q in range(1,NSteps):
        dCrkLen = norm(DmgNodeCoord[q,:]-DmgNodeCoord[q-1,:])
        CrkLen[q] = CrkLen[q-1] + dCrkLen
    
    for q in range(1,NSteps-1):
        Coeff = np.polyfit(Time_T[q-1:q+2],CrkLen[q-1:q+2], 1);
        CTVel[q] = Coeff[0]
    
    print('E')
    
    ResultPath = ScratchPath + 'Results_Run' + str(Run) + '/';
    np.save(ResultPath+'CrackTipVelData', np.array([CTVel, DmgNodeCoord, CrkLen, Time_T], dtype=object))



def calcCrackTipVelocity_Shear(ScratchPath, Run):
    
    MatDataPath             = ScratchPath + 'ModelData/Mat/'
    MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob           = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
    GlobNNode               = MeshData_Glob['GlobNNode']
    
    print('A')
    MatDataPath         = ScratchPath + 'ModelData/Mat/'
    Nodes_FileName      = MatDataPath + 'nodes.bin'
    Nodes               = np.fromfile(Nodes_FileName, dtype=np.float64).astype(float).reshape((GlobNNode,3),order='F')
    NNodes = np.shape(Nodes)[0]
    
    ResVecDataPath = ScratchPath + 'Results_Run' + str(Run) + '/ResVecData/'
    ResNodeId = readMPIFile(ResVecDataPath+'NodeId')
    Time_T = np.load(ResVecDataPath+'Time_T.npy')
    NSteps = len(Time_T)
    
    NSteps-= 10
    
    print('B')
    
    DmgNodeCoord = np.zeros([NSteps,2])
    ArrInd_2d = np.array([0,1])
    for i in range(NSteps):
        if i%100==0:    print(np.round(i*100/NSteps))
        InpData = readMPIFile(ResVecDataPath + 'D_' + str(i))
        D = np.zeros(GlobNNode, dtype=InpData.dtype)
        D[ResNodeId] = InpData
        
        RefNodes = Nodes[np.logical_and(D>=0.9, Nodes[:,1]>0.025),:]
        if len(RefNodes)>0:
            I = np.argmax(RefNodes[:,1])
            DmgNodeCoord[i,:] = RefNodes[I, ArrInd_2d];
    
    print('C')
    
    #Applying smoothing
    DmgNodeCoord_Smooth = np.zeros([NSteps,2]);
    so = 25
    for q in range(so,NSteps-so):
        DmgNodeCoord_Smooth[q,:] = np.mean(DmgNodeCoord[q-so:q+so+1,:],0)
    DmgNodeCoord = DmgNodeCoord_Smooth;
    
    DmgNodeCoord_Smooth = np.zeros([NSteps,2]);
    for q in range(so,NSteps-so):
        DmgNodeCoord_Smooth[q,:] = np.mean(DmgNodeCoord[q-so:q+so+1,:],0)
    DmgNodeCoord = DmgNodeCoord_Smooth;
    
    print('D')
    
    #Ref: Jian-Ying Wu et al. 2019. Phase-field modelling of fracture.
    CrkLen = np.zeros(NSteps)
    CTVel = np.zeros(NSteps)
    for q in range(1,NSteps):
        dCrkLen = norm(DmgNodeCoord[q,:]-DmgNodeCoord[q-1,:])
        CrkLen[q] = CrkLen[q-1] + dCrkLen
    
    for q in range(1,NSteps-1):
        Coeff = np.polyfit(Time_T[q-1:q+2],CrkLen[q-1:q+2], 1);
        CTVel[q] = Coeff[0]
    
    print('E')
    
    ResultPath = ScratchPath + 'Results_Run' + str(Run) + '/';
    np.save(ResultPath+'CrackTipVelData', np.array([CTVel, DmgNodeCoord, CrkLen, Time_T], dtype=object))




def calcCrackTipCoord_CrkArrest(ScratchPath, Run):
    
    MatDataPath             = ScratchPath + 'ModelData/Mat/'
    MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob           = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
    GlobNNode               = MeshData_Glob['GlobNNode']
    
    print('A')
    MatDataPath         = ScratchPath + 'ModelData/Mat/'
    Nodes_FileName      = MatDataPath + 'nodes.bin'
    Nodes               = np.fromfile(Nodes_FileName, dtype=np.float64).astype(float).reshape((GlobNNode,3),order='F')
    NNodes = np.shape(Nodes)[0]
    
    ResVecDataPath = ScratchPath + 'Results_Run' + str(Run) + '/ResVecData/'
    ResNodeId = readMPIFile(ResVecDataPath+'NodeId')
    Time_T = np.load(ResVecDataPath+'Time_T.npy')
    NSteps = len(Time_T)
    NSteps-= 1
    
    print('B')
    
    DmgNodeCoord = np.zeros([NSteps,2])
    ArrInd_2d = np.array([0,1])
    for i in range(NSteps):
        print(np.round(i*100/NSteps))
        InpData = readMPIFile(ResVecDataPath + 'D_' + str(i))
        D = np.zeros(GlobNNode, dtype=InpData.dtype)
        D[ResNodeId] = InpData
        
        RefNodes = Nodes[np.logical_and(D>=0.9, Nodes[:,1]<0.040),:]
        if len(RefNodes)>0:
            I = np.argmax(RefNodes[:,0])
            DmgNodeCoord[i,:] = RefNodes[I, ArrInd_2d];
    
    print('C')
    
    
    ResultPath = ScratchPath + 'Results_Run' + str(Run) + '/';
    np.save(ResultPath+'CrackTipCoordData', np.array([DmgNodeCoord, Time_T], dtype=object))








def getTimeHistoryData(ScratchPath, Run):
    
    MatDataPath             = ScratchPath + 'ModelData/Mat/'
    MeshData_Glob_FileName  = MatDataPath + 'MeshData_Glob.zpkl'
    MeshData_Glob           = pickle.loads(zlib.decompress(open(MeshData_Glob_FileName, 'rb').read()))
    GlobNNode               = MeshData_Glob['GlobNNode']
    GlobNDof                = MeshData_Glob['GlobNDof']

    MatDataPath         = ScratchPath + 'ModelData/Mat/'
    Nodes_FileName      = MatDataPath + 'nodes.bin'
    Nodes               = np.fromfile(Nodes_FileName, dtype=np.float64).astype(float).reshape((GlobNNode,3),order='F')
    NNodes = np.shape(Nodes)[0]
    
    ResVecDataPath = ScratchPath + 'Results_Run' + str(Run) + '/ResVecData/'
    ResNodeId = readMPIFile(ResVecDataPath+'NodeId')
    ResDof = readMPIFile(ResVecDataPath+'Dof')

    RefCoordList        =np.array([[0.000, 0.0225, 0.0225],
                                   [0.009, 0.0225, 0.0225],
                                   [0.018, 0.0225, 0.0225],
                                   [0.027, 0.0225, 0.0225],
                                   [0.036, 0.0225, 0.0225],
                                   [0.045, 0.0225, 0.0225]], dtype=float)
    Tol = 1e-12
    NCoords = len(RefCoordList)
    RefNodeIdList = []
    for i in range(NCoords):
        RefNodeId = np.arange(GlobNNode)[(np.abs(Nodes[:,0]-RefCoordList[i,0])<Tol)*(np.abs(Nodes[:,1]-RefCoordList[i,1])<Tol)*(np.abs(Nodes[:,2]-RefCoordList[i,2])<Tol)]
        RefNodeIdList.append(RefNodeId[0])
    
    if not len(RefNodeIdList) == NCoords:   raise Exception
    RefNodeIdList = np.array(RefNodeIdList)
    print(Nodes[RefNodeIdList,:])
    
    Time_T = np.load(ResVecDataPath+'Time_T.npy')
    NSteps = len(Time_T)
    TimeHistoryData = { 'T':Time_T,
                        'U':[],
                        'PS1': []}
    for i in range(NSteps):
        print(np.round(i*100/NSteps))
        
        #Displacement
        InpData         = readMPIFile(ResVecDataPath + 'U_' + str(i))
        A = np.zeros(GlobNDof, dtype=InpData.dtype)
        A[ResDof] = InpData
        Ux = A[::3][RefNodeIdList]
        TimeHistoryData['U'].append(Ux)
        
        #Stress
        InpData         = readMPIFile(ResVecDataPath + 'PS1_' + str(i))
        A = np.zeros(GlobNNode, dtype=InpData.dtype)
        A[ResNodeId] = InpData
        PS1 = A[RefNodeIdList]
        TimeHistoryData['PS1'].append(PS1)
        
    
    savemat(ScratchPath + 'Results_Run' + str(Run) + '/TimeHistoryData.mat', TimeHistoryData)

#getTimeHistoryData('/g/data/ud04/Ankit/DynDmg_CircIncl/', '11_Dmg')
            
            
    