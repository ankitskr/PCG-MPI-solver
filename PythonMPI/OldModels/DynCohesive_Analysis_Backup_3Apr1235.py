# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:10:24 2020

@author: z5166762
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
gc.collect()
gc.set_threshold(5600, 20, 20)
#gc.disable()

import matplotlib.pyplot as plt
from datetime import datetime
import sys
from time import time, sleep
import numpy as np
import os.path
import shutil

import pickle
import zlib

import mpi4py
from mpi4py import MPI

#mpi4py.rc.threads = False

#import logging
#from os.path import abspath
#from Cython_Array.Array_cy import updateLocFint, apply_sum
from scipy.io import savemat
from GeneralFunc import configTimeRecData, GaussLobattoIntegrationTable, GaussIntegrationTable
from scipy.interpolate import interp1d, interp2d



def calcPenaltyContactForceVector(IntfcElem, CntStiffMat_List, getSecStiff, IntgPnts_GapVectorList):
                
    N_IntegrationPoints     = IntfcElem['N_IntegrationPoints']
    InitCoordVector         = IntfcElem['InitCoordVector']
    Elem_NDOF               = IntfcElem['NDOF']
    IntfcArea               = IntfcElem['IntfcArea']
    Knn                     = IntfcElem['Knn']
    
    LocalStiffnessMatrix    = np.zeros([Elem_NDOF, Elem_NDOF])
    Ni, wi                  = GaussIntegrationTable(N_IntegrationPoints)
    
    NodalGapVector          = InitCoordVector + IntfcElem_UnVector_1
                
    #RefSlidingDirection = self.SlidingDirectionList[self.RefGLPoint]
    #RefSlidingDirection = self.SlidingDirection
    
    #Numerically integrating to obtain stiffness matrices
    q=0
    HasContact = False
    for p0 in range(N_IntegrationPoints):
        for p1 in range(N_IntegrationPoints):
            
            GapVector =  IntgPnts_GapVectorList[q]

            #Calculating Contact Condition
            #ShearCOD0 = GapVector[0]
            #ShearCOD1 = GapVector[1]
            NormalGap = GapVector[2]
            
            if NormalGap > 0: #Free
                ContactCondition = 0 #Free
            else:
                #ShearStress = self.Ks*abs(ShearCOD)
                #NormalStress = self.Kn*abs(NormalGap)
                #if ShearStress < self.ContactCohesion + self.FrictionCoefficient*NormalStress:
                #    ContactCondition = 1 #Stick
                #else:   ContactCondition = 2 #Slip
                ContactCondition = 1 #Stick
                HasContact = True
                
            #Configuring Penalty stiffnesses
            if ContactCondition == 0: 
                knn = 0.0; #ks = 0.0; ksn = 0.0; kns = 0.0
                
            elif ContactCondition == 1: #Stick
                knn = Knn; #ks = Ks; ksn = 0.0; kns = 0.0
                
                #knn = Knn*getSecStiff(abs(NormalGap))
                    
            else: 
                raise Exception
            #elif ContactCondition == 2: #Slip
            #    kn_s = self.Kn; ks_s = 0.0; 
            #    ksn_s = RefSlidingDirection*self.Kn*self.FrictionCoefficient
            #    kns_s = 0.0
            
            #Calculating Stiffness Matrices
            if ContactCondition > 0:
                CntStiffMat = CntStiffMat_List[q]
                LocalStiffnessMatrix += IntfcArea*knn*CntStiffMat
                
            q+=1
            
    
    if HasContact:
    
        #Transforming to Global Coordinate System
        #T = TransformationMatrix
        #StiffnessMatrix = np.dot(np.dot(T.T, LocalStiffnessMatrix), T)
        
        StiffnessMatrix = LocalStiffnessMatrix
        CntFVec = np.dot(StiffnessMatrix, NodalGapVector)
    
    else:   CntFVec = np.zeros(Elem_NDOF)
    
        
    return CntFVec



def getElemContactTraction(knn_ij, Mij_List, Elem_DeformedCoord, RefIntfcElem_SecStiffList):
    N_IntegrationPoints = len(Mij_List)
    
    Elem_CntTractionList = []
    Elem_GapList = []
    for p in range(N_IntegrationPoints):
        
        IntgPnt_Gap = np.dot(Mij_List[p], Elem_DeformedCoord)
        SecStiff = RefIntfcElem_SecStiffList[p]
        
        IntgPnt_CntTraction = np.dot(SecStiff*knn_ij, IntgPnt_Gap)
        Elem_CntTractionList.append(IntgPnt_CntTraction)
        Elem_GapList.append(IntgPnt_Gap)
        
    Elem_CntTraction = np.mean(Elem_CntTractionList, axis=0)
    Elem_Gap = np.mean(Elem_GapList, axis=0)
    
    return [Elem_Gap, Elem_CntTraction]




def initPenaltyStiffnessMatrix(MP_SubDomainData, N_IntegrationPoints, dt):
    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId == -1: #Interface Elements
        
            ElemList_LocDofVector   = RefTypeGroup['ElemList_LocDofVector']
            IntfcElemList           = RefTypeGroup['ElemList_IntfcElem']
            RefIntfcElem            = IntfcElemList[0]
            Ni, wi                  = GaussIntegrationTable(N_IntegrationPoints)
            N_IntfcElem             = len(IntfcElemList)
            
            Knn                     = 1.0
            IntfcArea               = 1.0
            
            #Saving stiffness matrix
            knn_ij = np.array([[0,  0,   0],
                               [0,  0,   0],
                               [0,  0, Knn]], dtype=float)
                            
            CntStiffMat_List = []
            Mij_List = []
            for p0 in range(N_IntegrationPoints):
                for p1 in range(N_IntegrationPoints):
                        
                    s0 = Ni[p0]
                    s1 = Ni[p1]
                    
                    N1 = 0.25*(1-s0)*(1-s1)
                    N2 = 0.25*(1+s0)*(1-s1)
                    N3 = 0.25*(1+s0)*(1+s1)
                    N4 = 0.25*(1-s0)*(1+s1)
                    
                    I = np.eye(3)
                    N = np.hstack([N1*I, N2*I, N3*I, N4*I])
                    Mij = np.hstack([-N, N])
                    
                    Knn_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(np.dot(Mij.T, knn_ij), Mij)
                    CntStiffMat_List.append(Knn_ij)
                    Mij_List.append(Mij)
            
            """
            #Contact secant stiffness
            Knn = RefIntfcElem['Knn']
            wc = None
            g0 = 5.0e-8
            GapList0 = g0*np.array([-1e-16, 0.5, 1.0], dtype=float)
            GapList1 = g0*np.array([1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0, 100, 150, 200, 250, 300, 400, 600, 800, 1200, 1600, 2000, 4000, 6000, 10000, 50000, 100000, 1000000], dtype=float)
            CnP0 = (0.5*Knn/g0)*GapList0**2
            CnP1 = Knn*(GapList1-0.5*g0)
            GapList = np.hstack([GapList0,GapList1])
            CnP = np.hstack([CnP0,CnP1])
            SecStiff = CnP/GapList
            SecStiff[0] = 0.0
            SecStiffFunc = interp1d(GapList, SecStiff, bounds_error=True)
            """
            
            
            #Cohesive secant stiffness
            Knn = RefIntfcElem['Knn']
            ft = 1e0
            Gf = 1
            wc = 2*Gf/ft
            w0 = ft/Knn;
            
            """
            NRef = wc/w0;
            GapList0 = w0*np.array([-1e16, 0.5, 1.0], dtype=float)
            GapList1 = w0*np.geomspace(1.01,NRef,1000)
            CnP0 = Knn*GapList0
            CnP1 = (ft/(wc-w0))*(wc-GapList1)
            GapList = np.hstack([GapList0,GapList1])
            CnP = np.hstack([CnP0,CnP1])
            SecStiff = CnP/GapList
            SecStiffFunc = interp1d(GapList, SecStiff, bounds_error=False, fill_value = 0.0)
            """
            
            def getSecStiff(NormalGap, ft=ft, wc=wc, Knn=Knn, w0=w0):
                
                if NormalGap <= w0:
                    SecStiff = Knn
                elif w0 < NormalGap <= wc:
                    NormalTraction = (ft/(wc-w0))*(wc-NormalGap)
                    SecStiff = NormalTraction/NormalGap 
                elif NormalGap > wc:
                    SecStiff = 0.0
                    
                return SecStiff
            
            
            #Saving variables
            RefTypeGroup['CntStiffMat_List'] = CntStiffMat_List
            RefTypeGroup['Mij_List'] = Mij_List
            RefTypeGroup['getSecStiff'] = getSecStiff
            RefTypeGroup['N_IntegrationPoints_2D'] = N_IntegrationPoints**2
            RefTypeGroup['Elem_NDOF'] = RefIntfcElem['NDOF']
            RefTypeGroup['knn_ij'] = knn_ij
            RefTypeGroup['wc'] = wc
            RefTypeGroup['DamageList'] = np.ones(N_IntfcElem)*RefIntfcElem['Damage']
            
            
            break



def getPenaltyWeightList(IntfcElemList, ElemList_IntgPntGapVec, getSecStiff):
    
    N_IntfcElem = len(IntfcElemList)
    ElemList_PenaltyWeight = np.zeros(N_IntfcElem)
    #ElemList_SlipPenaltyWeight = np.zeros(N_IntfcElem)
    
    for i in range(N_IntfcElem):
        NormalGap           = ElemList_IntgPntGapVec[2, i]
        IntfcElemArea    = IntfcElemList[i]['IntfcArea']
    
        if NormalGap <= 0:
        
            #if Stick is satisfied
            ElemList_PenaltyWeight[i] = IntfcElemArea
            if getSecStiff:    ElemList_PenaltyWeight[i] *= getSecStiff(abs(NormalGap))
            
            #if Slip is satisfied
            #ElemList_SlipPenaltyWeight[i] = Slip_PenaltyWeight

        
    return ElemList_PenaltyWeight
    

def getDamageWeightList(IntfcElemList, DamageList, ElemList_IntgPntGapVec, getSecStiff, wc):
    
    N_IntfcElem = len(IntfcElemList)
    ElemList_StiffnessWeight = np.zeros(N_IntfcElem)
    
    for i in range(N_IntfcElem):
        NormalGap           = ElemList_IntgPntGapVec[2, i]
        IntfcElemArea    = IntfcElemList[i]['IntfcArea']
        
        
        # SecK = 1e3
        # ElemList_StiffnessWeight[i] = IntfcElemArea*SecK
        
        
        SecK = getSecStiff(NormalGap)
        ElemList_StiffnessWeight[i] = IntfcElemArea*SecK
        
        # Damage = np.min([1.0,NormalGap/wc])
        
        # if DamageList[i]<Damage: #Loading (Increasing Damage)
            # DamageList[i]=Damage
            # ElemList_StiffnessWeight[i] = IntfcElemArea*SecK
        
        # else: #Loading/Unloading
            # if DamageList[i] == 1: #To handle contact-impact problem
                # ElemList_StiffnessWeight[i] = 0.0
            # else:
                # ElemList_StiffnessWeight[i] = (1.0-DamageList[i])*IntfcElemArea*getSecStiff(0) #<-- rectify for regularised stiffness
                # raise Exception
                    
            
            
    return ElemList_StiffnessWeight
    
        

def calcMPFint(MP_Un_1, MP_DeformedCoord_1, FintCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, Flat_ElemLocDof, NCount, MP_NbrMPIdVector, MP_TimeRecData):
    
    #Calculating Local Fint Vector for Octree cells
    MP_NDOF = len(MP_Un_1)    
    MP_LocFintVec = np.zeros(MP_NDOF, dtype=float)    
    MP_LocCntFintVec = np.zeros(MP_NDOF, dtype=float)    
    if FintCalcMode == 'outbin':
        Flat_ElemFint = np.zeros(NCount, dtype=float)
        I=0     
        
        
    MP_ElemCntStressDataList = []
    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        ElemList_LocDofVector = RefTypeGroup['ElemList_LocDofVector']
        ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
            
        if ElemTypeId == -1: #Interface Elements
        
            IntfcElemList = RefTypeGroup['ElemList_IntfcElem']
            CntStiffMat_List = RefTypeGroup['CntStiffMat_List']
            Mij_List = RefTypeGroup['Mij_List']
            DamageList = RefTypeGroup['DamageList']
            wc = RefTypeGroup['wc']
            getSecStiff = RefTypeGroup['getSecStiff']
            #getSecStiff = None
            N_IntegrationPoints_2D = RefTypeGroup['N_IntegrationPoints_2D']
            Elem_NDOF = RefTypeGroup['Elem_NDOF']
            N_IntfcElem = len(IntfcElemList)
            RefIntfcElemLocId = 0
            RefIntfcElem_SecStiffList = []
            
            ElemList_DeformedCoord =   MP_DeformedCoord_1[ElemList_LocDofVector]
            ElemList_CntFint = np.zeros([Elem_NDOF, N_IntfcElem])
            
            for p in range(N_IntegrationPoints_2D):
                
                #Calculating PenaltyWeight
                ElemList_IntgPntGapVec =  np.dot(Mij_List[p], ElemList_DeformedCoord)
                
                #ElemList_StiffnessWeight = getPenaltyWeightList(IntfcElemList, ElemList_IntgPntGapVec, getSecStiff)
                ElemList_StiffnessWeight = getDamageWeightList(IntfcElemList, DamageList, ElemList_IntgPntGapVec, getSecStiff, wc)
    
                #Calculating Contact Force
                CntKe = CntStiffMat_List[p]
                ElemList_CntFint += np.dot(CntKe, ElemList_StiffnessWeight*ElemList_DeformedCoord)
                
                #Extracting Data for calculating Contact Stress (Post-processing)
                SecStiff = ElemList_StiffnessWeight[RefIntfcElemLocId]/IntfcElemList[RefIntfcElemLocId]['IntfcArea']
                RefIntfcElem_SecStiffList.append(SecStiff)
                
            MP_LocCntFintVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_CntFint.ravel(), minlength=MP_NDOF)        
                
            
            #Calculating Contact Stress of Ref Elements
            Elem_DeformedCoord = ElemList_DeformedCoord[:, RefIntfcElemLocId]
            Elem_CntTracData = getElemContactTraction(RefTypeGroup['knn_ij'], Mij_List, Elem_DeformedCoord, RefIntfcElem_SecStiffList)
            
            MP_ElemCntStressDataList.append(Elem_CntTracData)
            
                
        else: #Octree Elements
            Ke = RefTypeGroup['ElemStiffMat']
            ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
            ElemList_Level = RefTypeGroup['ElemList_Level']
            
            ElemList_Un_1 =   MP_Un_1[ElemList_LocDofVector]
            ElemList_Un_1[ElemList_SignVector] *= -1.0        
            ElemList_Fint =   np.dot(Ke, ElemList_Level*ElemList_Un_1)
            
            
            ElemList_Fint[ElemList_SignVector] *= -1.0
            
            
            
            
            if FintCalcMode == 'inbin':            
                MP_LocFintVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_Fint.ravel(), minlength=MP_NDOF)        
            elif FintCalcMode == 'infor':
                apply_sum(ElemList_LocDofVector, MP_LocFintVec, ElemList_Fint)            
            elif FintCalcMode == 'outbin':
                N = len(ElemList_LocDofVector_Flat)
                Flat_ElemFint[I:I+N]=ElemList_Fint.ravel()
                I += N
    
    
    if FintCalcMode == 'outbin':
        MP_LocFintVec = np.bincount(Flat_ElemLocDof, weights=Flat_ElemFint, minlength=MP_NDOF)
    
    MP_LocFintVec += MP_LocCntFintVec
    
    
    #Calculating Overlapping Fint Vectors
    MP_OvrlpFintVecList = []
    MP_InvOvrlpFintVecList = []
    N_NbrMP = len(MP_OvrlpLocalDofVecList)    
    for j in range(N_NbrMP):
        MP_OvrlpFintVec = np.hstack([MP_LocFintVec[MP_OvrlpLocalDofVecList[j]], MP_LocCntFintVec[MP_OvrlpLocalDofVecList[j]]])
        MP_OvrlpFintVecList.append(MP_OvrlpFintVec)   
        
        N_NbrDof_j = len(MP_OvrlpLocalDofVecList[j]);
        MP_InvOvrlpFintVecList.append(np.zeros(2*N_NbrDof_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping Fint
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        SendReq     = Comm.Isend(MP_OvrlpFintVecList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpFintVecList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
     
    #Calculating Fint
    MP_FintVec = MP_LocFintVec
    MP_CntFintVec = MP_LocCntFintVec
    for j in range(N_NbrMP): 
        N_NbrDof_j = len(MP_OvrlpLocalDofVecList[j]);
        MP_FintVec[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpFintVecList[j][:N_NbrDof_j]
        MP_CntFintVec[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpFintVecList[j][N_NbrDof_j:] 
                    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
     
    return MP_FintVec, MP_CntFintVec, MP_ElemCntStressDataList 




def MPI_SUM(MP_RefVar, MP_TimeRecData):
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    Glob_RefVar = Comm.allreduce(MP_RefVar, op=MPI.SUM)
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    return Glob_RefVar

    
    

def updateTime(MP_TimeRecData, Ref, TimeStepCount=None):
    
    if Ref == 'UpdateList':
        MP_TimeRecData['TimeStepCountList'].append(TimeStepCount)
        MP_TimeRecData['dT_CalcList'].append(MP_TimeRecData['dT_Calc'])
        MP_TimeRecData['dT_CommWaitList'].append(MP_TimeRecData['dT_CommWait'])
    else:  
        t1 = time()
        MP_TimeRecData[Ref] += t1 - MP_TimeRecData['t0']
        MP_TimeRecData['t0'] = t1
        

    
    
    
    

def calcMPDispVec(MP_FintVector, MP_RefLoadVector, MP_InvLumpedMassVector, MP_FixedLocalDofVector, MP_Un_2, MP_Un_1, dt, DeltaLambda_1, Damping_Alpha):
   
    MP_FextVector = DeltaLambda_1*MP_RefLoadVector
    DampTerm = 0.5*Damping_Alpha*dt
    MP_Un = (1.0/(1.0+DampTerm))*(2.0*MP_Un_1 - (1-DampTerm)*MP_Un_2 + dt*dt*MP_InvLumpedMassVector*(MP_FextVector - MP_FintVector))
    MP_Un[MP_FixedLocalDofVector] = 0.0
    
    return MP_Un



    
def getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData, RecTime = True):
    
    if RecTime: updateTime(MP_TimeRecData, 'dT_Calc')
    SendReq = Comm.Isend(MP_Un, dest=0, tag=Rank)            
    if Rank == 0:                
        for j in range(N_Workers):    Comm.Recv(MPList_Un[j], source=j, tag=j)                
    SendReq.Wait()
    RecTime:    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    GlobUnVector = []
    if Rank == 0:
        GathUnVector = np.hstack(MPList_Un)
        GlobUnVector = np.zeros(GlobNDof, dtype=float)
        GlobUnVector[GathDofVector] = GathUnVector
            
    return GlobUnVector



def plotDispVecData(PlotFileName, TimeList, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList, TimeList_PlotDispVector.T)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    

    

def exportDispVecData(OutputFileName, ExportCount, Time_dT, GlobUnVector, Splits = 1):
    
    DispVecFileName = OutputFileName+'_'+str(ExportCount)+'_'
    
    if Rank == 0:
        
        J = Splits
        N = int(len(GlobUnVector))
        Nj = int(N/J)
        for j in range(J):
            if j==0:
                N1 = 0; N2 = Nj;
            elif j == J-1:
                N1 = N2; N2 = N;
            else:
                N1 = N2; N2 = (j+1)*Nj;
            
            DispData_j = {'T': Time_dT, 'U': GlobUnVector[N1:N2]}
            savemat(DispVecFileName+str(j+1)+'.mat', DispData_j)



    



if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    #-------------------------------------------------------------------------------
    #print(Initializing ModelData..')
    MP_TimeRecData = {'dT_FileRead':        0.0,
                      'dT_Calc':            0.0,
                      'dT_CommWait':        0.0,
                      'dT_CalcList':        [],
                      'dT_CommWaitList':    [],
                      'TimeStepCountList':  [],
                      't0':                 time()}
    
    N_MshPrt =          N_Workers
    ModelName =         sys.argv[1]
    ScratchPath =       sys.argv[2]
    R0 =                sys.argv[3]
    SpeedTestFlag =     int(sys.argv[4])
    PBS_JobId =         sys.argv[5]
    
    
    #Creating directories
    ResultPath          = ScratchPath + 'Results_Run' + str(R0)
    if SpeedTestFlag == 1:  ResultPath += '_SpeedTest/'
    else:                   ResultPath += '/'
        
    PlotPath = ResultPath + 'PlotData/'
    DispVecPath = ResultPath + 'DispVecData/'
    if Rank==0:    
        if not os.path.exists(ResultPath):            
            os.makedirs(PlotPath)        
            os.makedirs(DispVecPath)
    
 
    OutputFileName = DispVecPath + ModelName
    PlotFileName = PlotPath + ModelName
    
    #Reading Model Data Files
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(N_MshPrt)  + '/'
    RefMeshPart_FileName = PyDataPath + str(Rank) + '.zpkl'
    Cmpr_RefMeshPart = open(RefMeshPart_FileName, 'rb').read()
    RefMeshPart = pickle.loads(zlib.decompress(Cmpr_RefMeshPart))
    
    MP_SubDomainData =                   RefMeshPart['SubDomainData']
    MP_NbrMPIdVector =                     RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList =            RefMeshPart['OvrlpLocalDofVecList']
    MP_NDOF =                            RefMeshPart['NDOF']
    MP_DOFIdVector =                    RefMeshPart['DofVector']
    MP_RefLoadVector =                  RefMeshPart['RefLoadVector']
    MP_NodeCoordVec =                   RefMeshPart['NodeCoordVec']
    MP_InvLumpedMassVector =             RefMeshPart['InvDiagM']
    MP_FixedLocalDofVector =            RefMeshPart['FixedLocDofVector']
    GathDofVector =                    RefMeshPart['GathDofVector']
    MPList_NDofVec =                   RefMeshPart['MPList_NDofVec']
    MPList_RefPlotDofIndicesList =      RefMeshPart['MPList_RefPlotDofIndicesList']
    MP_RefPlotData =                    RefMeshPart['RefPlotData']
    MP_WeightVector =                   RefMeshPart['WeightVector']
    GlobData =                           RefMeshPart['GlobData']
        
    GlobNDof =                           GlobData['GlobNDof']
    MaxTime =                            GlobData['MaxTime']
    Damping_Alpha =                      GlobData['Damping_Alpha']
    dt =                                 GlobData['TimeStepSize']
    DeltaLambdaList =                    GlobData['DeltaLambdaList']
    dT_Export =                          GlobData['dT_Export']
    PlotFlag =                           GlobData['PlotFlag']
    ExportFrms =                        GlobData['ExportFrms']
    ExportFlag =                         GlobData['ExportFlag']
    FintCalcMode =                       GlobData['FintCalcMode']
    N_IntegrationPoints =                GlobData['N_IntegrationPoints']
    
    EnergyCalcFlag =                     1
    MP_LumpedMassVector =                1.0/MP_InvLumpedMassVector
    
    
    if SpeedTestFlag==1:  
        PlotFlag = 0; ExportFlag = 0; MaxTime=1000*dt; FintCalcMode = 'outbin'; EnergyCalcFlag = 0;
    
    if not FintCalcMode in ['inbin', 'infor', 'outbin']:  
        raise ValueError("FintCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    ExportKeyFrm = round(dT_Export/dt)
    RefMaxTimeStepCount = int(np.ceil(MaxTime/dt)) + 1
    
    if Rank==0:
        print(ExportFlag, FintCalcMode)
        print('dt', dt)
        print('ExportKeyFrm',ExportKeyFrm)   
    
    
    #Initializing Variables
    MPList_Un = []
    if Rank==0:
        N_MeshParts = len(MPList_NDofVec)
        MPList_Un = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]
            
    
    #A small sleep to avoid hanging
    sleep(Rank*1e-4)
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    
    #Initializing Variables
    N_NbrMP = len(MP_NbrMPIdVector)
    N_OvrlpLocalDofVecList = [len(MP_OvrlpLocalDofVecList[j]) for j in range(N_NbrMP)]
    N_NbrDof               = np.sum(N_OvrlpLocalDofVecList)
    
    NCount = 0
    N_Type = len(MP_SubDomainData['StrucDataList'])
    for j in range(N_Type):    NCount += len(MP_SubDomainData['StrucDataList'][j]['ElemList_LocDofVector_Flat'])
    
    
    Flat_ElemLocDof = np.zeros(NCount, dtype=int)
    I=0
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]  
        ElemTypeId = RefTypeGroup['ElemTypeId']
        if ElemTypeId >= 0:
            ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
            N = len(ElemList_LocDofVector_Flat)
            Flat_ElemLocDof[I:I+N]=ElemList_LocDofVector_Flat
            I += N
    
    
    MP_Un_1 = (1e-200)*np.random.rand(MP_NDOF)
    MP_Un = (1e-200)*np.random.rand(MP_NDOF)
    
    initPenaltyStiffnessMatrix(MP_SubDomainData, N_IntegrationPoints, dt)
    
    ExportCount = 1
    TimeList = [i*dt for i in range(RefMaxTimeStepCount)]
    
    
    if PlotFlag == 1:
        MP_PlotLocalDofVec  = MP_RefPlotData['LocalDofVec']
        MP_PlotNDofs       = len(MP_PlotLocalDofVec)
        RefPlotDofVec       = MP_RefPlotData['RefPlotDofVec']
        qpoint              = MP_RefPlotData['qpoint']
        TestPlotFlag        = MP_RefPlotData['TestPlotFlag']
        
        if MP_PlotNDofs > 0:
            MP_PlotDispVector               = np.zeros([MP_PlotNDofs, RefMaxTimeStepCount])
            MP_PlotDispVector[:,0]          = MP_Un[MP_PlotLocalDofVec]
        else:   
            MP_PlotDispVector               = []
        
        MP_PlotCntStressData               = []
        #MP_PlotCntStrData               = []
        #MP_PlotNormGapData               = []
            
    
    if ExportFlag == 1:
        
        ExportNow = False
        if ExportKeyFrm>0 or 0 in ExportFrms:
            ExportNow = True
        
        if ExportNow:
            Time_dT = 0*dt
            
            GlobUnVector = getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData)
            exportDispVecData(OutputFileName, ExportCount, Time_dT, GlobUnVector)
            
            ExportCount += 1
                
    
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')
    for TimeStepCount in range(1, RefMaxTimeStepCount):
        
        print('---------------------')
        print('TimeStepCount', TimeStepCount)
        print(MP_Un)
        
        MP_Un_2 = MP_Un_1
        MP_Un_1 = MP_Un
        
        #Calculating Fint
        MP_DeformedCoord_1 = MP_NodeCoordVec + MP_Un_1
        MP_FintVec, MP_CntFintVec, MP_ElemCntStressDataList    = calcMPFint(MP_Un_1, MP_DeformedCoord_1, FintCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, Flat_ElemLocDof, NCount, MP_NbrMPIdVector, MP_TimeRecData)
        
        #Calculating Displacement Vector
        DeltaLambda_1 = DeltaLambdaList[TimeStepCount-1]
        MP_Un = calcMPDispVec(MP_FintVec, MP_RefLoadVector, MP_InvLumpedMassVector, MP_FixedLocalDofVector, MP_Un_2, MP_Un_1, dt, DeltaLambda_1, Damping_Alpha)
        
        #updateTime(MP_TimeRecData, 'UpdateList', TimeStepCount=TimeStepCount)
        
        if PlotFlag == 1:
            #if Rank==0: print(TimeStepCount)
            if MP_PlotNDofs>0:
                MP_PlotDispVector[:,TimeStepCount] = MP_Un[MP_PlotLocalDofVec]
                
            if len(MP_ElemCntStressDataList)>0:
                MP_PlotCntStressData.append([MP_ElemCntStressDataList[0][0][2], MP_ElemCntStressDataList[0][1][2]])
                #MP_PlotCntStrData.append(MP_ElemCntStrList[0])
                #MP_PlotNormGapData.append(MP_ElemNormGapList[0])
                
            if (TimeStepCount)%1000==0:
                if TestPlotFlag:
                    plotDispVecData(PlotFileName+'_'+str(TimeStepCount), TimeList, MP_PlotDispVector)
            
            
        if ExportFlag == 1:        
            
            ExportNow = False
            if ExportKeyFrm>0:
                if (TimeStepCount)%ExportKeyFrm==0: ExportNow = True
                
            if TimeStepCount in ExportFrms:
                ExportNow = True
            
            if ExportNow:
                Time_dT = TimeStepCount*dt
                
                GlobUnVector = getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData)
                exportDispVecData(OutputFileName, ExportCount, Time_dT, GlobUnVector)
                
                ExportCount += 1
        
    
    
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('TotalTimeStepCount', RefMaxTimeStepCount) 
    
    
    #Saving CPU Time
    MP_TimeRecData['dT_Total_Verify'] = t0_End - t0_Start 
    MP_TimeRecData['t0_Start'] = t0_Start
    MP_TimeRecData['t0_End'] = t0_End
    MP_TimeRecData['MP_NCount'] = NCount
    MP_TimeRecData['MP_NDOF'] = MP_NDOF
    MP_TimeRecData['N_NbrDof'] = N_NbrDof    
    MPList_TimeRecData = Comm.gather(MP_TimeRecData, root=0)
    
    if PlotFlag == 1:   
        MPList_PlotDispVector = Comm.gather(MP_PlotDispVector, root=0)
        MPList_PlotCntStressData = Comm.gather(MP_PlotCntStressData, root=0)
        #MPList_PlotCntStrData = Comm.gather(MP_PlotCntStrData, root=0)
        #MPList_PlotNormGapData = Comm.gather(MP_PlotNormGapData, root=0)
        
    
    if Rank == 0:        
        TimeData = configTimeRecData(MPList_TimeRecData)
        TimeData['PBS_JobId'] = PBS_JobId
        
        TimeDataFileName = OutputFileName + '_MP' +  str(N_MshPrt) + '_' + FintCalcMode + '_TimeData'
        np.savez_compressed(TimeDataFileName, TimeData = TimeData)
        savemat(TimeDataFileName +'.mat', TimeData)
        
        #Exporting Plots        
        if PlotFlag == 1:
            N_TotalPlotDofs     = len(RefPlotDofVec)
            TimeList_PlotDispVector = np.zeros([N_TotalPlotDofs, RefMaxTimeStepCount])
            TimeList_CntStressData = []
            TimeList_CntStrData = []
            TimeList_NormGapData = []
        
            for i in range(N_Workers):
                MP_PlotDispVector_i = MPList_PlotDispVector[i]
                
                RefPlotDofIndices_i = MPList_RefPlotDofIndicesList[i]
                N_PlotDofs_i = len(MP_PlotDispVector_i)
                if N_PlotDofs_i>0:
                    for j in range(N_PlotDofs_i):
                        TimeList_PlotDispVector[RefPlotDofIndices_i[j],:] = MP_PlotDispVector_i[j]
                
                
                MP_PlotCntStressData_i = MPList_PlotCntStressData[i]
                #MP_PlotCntStrData_i = MPList_PlotCntStrData[i]
                #MP_PlotNormGapData_i = MPList_PlotNormGapData[i]
                
                N_PlotCntStressData = len(MP_PlotCntStressData_i)
                if N_PlotCntStressData>0:
                    TimeList_CntGapList = [MP_PlotCntStressData_i[j][0] for j in range(N_PlotCntStressData)]
                    TimeList_CntStressList = [MP_PlotCntStressData_i[j][1] for j in range(N_PlotCntStressData)]
                    #TimeList_CntStrData.append(MP_PlotCntStrData_i)
                    #TimeList_NormGapData.append(MP_PlotNormGapData_i)
                    
                
            #Saving Data File
            PlotTimeData = {'Plot_T': TimeList, 
                            'Plot_U': TimeList_PlotDispVector, 
                            'Plot_CntStressData': [TimeList_CntGapList, TimeList_CntStressList], 
                            #'CntStrData': TimeList_CntStrData, 
                            #'NormGapData': TimeList_NormGapData, 
                            'Plot_Dof': RefPlotDofVec+1, 
                            'qpoint': qpoint}
            np.savez_compressed(PlotFileName+'_PlotData', PlotData = PlotTimeData)
            savemat(PlotFileName+'_PlotData.mat', PlotTimeData)
            
            fig = plt.figure()
            plt.plot(TimeList_CntGapList, TimeList_CntStressList)
            # plt.xlim([0, 14])
            # plt.ylim([-0.5, 3.0])
            fig.savefig(PlotFileName+'_PlotData.png', dpi = 480, bbox_inches='tight')
            plt.close()
        
        
            
    
   
    #Printing Results
    GlobUnVector = getGlobDispVec(MP_Un, MPList_Un, GathDofVector, GlobNDof, MP_TimeRecData, RecTime = False)
    if Rank == 0:
        np.set_printoptions(precision=12)
        print('\n\n\n')
        I = np.argmax(GlobUnVector)
        print('I', I)
        for i in range(10): print(GlobUnVector[I+i-5])
    
    
    
    

