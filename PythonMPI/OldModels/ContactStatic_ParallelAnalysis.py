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

#Defining Constants
eps = np.finfo(float).eps


def initNodalGapMatrix(RefMeshPart):
    
    MP_SubDomainData                    = RefMeshPart['SubDomainData']
    N_IntegrationPoints                 = 2
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId == -1: #Interface Elements
            Ni, wi                  = GaussLobattoIntegrationTable(N_IntegrationPoints)
            NodalGapCalcMatrix = []
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
                    NodalGapCalcMatrix.append(Mij)
                    
            NodalGapCalcMatrix = np.vstack(NodalGapCalcMatrix)
            RefTypeGroup['NodalGapCalcMatrix'] = NodalGapCalcMatrix
            break
    



def initPenaltyStiffnessMatrix(RefMeshPart):
    
    MP_SubDomainData =                   RefMeshPart['SubDomainData']
    N_IntegrationPoints =                GlobData['N_IntegrationPoints']
    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId == -1: #Interface Elements
        
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
            ft = 1e3
            Gf = 10
            wc = 2*Gf/ft
            wo = ft/Knn;
            """
            NRef = wc/wo;
            GapList0 = wo*np.array([-1e16, 0.5, 1.0], dtype=float)
            GapList1 = wo*np.geomspace(1.01,NRef,1000)
            CnP0 = Knn*GapList0
            CnP1 = (ft/(wc-wo))*(wc-GapList1)
            GapList = np.hstack([GapList0,GapList1])
            CnP = np.hstack([CnP0,CnP1])
            SecStiff = CnP/GapList
            SecStiffFunc = interp1d(GapList, SecStiff, bounds_error=False, fill_value = 0.0)
            """
            
            def getSecStiff(NormalGap, ft=ft, wc=wc, Knn=Knn):
                wo = ft/Knn;
                if NormalGap <= wo:
                    SecStiff = Knn
                elif wo < NormalGap <= wc:
                    NormalTraction = (ft/(wc-wo))*(wc-NormalGap)
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
            RefTypeGroup['DamageList'] = np.zeros(N_IntfcElem)
            
            break


    

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





def getPenaltyWeightList(IntfcElemList, ElemList_IntgPntGapVec, getSecStiff):
    
    N_IntfcElem = len(IntfcElemList)
    ElemList_PenaltyWeight = np.zeros(N_IntfcElem)
    ElemList_ContactBool = np.zeros(N_IntfcElem)
    #ElemList_SlipPenaltyWeight = np.zeros(N_IntfcElem)
    
    for i in range(N_IntfcElem):
        NormalGap           = ElemList_IntgPntGapVec[2, i]
        IntfcElemArea    = IntfcElemList[i]['IntfcArea']
    
        if NormalGap <= 0:
        
            #if Stick is satisfied
            ElemList_PenaltyWeight[i] = IntfcElemArea
            if getSecStiff:    ElemList_PenaltyWeight[i] *= getSecStiff(abs(NormalGap))
            
            ElemList_ContactBool[i] = 1.0
            
            #if Slip is satisfied
            #ElemList_SlipPenaltyWeight[i] = Slip_PenaltyWeight

        
    return ElemList_PenaltyWeight, ElemList_ContactBool
    


def getDamageWeightList(IntfcElemList, DamageList, ElemList_IntgPntGapVec, getSecStiff, wc):
    
    N_IntfcElem = len(IntfcElemList)
    ElemList_StiffnessWeight = np.zeros(N_IntfcElem)
    
    for i in range(N_IntfcElem):
        NormalGap           = ElemList_IntgPntGapVec[2, i]
        IntfcElemArea    = IntfcElemList[i]['IntfcArea']
        
    
        if NormalGap <= 0:
            ElemList_StiffnessWeight[i] = IntfcElemArea*getSecStiff(NormalGap)
            
        else:
            Damage = np.min([1.0,NormalGap/wc])
            
            if DamageList[i]<Damage: #Loading (Increasing Damage)
                DamageList[i]=Damage
                ElemList_StiffnessWeight[i] = IntfcElemArea*getSecStiff(NormalGap)
            
            else: #Loading/Unloading
                ElemList_StiffnessWeight[i] = (1.0-DamageList[i])*IntfcElemArea*getSecStiff(0)
            
            
            
    return ElemList_StiffnessWeight
    
        

def calcMPFint(MP_Un, MP_DeformedCoord, RefMeshPart, MP_TimeRecData, A0=0.0, B0=1.0):
    
    #Reading Variables
    MP_SubDomainData                     = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector                     = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList              = RefMeshPart['OvrlpLocalDofVecList']
    Flat_ElemLocDof                      = RefMeshPart['Flat_ElemLocDof']
    NCount                               = RefMeshPart['NCount']
    FintCalcMode                         = RefMeshPart['GlobData']['FintCalcMode']
    
    #Calculating Local Fint Vector for Octree cells
    MP_NDOF = len(MP_Un)    
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
            #getSecStiff = RefTypeGroup['getSecStiff']
            getSecStiff = None
            N_IntegrationPoints_2D = RefTypeGroup['N_IntegrationPoints_2D']
            Elem_NDOF = RefTypeGroup['Elem_NDOF']
            N_IntfcElem = len(IntfcElemList)
            RefIntfcElemLocId = 0
            RefIntfcElem_SecStiffList = []
            
            ElemList_DeformedCoord =   MP_DeformedCoord[ElemList_LocDofVector]
            ElemList_CntFint = np.zeros([Elem_NDOF, N_IntfcElem])
            
            for p in range(N_IntegrationPoints_2D):
                
                #Calculating PenaltyWeight
                ElemList_IntgPntGapVec =  np.dot(Mij_List[p], ElemList_DeformedCoord)
                
                ElemList_StiffnessWeight, ElemList_ContactBool = getPenaltyWeightList(IntfcElemList, ElemList_IntgPntGapVec, getSecStiff)
                #ElemList_StiffnessWeight = getDamageWeightList(IntfcElemList, DamageList, ElemList_IntgPntGapVec, getSecStiff, wc)
    
                RefTypeGroup['ElemList_ContactBool'] = ElemList_ContactBool
            
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
            Ke = RefTypeGroup['ElemStiffMat']; Me = RefTypeGroup['ElemMassMat']
            ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
            ElemList_Level = RefTypeGroup['ElemList_Level']
            ElemList_LevelCubed = RefTypeGroup['ElemList_LevelCubed']
        
            ElemList_Un =   MP_Un[ElemList_LocDofVector]
            ElemList_Un[ElemList_SignVector] *= -1.0        
            if B0==0.0:     ElemList_Fint =  A0*np.dot(Me, ElemList_LevelCubed*ElemList_Un)
            elif A0==0.0:   ElemList_Fint =  B0*np.dot(Ke, ElemList_Level*ElemList_Un)
            else:           ElemList_Fint =  A0*np.dot(Me, ElemList_LevelCubed*ElemList_Un) + B0*np.dot(Ke, ElemList_Level*ElemList_Un)
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

    



def PCG(MP_X0, MP_Fc, RefMeshPart, MP_TimeRecData, A0=None, B0=None):

    MP_WeightVector                      = RefMeshPart['WeightVector']
    MP_NDOF                              = RefMeshPart['NDOF']
    MP_LocDof_eff                        = RefMeshPart['LocDof_eff']
    MP_LumpedMassVector                  = RefMeshPart['LumpedMassVector']
    MP_InvDiagPreCondVector0             = RefMeshPart['InvDiagPreCondVector0']
    MP_InvDiagPreCondVector1             = RefMeshPart['InvDiagPreCondVector1']
    MPList_RefPlotDofIndicesList         = RefMeshPart['MPList_RefPlotDofIndicesList']
    GathDofVector                        = RefMeshPart['GathDofVector']
    GlobData                             = RefMeshPart['GlobData']
    
    GlobNDof_eff                        = GlobData['GlobNDof_eff']
    GlobNDof                            = GlobData['GlobNDof']
    MaxIter                             = GlobData['MaxIter']    
    Tol                                 = GlobData['Tol']
    ExistDP0                            = GlobData['ExistDP0']
    ExistDP1                            = GlobData['ExistDP1']
    UseLumpedMass                       = GlobData['UseLumpedMass']
    fb1a                                 = GlobData['fb1a']
    
    if A0==None and B0==None:
        if UseLumpedMass:   A0 = 0.0; B0 = 1.0
        else:               A0 = 1.0; B0 = fb1a
    
    #Initializing Variables
    MP_Fc                                = MP_Fc[MP_LocDof_eff]
    MP_X0                                = MP_X0[MP_LocDof_eff]
    MP_X                                 = MP_X0
    MP_XMin                              = MP_X                                  #Iterate which has minimal residual so far
    MP_SqRefLoad                         = np.dot(MP_Fc, MP_Fc*MP_WeightVector)
    SqRefLoad                            = MPI_SUM(MP_SqRefLoad, MP_TimeRecData)
    NormRefLoadVector                    = np.sqrt(SqRefLoad) #n2b
    TolB                                 = Tol*NormRefLoadVector
    
    #Check for all zero right hand side vector => all zero solution
    if NormRefLoadVector == 0:                      # if rhs vector is all zeros        
        MP_X_Unq                        = np.zeros(MP_NDOF);  # then  solution is all zeros
        MP_X_Unq[MP_LocDof_eff]         = MP_X
        Flag                            = 0;                                   # a valid solution has been obtained
        RelRes                          = 0;                                   # the relative residual is actually 0/0
        Iter                            = 0;                                   # no iterations need be performed
        ResVec                          = [0];                               # resvec(1) = norm(b-A*x) = norm(0)
        
        return MP_X_Unq, Flag, RelRes, Iter
    
    
    #Initial Settings    
    Flag                                        = 1
    Rho                                         = 1.0 #Dummy
    Stag                                        = 0  # stagnation of the method
    MoreSteps                                   = 0
    MaxStagSteps                                = 3
    MaxMSteps                                   = min([int(GlobNDof_eff/50),5,GlobNDof_eff-MaxIter]);
    iMin                                        = 0
    Iter                                        = 0 
    
    MP_X_Unq                                    = np.zeros(MP_NDOF)
    MP_P_Unq                                    = np.zeros(MP_NDOF)
    
    MP_X_Unq[MP_LocDof_eff]                     = MP_X
    MP_RefLoadVector0_Unq                       = calcMPFint(MP_X_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
    if UseLumpedMass:      
        MP_RefLoadVector0_Unq                   = MP_LumpedMassVector*MP_X_Unq + fb1a*MP_RefLoadVector0_Unq
    MP_RefLoadVector0                           = MP_RefLoadVector0_Unq[MP_LocDof_eff]
    MP_R                                        = MP_Fc - MP_RefLoadVector0
    MP_SqR                                      = np.dot(MP_R, MP_R*MP_WeightVector)
    NormR                                       = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
    NormRMin                                    = NormR
    
    #Checking if the initial guess is a good enough solution
    if NormR <= TolB:
        Flag                            = 0;
        RelRes                          = NormR/NormRefLoadVector;
        Iter                            = 0;
        #ResVec                          = [NormR];
        return MP_X_Unq, Flag, RelRes, Iter
    
    """
    if Rank == 0:   
        ResVec                       = np.zeros(GlobNDof_eff+1)        
        ResVec[0]                    = NormR    
        
        #print(0, [NormR, TolB], [Stag, MaxStagSteps])
    """
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')    
    
    for i in range(MaxIter):
        
        #Calculating Z
        if ExistDP0: #Diagonal Preconditioner DP0            
            MP_Y           = MP_InvDiagPreCondVector0*MP_R
            if np.any(np.isinf(MP_Y)):                
                Flag = 2
                break        
        else:    MP_Y      = MP_R
            
        if ExistDP1: #Diagonal Preconditioner DP1            
            MP_Z           = MP_InvDiagPreCondVector1*MP_Y
            if np.any(np.isinf(MP_Z)):                
                Flag = 2
                break        
        else:    MP_Z      = MP_Y
        
        #Calculating Rho
        Rho_1                           = Rho
        MP_Rho                          = np.dot(MP_Z,MP_R*MP_WeightVector)
        Rho                             = MPI_SUM(MP_Rho, MP_TimeRecData)
    
        
        if Rho == 0 or np.isinf(Rho):
            Flag = 4
            break
        
        #Calculating P and Beta
        if i == 0:
            MP_P         = MP_Z
        else:
            Beta                        = Rho/Rho_1   
            if Beta == 0 or np.isinf(Beta):
                Flag = 4
                break
            MP_P         = MP_Z + Beta*MP_P         
            
        #Calculating Q
        MP_P_Unq[MP_LocDof_eff]              = MP_P    
        MP_Q_Unq                             = calcMPFint(MP_P_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
        if UseLumpedMass:      
            MP_Q_Unq                         = MP_LumpedMassVector*MP_P_Unq + fb1a*MP_Q_Unq
        MP_Q                                 =  MP_Q_Unq[MP_LocDof_eff]
        
        #Calculating PQ and Alpha
        MP_PQ                            = np.dot(MP_P,MP_Q*MP_WeightVector)
        PQ                               = MPI_SUM(MP_PQ, MP_TimeRecData)
        if PQ <= 0 or np.isinf(PQ):
            Flag = 4
            break
        else:    Alpha                   = Rho/PQ    
        if np.isinf(Alpha):
            Flag = 4
            break
        
        #Calculating R
        MP_R                              -= Alpha*MP_Q
        
        #Calculating Convergence Variables
        MP_SqP                           = np.dot(MP_P, MP_P*MP_WeightVector)
        MP_SqX                           = np.dot(MP_X, MP_X*MP_WeightVector)
        MP_SqR                           = np.dot(MP_R, MP_R*MP_WeightVector)
        [NormP, NormX, NormR]            = np.sqrt(MPI_SUM(np.array([MP_SqP, MP_SqX, MP_SqR]), MP_TimeRecData))
        
        #Stagnation
        if (NormP*abs(Alpha) < eps*NormX):  Stag += 1
        else:                               Stag = 0
        
        #Calculating X
        MP_X                              += Alpha*MP_P
        
        NormR_Act                       = NormR
        
        """
        if Rank==0:
            #savemat(OutputFileName+'_Log.mat', {'NormR':NormR, 'TolB': TolB, 'i': i})
            #if i%1==0:    print(i, [NormR, TolB], [Stag, MaxStagSteps])            
            ResVec[i+1]                    = NormR_Act
        """
        
        if NormR <= TolB or Stag >= MaxStagSteps or MoreSteps > 0:
            MP_X_Unq[MP_LocDof_eff]             = MP_X
            MP_Fint_Unq                         = calcMPFint(MP_X_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
            if UseLumpedMass:      
                MP_Fint_Unq                     = MP_LumpedMassVector*MP_X_Unq + fb1a*MP_Fint_Unq
            MP_Fint                             = MP_Fint_Unq[MP_LocDof_eff]
            MP_R                                = MP_Fc - MP_Fint
            MP_SqR_Act                          = np.dot(MP_R, MP_R*MP_WeightVector)
            NormR_Act                           = np.sqrt(MPI_SUM(MP_SqR_Act, MP_TimeRecData))
        
                    
            #if Rank==0:                     ResVec[i+1] = NormR_Act                
            
            #Converged
            if NormR_Act <= TolB: #Act = Actual                
                Flag = 0
                Iter = i
                break            
            else:            
                if Stag >= MaxStagSteps and MoreSteps == 0: Stag = 0
                MoreSteps           += 1            
                #Stagnated
                if MoreSteps >= MaxMSteps: 
                    raise Warning('PCG : TooSmallTolerance')
                    Flag = 3
                    Iter = i
                    break
        
        #Update Minimal Norm Quanitites
        if NormR_Act < NormRMin:            
            NormRMin                    = NormR_Act
            MP_XMin                      = np.array(MP_X)
            iMin                        = i
        
        if Stag >= MaxStagSteps:
            Flag = 3
            break
    
        
    #Finalizing Results
    if Flag == 0:
        RelRes = NormR_Act/NormRefLoadVector
    else:        
        MP_X_Unq[MP_LocDof_eff]             = MP_XMin
        MP_Fint_Unq                         = calcMPFint(MP_X_Unq, RefMeshPart, MP_TimeRecData, A0=A0, B0=B0)
        if UseLumpedMass:      
            MP_Fint_Unq                     = MP_LumpedMassVector*MP_X_Unq + fb1a*MP_Fint_Unq
        MP_Fint                             = MP_Fint_Unq[MP_LocDof_eff]
        MP_R                                = MP_Fc - MP_Fint
        MP_SqR                              = np.dot(MP_R, MP_R*MP_WeightVector)
        NormR                               = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
        
        if NormR < NormR_Act:            
            MP_X = MP_XMin
            Iter = iMin
            RelRes = NormR/NormRefLoadVector        
        else:            
            Iter = i
            RelRes = NormR_Act/NormRefLoadVector
    
    Iter += 1 #So that the iteration matches with Matlab
    
    """
    #Truncate the zeros from resvec
    if Rank == 0:            
        if Flag <= 1 or Flag ==3:        ResVec     = ResVec[:i+2]
        else:                            ResVec     = ResVec[:i+1]
    """
    
    return MP_X_Unq, Flag, RelRes, Iter
    
    

def updateTime(MP_TimeRecData, Ref, TimeStepCount=None):
    
    if Ref == 'UpdateList':
        MP_TimeRecData['TimeStepCountList'].append(TimeStepCount)
        MP_TimeRecData['dT_CalcList'].append(MP_TimeRecData['dT_Calc'])
        MP_TimeRecData['dT_CommWaitList'].append(MP_TimeRecData['dT_CommWait'])
    else:  
        t1 = time()
        MP_TimeRecData[Ref] += t1 - MP_TimeRecData['t0']
        MP_TimeRecData['t0'] = t1
        

    
  
    

def calcNodalGapVec(MP_Un, RefMeshPart, MP_TimeRecData):
    
    #Reading Variables
    MP_SubDomainData = RefMeshPart['SubDomainData']
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    NodalGapVec = []
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId == -1: #Interface Elements
            
            NodalGapCalcMatrix = RefTypeGroup['NodalGapCalcMatrix']
            ElemList_ContactBool = RefTypeGroup['ElemList_ContactBool']
            
            ElemList_DeformedCoord =   MP_DeformedCoord[ElemList_LocDofVector]
            ElemList_NodalPntGapVec =  np.dot(NodalGapCalcMatrix, ElemList_ContactBool*ElemList_DeformedCoord)
            
            NodalGapVec.append(ElemList_NodalPntGapVec)
    
    NodalGapVec = np.vstack(NodalGapVec)
    
    return NodalGapVec
  
    




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





def plotDispVecData(PlotFileName, TimeList_PlotdT, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList_PlotdT, TimeList_PlotDispVector.T)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    


    



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
    
    MP_WeightVector =                    RefMeshPart['WeightVector']
    MP_LocDof_eff =                      RefMeshPart['LocDof_eff']
    MP_DPDiagKVector =                   RefMeshPart['DPDiagKVector']
    MP_NDOF =                            RefMeshPart['NDOF']
    MP_RefLoadVector =                   RefMeshPart['RefTransientLoadVector']
    MP_NodeCoordVec =                    RefMeshPart['NodeCoordVec']
    MPList_NDofVec =                     RefMeshPart['MPList_NDOFIdVec']
    MPList_RefPlotDofIndicesList =       RefMeshPart['MPList_RefPlotDofIndicesList']
    MP_RefPlotData =                     RefMeshPart['RefPlotData']
    MP_WeightVector =                    RefMeshPart['WeightVector']
    GlobData =                           RefMeshPart['GlobData']
        
    GlobNDOF =                           GlobData['GlobNDOF']
    N_TimeSteps =                        GlobData['N_TimeSteps']
    N_Exports =                          GlobData['dT_Export']
    PlotFlag =                           GlobData['PlotFlag']
    ExportFlag =                         GlobData['ExportFlag']
    FintCalcMode =                       GlobData['FintCalcMode']
    ArcLengthParameters =                GlobData['ArcLengthParameters']
    PCGParameters =                      GlobData['PCGParameters']
    
    MaxIterCount =                      ArcLengthParameters['MaxIterCount']
    RelTol =                            ArcLengthParameters['RelTol']
    m =                                 ArcLengthParameters['m']
    Nd0 =                               ArcLengthParameters['Nd0']
    n =                                 ArcLengthParameters['n']
    p =                                 ArcLengthParameters['p']
    DelLambda_ini =                     ArcLengthParameters['del_lambda_ini']
    
    
    
    if SpeedTestFlag==1:  
        PlotFlag = 0; ExportFlag = 0; N_TimeSteps=10; FintCalcMode = 'outbin'
    
    if Rank == 0:    print(PlotFlag, ExportFlag, FintCalcMode)
    
    if not FintCalcMode in ['inbin', 'infor', 'outbin']:  raise ValueError("FintCalcMode must be 'inbin', 'infor' or 'outbin'")
    
    ExportKeyFrm = round(N_TimeSteps/N_Exports)
    
    #Initializing Variables
    MPList_TempUn = []
    if Rank==0:
        N_MeshParts = len(MPList_NDofVec)
        MPList_TempUn = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]
                
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    MP_Un = (1e-200)*np.random.rand(MP_NDOF)
    
    initNodalGapMatrix(RefMeshPart)
    initPenaltyStiffnessMatrix(RefMeshPart)
    
    RefMeshPart['WeightVector'] = MP_WeightVector[MP_LocDof_eff]
    InvDiagPreConditioner = 1.0/MP_DPDiagKVector
    RefMeshPart['InvDiagPreCondVector0'] = InvDiagPreConditioner[MP_LocDof_eff]
    RefMeshPart['InvDiagPreCondVector1'] = []
    
    PCG_Tol = PCGParameters['Tol']
    if PCG_Tol < eps:        
        raise Warning('PCG : Too Small Tolerance')
        PCGParameters['Tol'] = eps        
    elif PCG_Tol > 1:        
        raise Warning('PCG : Too Big Tolerance')
        PCGParameters['Tol'] = 1 - eps
    
    
    ExportCount = 0
    
    if PlotFlag == 1:
        MP_PlotLocalDofVec  = MP_RefPlotData['LocalDofVec']
        MP_PlotNDofs       = len(MP_PlotLocalDofVec)
        RefPlotDofVec       = MP_RefPlotData['RefPlotDofVec']
        qpoint              = MP_RefPlotData['qpoint']
        TestPlotFlag        = MP_RefPlotData['TestPlotFlag']
        
        if MP_PlotNDofs > 0:
            MP_PlotDispVector               = np.zeros([MP_PlotNDofs, N_TimeSteps])
            MP_PlotDispVector[:,0]          = MP_Un[MP_PlotLocalDofVec]
        else:   
            MP_PlotDispVector               = []
        
        MP_PlotCntStressData               = []
        #MP_PlotCntStrData               = []
        #MP_PlotNormGapData               = []
    
        
    #Arc Length Parameters
    MP_SqRefLoad                         = np.dot(MP_RefLoadVector, MP_RefLoadVector*MP_WeightVector)
    SqRefLoad                            = MPI_SUM(MP_SqRefLoad, MP_TimeRecData)
    NormRefLoadVector                    = np.sqrt(SqRefLoad)
    TolF                                 = RelTol*NormRefLoadVector
    
    #MOVE INTO FOR LOOP
    Nintfc =            len(ActiveIntfElem) 
    Nd =                Nintfc**n + Nd0

    
    #Initializing solver variables
    MP_DelUp = PCG(np.zeros(MP_NDOF), MP_RefLoadVector, RefMeshPart, MP_TimeRecData, A0=0.0, B0=1.0)
    MP_LocDelUp = calcNodalGapVec(MP_DelUp, RefMeshPart, MP_TimeRecData)
    MP_SqLocDelUp = np.dot(MP_LocDelUp, MP_LocDelUp*MP_WeightVector)
    NormLocDelUp = np.sqrt(MPI_SUM(MP_SqLocDelUp, MP_TimeRecData))
    DelLambda = DelLambda_ini
    Delta_l = DelLambda*NormLocDelUp
    
    LastIterCount = Nd
    DeltaLambda = 0.0
    
    #Initializing Residual
    MP_Fext = DeltaLambda*MP_RefLoadVector
    MP_Fint = calcMPFint(MP_Un, MP_DeformedCoord, RefMeshPart, MP_TimeRecData)
    MP_R = MP_Fint - MP_Fext
    MP_SqR = np.dot(MP_R, MP_R*MP_WeightVector)
    NormR  = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
    
    if NormR<TolF:    raise Exception
        
    
    for TimeStepCount in range(RefMaxTimeStepCount):
        
        MP_DeltaUn =      np.zeros(MP_NDOF)
        Delta_l *=      (float(Nd)/LastIterCount)**m
        
        if TimeStepCount%50 == 0:
            print('')
            print('LastIterCount', LastIterCount, delta_l)
        
        
        
        for IterCount in range(MaxIterCount):
            
            MP_DelUp = PCG(MP_DelUp, MP_RefLoadVector, RefMeshPart, MP_TimeRecData, A0=0.0, B0=1.0)
            MP_DelUf = -1.0*PCG(-1.0*MP_DelUf, MP_R, RefMeshPart, MP_TimeRecData, A0=0.0, B0=1.0)
            
            MP_LocDelUp = calcNodalGapVec(MP_DelUp, RefMeshPart, MP_TimeRecData)
            MP_LocDelUf = calcNodalGapVec(MP_DelUf, RefMeshPart, MP_TimeRecData)
            MP_LocDeltaUn = calcNodalGapVec(MP_DeltaUn, RefMeshPart, MP_TimeRecData)
            
            if IterCount == 0:
                DelLambda = Delta_l/NormLocDelUp
                DelLambda0 = DelLambda
                
            else:
                
#                    del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
#                    del_lambda = (self.delta_l**2 - np.dot(Local_delta_u, Local_delta_u + Local_del_uf))/np.dot(Local_delta_u, Local_del_up)
                
                DelLambda_Up = MPI_SUM(np.dot(MP_LocDeltaUn, MP_LocDelUf*MP_WeightVector), MP_TimeRecData)
                DelLambda_Dn = MPI_SUM(np.dot(MP_LocDeltaUn, MP_LocDelUp*MP_WeightVector), MP_TimeRecData)
                DelLambda = -1.0*DelLambda_Up/DelLambda_Dn
                
            
            MP_DelUn = MP_DelUf + DelLambda*MP_DelUp
            
            #Updating increments
            MP_DeltaUn +=  MP_DelUn
            DeltaLambda += DelLambda
            
            MP_Un += MP_DelUn
            MP_Fext = DeltaLambda*MP_RefLoadVector
            
            MP_Fint = calcMPFint(MP_Un, MP_DeformedCoord, RefMeshPart, MP_TimeRecData)
            
            MP_R = MP_Fint - MP_Fext
            MP_SqR = np.dot(MP_R, MP_R*MP_WeightVector)
            NormR  = np.sqrt(MPI_SUM(MP_SqR, MP_TimeRecData))
            
            if NormR<TolF:
                LastIterCount = IterCount
                break
        
        else:
        
            print('Convergence was not achieved;',  FailedConvergenceCount, NormR/Norm_RefLoadVector)
            FailedConvergenceCount += 1
            LastIterCount = Nd*p
            
        
        if TimeStepCount >= MaxTimeStepCount or FailedConvergenceCount >= MaxFailedConvergenceCount:
            AnalysisFinished = True
            break
    
        
        
    if PlotFlag == 1:
        #if Rank==0: print(TimeStepCount)
        if N_MP_PlotDofs>0:
            MP_PlotDispVector[:,TimeChunkCount+1] = MP_UnVector[MP_PlotLocalDofVec]
            
        if len(MP_ElemCntStressDataList)>0:
            MP_PlotCntStressData.append([MP_ElemCntStressDataList[0][0][2], MP_ElemCntStressDataList[0][1][2]])
            #MP_PlotCntStrData.append(MP_ElemCntStrList[0])
            #MP_PlotNormGapData.append(MP_ElemNormGapList[0])
            
        if (TimeChunkCount+1)%1000==0:
            if TestPlotFlag:
                plotDispVecData(PlotFileName+'_'+str(TimeChunkCount+1), TimeList_PlotdT, MP_PlotDispVector)
        
        
    if ExportFlag == 1:        
        if (TimeChunkCount+1)%UpdateKFrm==0:            
            updateTime(MP_TimeRecData, 'dT_Calc')
            SendReq = Comm.Isend(MP_UnVector, dest=0, tag=Rank)            
            if Rank == 0:                
                for j in range(N_Workers):    Comm.Recv(MPList_UnVector_1[j], source=j, tag=j)                
            SendReq.Wait()
            updateTime(MP_TimeRecData, 'dT_CommWait')
            
            if Rank == 0:
                GathUnVector = np.hstack(MPList_UnVector_1)
                Glob_RefDispVector_1 = getGlobDispVec(GathUnVector, GathDofVector, GlobNDOF)
                
                Time_dT = (TimeChunkCount+1)*dT
                exportDispVecData(OutputFileName, ExportCount, Glob_RefDispVector_1, Time_dT)
                ExportCount += 1
                
        
    
    
    TotalTimeStepCount = TimeStepCount-1
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('TotalTimeStepCount', TotalTimeStepCount) 
    
    
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
            TimeList_PlotDispVector = np.zeros([N_TotalPlotDofs, N_TimeChunk+1])
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
            PlotTimeData = {'Plot_T': TimeList_PlotdT, 
                            'Plot_U': TimeList_PlotDispVector, 
                            'Plot_CntStressData': [TimeList_CntGapList, TimeList_CntStressList], 
                            'Plot_KE': TimeList_KE,
                            'Plot_PE': TimeList_PE,
                            #'CntStrData': TimeList_CntStrData, 
                            #'NormGapData': TimeList_NormGapData, 
                            'Plot_Dof': RefPlotDofVec+1, 
                            'qpoint': qpoint}
            np.savez_compressed(PlotFileName+'_PlotData', PlotData = PlotTimeData)
            savemat(PlotFileName+'_PlotData.mat', PlotTimeData)
        
        
            
    
    #Printing Results
    SendReq = Comm.Isend(MP_UnVector, dest=0, tag=Rank)
    if Rank == 0:
        np.set_printoptions(precision=12)
        
        for j in range(N_Workers):    Comm.Recv(MPList_UnVector_1[j], source=j, tag=j)
        GathUnVector = np.hstack(MPList_UnVector_1)
        
        print('\n\n\n')
        Glob_RefDispVector_1 = getGlobDispVec(GathUnVector, GathDofVector, GlobNDOF)
        MaxUn = np.max(Glob_RefDispVector_1)
        I = np.where(Glob_RefDispVector_1==MaxUn)[0][0]
        print('Result', I, Glob_RefDispVector_1[I-5:I+5])
    SendReq.Wait()
    
    
    

