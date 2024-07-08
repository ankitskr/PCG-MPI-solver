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


import matplotlib.pyplot as plt
from datetime import datetime
import sys
from time import time, sleep
import numpy as np
import os.path
import shutil
import scipy.io

import pickle
import zlib

import mpi4py
from mpi4py import MPI

#mpi4py.rc.threads = False

#import logging
#from os.path import abspath
#from Cython_Array.Array_cy import updateLocFint, apply_sum
from scipy.io import savemat
from GeneralFunc import configTimeRecData, getPrincipalStrain, getPrincipalStress, GaussLobattoIntegrationTable, GaussIntegrationTable, readMPIFile, readMPIFile_parallel, writeMPIFile_parallel
from scipy.interpolate import interp1d, interp2d


  
def calcMPFint(MP_Un, RefMeshPart, MP_TimeRecData, DmgCkRef=True):
    
    #Reading Variables
    MP_SubDomainData                     = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector                     = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList              = RefMeshPart['OvrlpLocalDofVecList']
    Flat_ElemLocDof                      = RefMeshPart['Flat_ElemLocDof']
    NCount                               = RefMeshPart['NCount']
    MP_NDOF                              = RefMeshPart['NDOF']
    RefVecCalcMode                       = RefMeshPart['GlobData']['FintCalcMode']
    
    
    #Calculating Local RefVec Vector for Octree cells
    MP_LocRefVec = np.zeros(MP_NDOF, dtype=float)   
    if RefVecCalcMode == 'outbin':
        Flat_ElemRefVec = np.zeros(NCount, dtype=float)
        I=0     
        
        
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemList_LocDofVector = RefTypeGroup['ElemList_LocDofVector']
        ElemList_LocDofVector_Flat = RefTypeGroup['ElemList_LocDofVector_Flat']
    
        Ke                  = RefTypeGroup['ElemStiffMat']
        ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
        if DmgCkRef:    ElemList_DmgCk      = RefTypeGroup['ElemList_DmgCk']
        else:           ElemList_DmgCk      = RefTypeGroup['ElemList_DmgCk_1']
            
        ElemList_Un         = MP_Un[ElemList_LocDofVector]
        ElemList_Un[ElemList_SignVector] *= -1.0      
        ElemList_RefVec     = np.dot(Ke, ElemList_DmgCk*ElemList_Un)
        ElemList_RefVec[ElemList_SignVector] *= -1.0
       
        if RefVecCalcMode == 'inbin':            
            MP_LocRefVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_RefVec.ravel(), minlength=MP_NDOF)        
        elif RefVecCalcMode == 'infor':
            apply_sum(ElemList_LocDofVector, MP_LocRefVec, ElemList_RefVec)            
        elif RefVecCalcMode == 'outbin':
            N = len(ElemList_LocDofVector_Flat)
            Flat_ElemRefVec[I:I+N]=ElemList_RefVec.ravel()
            I += N

    
    if RefVecCalcMode == 'outbin':
        MP_LocRefVec = np.bincount(Flat_ElemLocDof, weights=Flat_ElemRefVec, minlength=MP_NDOF)
    
    
    #Calculating Overlapping RefVec Vectors
    MP_OvrlpRefVecList = []
    MP_InvOvrlpRefVecList = []
    N_NbrMP = len(MP_OvrlpLocalDofVecList)    
    for j in range(N_NbrMP):
        MP_OvrlpRefVec = MP_LocRefVec[MP_OvrlpLocalDofVecList[j]]
        MP_OvrlpRefVecList.append(MP_OvrlpRefVec)   
        
        N_NbrDof_j = len(MP_OvrlpLocalDofVecList[j]);
        MP_InvOvrlpRefVecList.append(np.zeros(N_NbrDof_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping RefVec
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id    = MP_NbrMPIdVector[j]
        SendReq     = Comm.Isend(MP_OvrlpRefVecList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpRefVecList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
     
    #Calculating RefVec
    MP_RefVec = MP_LocRefVec
    for j in range(N_NbrMP):
        MP_RefVec[MP_OvrlpLocalDofVecList[j]] += MP_InvOvrlpRefVecList[j]
                    
     
    return MP_RefVec


def updateOctreeCellDamage_None(MP_Un, RefMeshPart, MP_TimeRecData, MP_DmgTimeRecData):
    
    MP_SubDomainData    = RefMeshPart['SubDomainData']
    MP_TypeGroupList    = MP_SubDomainData['StrucDataList']
    N_Type              = len(MP_TypeGroupList) 
    
    for j in range(N_Type):        
        RefTypeGroup            = MP_TypeGroupList[j]   
        
        ElemList_LocDofVector   = RefTypeGroup['ElemList_LocDofVector']
        StrainMode              = RefTypeGroup['ElemStrainModeMat']
        ElemList_SignVector     = RefTypeGroup['ElemList_SignVector']
        ElemList_Ce             = RefTypeGroup['ElemList_Ce']
        
        ElemList_Un         =   MP_Un[ElemList_LocDofVector]
        ElemList_Un[ElemList_SignVector] *= -1.0        
        ElemList_LocStrainVec  =   np.dot(StrainMode, ElemList_Ce*ElemList_Un) #Strains have been computed in the local coordinate system of each pattern
        RefTypeGroup['ElemList_LocStrainVec'] = ElemList_LocStrainVec #Saving to calculate (damaged) element stress later, if required
        
    

def updateOctreeCellDamage_Bilateral(MP_Un, RefMeshPart, MP_TimeRecData, MP_DmgTimeRecData):

    
    def calcEqvStrain(ElemList_LocStrainVec, v, EqvStrainModel):
        
        ES_Name = EqvStrainModel['Name']
        
        Eps11 = ElemList_LocStrainVec[0, :];
        Eps22 = ElemList_LocStrainVec[1, :];
        Eps33 = ElemList_LocStrainVec[2, :];
        Eps23 = ElemList_LocStrainVec[3, :];
        Eps13 = ElemList_LocStrainVec[4, :];
        Eps12 = ElemList_LocStrainVec[5, :];
        
        if ES_Name == 'VonMises':
            
            k = EqvStrainModel['k']
            
            a1 = 1.0/(2.0*k);
            a2 = (k-1.0)/(1.0-2.0*v);
            a3 = 12.0*k/((1.0+v)*(1.0+v));
            I1 = Eps11 + Eps22 + Eps33
            #I2 = Eps12*Eps12 + Eps13*Eps13 + Eps23*Eps23 - Eps11*Eps22 - Eps22*Eps33 - Eps33*Eps11
            J2 = (1.0/3.0)*(Eps11*Eps11 + Eps22*Eps22 + Eps33*Eps33 - Eps11*Eps22 - Eps11*Eps33 - Eps22*Eps33) + (Eps12*Eps12 + Eps13*Eps13 + Eps23*Eps23)
            ElemList_EqvStrain = a1*(a2*I1 + np.sqrt(a2*a2*I1*I1 + a3*J2));
        
        else: raise Exception
        
        return ElemList_EqvStrain
    
    
    def calcOmega(Kappa, SofteningModel, ep0, epf, Alpha, Beta):
        
        if SofteningModel == 'Linear':
            Omega = (epf/(epf-ep0))*(1.0-ep0/Kappa)
            Omega[Kappa > epf] = 1.0
            
        elif SofteningModel == 'Exponential':
            Omega = 1.0 - (ep0/Kappa)*np.exp((Kappa-ep0)/(ep0-epf))
            
        elif SofteningModel == 'ModExponential':
            Omega = 1.0 - (ep0/Kappa)*(1.0-Alpha+Alpha*np.exp(Beta*(ep0-Kappa))) 
            
        else: raise Exception
        
        Omega[Kappa < ep0] = 0.0
        Omega = np.round(Omega,8)
        Omega[Omega == 1.0] = 1.0 - 1e-9
        
        return Omega
    
    
    updateTime(MP_DmgTimeRecData, 'dT_Elast')
    
    DmgProp             = RefMeshPart['DmgProp']
    MatProp             = RefMeshPart['MatProp']
    MP_SubDomainData    = RefMeshPart['SubDomainData']
    
    EqvStrainModel      = DmgProp['EqvStrainModel']
    DmgType             = DmgProp['Type']
    SofteningModel      = DmgProp['SofteningModel']
    MP_TypeGroupList    = MP_SubDomainData['StrucDataList']
    N_Type              = len(MP_TypeGroupList) 
    
    MP_NElem            = RefMeshPart['NElem']
    MP_EqvStrainList    = np.zeros(MP_NElem)
    
    for j in range(N_Type):        
        RefTypeGroup            = MP_TypeGroupList[j]   
        
        ElemList_LocElemId      = RefTypeGroup['ElemList_LocElemId']
        ElemList_LocDofVector   = RefTypeGroup['ElemList_LocDofVector']
        StrainMode              = RefTypeGroup['ElemStrainModeMat']
        ElemList_SignVector     = RefTypeGroup['ElemList_SignVector']
        ElemList_Ce             = RefTypeGroup['ElemList_Ce']
        PoissonRatio            = RefTypeGroup['PoissonRatio']
        
        ElemList_Un         =   MP_Un[ElemList_LocDofVector]
        ElemList_Un[ElemList_SignVector] *= -1.0        
        ElemList_LocStrainVec  =   np.dot(StrainMode, ElemList_Ce*ElemList_Un) #Strains have been computed in the local coordinate system of each pattern
        RefTypeGroup['ElemList_LocStrainVec'] = ElemList_LocStrainVec #Saving to calculate (damaged) element stress later, if required
        
        ElemList_EqvStrain  = calcEqvStrain(ElemList_LocStrainVec, PoissonRatio, EqvStrainModel)
        MP_EqvStrainList[ElemList_LocElemId] = ElemList_EqvStrain
    
    
    if DmgType == 'NonLocal':
    
        MP_NLSpWeightMatrix         = RefMeshPart['NLSpWeightMatrix']
        MP_OvrlpLocElemIdVecList    = RefMeshPart['OvrlpLocElemIdVecList']
        NL_InvOvrlpLocElemIdVecList = RefMeshPart['NL_InvOvrlpLocElemIdVecList']
        MP_NLElemLocIdVec           = RefMeshPart['NL_ElemLocIdVec']
        NL_NbrMPIdVector            = RefMeshPart['NL_NbrMPIdVector']
        NL_NElem                    = RefMeshPart['NL_NElem']
        
        MP_OvrlpEqvStrainVecList = []
        MP_InvOvrlpEqvStrainVecList = []
        N_NbrMP = len(MP_OvrlpLocElemIdVecList)    
        for j in range(N_NbrMP):
            MP_OvrlpEqvStrainVec = MP_EqvStrainList[MP_OvrlpLocElemIdVecList[j]]
            MP_OvrlpEqvStrainVecList.append(MP_OvrlpEqvStrainVec)   
            
            N_NbrDof_j = len(NL_InvOvrlpLocElemIdVecList[j]);
            MP_InvOvrlpEqvStrainVecList.append(np.zeros(N_NbrDof_j))
        
        updateTime(MP_TimeRecData, 'dT_Calc')
        
        #Send-Recv
        SendReqList = []
        for j in range(N_NbrMP):        
            NbrMP_Id    = NL_NbrMPIdVector[j]
            SendReq     = Comm.Isend(MP_OvrlpEqvStrainVecList[j], dest=NbrMP_Id, tag=Rank)
            SendReqList.append(SendReq)
            
        for j in range(N_NbrMP):    
            NbrMP_Id = NL_NbrMPIdVector[j]
            Comm.Recv(MP_InvOvrlpEqvStrainVecList[j], source=NbrMP_Id, tag=NbrMP_Id)
            
        MPI.Request.Waitall(SendReqList)    
        updateTime(MP_TimeRecData, 'dT_CommWait')
        
        #Calculating NonLocal EqvStrainList
        NL_EqvStrainList = np.zeros(NL_NElem)
        NL_EqvStrainList[MP_NLElemLocIdVec] = MP_EqvStrainList
        for j in range(N_NbrMP):
            NL_EqvStrainList[NL_InvOvrlpLocElemIdVecList[j]] = MP_InvOvrlpEqvStrainVecList[j]
        
        MP_EqvStrainList = MP_NLSpWeightMatrix.dot(NL_EqvStrainList) #Non-local equivalent strain
    

    for j in range(N_Type):        
        RefTypeGroup                = MP_TypeGroupList[j]   
        
        ElemList_LocElemId          = RefTypeGroup['ElemList_LocElemId']
        ElemList_Kappa              = RefTypeGroup['ElemList_Kappa']
        ElemList_Ck                 = RefTypeGroup['ElemList_Ck']
        
        ElemList_ep0                = RefTypeGroup['ElemList_ep0']
        ElemList_epf                = RefTypeGroup['ElemList_epf']
        ElemList_Alpha              = RefTypeGroup['ElemList_Alpha']
        ElemList_Beta               = RefTypeGroup['ElemList_Beta']
        
        ElemList_EqvStrain          = MP_EqvStrainList[ElemList_LocElemId]
        Io = ElemList_EqvStrain > ElemList_Kappa
        ElemList_Kappa[Io] = ElemList_EqvStrain[Io]
        ElemList_Omega = calcOmega(ElemList_Kappa, SofteningModel, ElemList_ep0, ElemList_epf, ElemList_Alpha, ElemList_Beta)
        RefTypeGroup['ElemList_DmgCk_1'] = RefTypeGroup['ElemList_DmgCk']
        RefTypeGroup['ElemList_DmgCk'] = (1.0-ElemList_Omega)*ElemList_Ck
        RefTypeGroup['ElemList_Omega'] = ElemList_Omega
        RefTypeGroup['ElemList_EqvStrain'] = ElemList_EqvStrain
        
    
    updateTime(MP_DmgTimeRecData, 'dT_Dmg')
    
    
    

def updateOctreeCellDamage_Unilateral(MP_Un, RefMeshPart, MP_TimeRecData, MP_DmgTimeRecData):

    def calcTriaxialFactor(ElemList_EffStressVec):
        
        raise Exception("Verify this code with Mazar's failure curve in stress space")
        
        ElemList_EffPStressVec = getPrincipalStress(ElemList_EffStressVec)
        
        ElemList_EffPStressVec_Abs = np.abs(ElemList_EffPStressVec)
        ElemList_EffPStressVec_Macaulay = 0.5*(ElemList_EffPStressVec_Abs + ElemList_EffPStressVec)
        ElemList_r = np.sum(ElemList_EffPStressVec_Macaulay,axis=0)/(np.sum(ElemList_EffPStressVec_Abs,axis=0)+1.0e-16) #Summing the normal stresses
        
        return ElemList_r
    
        
    def calcEqvStrain(ElemList_LocStrainVec, v, EqvStrainModel):
        
        ES_Name = EqvStrainModel['Name']
        
        Eps11 = ElemList_LocStrainVec[0, :]
        Eps22 = ElemList_LocStrainVec[1, :]
        Eps33 = ElemList_LocStrainVec[2, :]
        Eps23 = ElemList_LocStrainVec[3, :]
        Eps13 = ElemList_LocStrainVec[4, :]
        Eps12 = ElemList_LocStrainVec[5, :]
    
        I1 = Eps11 + Eps22 + Eps33
        I2 = Eps12*Eps12 + Eps13*Eps13 + Eps23*Eps23 - Eps11*Eps22 - Eps22*Eps33 - Eps33*Eps11
        J2 = I1*I1 + 3.0*I2
        
        if ES_Name == 'Mazars':
            sqrt_J2 = np.sqrt(J2)
            a1 = 1.0/(1.0-2.0*v)
            a2 = 1.0/(1.0+v)
            
            ElemList_EqvStrain_c = 0.2*(a1*I1 + 6.0*a2*sqrt_J2)
            ElemList_EqvStrain_t = 0.5*(a1*I1 + a2*sqrt_J2)
        
        else:   raise Exception
            
        return ElemList_EqvStrain_c, ElemList_EqvStrain_t
    
    
    def calcOmega(RefTypeGroup, SofteningModel):
        
        if SofteningModel == 'Mazars':

            Kappa_c = RefTypeGroup['ElemList_Kappa_c']
            Kappa_t = RefTypeGroup['ElemList_Kappa_t']
            ep0_c   = RefTypeGroup['ElemList_ep0_c']
            ep0_t   = RefTypeGroup['ElemList_ep0_t']
            A_c     = RefTypeGroup['ElemList_A_c']
            A_t     = RefTypeGroup['ElemList_A_t']
            B_c     = RefTypeGroup['ElemList_B_c']
            B_t     = RefTypeGroup['ElemList_B_t']
            k       = RefTypeGroup['ElemList_k']
            r       = RefTypeGroup['ElemList_r']
            N_Elem   = RefTypeGroup['N_Elem']
            
            Y_tmp   = np.zeros([N_Elem,2], dtype=float)
            
            Y_tmp[:,0]  = ep0_c
            Y_tmp[:,1]  = Kappa_c
            Y_c         = np.max(Y_tmp, axis=1)
            
            Y_tmp[:,0]  = ep0_t 
            Y_tmp[:,1]  = Kappa_t
            Y_t         = np.max(Y_tmp, axis=1)
            
            Y   = r*Y_t + (1.0-r)*Y_c
            Y0  = r*ep0_t + (1.0-r)*ep0_c
            
            rr  = r*r
            Ao  = A_t*(2.0*rr*(1.0-2.0*k) - r*(1.0-4.0*k)) + A_c*(2.0*rr - 3.0*r + 1.0)
            fr  = r**(rr - 2.0*r + 2.0)
            Bo  = fr*B_t + (1.0-fr)*B_c
            
            Omega = 1.0 - (1.0-Ao)*Y0/Y - Ao*np.exp(-Bo*(Y-Y0))
        
        else:   raise Exception
        
        Omega = np.round(Omega,8)
        Omega[Omega == 1.0] = 1.0 - 1e-9
        
        return Omega
    
    
    updateTime(MP_DmgTimeRecData, 'dT_Elast')
    
    DmgProp             = RefMeshPart['DmgProp']
    MatProp             = RefMeshPart['MatProp']
    MP_SubDomainData    = RefMeshPart['SubDomainData']
    
    EqvStrainModel      = DmgProp['EqvStrainModel']
    DmgType             = DmgProp['Type']
    SofteningModel      = DmgProp['SofteningModel']
    MP_TypeGroupList    = MP_SubDomainData['StrucDataList']
    N_Type              = len(MP_TypeGroupList) 
    
    MP_NElem            = RefMeshPart['NElem']
    MP_EqvStrainList    = np.zeros([MP_NElem, 2], dtype=float)
    
    #Computing Triaxial Factor and  Equivalent Strains
    for j in range(N_Type):        
        RefTypeGroup            = MP_TypeGroupList[j]   
        
        ElemList_LocElemId      = RefTypeGroup['ElemList_LocElemId']
        ElemList_LocDofVector   = RefTypeGroup['ElemList_LocDofVector']
        StrainMode              = RefTypeGroup['ElemStrainModeMat']
        ElemList_SignVector     = RefTypeGroup['ElemList_SignVector']
        ElemList_Ce             = RefTypeGroup['ElemList_Ce']
        ElemList_E              = RefTypeGroup['ElemList_E']
        PoissonRatio            = RefTypeGroup['PoissonRatio']
        ElasticityMat           = RefTypeGroup['ElasticityMat']
        
        ElemList_Un         =   MP_Un[ElemList_LocDofVector]
        ElemList_Un[ElemList_SignVector] *= -1.0        
        ElemList_LocStrainVec  =   np.dot(StrainMode, ElemList_Ce*ElemList_Un) #Strains have been computed in the local coordinate system of each pattern
        RefTypeGroup['ElemList_LocStrainVec'] = ElemList_LocStrainVec #Saving to calculate (damaged) element stress later, if required
        
        #Triaxial factor
        ElemList_EffStressVec = ElemList_E*np.dot(ElasticityMat, ElemList_LocStrainVec)
        RefTypeGroup['ElemList_r'] = calcTriaxialFactor(ElemList_EffStressVec)
        
        #Equivalent strains
        ElemList_EqvStrain_c, ElemList_EqvStrain_t  = calcEqvStrain(ElemList_LocStrainVec, PoissonRatio, EqvStrainModel)
        MP_EqvStrainList[ElemList_LocElemId, 0] = ElemList_EqvStrain_c
        MP_EqvStrainList[ElemList_LocElemId, 1] = ElemList_EqvStrain_t
        
    #Computing Non-local strains
    if DmgType == 'NonLocal':
    
        MP_NLSpWeightMatrix         = RefMeshPart['NLSpWeightMatrix']
        MP_OvrlpLocElemIdVecList    = RefMeshPart['OvrlpLocElemIdVecList']
        NL_InvOvrlpLocElemIdVecList = RefMeshPart['NL_InvOvrlpLocElemIdVecList']
        MP_NLElemLocIdVec           = RefMeshPart['NL_ElemLocIdVec']
        NL_NbrMPIdVector            = RefMeshPart['NL_NbrMPIdVector']
        NL_NElem                    = RefMeshPart['NL_NElem']
        
        MP_OvrlpEqvStrainVecList = []
        MP_InvOvrlpEqvStrainVecList = []
        N_NbrMP = len(MP_OvrlpLocElemIdVecList)    
        for j in range(N_NbrMP):
            MP_OvrlpEqvStrainVec = MP_EqvStrainList[MP_OvrlpLocElemIdVecList[j]]
            MP_OvrlpEqvStrainVecList.append(MP_OvrlpEqvStrainVec)   
            
            N_NbrDof_j = len(NL_InvOvrlpLocElemIdVecList[j]);
            MP_InvOvrlpEqvStrainVecList.append(np.zeros([N_NbrDof_j,2], dtype=float))
        
        updateTime(MP_TimeRecData, 'dT_Calc')
        
        #Send-Recv
        SendReqList = []
        for j in range(N_NbrMP):        
            NbrMP_Id    = NL_NbrMPIdVector[j]
            SendReq     = Comm.Isend(MP_OvrlpEqvStrainVecList[j], dest=NbrMP_Id, tag=Rank)
            SendReqList.append(SendReq)
            
        for j in range(N_NbrMP):    
            NbrMP_Id = NL_NbrMPIdVector[j]
            Comm.Recv(MP_InvOvrlpEqvStrainVecList[j], source=NbrMP_Id, tag=NbrMP_Id)
            
        MPI.Request.Waitall(SendReqList)    
        updateTime(MP_TimeRecData, 'dT_CommWait')
        
        #Calculating NonLocal EqvStrainList
        NL_EqvStrainList = np.zeros([NL_NElem, 2], dtype=float)
        NL_EqvStrainList[MP_NLElemLocIdVec, :] = MP_EqvStrainList
        for j in range(N_NbrMP):
            NL_EqvStrainList[NL_InvOvrlpLocElemIdVecList[j], :] = MP_InvOvrlpEqvStrainVecList[j]
        
        MP_EqvStrainList = MP_NLSpWeightMatrix.dot(NL_EqvStrainList)
    
    
    #Computing Damage
    for j in range(N_Type):        
        RefTypeGroup                = MP_TypeGroupList[j]   
        
        ElemList_LocElemId          = RefTypeGroup['ElemList_LocElemId']
        ElemList_Kappa_c            = RefTypeGroup['ElemList_Kappa_c']
        ElemList_Kappa_t            = RefTypeGroup['ElemList_Kappa_t']
        ElemList_Ck                 = RefTypeGroup['ElemList_Ck']
        
        #Compression
        ElemList_EqvStrain_c        = MP_EqvStrainList[ElemList_LocElemId, 0]
        I_c = ElemList_EqvStrain_c > ElemList_Kappa_c
        ElemList_Kappa_c[I_c] = ElemList_EqvStrain_c[I_c]
        
        #Tension
        ElemList_EqvStrain_t        = MP_EqvStrainList[ElemList_LocElemId, 1]
        I_t = ElemList_EqvStrain_t > ElemList_Kappa_t
        ElemList_Kappa_t[I_t] = ElemList_EqvStrain_t[I_t]
    
        ElemList_Omega = calcOmega(RefTypeGroup, SofteningModel)
        RefTypeGroup['ElemList_Omega'] = ElemList_Omega
        RefTypeGroup['ElemList_DmgCk'] = (1.0-ElemList_Omega)*ElemList_Ck
        
    
    updateTime(MP_DmgTimeRecData, 'dT_Dmg')
    



def calcMPDispVec(MP_Un_2, MP_Un_1, MP_FintVec, MP_Fext, RefMeshPart):
    
    Damping_Alpha           = 0.0
    dt                      = RefMeshPart['GlobData']['dt']
    MP_LocFixedDof          = RefMeshPart['LocFixedDof']
    MP_InvLumpedMassVector  = RefMeshPart['InvDiagM']
    MP_Vd                   = RefMeshPart['Vd']
    
    DampTerm = 0.5*Damping_Alpha*dt
    MP_Un = (1.0/(1.0+DampTerm))*(2.0*MP_Un_1 - (1-DampTerm)*MP_Un_2 + dt*dt*MP_InvLumpedMassVector*(MP_Fext - MP_FintVec))
    MP_Un[MP_LocFixedDof] = MP_Un_2[MP_LocFixedDof] +2*dt*MP_Vd[MP_LocFixedDof]
    
    return MP_Un




def calcEnergy(RefMeshPart, MP_Un_1, MP_FintVec_1):
    
    #Elastic Strain Energy
    MP_PE_1 = 0.5*np.dot(MP_Un_1, MP_FintVec_1*RefMeshPart['DofWeightVector'])
    PE_1 = MPI_SUM(MP_PE_1, MP_TimeRecData)
    RefMeshPart['PEList'].append(PE_1)
    
    #Dissipation 
    MP_FintVec_1p = calcMPFint(MP_Un_1, RefMeshPart, MP_TimeRecData, DmgCkRef=False)
    MP_PE_1p = 0.5*np.dot(MP_Un_1, MP_FintVec_1p*RefMeshPart['DofWeightVector'])
    PE_1p = MPI_SUM(MP_PE_1p, MP_TimeRecData)
    RefMeshPart['DE'] += (PE_1p - PE_1) 
    RefMeshPart['DEList'].append(RefMeshPart['DE']) 
    



def MPI_SUM(MP_RefVar, MP_TimeRecData):
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    Glob_RefVar = Comm.allreduce(MP_RefVar, op=MPI.SUM)
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    return Glob_RefVar
   


def updateTime(RefTimeRecData, RefKey, TimeStepCount=None):
    
    if RefKey == 'UpdateList':
        RefTimeRecData['TimeStepCountList'].append(TimeStepCount)
        RefTimeRecData['dT_CalcList'].append(RefTimeRecData['dT_Calc'])
        RefTimeRecData['dT_CommWaitList'].append(RefTimeRecData['dT_CommWait'])
    else:  
        t1 = time()
        RefTimeRecData[RefKey] += t1 - RefTimeRecData['t0']
        RefTimeRecData['t0'] = t1
        


def plotDispVecData(PlotFileName, TimeList, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList, TimeList_PlotDispVector.T)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    
  
def getNodalScalarVar(RefMeshPart, MP_TimeRecData, Ref):
    
    MP_SubDomainData            = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector            = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalNodeIdVecList  = RefMeshPart['OvrlpLocalNodeIdVecList']
    MP_NNode                    = RefMeshPart['NNode']
    
    MP_TypeGroupList            = MP_SubDomainData['StrucDataList']
    N_Type                      = len(MP_TypeGroupList) 
    
    MP_VarSum                = np.zeros(MP_NNode)
    MP_VarCount              = np.zeros(MP_NNode)
    
    for j in range(N_Type):        
        RefTypeGroup                = MP_TypeGroupList[j]    
        NNodes_ElemType             = RefTypeGroup['NNodes']
        ElemList_LocNodeIdVector    = RefTypeGroup['ElemList_LocNodeIdVector']
        ElemList_Omega              = RefTypeGroup['ElemList_Omega'] #Dmg
        N_Elem                      = len(ElemList_Omega)
        
        if Ref=='Dmg':
            ElemList_RefVar = ElemList_Omega
        elif Ref=='EqvStrain':
            ElemList_RefVar = RefTypeGroup['ElemList_EqvStrain']
        
        ElemList_VarVec     = ElemList_RefVar*np.ones(NNodes_ElemType)[np.newaxis].T
        MP_VarSum += np.bincount(ElemList_LocNodeIdVector.ravel(), weights=ElemList_VarVec.ravel(), minlength=MP_NNode)
        
        FlatElemList_VarCount     = np.ones(NNodes_ElemType*N_Elem)
        MP_VarCount += np.bincount(ElemList_LocNodeIdVector.ravel(), weights=FlatElemList_VarCount, minlength=MP_NNode)
        
    
    #Calculating Overlapping Fint Vectors
    N_NbrMP = len(MP_NbrMPIdVector)
            
    MP_OvrlpVarList = []
    MP_InvOvrlpVarList = []
    for j in range(N_NbrMP):
        MP_OvrlpVar = np.hstack([MP_VarSum[MP_OvrlpLocalNodeIdVecList[j]], MP_VarCount[MP_OvrlpLocalNodeIdVecList[j]]])
        MP_OvrlpVarList.append(MP_OvrlpVar)   
        
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_InvOvrlpVarList.append(np.zeros(2*N_NbrNode_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping Var
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id        = MP_NbrMPIdVector[j]
        SendReq         = Comm.Isend(MP_OvrlpVarList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id        = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpVarList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    #Updating Var at the Meshpart boundary
    for j in range(N_NbrMP): 
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_VarSum[MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpVarList[j][:N_NbrNode_j]
        MP_VarCount[MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpVarList[j][N_NbrNode_j:] 
    
    MP_Var = MP_VarSum/(MP_VarCount+1e-15) #1e-15 is added for VarCount = 0
    
    
    return MP_Var
    

  
def getNodalPS(RefMeshPart, MP_TimeRecData, Ref):

    MP_SubDomainData            = RefMeshPart['SubDomainData']
    MP_NbrMPIdVector            = RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalNodeIdVecList  = RefMeshPart['OvrlpLocalNodeIdVecList']
    MP_NNode                    = RefMeshPart['NNode']
    
    MP_TypeGroupList            = MP_SubDomainData['StrucDataList']
    N_Type                      = len(MP_TypeGroupList) 
    
    MP_PSSum                   = np.zeros([3,MP_NNode])
    MP_PSCount                 = np.zeros(MP_NNode)
    
    for j in range(N_Type):        
        RefTypeGroup                = MP_TypeGroupList[j]    
        NNodes_ElemType             = RefTypeGroup['NNodes']
        ElemList_LocNodeIdVector    = RefTypeGroup['ElemList_LocNodeIdVector']
        ElemList_Omega              = RefTypeGroup['ElemList_Omega'] #Dmg
        ElasticityMat               = RefTypeGroup['ElasticityMat']
        ElemList_E                  = RefTypeGroup['ElemList_E']
        ElemList_LocStrainVec       = RefTypeGroup['ElemList_LocStrainVec']
        
        if Ref=='Stress':       
            ElemList_LocStressVec   = (1.0-ElemList_Omega)*ElemList_E*np.dot(ElasticityMat, ElemList_LocStrainVec)
            ElemList_PSVec          = getPrincipalStress(ElemList_LocStressVec) 
        
        elif Ref=='Strain':     
            ElemList_PSVec          = getPrincipalStrain(ElemList_LocStrainVec) 
        
        else:   raise Exception
        
        #Calculating Nodal PS
        ElemList_LocNodeIdVector_Flat  = ElemList_LocNodeIdVector.ravel()
        for i in range(3):
            ElemList_PS_i      = ElemList_PSVec[i,:]*np.ones(NNodes_ElemType)[np.newaxis].T
            MP_PSSum[i,:] += np.bincount(ElemList_LocNodeIdVector_Flat, weights=ElemList_PS_i.ravel(), minlength=MP_NNode)
            
        N_Elem                   = len(ElemList_Omega)
        FlatElemList_PSCount     = np.ones(NNodes_ElemType*N_Elem)
        MP_PSCount += np.bincount(ElemList_LocNodeIdVector_Flat, weights=FlatElemList_PSCount, minlength=MP_NNode)
        
    
    #Calculating Overlapping Fint Vectors
    N_NbrMP = len(MP_NbrMPIdVector)
            
    MP_OvrlpPSList = []
    MP_InvOvrlpPSList = []
    for j in range(N_NbrMP):
        MP_OvrlpPS = np.hstack([MP_PSSum[:, MP_OvrlpLocalNodeIdVecList[j]].ravel(), MP_PSCount[MP_OvrlpLocalNodeIdVecList[j]]])
        MP_OvrlpPSList.append(MP_OvrlpPS)
        
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_InvOvrlpPSList.append(np.zeros(4*N_NbrNode_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping PS
    SendReqList = []
    for j in range(N_NbrMP):        
        NbrMP_Id        = MP_NbrMPIdVector[j]
        SendReq         = Comm.Isend(MP_OvrlpPSList[j], dest=NbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_NbrMP):    
        NbrMP_Id        = MP_NbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpPSList[j], source=NbrMP_Id, tag=NbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    #Updating PS at the Meshpart boundary
    for j in range(N_NbrMP): 
        N_NbrNode_j = len(MP_OvrlpLocalNodeIdVecList[j]);
        MP_PSSum[0, MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][:N_NbrNode_j]
        MP_PSSum[1, MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][N_NbrNode_j:2*N_NbrNode_j]
        MP_PSSum[2, MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][2*N_NbrNode_j:3*N_NbrNode_j]
        MP_PSCount[MP_OvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpPSList[j][3*N_NbrNode_j:] 
    
    MP_PS = MP_PSSum/(MP_PSCount+1e-15) #1e-15 is added for PSCount = 0
    
    
    return MP_PS
    



def exportFiles(RefMeshPart, ResVecPath, MP_Un, T_i, TimeList_T, MP_TimeRecData):
    
    ExportCount                 = RefMeshPart['ExportCount']
    GlobData                    = RefMeshPart['GlobData']
    MP_DofWeightVector_Export   = RefMeshPart['DofWeightVector_Export']
    MP_NodeWeightVector_Export  = RefMeshPart['NodeWeightVector_Export']
    
    if 'U' in GlobData['ExportVars']:
        writeMPIFile_parallel(ResVecPath + 'U_' + str(ExportCount), MP_Un[MP_DofWeightVector_Export], Comm)
    
    if 'D' in GlobData['ExportVars']:
        MP_Dmg = getNodalScalarVar(RefMeshPart, MP_TimeRecData, 'Dmg')
        writeMPIFile_parallel(ResVecPath + 'D_' + str(ExportCount), MP_Dmg[MP_NodeWeightVector_Export], Comm)
    
    if 'ES' in GlobData['ExportVars']:
        MP_ES = getNodalScalarVar(RefMeshPart, MP_TimeRecData, 'EqvStrain')
        writeMPIFile_parallel(ResVecPath + 'ES_' + str(ExportCount), MP_ES[MP_NodeWeightVector_Export], Comm)
    
    if 'PE' in GlobData['ExportVars']:
        MP_PE = getNodalPS(RefMeshPart, MP_TimeRecData, 'Strain')
        if 'PE1' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PE1_' + str(ExportCount), MP_PE[0, MP_NodeWeightVector_Export], Comm)
        if 'PE2' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PE2_' + str(ExportCount), MP_PE[1, MP_NodeWeightVector_Export], Comm)
        if 'PE3' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PE3_' + str(ExportCount), MP_PE[2, MP_NodeWeightVector_Export], Comm)
    
    if 'PS' in GlobData['ExportVars']:
        MP_PS = getNodalPS(RefMeshPart, MP_TimeRecData, 'Stress')
        if 'PS1' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PS1_' + str(ExportCount), MP_PS[0, MP_NodeWeightVector_Export], Comm)
        if 'PS2' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PS2_' + str(ExportCount), MP_PS[1, MP_NodeWeightVector_Export], Comm)
        if 'PS3' in GlobData['ExportVars']:     writeMPIFile_parallel(ResVecPath + 'PS3_' + str(ExportCount), MP_PS[2, MP_NodeWeightVector_Export], Comm)
    
    if Rank==0: 
        TimeList_T.append(T_i)
        np.save(ResVecPath + 'Time_T', TimeList_T)
     
    RefMeshPart['ExportCount'] += 1

    
    
    
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
    
    MP_DmgTimeRecData = {'dT_FileRead':   0.0,
                         'dT_Elast':      0.0,
                         'dT_Dmg':        0.0,
                         't0':            time()}
    
    
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
    ResVecPath = ResultPath + 'ResVecData/'
    if Rank==0:
        if os.path.exists(ResultPath):
            try:        os.rename(ResultPath, ResultPath[:-1]+'_'+ datetime.now().strftime('%d%m%Y_%H%M%S'))
            except:     raise Exception('Result Path in use!')
        os.makedirs(PlotPath)        
        os.makedirs(ResVecPath)
    Comm.barrier()
    
    PlotFileName = PlotPath + ModelName
    
    t1_ = time()
    #A small sleep to avoid hanging
    sleep(Rank*1e-4)
    
    #Reading Model Data Files
    PyDataPath = ScratchPath + 'ModelData/MPI/'
    Data_FileName = PyDataPath + str(N_MshPrt)
    
    """
    Data_Buffer = readMPIFile_parallel(Data_FileName, Comm)
    """
    
    #"""
    metadat = np.load(Data_FileName+'_metadat.npy', allow_pickle=True).item()
    Nf = metadat['NfData'][Rank]; DType = metadat['DTypeData'][Rank]
    MetaData = [Nf, DType]
    Data_Buffer = readMPIFile(Data_FileName+ '_' + str(Rank), MetaData)
    #"""
    
    RefMeshPart = pickle.loads(zlib.decompress(Data_Buffer.tobytes()))
    
    Comm.barrier() 
    if Rank==0: print('Time (sec) taken to read files: ', np.round(time()-t1_,2))
    
    #Reading Global data file
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    GlobSettingsFile = MatDataPath + 'GlobSettings.mat'
    GlobSettings = scipy.io.loadmat(GlobSettingsFile)
    
    MP_NDOF                             = RefMeshPart['NDOF']
    MP_NNode                            = RefMeshPart['NNode']
    MP_RefLoadVector                    = RefMeshPart['RefLoadVector']
    MPList_RefPlotDofIndicesList        = RefMeshPart['MPList_RefPlotDofIndicesList']
    MP_RefPlotData                      = RefMeshPart['RefPlotData']
    NCount                              = RefMeshPart['NCount']
    N_NbrDof                            = RefMeshPart['N_NbrDof']
    MP_DofWeightVector                  = RefMeshPart['DofWeightVector']
    MP_NodeWeightVector                 = RefMeshPart['NodeWeightVector']
    MP_DofVector                        = RefMeshPart['DofVector']
    MP_NodeIdVector                     = RefMeshPart['NodeIdVector']
    
    #DirchletBCExists                    = RefMeshPart['DirchletBCExists']
    
    RefMaxTimeStepCount                 = GlobSettings['RefMaxTimeStepCount'][0][0]
    TimeStepDelta                       = GlobSettings['TimeStepDelta'][0]
    
    ExportFrms                          = np.array(GlobSettings['ExportFrms'], dtype=int) 
    if len(ExportFrms)>0:    
        ExportFrms                      = ExportFrms[0] - 1
    
    ExportKeyFrm                        = int(GlobSettings['ExportFrmRate'][0][0])
    # ExportKeyFrm                        = int(RefMaxTimeStepCount/ExportFrmRate)
    
    
    PlotFlag                            = GlobSettings['PlotFlag'][0][0]   
    ExportFlag                          = GlobSettings['ExportFlag'][0][0]  
    FintCalcMode                        = GlobSettings['FintCalcMode'][0] 
    
    GlobData                            = RefMeshPart['GlobData']
    GlobData['FintCalcMode']            = GlobSettings['FintCalcMode'][0]
    GlobData['ExportVars']              = GlobSettings['ExportVars'][0]
    dt                                  = GlobData['dt']
    
    EnergyFlag                          = 1
    if SpeedTestFlag==1:  
        PlotFlag = 0; ExportFlag = 0; FintCalcMode = 'outbin'; EnergyFlag = 0;
    
    if not FintCalcMode in ['inbin', 'infor', 'outbin']:  
        raise ValueError("FintCalcMode must be 'inbin', 'infor' or 'outbin'")
    
        
    eps = np.finfo(float).eps
    
    if Rank==0:
        print(ExportFlag, FintCalcMode)
        
        
    #A small sleep to avoid hanging
    sleep(Rank*1e-4)
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    updateTime(MP_DmgTimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    
    #Initializing Variables
    if not RefMeshPart['DmgProp']['Type'] == 'None':
        Unilateral = RefMeshPart['DmgProp']['EqvStrainModel']['Unilateral']
        if Unilateral == 1:     updateOctreeCellDamage = updateOctreeCellDamage_Unilateral
        else:                   updateOctreeCellDamage = updateOctreeCellDamage_Bilateral
    else:   updateOctreeCellDamage = updateOctreeCellDamage_None
    
    
    MP_Un_1 = (1e-200)*np.random.rand(MP_NDOF)
    MP_Un = (1e-200)*np.random.rand(MP_NDOF)
    
    TimeList = [i*dt for i in range(RefMaxTimeStepCount)]
    TimeStepCount = 0
    TimeList_T = []
    
    RefMeshPart['PEList'] = []
    RefMeshPart['DE'] = 0.0
    RefMeshPart['DEList'] = []
    
    if PlotFlag == 1:
        MP_PlotLocalDofVec  = MP_RefPlotData['LocalDofVec']
        MP_PlotNDofs       = len(MP_PlotLocalDofVec)
        RefPlotDofVec       = MP_RefPlotData['RefPlotDofVec']
        qpoint              = MP_RefPlotData['qpoint']
        TestPlotFlag        = MP_RefPlotData['TestPlotFlag']
        
        if MP_PlotNDofs > 0:
            MP_PlotDispVector               = np.zeros([MP_PlotNDofs, RefMaxTimeStepCount])
            MP_PlotDispVector[:,0]          = MP_Un[MP_PlotLocalDofVec]
            
            MP_PlotLoadVector               = np.zeros(RefMaxTimeStepCount)
            
        else:   
            MP_PlotDispVector               = []
            MP_PlotLoadVector               = []
     
    
    if ExportFlag == 1:
        
        #Computing vectors for MPI-based export 
        MP_DofWeightVector_Export = MP_DofWeightVector.astype(bool)
        MP_NodeWeightVector_Export = MP_NodeWeightVector.astype(bool)
        
        writeMPIFile_parallel(ResVecPath+'Dof', MP_DofVector[MP_DofWeightVector_Export], Comm)
        writeMPIFile_parallel(ResVecPath+'NodeId', MP_NodeIdVector[MP_NodeWeightVector_Export], Comm)

        #Exporting data
        RefMeshPart['ExportCount'] = 0
        RefMeshPart['DofWeightVector_Export'] = MP_DofWeightVector_Export
        RefMeshPart['NodeWeightVector_Export'] = MP_NodeWeightVector_Export
        
        updateOctreeCellDamage(MP_Un, RefMeshPart, MP_TimeRecData, MP_DmgTimeRecData)
        
        ExportNow = False
        if ExportKeyFrm>0 or 0 in ExportFrms:
            ExportNow = True
        
        if ExportNow:
            exportFiles(RefMeshPart, ResVecPath, MP_Un, TimeList[TimeStepCount], TimeList_T, MP_TimeRecData)
            
    
    if RefMaxTimeStepCount>1:   Delta = 1.0/(RefMaxTimeStepCount-1)
    else:                       Delta = 0.0
    
    for TimeStepCount in range(1, RefMaxTimeStepCount):
        
        if Rank==0:
            if TimeStepCount%500==0:    
                print('TimeStepCount', TimeStepCount)
                print('Time', np.round([MP_TimeRecData['dT_Calc'], MP_TimeRecData['dT_CommWait']],1))
        
            
        MP_Un_2 = MP_Un_1
        MP_Un_1 = MP_Un
        
        updateOctreeCellDamage(MP_Un_1, RefMeshPart, MP_TimeRecData, MP_DmgTimeRecData)
        
        DeltaLambda_1 = TimeStepDelta[TimeStepCount-1]
        MP_Fext_1 = MP_RefLoadVector*DeltaLambda_1
        MP_FintVec_1 = calcMPFint(MP_Un_1, RefMeshPart, MP_TimeRecData)
        MP_Un = calcMPDispVec(MP_Un_2, MP_Un_1, MP_FintVec_1, MP_Fext_1, RefMeshPart)
        
        
        if EnergyFlag == 1:
            calcEnergy(RefMeshPart, MP_Un_1, MP_FintVec_1)
            
        
        if PlotFlag == 1:
            #if Rank==0: print(TimeStepCount)
            
            # MP_Rc   = calcMPFint(MP_Un, RefMeshPart, MP_TimeRecData) - MP_Fi
            # MP_Rc   = MP_Rc*MP_WeightVector
            # MP_Load =  np.sum(MP_Rc[MP_PlotLocalDofVec])
            # Load    = MPI_SUM(MP_Load, MP_TimeRecData)
            # EStress = Load/(1.0*1.0)
            EStress = 0.0
            
            if MP_PlotNDofs>0:
                MP_PlotLoadVector[TimeStepCount] = EStress
                MP_PlotDispVector[:,TimeStepCount] = MP_Un[MP_PlotLocalDofVec]
                
            
        if ExportFlag == 1:        
            
            ExportNow = False
            if ExportKeyFrm>0:
                if (TimeStepCount)%ExportKeyFrm==0: ExportNow = True
               
            if TimeStepCount in ExportFrms:
                ExportNow = True
            
            if ExportNow:
                exportFiles(RefMeshPart, ResVecPath, MP_Un, TimeList[TimeStepCount], TimeList_T, MP_TimeRecData)
                
        
        
      
    updateTime(MP_DmgTimeRecData, 'dT_Elast')  
    
    t0_End = time()
    
    if Rank==0:    
        print('Analysis Finished Sucessfully..')
        print('TotalTimeStepCount', RefMaxTimeStepCount) 
    
    
    #Saving CPU Time
    MP_TimeRecData['dT_Total_Verify'] = t0_End - t0_Start 
    MP_TimeRecData['t0_Start'] = t0_Start
    MP_TimeRecData['t0_End'] = t0_End
    MP_TimeRecData['dT_Elast'] = MP_DmgTimeRecData['dT_Elast']
    MP_TimeRecData['dT_Dmg'] = MP_DmgTimeRecData['dT_Dmg']
    
    MP_TimeRecData['MP_NCount'] = NCount
    MP_TimeRecData['MP_NDOF'] = MP_NDOF
    MP_TimeRecData['N_NbrDof'] = N_NbrDof    
    
    
    MPList_TimeRecData = Comm.gather(MP_TimeRecData, root=0)
    
    if PlotFlag == 1:   
        MPList_PlotDispVector = Comm.gather(MP_PlotDispVector, root=0)
        MPList_PlotLoadVector = Comm.gather(MP_PlotLoadVector, root=0)
        
    
    if Rank == 0:        
        TimeData = configTimeRecData(MPList_TimeRecData, ExportLoadUnbalanceData=True, Dmg=True)
        TimeData['PBS_JobId'] = PBS_JobId
        
        TimeDataFileName = PlotFileName + '_MP' +  str(N_MshPrt) + '_' + FintCalcMode + '_TimeData'
        np.savez_compressed(TimeDataFileName, TimeData = TimeData)
        savemat(TimeDataFileName +'.mat', TimeData)
        
        #Exporting Plots        
        if PlotFlag == 1:
            N_TotalPlotDofs     = len(RefPlotDofVec)
            TimeList_PlotDispVector = np.zeros([N_TotalPlotDofs, RefMaxTimeStepCount])
            TimeList_PlotLoadVector = np.zeros(RefMaxTimeStepCount)
            
            for i in range(N_Workers):
                MP_PlotDispVector_i = MPList_PlotDispVector[i]
                MP_PlotLoadVector_i = MPList_PlotLoadVector[i]
                
                RefPlotDofIndices_i = MPList_RefPlotDofIndicesList[i]
                N_PlotDofs_i = len(MP_PlotDispVector_i)
                if N_PlotDofs_i>0:
                    for j in range(N_PlotDofs_i):
                        TimeList_PlotDispVector[RefPlotDofIndices_i[j],:] = MP_PlotDispVector_i[j]
                
                if len(MP_PlotLoadVector_i)>0:
                    TimeList_PlotLoadVector = MP_PlotLoadVector_i
                        
                
            #Saving Data File
            PlotTimeData = {'Plot_T':   TimeList, 
                            'Plot_U':   TimeList_PlotDispVector, 
                            'Plot_L':   TimeList_PlotLoadVector,
                            'Plot_Dof': RefPlotDofVec+1, 
                            'qpoint':   qpoint,
                            'PEList':   RefMeshPart['PEList']}
            np.savez_compressed(PlotFileName+'_PlotData', PlotData = PlotTimeData)
            savemat(PlotFileName+'_PlotData.mat', PlotTimeData)
        
        
        if calcEnergy:
            EnergyData = {'Plot_T':   TimeList[:-1], 
                          'PEList':   RefMeshPart['PEList'], 
                          'DEList':   RefMeshPart['DEList']}
            np.savez_compressed(PlotFileName+'_EnergyData', EnergyData = EnergyData)
            savemat(PlotFileName+'_EnergyData.mat', EnergyData)
        
        
        
            
