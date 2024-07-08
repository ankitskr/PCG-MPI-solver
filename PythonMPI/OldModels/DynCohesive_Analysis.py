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
from GeneralFunc import configTimeRecData, GaussLobattoIntegrationTable, GaussIntegrationTable
from scipy.interpolate import interp1d, interp2d





def getIntfcMatData():


    kss_ij = np.array([[1,  0,   0],
                       [0,  0,   0],
                       [0,  0,   0]], dtype=float)
    
    kpp_ij = np.array([[0,  0,   0],
                       [0,  1,   0],
                       [0,  0,   0]], dtype=float)
    
    knn_ij = np.array([[0,  0,   0],
                       [0,  0,   0],
                       [0,  0,   1]], dtype=float)
    
    ksn_ij = np.array([[0,  0,   1],
                       [0,  0,   0],
                       [0,  0,   0]], dtype=float)
    
    kpn_ij = np.array([[0,  0,   0],
                       [0,  0,   1],
                       [0,  0,   0]], dtype=float)
    
    css_ij = np.array([[1],
                       [0],
                       [0]], dtype=float)
    
    cpp_ij = np.array([[0],
                       [1],
                       [0]], dtype=float)
    
    
    CntStiffMatData = {}
    CntStiffMatData['kss'] = kss_ij
    CntStiffMatData['kpp'] = kpp_ij
    CntStiffMatData['knn'] = knn_ij
    CntStiffMatData['ksn'] = ksn_ij
    CntStiffMatData['kpn'] = kpn_ij
    CntStiffMatData['Kss'] = []
    CntStiffMatData['Kpp'] = []
    CntStiffMatData['Knn'] = []
    CntStiffMatData['Ksn'] = []
    CntStiffMatData['Kpn'] = []
    
    CntCohMatData = {}
    CntCohMatData['css'] = css_ij.T[0]
    CntCohMatData['cpp'] = cpp_ij.T[0]
    CntCohMatData['Css'] = []
    CntCohMatData['Cpp'] = []
     
    return CntStiffMatData, CntCohMatData

    
    
    
def initPenaltyData(MP_SubDomainData, IntegrationOrder, dt):
    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId in [-2, -1]: #Interface Elements
            IntfcElemList = RefTypeGroup['ElemList_IntfcElem']
            
            PenaltyData = {}
            RefTypeGroup['PenaltyData'] = PenaltyData
            
            RefIntfcElem            = IntfcElemList[0]
            N_IntfcElem             = len(IntfcElemList)
            IntfcArea               = 1.0
        
            CntStiffMatData, CntCohMatData = getIntfcMatData()
        
            kss_ij = CntStiffMatData['kss']
            kpp_ij = CntStiffMatData['kpp']
            knn_ij = CntStiffMatData['knn']
            ksn_ij = CntStiffMatData['ksn']
            kpn_ij = CntStiffMatData['kpn']
            css_ij = CntCohMatData['css']
            cpp_ij = CntCohMatData['cpp']
            
            Mij_List = []
            
            if ElemTypeId == -2:
                
                #https://www.lncc.br/~alm/public/integexp.pdf
                #http://people.ucalgary.ca/~aknigh/fea/fea/triangles/num_ex.html
                #http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
                #GAUSSIAN QUADRATURE FORMULAS FOR TRIANGLES. G. R. COWPER
                if IntegrationOrder == 1:
                    wi = [1.0]
                    Ni0 = [1.0/3.0]
                    Ni1 = [1.0/3.0]
                    N_IntegrationPoints_2D = 1
                
                elif IntegrationOrder == 2:
                    wi =  [1.0/3.0, 1.0/3.0, 1.0/3.0]
                    Ni0 = [1.0/6.0, 2.0/3.0, 1.0/6.0]
                    Ni1 = [1.0/6.0, 1.0/6.0, 2.0/3.0]
                    N_IntegrationPoints_2D = 3
                
                else:   raise Exception ('Higher order integration not implemented')
                
                
                for p in range(N_IntegrationPoints_2D):
                
                    s0 = Ni0[p]
                    s1 = Ni1[p]
                    
                    N1 = 1.0 - s0 - s1
                    N2 = s0
                    N3 = s1
                    
                    I = np.eye(3)
                    N = np.hstack([N1*I, N2*I, N3*I])
                    Mij = np.hstack([-N, N])
                    
                    Kss_ij = IntfcArea*wi[p]*np.dot(np.dot(Mij.T, kss_ij), Mij)
                    Kpp_ij = IntfcArea*wi[p]*np.dot(np.dot(Mij.T, kpp_ij), Mij)
                    Knn_ij = IntfcArea*wi[p]*np.dot(np.dot(Mij.T, knn_ij), Mij)
                    Ksn_ij = IntfcArea*wi[p]*np.dot(np.dot(Mij.T, ksn_ij), Mij)
                    Kpn_ij = IntfcArea*wi[p]*np.dot(np.dot(Mij.T, kpn_ij), Mij)
                    
                    Css_ij = IntfcArea*wi[p]*np.dot(Mij.T, css_ij).T[0]
                    Cpp_ij = IntfcArea*wi[p]*np.dot(Mij.T, cpp_ij).T[0]
                    
                    CntStiffMatData['Kss'].append(Kss_ij)
                    CntStiffMatData['Kpp'].append(Kpp_ij)
                    CntStiffMatData['Knn'].append(Knn_ij)
                    CntStiffMatData['Ksn'].append(Ksn_ij)
                    CntStiffMatData['Kpn'].append(Kpn_ij)
                    CntCohMatData['Css'].append(Css_ij)
                    CntCohMatData['Cpp'].append(Cpp_ij)
                    Mij_List.append(Mij)
        
            
            elif ElemTypeId == -1:
            
                if IntegrationOrder==1: N_IntegrationPoints = 1
                elif IntegrationOrder==2: N_IntegrationPoints = 2
                else:   raise Exception ('Higher order integration not implemented')
                
                N_IntegrationPoints_2D = N_IntegrationPoints**2
                Ni, wi                  = GaussIntegrationTable(N_IntegrationPoints)
                
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
                        
                        Kss_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(np.dot(Mij.T, kss_ij), Mij)
                        Kpp_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(np.dot(Mij.T, kpp_ij), Mij)
                        Knn_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(np.dot(Mij.T, knn_ij), Mij)
                        Ksn_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(np.dot(Mij.T, ksn_ij), Mij)
                        Kpn_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(np.dot(Mij.T, kpn_ij), Mij)
                        
                        Css_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(Mij.T, css_ij).T[0]
                        Cpp_ij = 0.25*IntfcArea*wi[p0]*wi[p1]*np.dot(Mij.T, cpp_ij).T[0]
                        
                        CntStiffMatData['Kss'].append(Kss_ij)
                        CntStiffMatData['Kpp'].append(Kpp_ij)
                        CntStiffMatData['Knn'].append(Knn_ij)
                        CntStiffMatData['Ksn'].append(Ksn_ij)
                        CntStiffMatData['Kpn'].append(Kpn_ij)
                        CntCohMatData['Css'].append(Css_ij)
                        CntCohMatData['Cpp'].append(Cpp_ij)
                        Mij_List.append(Mij)
                
            
            
            #Cohesive secant stiffness/ Friction Parameters
            TSL_Param             = RefIntfcElem['TSL_Param']
            TSL_Param['dnc']      = 2*TSL_Param['GIc']/TSL_Param['ft']
            TSL_Param['dn0']      = TSL_Param['ft'] /TSL_Param['Kn0']
            TSL_Param['dsc']      = 2*TSL_Param['GIIc']/TSL_Param['fs']
            TSL_Param['ds0']      = TSL_Param['fs']/TSL_Param['Ks0']
            
            CntCoh  = 0.0e6
            FrCoeff = 0.0
            
            
            
            #Saving variables
            PenaltyData['TSL_Param']                = TSL_Param
            PenaltyData['IntfcElemList']            = IntfcElemList
            PenaltyData['CntStiffMatData']          = CntStiffMatData
            PenaltyData['CntCohMatData']            = CntCohMatData
            PenaltyData['Mij_List']                 = Mij_List
            PenaltyData['N_IntegrationPoints_2D']   = N_IntegrationPoints_2D
            PenaltyData['Elem_NDOF']                = RefIntfcElem['NDOF']
            PenaltyData['N_IntfcElem']              = N_IntfcElem
            PenaltyData['knn_ij']                   = knn_ij
            PenaltyData['kss_ij']                   = kss_ij
            PenaltyData['kpp_ij']                   = kpp_ij
            PenaltyData['CntCoh']                   = CntCoh
            PenaltyData['FrCoeff']                  = FrCoeff
            PenaltyData['dt']                       = dt
            PenaltyData['DamageData']               = np.array([[IntfcElem['Damage'] for i in range(N_IntegrationPoints_2D)] for IntfcElem in IntfcElemList], dtype=float).T
            PenaltyData['GapVecData']               = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 3])
            PenaltyData['KssWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            PenaltyData['KppWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            PenaltyData['KnnWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            PenaltyData['KsnWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            PenaltyData['KpnWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            PenaltyData['CssWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            PenaltyData['CppWeightData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
            
            PenaltyData['StickSlipData']            = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 2])
            PenaltyData['StickSlipData'][:,:,0]     = 1 #Assuming Stick condition at start

   
   
    
    
def updateMixedModeParam(TSL_Param, GapList, DmgVarMax, Method=2):
    
    Kn0                     = TSL_Param['Kn0']
    Ks0 = Kp0               = TSL_Param['Ks0']
    dn0                     = TSL_Param['dn0']
    ds0                     = TSL_Param['ds0']
    GIc                     = TSL_Param['GIc']
    GIIc                    = TSL_Param['GIIc']
    ds, dp, dn              = GapList
    
    dsh2 = ds*ds + dp*dp
    dsh = (dsh2)**0.5 #Net shear gap
    
    if dn>0:
        beta = dsh/dn #Mix-modity
        beta2 = beta*beta
        dm_0 = dn0*ds0*((1+beta2)/(ds0*ds0 +beta2*dn0*dn0))**0.5
        dm = (dsh2 + dn*dn)**0.5
        dm_f = (2*(1+beta2)/dm_0)/(Kn0/GIc + Ks0*beta2/GIIc)
    else:      
        dm_0 = ds0
        dm = dsh
        dm_f = 2*GIIc/(dm_0*Ks0)
    
    if Method==1:
        dm_max = DmgVarMax
        if dm>dm_max:   dm_max=dm
        DmgVarMax = dm_max
        D_max = (dm_f/dm_max)*(dm_max - dm_0)/(dm_f - dm_0)
    elif Method==2:
        D_max = DmgVarMax
        if abs(dm)>0:
            D = (dm_f/dm)*(dm - dm_0)/(dm_f - dm_0)
            if D>D_max:     D_max=D
            if D_max>1.0:   D_max=1.0
            elif D_max<0.0:   D_max=0.0
            DmgVarMax = D_max
        dm_max = (dm_f*dm_0)/(dm_f - D_max*(dm_f - dm_0))
    else:   raise Exception
    
    if dm_max <= dm_0:
        Kss = Ks0
        Kpp = Kp0
        Knn = Kn0
    
    elif dm_0 < dm_max <= dm_f:
        Kss = (1-D_max)*Ks0
        Kpp = (1-D_max)*Kp0
        if dn>0: Knn = (1-D_max)*Kn0
        else:    Knn = Kn0
    
    else:
        Kss = Kpp = 0.0
        if dn>0: Knn = 0.0
        else:    Knn = Kn0
    
    return Kss, Kpp, Knn, DmgVarMax, D_max
   
   
def updatePenaltyData(PenaltyData, IntfcElemList_Un_1):
    
    Mij_List                = PenaltyData['Mij_List']
    IntfcElemList           = PenaltyData['IntfcElemList']
    GapVecData0             = PenaltyData['GapVecData']
    StickSlipData           = PenaltyData['StickSlipData']
    DamageData              = PenaltyData['DamageData']
    N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
    N_IntfcElem             = PenaltyData['N_IntfcElem']
    TSL_Param               = PenaltyData['TSL_Param']
    CntCoh                  = PenaltyData['CntCoh']
    FrCoeff                 = PenaltyData['FrCoeff']
    KssWeightData           = PenaltyData['KssWeightData']
    KppWeightData           = PenaltyData['KppWeightData']
    KnnWeightData           = PenaltyData['KnnWeightData']
    KsnWeightData           = PenaltyData['KsnWeightData']
    KpnWeightData           = PenaltyData['KpnWeightData']
    CssWeightData           = PenaltyData['CssWeightData']
    CppWeightData           = PenaltyData['CppWeightData']
    Kn0                     = PenaltyData['TSL_Param']['Kn0']
    Ks0                     = PenaltyData['TSL_Param']['Ks0']
    dt                      = PenaltyData['dt']
    
    dnc                     = TSL_Param['dnc']
    dsc                     = TSL_Param['dsc']
    
    GapVecData              = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 3])
            
    PenaltyData['GapVecData']       = GapVecData
    
    
    
    for p in range(N_IntegrationPoints_2D):
        IntfcElemList_GapVec =  np.dot(Mij_List[p], IntfcElemList_Un_1)
        GapVecData[p, :, :] = IntfcElemList_GapVec.T
        
        #print(IntfcElemList_GapVec)
        
        for i in range(N_IntfcElem):
            SlideGap_s          = IntfcElemList_GapVec[0, i]
            SlideGap_p          = IntfcElemList_GapVec[1, i]
            SlideGap            = (SlideGap_s*SlideGap_s + SlideGap_p*SlideGap_p)**0.5
            NormalGap           = IntfcElemList_GapVec[2, i]
            GapList             = IntfcElemList_GapVec[:,i]
            
            IntfcElemArea       = IntfcElemList[i]['IntfcArea']
            
            Kss, Kpp, Knn, DmgVarMax, D_max = updateMixedModeParam(TSL_Param, GapList, DamageData[p,i], Method=2)
            DamageData[p,i] = DmgVarMax
            
            KssWeightData[p,i] = IntfcElemArea*Kss
            KppWeightData[p,i] = IntfcElemArea*Kpp
            KnnWeightData[p,i] = IntfcElemArea*Knn
            
            if NormalGap <= 0 and (CntCoh > 0 or FrCoeff > 0):
                
                CoulombFlag = False
                if StickSlipData[p, i, 0] == 1: #Checking Stick in previous step
                    #STICK
                    RefSlideGap = StickSlipData[p, i, 1]
                    if Ks0*abs(SlideGap-RefSlideGap) < CntCoh + FrCoeff*Kn0*abs(NormalGap):
                        KssWeightData[p,i] = IntfcElemArea*Ks0
                        KppWeightData[p,i] = IntfcElemArea*Ks0
                        KsnWeightData[p,i] = 0.0
                        KpnWeightData[p,i] = 0.0
                        CssWeightData[p,i] = 0.0
                        CppWeightData[p,i] = 0.0
                    else:   
                        StickSlipData[p, i, 0] = 0
                        CoulombFlag = True
                    
                    
                if StickSlipData[p, i, 0]==0: #Checking Slip
                        
                    #Gaps (in previous step)
                    SlideGap_s0 = GapVecData0[p, i, 0]
                    SlideGap_p0 = GapVecData0[p, i, 1]
                    
                    #Incremental gaps
                    ds = SlideGap_s - SlideGap_s0
                    dp = SlideGap_p - SlideGap_p0
                    dm = (ds*ds + dp*dp)**0.5
                    if dm == 0: dm = 1e-15
                    
                    #SLIP
                    SignedComp_ss = np.sign(ds)*abs(ds)/dm
                    SignedComp_pp = np.sign(dp)*abs(dp)/dm
                    KssWeightData[p,i] = 0.0
                    KppWeightData[p,i] = 0.0
                    KsnWeightData[p,i] = -IntfcElemArea*FrCoeff*Kn0*SignedComp_ss #-ve sign is used as NormGap is also -ve
                    KpnWeightData[p,i] = -IntfcElemArea*FrCoeff*Kn0*SignedComp_pp #-ve sign is used as NormGap is also -ve
                    CssWeightData[p,i] = CntCoh*SignedComp_ss
                    CppWeightData[p,i] = CntCoh*SignedComp_pp
                    
                    #Assigning Stick for small slide velocity
                    if CoulombFlag == False:
                        dv = dm/dt #Slide velocity
                        if dv < 1e-12: 
                            StickSlipData[p, i, 0] = 1
                            StickSlipData[p, i, 1] = SlideGap #Resetting the stick-point
                        
                
                
                
                

  
def calcMPFint(MP_Un_1, FintCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, Flat_ElemLocDof, NCount, MP_NbrMPIdVector, MP_TimeRecData):
    
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
            
        if ElemTypeId in [-2, -1]: #Interface Elements
            
            PenaltyData             = RefTypeGroup['PenaltyData']
            ElemList_Un_1           = MP_Un_1[ElemList_LocDofVector]
            updatePenaltyData(PenaltyData, ElemList_Un_1)
            
            CntStiffMatData         = PenaltyData['CntStiffMatData']
            CntCohMatData           = PenaltyData['CntCohMatData']
            N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
            Elem_NDOF               = PenaltyData['Elem_NDOF']
            KssWeightData           = PenaltyData['KssWeightData']
            KppWeightData           = PenaltyData['KppWeightData']
            KnnWeightData           = PenaltyData['KnnWeightData']
            KsnWeightData           = PenaltyData['KsnWeightData']
            KpnWeightData           = PenaltyData['KpnWeightData']
            CssWeightData           = PenaltyData['CssWeightData']
            CppWeightData           = PenaltyData['CppWeightData']
            IntfcElemList           = PenaltyData['IntfcElemList']
            N_IntfcElem             = len(IntfcElemList)
            
            
            RefIntfcElemLocId = 0
            RefIntfcElem_SecStiffList = []
            
            ElemList_CntFint = np.zeros([Elem_NDOF, N_IntfcElem], dtype=float)
            #ElemList_CntCoh = np.zeros([Elem_NDOF, N_IntfcElem], dtype=float)
            
            for p in range(N_IntegrationPoints_2D):
                
                # *************************
                # Use Sign Vector (like octrees) to rotate the elements
                # *************************
                
                #Calculating Contact Force
                ElemList_CntFint += np.dot(CntStiffMatData['Kss'][p], KssWeightData[p,:]*ElemList_Un_1)  + \
                                    np.dot(CntStiffMatData['Kpp'][p], KppWeightData[p,:]*ElemList_Un_1)  + \
                                    np.dot(CntStiffMatData['Knn'][p], KnnWeightData[p,:]*ElemList_Un_1)  + \
                                    np.dot(CntStiffMatData['Ksn'][p], KsnWeightData[p,:]*ElemList_Un_1)  + \
                                    np.dot(CntStiffMatData['Kpn'][p], KpnWeightData[p,:]*ElemList_Un_1)
                
                #for i = 1:N_Elem
                #    ElemList_CntCoh[:,i] += CntCohMatData['Css'][p]*CssWeightData[p,i] + \
                #                         CntCohMatData['Cpp'][p]*CppWeightData[p,i]
                
                
                
            MP_LocCntFintVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_CntFint.ravel(), minlength=MP_NDOF)        
                
            #MP_LocCntFintVec += np.bincount(ElemList_LocDofVector_Flat, weights=ElemList_CntCoh.ravel(), minlength=MP_NDOF)        
            
            
                
        else: #Octree Elements
            Ke = RefTypeGroup['ElemStiffMat']
            ElemList_SignVector = RefTypeGroup['ElemList_SignVector']
            ElemList_Ck = RefTypeGroup['ElemList_Ck']
            
            ElemList_Un_1 =   MP_Un_1[ElemList_LocDofVector]
            ElemList_Un_1[ElemList_SignVector] *= -1.0        
            ElemList_Fint =   np.dot(Ke, ElemList_Ck*ElemList_Un_1)
            
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

    

def getNodalStress(PenaltyData):

    GapVecData              = PenaltyData['GapVecData']
    IntfcElemList           = PenaltyData['IntfcElemList']
    N_IntegrationPoints     = PenaltyData['N_IntegrationPoints']
    N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
    CntStiffMatData         = PenaltyData['CntStiffMatData']
    CntCohMatData           = PenaltyData['CntCohMatData']
    KssWeightData           = PenaltyData['KssWeightData']
    KppWeightData           = PenaltyData['KppWeightData']
    KnnWeightData           = PenaltyData['KnnWeightData']
    KsnWeightData           = PenaltyData['KsnWeightData']
    KpnWeightData           = PenaltyData['KpnWeightData']
    CssWeightData           = PenaltyData['CssWeightData']
    CppWeightData           = PenaltyData['CppWeightData']
    
    N_IntfcElem             = len(IntfcElemList)
    
    #Defining Extrapolation points
    #https://quickfem.com/wp-content/uploads/IFEM.Ch28.pdf
    if N_IntegrationPoints == 1:
        Ms = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)[np.newaxis].T
    
    elif N_IntegrationPoints == 2:
        Ni = [-3**0.5, 3**0.5]
        Ms = []
        for p0 in range(N_IntegrationPoints):
            for p1 in range(N_IntegrationPoints):
                s0 = Ni[p0]
                s1 = Ni[p1]
                N1 = 0.25*(1-s0)*(1-s1)
                N2 = 0.25*(1+s0)*(1-s1)
                N3 = 0.25*(1+s0)*(1+s1)
                N4 = 0.25*(1-s0)*(1+s1)
                Ms.append([N1, N2, N3, N4])
        Ms = np.array(Ms, dtype=float).T
        
    else: raise NotImplementedError
    
    
    Intfc_NodeIdList = np.unique(np.hstack([IntfcElemList[i]['NodeIdList'][:4] for i in range(N_IntfcElem)]))
    N_IntfcNodes = len(Intfc_NodeIdList)
    IntfcNodeList_StressList = [[] for i in range(N_IntfcNodes)]
    IntfcNodeList_CoordList = [[] for i in range(N_IntfcNodes)]
    for i in range(N_IntfcElem):
        IntfcElem   = IntfcElemList[i]
        IntfcElemArea   = IntfcElem['IntfcArea']
        GP_StressVec    = np.zeros([N_IntegrationPoints_2D, 3], dtype=float)
        for p in range(N_IntegrationPoints_2D):
            k_p  = (CntStiffMatData['kss']*KssWeightData[p,i] + \
                    CntStiffMatData['kpp']*KppWeightData[p,i] + \
                    CntStiffMatData['knn']*KnnWeightData[p,i] + \
                    CntStiffMatData['ksn']*KsnWeightData[p,i] + \
                    CntStiffMatData['kpn']*KpnWeightData[p,i])/IntfcElemArea
            
            #c_p  = (CntCohMatData['css']*CssWeightData[p,i] + \
            #        CntCohMatData['cpp']*CppWeightData[p,i])/IntfcElemArea
        
            GapVec_p = GapVecData[p,i,:]
            
            GP_StressVec[p,:] = np.dot(k_p, GapVec_p)
            #GP_StressVec[p,:] = np.dot(k_p, GapVec_p) + c_p
        
        IntfcElem_NodalStressVec = np.dot(Ms, GP_StressVec)
        N_LowerNodes = int(IntfcElem['N_Node']/2)
        for j in range(N_LowerNodes):
            J = np.where(Intfc_NodeIdList==IntfcElem['NodeIdList'][j])[0][0]
            IntfcNodeList_StressList[J].append(IntfcElem_NodalStressVec[j,:])
            
            if len(IntfcNodeList_CoordList[J]) == 0:
                IntfcNodeList_CoordList[J] = IntfcElem['CoordList'][j]
    
    for j in range(N_IntfcNodes):
        IntfcNodeList_StressList[j] = np.mean(IntfcNodeList_StressList[j], axis=0)
    
    IntfcNodeList_StressList = np.array(IntfcNodeList_StressList, dtype=float)
    IntfcNodeList_CoordList = np.array(IntfcNodeList_CoordList, dtype=float)
   
    
    RefYCoord = 0.25
    RefIndexList = np.where(np.abs(IntfcNodeList_CoordList[:,1]-RefYCoord)<1e-5)[0]
    RefNodeCoordList = IntfcNodeList_CoordList[RefIndexList]
    RefTractionList = IntfcNodeList_StressList[RefIndexList]
    
    XSortIndexList = np.argsort(RefNodeCoordList[:,0])
    RefTractionList = RefTractionList[XSortIndexList]
    RefNodeCoordList = RefNodeCoordList[XSortIndexList]
    
    #print(IntfcNodeList_CoordList[:,1]-RefYCoord)
    np.save(PlotFileName+'_DCB_Friction.npy', [RefNodeCoordList, RefTractionList])
    
    fig = plt.figure()
    plt.plot(RefNodeCoordList[:,0], RefTractionList[:,0]*1e-6)
    plt.xlim(0,4)
    plt.ylim(-0.15,0)
    fig.savefig(PlotFileName+'StrDis.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = IntfcNodeList_CoordList[:,0]
    y = IntfcNodeList_CoordList[:,1]
    z = IntfcNodeList_StressList[:,0]
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
    fig.savefig(PlotFileName+'3D.png', dpi = 480, bbox_inches='tight')
    plt.close()
    """
        

 
def initIntfcNodalDamageData(MP_SubDomainData):
    
    MP_TypeGroupList = MP_SubDomainData['StrucDataList']
    N_Type = len(MP_TypeGroupList) 
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId in [-2, -1]:
            
            if ElemTypeId == -2: #Interface Elements (Triangular)
                IntfcElemList = RefTypeGroup['ElemList_IntfcElem']
                PenaltyData = RefTypeGroup['PenaltyData']
                        
                N_IntegrationPoints_2D     = PenaltyData['N_IntegrationPoints_2D']
                if N_IntegrationPoints_2D == 1:
                    Ms = np.array([1.0, 1.0, 1.0], dtype=float)[np.newaxis].T
                
                elif N_IntegrationPoints_2D == 3:
                    Ni0 = [-1.0/3.0,  5.0/3.0, -1.0/3.0]
                    Ni1 = [-1.0/3.0, -1.0/3.0,  5.0/3.0]
                    Ms = []
                    for p in range(N_IntegrationPoints_2D):
                        s0 = Ni0[p]
                        s1 = Ni1[p]
                        N1 = 1.0 - s0 - s1
                        N2 = s0
                        N3 = s1
                        Ms.append([N1, N2, N3])
                        
                    Ms = np.array(Ms, dtype=float)
                    
                else: raise NotImplementedError
                
            elif ElemTypeId == -1: #Interface Elements (Quadrilateral)
                IntfcElemList = RefTypeGroup['ElemList_IntfcElem']
                PenaltyData = RefTypeGroup['PenaltyData']
                        
                N_IntegrationPoints_2D     = PenaltyData['N_IntegrationPoints_2D']
                
                #Defining Extrapolation points
                #https://quickfem.com/wp-content/uploads/IFEM.Ch28.pdf
                if N_IntegrationPoints_2D == 1:
                    Ms = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)[np.newaxis].T
                
                elif N_IntegrationPoints_2D == 4:
                    Ni0 = [-3**0.5,  3**0.5, 3**0.5, -3**0.5]
                    Ni1 = [-3**0.5, -3**0.5, 3**0.5,  3**0.5]
                    Ms = []
                    for p in range(N_IntegrationPoints_2D):
                        s0 = Ni0[p]
                        s1 = Ni1[p]
                        N1 = 0.25*(1-s0)*(1-s1)
                        N2 = 0.25*(1+s0)*(1-s1)
                        N3 = 0.25*(1+s0)*(1+s1)
                        N4 = 0.25*(1-s0)*(1+s1)
                        Ms.append([N1, N2, N3, N4])
                    Ms = np.array(Ms, dtype=float)
                    
                else: raise NotImplementedError
                    
            PenaltyData['Ms']           = Ms
            PenaltyData['N_LowerNodes'] = len(Ms)
            
    
def getNodalDamage(MP_SubDomainData, MP_IntfcOvrlpLocalNodeIdVecList, MP_IntfcNbrMPIdVector, MP_NNode):
    
    MP_TypeGroupList            = MP_SubDomainData['StrucDataList']
    N_Type                      = len(MP_TypeGroupList) 
    MP_DamageSum                = np.zeros(MP_NNode)
    MP_DamageCount              = np.zeros(MP_NNode)
            
    for j in range(N_Type):        
        RefTypeGroup = MP_TypeGroupList[j]    
        ElemTypeId = RefTypeGroup['ElemTypeId']
        
        if ElemTypeId in [-2, -1]: #Interface Elements
            IntfcElemList = RefTypeGroup['ElemList_IntfcElem']
            PenaltyData = RefTypeGroup['PenaltyData']
            
            MP_IntfcElemList            = PenaltyData['IntfcElemList']
            MP_DamageData               = PenaltyData['DamageData']
            N_LowerNodes                = PenaltyData['N_LowerNodes']
            Ms                          = PenaltyData['Ms']
            N_IntfcElem                 = len(MP_IntfcElemList)
            
            
            #Calculating Damage
            for i in range(N_IntfcElem):
                IntfcElem   = MP_IntfcElemList[i]
                GP_DamageVec = MP_DamageData[:,i][np.newaxis].T
                IntfcElem_NodalDamageVec = np.dot(Ms, GP_DamageVec)
                for j in range(N_LowerNodes):
                    J0 = IntfcElem['LocNodeIdList'][j]
                    J1 = IntfcElem['LocNodeIdList'][j+N_LowerNodes]
                    MP_DamageSum[J0] += IntfcElem_NodalDamageVec[j,0]
                    MP_DamageSum[J1] += IntfcElem_NodalDamageVec[j,0]
                    MP_DamageCount[J0] += 1
                    MP_DamageCount[J1] += 1
                
    
    #Calculating Overlapping Fint Vectors
    N_IntfcNbrMP = len(MP_IntfcNbrMPIdVector)
            
    MP_OvrlpDamageList = []
    MP_InvOvrlpDamageList = []
    for j in range(N_IntfcNbrMP):
        MP_OvrlpDamage = np.hstack([MP_DamageSum[MP_IntfcOvrlpLocalNodeIdVecList[j]], MP_DamageCount[MP_IntfcOvrlpLocalNodeIdVecList[j]]])
        MP_OvrlpDamageList.append(MP_OvrlpDamage)   
        
        N_IntfcNbrDof_j = len(MP_IntfcOvrlpLocalNodeIdVecList[j]);
        MP_InvOvrlpDamageList.append(np.zeros(2*N_IntfcNbrDof_j))
    
    updateTime(MP_TimeRecData, 'dT_Calc')
    
    
    #Communicating Overlapping Damage
    SendReqList = []
    for j in range(N_IntfcNbrMP):        
        IntfcNbrMP_Id    = MP_IntfcNbrMPIdVector[j]
        SendReq     = Comm.Isend(MP_OvrlpDamageList[j], dest=IntfcNbrMP_Id, tag=Rank)
        SendReqList.append(SendReq)
        
    for j in range(N_IntfcNbrMP):    
        IntfcNbrMP_Id = MP_IntfcNbrMPIdVector[j]
        Comm.Recv(MP_InvOvrlpDamageList[j], source=IntfcNbrMP_Id, tag=IntfcNbrMP_Id)
        
    MPI.Request.Waitall(SendReqList)    
    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    #Updating Damage at the Meshpart boundary
    for j in range(N_IntfcNbrMP): 
        N_IntfcNbrDof_j = len(MP_IntfcOvrlpLocalNodeIdVecList[j]);
        MP_DamageSum[MP_IntfcOvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpDamageList[j][:N_IntfcNbrDof_j]
        MP_DamageCount[MP_IntfcOvrlpLocalNodeIdVecList[j]] += MP_InvOvrlpDamageList[j][N_IntfcNbrDof_j:] 
    
    MP_Damage = MP_DamageSum/(MP_DamageCount+1e-15) #1e-15 is added for DamageCount = 0
    
    return MP_Damage
    
    
    
    
        

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
    if RecTime:    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    GlobUnVector = []
    if Rank == 0:
        GathUnVector = np.hstack(MPList_Un)
        GlobUnVector = np.zeros(GlobNDof, dtype=float)
        GlobUnVector[GathDofVector] = GathUnVector
            
    return GlobUnVector


def getGlobDamageVec(MP_Damage, MPList_Damage, MP_IntfcLocalNodeIdList, MPList_IntfcNodeIdVector, GlobNNode, MP_TimeRecData, RecTime = True):
    
    if RecTime: updateTime(MP_TimeRecData, 'dT_Calc')
    SendReq = Comm.Isend(MP_Damage[MP_IntfcLocalNodeIdList], dest=0, tag=Rank)  
    if Rank == 0:
        for j in range(N_Workers):  Comm.Recv(MPList_Damage[j], source=j, tag=j)
    SendReq.Wait()
    if RecTime:    updateTime(MP_TimeRecData, 'dT_CommWait')
    
    GlobDamageVector = []
    if Rank == 0:
        GlobDamageVector = np.zeros(GlobNNode, dtype=float)
        for j in range(N_Workers):
            GlobDamageVector[MPList_IntfcNodeIdVector[j]] = MPList_Damage[j]
        
    return GlobDamageVector


def plotDispVecData(PlotFileName, TimeList, TimeList_PlotDispVector):
    
    fig = plt.figure()
    plt.plot(TimeList, TimeList_PlotDispVector.T)
#    plt.xlim([0, 14])
#    plt.ylim([-0.5, 3.0])
#    plt.show()    
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    
def exportGlobVecData(OutputFileName, ExportCount, Time_dT, GlobVector, Key, Splits = 16):
    
    FileName = OutputFileName+'_'+str(ExportCount)+'_'
    
    if Rank == 0:
        J = Splits
        N = int(len(GlobVector))
        Nj = int(N/J)
        for j in range(J):
            if j==0:
                N1 = 0; N2 = Nj;
            elif j == J-1:
                N1 = N2; N2 = N;
            else:
                N1 = N2; N2 = (j+1)*Nj;
            
            Data_j = {'T': Time_dT, Key: GlobVector[N1:N2]}
            savemat(FileName+str(j+1)+'.mat', Data_j)



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
    DmgVecPath = ResultPath + 'DmgVecData/'
    if Rank==0:    
        if not os.path.exists(ResultPath):            
            os.makedirs(PlotPath)        
            os.makedirs(DispVecPath)
            os.makedirs(DmgVecPath)
    
 
    DispOutputFileName = DispVecPath + ModelName
    DmgOutputFileName = DmgVecPath + ModelName
    PlotFileName = PlotPath + ModelName
    
    #Reading Model Data Files
    PyDataPath = ScratchPath + 'ModelData/' + 'MP' +  str(N_MshPrt)  + '/'
    RefMeshPart_FileName = PyDataPath + str(Rank) + '.zpkl'
    Cmpr_RefMeshPart = open(RefMeshPart_FileName, 'rb').read()
    RefMeshPart = pickle.loads(zlib.decompress(Cmpr_RefMeshPart))
    
    MP_SubDomainData =                  RefMeshPart['SubDomainData']
    MP_NbrMPIdVector =                  RefMeshPart['NbrMPIdVector']
    MP_OvrlpLocalDofVecList =           RefMeshPart['OvrlpLocalDofVecList']
    MP_NDOF =                           RefMeshPart['NDOF']
    MP_NNode =                          RefMeshPart['NNode']
    MP_DOFIdVector =                    RefMeshPart['DofVector']
    MP_RefLoadVector =                  RefMeshPart['RefLoadVector']
    MP_InvLumpedMassVector =            RefMeshPart['InvDiagM']
    MP_FixedLocalDofVector =            RefMeshPart['FixedLocDofVector']
    GathDofVector =                     RefMeshPart['GathDofVector']
    MP_IntfcLocalNodeIdList =           RefMeshPart['IntfcLocalNodeIdList']
    MPList_IntfcNodeIdVector =          RefMeshPart['MPList_IntfcNodeIdVector']
    MPList_NDofVec =                    RefMeshPart['MPList_NDofVec']
    MPList_IntfcNNode =                 RefMeshPart['MPList_IntfcNNode']
    MPList_RefPlotDofIndicesList =      RefMeshPart['MPList_RefPlotDofIndicesList']
    MP_RefPlotData =                    RefMeshPart['RefPlotData']
    Flat_ElemLocDof =                   RefMeshPart['Flat_ElemLocDof']
    NCount =                            RefMeshPart['NCount']
    N_NbrDof =                          RefMeshPart['N_NbrDof']
    
    
    MP_IntfcNbrMPIdVector =             RefMeshPart['IntfcNbrMPIdVector']
    MP_IntfcOvrlpLocalNodeIdVecList =        RefMeshPart['IntfcOvrlpLocalNodeIdVecList']
    
    #Reading Global data file
    MatDataPath = ScratchPath + 'ModelData/Mat/'
    GlobSettingsFile = MatDataPath + 'GlobSettings.mat'
    GlobSettings = scipy.io.loadmat(GlobSettingsFile)
    ExportFrms                          = np.array(GlobSettings['ExportFrms'], dtype=int) 
    if len(ExportFrms)>0:    ExportFrms = ExportFrms[0] - 1
    MaxTime                             = GlobSettings['Tmax'][0][0] 
    DeltaLambdaList                     = np.array(GlobSettings['ft'][0], dtype=float)
    dt                                  = GlobSettings['dt'][0][0] 
    dT_Export                           = GlobSettings['dT_Export'][0][0] 
    PlotFlag                            = GlobSettings['PlotFlag'][0][0]   
    ExportFlag                          = GlobSettings['ExportFlag'][0][0]  
    FintCalcMode                        = GlobSettings['FintCalcMode'][0] 
    IntegrationOrder                    = GlobSettings['IntegrationOrder'][0][0]
    
    GlobData =                           RefMeshPart['GlobData']
    GlobNDof =                           GlobData['GlobNDof']
    if not GlobData['GlobNDof']%3==0:    raise Exception
    GlobNNode =                          int(GlobData['GlobNDof']/3)
    Damping_Alpha =                      GlobData['Damping_Alpha']
    
    MP_LumpedMassVector =                1.0/MP_InvLumpedMassVector
    
    if SpeedTestFlag==1:  
        PlotFlag = 0; ExportFlag = 0; MaxTime=20000*dt; FintCalcMode = 'outbin'; EnergyCalcFlag = 0;
    
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
    MPList_Damage = []
    if Rank==0:
        N_MeshParts = len(MPList_NDofVec)
        MPList_Un = [np.zeros(MPList_NDofVec[i]) for i in range(N_MeshParts)]
        MPList_Damage = [np.zeros(MPList_IntfcNNode[i]) for i in range(N_MeshParts)]
            
    
    #A small sleep to avoid hanging
    sleep(Rank*1e-4)
    
    #Barrier so that all processes start at same time
    Comm.barrier()    
    updateTime(MP_TimeRecData, 'dT_FileRead')    
    t0_Start = time()
    
    
    #Initializing Variables
    MP_Un_1 = (1e-200)*np.random.rand(MP_NDOF)
    MP_Un = (1e-200)*np.random.rand(MP_NDOF)
    
    initPenaltyData(MP_SubDomainData, IntegrationOrder, dt)
    initIntfcNodalDamageData(MP_SubDomainData)
    
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
            exportGlobVecData(DispOutputFileName + '_U', ExportCount, Time_dT, GlobUnVector, 'U')
            
            MP_Damage = getNodalDamage(MP_SubDomainData, MP_IntfcOvrlpLocalNodeIdVecList, MP_IntfcNbrMPIdVector, MP_NNode)
            GlobDamageVector = getGlobDamageVec(MP_Damage, MPList_Damage, MP_IntfcLocalNodeIdList, MPList_IntfcNodeIdVector, GlobNNode, MP_TimeRecData)
            exportGlobVecData(DmgOutputFileName +'_Dmg', ExportCount, Time_dT, GlobDamageVector, 'Dmg')
            
            ExportCount += 1
                
    
    #-------------------------------------------------------------------------------
    #print(Rank, 'Starting parallel computation..')
    for TimeStepCount in range(1, RefMaxTimeStepCount):
        
        if Rank==0:
            if TimeStepCount%100==0:
                print('---------------------')
                print('TimeStepCount', TimeStepCount)
                #print(MP_Un)
            
        MP_Un_2 = MP_Un_1
        MP_Un_1 = MP_Un
        
        #Calculating Fint
        MP_FintVec, MP_CntFintVec, MP_ElemCntStressDataList    = calcMPFint(MP_Un_1, FintCalcMode, MP_SubDomainData, MP_OvrlpLocalDofVecList, Flat_ElemLocDof, NCount, MP_NbrMPIdVector, MP_TimeRecData)
        
        #Calculating Displacement Vector
        DeltaLambda_1 = DeltaLambdaList[TimeStepCount-1]
        MP_Un = calcMPDispVec(MP_FintVec, MP_RefLoadVector, MP_InvLumpedMassVector, MP_FixedLocalDofVector, MP_Un_2, MP_Un_1, dt, DeltaLambda_1, Damping_Alpha)
        
    
        
        #updateTime(MP_TimeRecData, 'UpdateList', TimeStepCount=TimeStepCount)
        
        if PlotFlag == 1:
            #if Rank==0: print(TimeStepCount)
            if MP_PlotNDofs>0:
                MP_PlotDispVector[:,TimeStepCount] = MP_Un[MP_PlotLocalDofVec]
                
            if len(MP_ElemCntStressDataList)>0:
                Dir = 0 #x=0,y=1,z=2
                MP_PlotCntStressData.append([MP_ElemCntStressDataList[0][0][Dir], MP_ElemCntStressDataList[0][1][Dir]])
                #MP_PlotCntStrData.append(MP_ElemCntStrList[0])
                #MP_PlotNormGapData.append(MP_ElemNormGapList[0])
                
            if (TimeStepCount)%2000==0:
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
                exportGlobVecData(DispOutputFileName + '_U', ExportCount, Time_dT, GlobUnVector, 'U')
                
                MP_Damage = getNodalDamage(MP_SubDomainData, MP_IntfcOvrlpLocalNodeIdVecList, MP_IntfcNbrMPIdVector, MP_NNode)
                GlobDamageVector = getGlobDamageVec(MP_Damage, MPList_Damage, MP_IntfcLocalNodeIdList, MPList_IntfcNodeIdVector, GlobNNode, MP_TimeRecData)
                exportGlobVecData(DmgOutputFileName +'_Dmg', ExportCount, Time_dT, GlobDamageVector, 'Dmg')
                
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
        
        TimeDataFileName = PlotFileName + '_MP' +  str(N_MshPrt) + '_' + FintCalcMode + '_TimeData'
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
                else:
                    TimeList_CntGapList = []
                    TimeList_CntStressList = []
                
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
            plt.plot(TimeList_CntGapList, TimeList_CntStressList, '-')
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
    
    
    
    

