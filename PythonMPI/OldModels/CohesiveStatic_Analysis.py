# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:10:24 2020

@author: z5166762
"""

import matplotlib.pyplot as plt
from datetime import datetime
import sys
from time import time, sleep
import numpy as np
import os.path
import shutil

import pickle
import zlib

from scipy.io import savemat
from GeneralFunc import configTimeRecData, GaussLobattoIntegrationTable, GaussIntegrationTable
from scipy.interpolate import interp1d, interp2d
from numpy.linalg import norm, inv



def initNodalGapCalcMatrix():
    
    N_IntegrationPoints = 2 # 2 for nodal calculation in GL Integration
    Ni, wi = GaussLobattoIntegrationTable(N_IntegrationPoints)
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
    
    return NodalGapCalcMatrix
    

def initPenaltyData(IntfcElemList, N_IntegrationPoints):

    RefIntfcElem            = IntfcElemList[0]
    Ni, wi                  = GaussIntegrationTable(N_IntegrationPoints)
    N_IntfcElem             = len(IntfcElemList)
    
    IntfcArea               = 1.0
    
    #Saving stiffness matrices
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
    
    
    #Cohesive secant stiffness/ Friction Parameters
    TSL_Param = {}
    
    #Mode-I
    TSL_Param['Kn0']     = 1.0e9
    TSL_Param['ft']      = 10e3
    TSL_Param['GIc']     = 1.0
    TSL_Param['dnc']      = 2*TSL_Param['GIc']/TSL_Param['ft']
    TSL_Param['dn0']      = TSL_Param['ft'] /TSL_Param['Kn0']
    
    #Mode-II
    TSL_Param['Ks0']     = 0.5e9
    TSL_Param['fs']      = 18e3
    TSL_Param['GIIc']    = 1.8
    TSL_Param['dsc']      = 2*TSL_Param['GIIc']/TSL_Param['fs']
    TSL_Param['ds0']      = TSL_Param['fs']/TSL_Param['Ks0']
    
    """
    TSL_Param = {}
    
    TSL_Param['Kn0']     = 1.0e9
    TSL_Param['ft']      = 20e3
    TSL_Param['GfI']     = 1.0
    TSL_Param['dnc']      = 2*TSL_Param['GfI']/TSL_Param['ft']
    TSL_Param['dn0']      = TSL_Param['ft'] /TSL_Param['Kn0']
    
    TSL_Param['Ks0']     = 1.0e9
    TSL_Param['fs']      = 20e3
    TSL_Param['GfII']    = 1.0
    TSL_Param['dsc']      = 2*TSL_Param['GfII']/TSL_Param['fs']
    TSL_Param['ds0']      = TSL_Param['fs']/TSL_Param['Ks0']
    """
    
    CntCoh  = 0.0e6
    FrCoeff = 0.0
    
    
    def getNormSecStiff(NormalGap, TSL_Param=TSL_Param):
        
        dn0 = TSL_Param['dn0']
        dnc = TSL_Param['dnc']
        ft  = TSL_Param['ft']
        Kn0 = TSL_Param['Kn0']
        if NormalGap <= dn0:
            SecStiff = Kn0
        elif dn0 < NormalGap <= dnc:
            NormalTraction = (ft/(dnc-dn0))*(dnc-NormalGap)
            SecStiff = NormalTraction/NormalGap 
        elif NormalGap > dnc:
            SecStiff = 0.0
        
        return SecStiff
    
    
    def getShearSecStiff(SlideGap, TSL_Param=TSL_Param):
        
        ds0 = TSL_Param['ds0']
        dsc = TSL_Param['dsc']
        fs  = TSL_Param['fs']
        Ks0 = TSL_Param['Ks0']
        
        if SlideGap <= ds0:
            SecStiff = Ks0
        elif ds0 < SlideGap <= dsc:
            ShearTraction = (fs/(dsc-ds0))*(dsc-SlideGap)
            SecStiff = ShearTraction/SlideGap 
        elif SlideGap > dsc:
            SecStiff = 0.0
        
        return SecStiff
    
    """
    def getShearSecStiff(SlideGap, NormalGap, fs=fs, sc=sc, Kss=Kss):
        
        if NormalGap>=0:
            so = fs/Kss;
            if SlideGap <= so:
                SecStiff = Kss
            elif so < SlideGap <= sc:
                ShearTraction = (fs/(sc-so))*(sc-SlideGap)
                SecStiff = ShearTraction/SlideGap 
            elif SlideGap > sc:
                SecStiff = 0.0
        else:
            SecStiff = Kss
            
        return SecStiff
    """
    
    #Saving variables
    PenaltyData = {}
    PenaltyData['TSL_Param']                = TSL_Param
    PenaltyData['CntStiffMatData']          = CntStiffMatData
    PenaltyData['CntCohMatData']            = CntCohMatData
    PenaltyData['Mij_List']                 = Mij_List
    PenaltyData['getNormSecStiff']          = getNormSecStiff
    PenaltyData['getShearSecStiff']         = getShearSecStiff
    PenaltyData['N_IntegrationPoints_2D']   = N_IntegrationPoints**2
    PenaltyData['N_IntegrationPoints']      = N_IntegrationPoints
    PenaltyData['Elem_NDOF']                = RefIntfcElem['NDOF']
    PenaltyData['N_IntfcElem']              = N_IntfcElem
    PenaltyData['knn_ij']                   = knn_ij
    PenaltyData['CntCoh']                   = CntCoh
    PenaltyData['FrCoeff']                  = FrCoeff
    PenaltyData['DamageData']               = np.array([[IntfcElem['Damage'] for i in range(N_IntegrationPoints**2)] for IntfcElem in IntfcElemList], dtype=float).T
    PenaltyData['GapVecData']               = np.zeros([N_IntegrationPoints**2, N_IntfcElem, 3])
    PenaltyData['KssWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KppWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KnnWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KsnWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KpnWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['CssWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['CppWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    
    
    return PenaltyData

   
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
            if D>D_max:   D_max=D
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
   
   
def updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un):
    
    Mij_List                = PenaltyData['Mij_List']
    getShearSecStiff        = PenaltyData['getShearSecStiff']
    getNormSecStiff         = PenaltyData['getNormSecStiff']
    GapVecData0             = PenaltyData['GapVecData']
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
    
    dnc                     = TSL_Param['dnc']
    dsc                     = TSL_Param['dsc']
    
    GapVecData                      = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 3])
    IntfcElemList_HasStiffness      = np.zeros(N_IntfcElem)
    
    PenaltyData['GapVecData']                   = GapVecData
    PenaltyData['IntfcElemList_HasStiffness']   = IntfcElemList_HasStiffness
    
    
    for p in range(N_IntegrationPoints_2D):
        IntfcElemList_GapVec =  np.dot(Mij_List[p], IntfcElemList_Un)
        GapVecData[p, :, :] = IntfcElemList_GapVec.T
        
        #print(IntfcElemList_GapVec)
        
        for i in range(N_IntfcElem):
            #SlideGap_s          = IntfcElemList_GapVec[0, i]
            #SlideGap_p          = IntfcElemList_GapVec[1, i]
            #SlideGap            = norm([SlideGap_s, SlideGap_p])
            NormalGap           = IntfcElemList_GapVec[2, i]
            GapList             = IntfcElemList_GapVec[:,i]
            
            IntfcElemArea       = IntfcElemList[i]['IntfcArea']
            
            #if SlideGap == 0:  
            #    SlideGap = SlideGap_s = SlideGap_p = 1e-15
            
            KsnWeightData[p,i] = 0.0
            KpnWeightData[p,i] = 0.0
            
            Kss, Kpp, Knn, DmgVarMax, D_max = updateMixedModeParam(TSL_Param, GapList, DamageData[p,i], Method=2)
            DamageData[p,i] = DmgVarMax
            
            KssWeightData[p,i] = IntfcElemArea*Kss
            KppWeightData[p,i] = IntfcElemArea*Kpp
            KnnWeightData[p,i] = IntfcElemArea*Knn
            
            if D_max < 1.0:
                IntfcElemList_HasStiffness[i] = 1.0
            
            if NormalGap <= 0: # Keep "<=" or "<" ?
                IntfcElemList_HasStiffness[i] = 1.0
                
                """
                #Coulomb Friction
                Knn = getNormSecStiff(NormalGap)
                KnnWeightData[p, i] = IntfcElemArea*Knn
                IntfcElem_FrictionalStress = Ks0*SlideGap
                IntfcElem_NormalStress = Knn*abs(NormalGap)
                
                '''
                #SLIP: BRUTE-FORCE
                KsnWeightData[p,i] = IntfcElemArea*FrCoeff*Knn
                KpnWeightData[p,i] = 0.0
                '''
                
                if IntfcElem_FrictionalStress < CntCoh + FrCoeff*IntfcElem_NormalStress:
                    #STICK
                    KssWeightData[p,i] = IntfcElemArea*Ksm*abs(SlideGap_s)/SlideGap
                    KppWeightData[p,i] = IntfcElemArea*Ksm*abs(SlideGap_p)/SlideGap
                    
                else:
                    #SLIP
                    SlideGap_s0 = GapVecData0[p, i, 0]
                    SlideGap_p0 = GapVecData0[p, i, 1]
                    
                    SignedComp_ss = np.sign(SlideGap_s - SlideGap_s0)*abs(SlideGap_s)/SlideGap
                    SignedComp_pp = np.sign(SlideGap_p - SlideGap_p0)*abs(SlideGap_p)/SlideGap
                    
                    KsnWeightData[p,i] = -IntfcElemArea*FrCoeff*Knn*SignedComp_ss #-ve sign is used as NormGap is also -ve
                    KpnWeightData[p,i] = -IntfcElemArea*FrCoeff*Knn*SignedComp_pp #-ve sign is used as NormGap is also -ve
                    
                    CssWeightData[p,i] = CntCoh*SignedComp_ss
                    CppWeightData[p,i] = CntCoh*SignedComp_pp
                    
                """
            
            #print(NormalGap, Knn, Kss, Kpp)

def updateGlobalSecK(PenaltyData, IntfcElemList, GlobK, ConstrainedDOFIdVector):
    
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
    
    
    N_IntfcElem             = len(IntfcElemList)
    RefIntfcElemLocId       = 0
    RefIntfcElem_SecStiffList = []
    
    
    N_IntfcElem = len(IntfcElemList)
    GlobSecK = np.array(GlobK, dtype = float)
    GlobCntCohLoadVec = np.zeros(len(GlobK))
    for i in range(N_IntfcElem):
        IntfcElem = IntfcElemList[i]
        Intfc_DofIdList = IntfcElem['DofIdList']
        Intfc_K = np.zeros([Elem_NDOF, Elem_NDOF], dtype=float)
        Intfc_CntCoh = np.zeros(Elem_NDOF, dtype=float)
        for p in range(N_IntegrationPoints_2D):
            Intfc_K +=  CntStiffMatData['Kss'][p]*KssWeightData[p,i] + \
                        CntStiffMatData['Kpp'][p]*KppWeightData[p,i] + \
                        CntStiffMatData['Knn'][p]*KnnWeightData[p,i] + \
                        CntStiffMatData['Ksn'][p]*KsnWeightData[p,i] + \
                        CntStiffMatData['Kpn'][p]*KpnWeightData[p,i]
            
            Intfc_CntCoh += CntCohMatData['Css'][p]*CssWeightData[p,i] + \
                            CntCohMatData['Cpp'][p]*CppWeightData[p,i]
                        
            
            """
            #Extracting Data for calculating Contact Stress (Post-processing)
            SecStiff = ElemList_StiffnessWeight[RefIntfcElemLocId]/IntfcElemList[RefIntfcElemLocId]['IntfcArea']
            RefIntfcElem_SecStiffList.append(SecStiff)
            """
        
        
        #Calculating Global Secant Stiffness Matrix
        I = np.meshgrid(Intfc_DofIdList, Intfc_DofIdList, indexing='ij')
        GlobSecK[tuple(I)] += Intfc_K
        
        GlobCntCohLoadVec[Intfc_DofIdList] += Intfc_CntCoh
        """
        
        for io in range(Elem_NDOF):
            Dof_i = Intfc_DofIdList[io]
            for jo in range(Elem_NDOF):
                Dof_j = Intfc_DofIdList[jo]
                GlobSecK[Dof_i, Dof_j] += Intfc_K[io, jo]
    
        """
        
        #print(IntfcElem['NodeIdList'])
        #print(Intfc_DofIdList[0::3]/3+1)
        
    
    
    #Applying BC
    for i in range(GlobNDOF): #looping over rows 
        for j in ConstrainedDOFIdVector: #looping over columns
            if i == j:  GlobSecK[i, i] = 1.0  #Making diagonal members of GlobalStiffnessMatrix = 1, if displacement Boundary Constraint is applied. (So as to make the GlobalStiffnessMatrix invertible)
            else:       GlobSecK[i, j] = 0.0  #Making column vector = 0, if displacement Boundary Constraint is applied.        
    
    
    
    """
    RefLoad = np.zeros(Elem_NDOF)
    NodeId0 = np.array([0,1,2,3])
    NodeId1 = np.array([4,5,6,7])
    #DofId0 = NodeId0*3+2
    DofId1 = NodeId1*3+2
    #RefLoad[DofId0] = -1e9/4
    RefLoad[DofId1] = 1e9/4
    
    ConstDof = np.hstack([NodeId0*3, NodeId0*3+1, NodeId0*3+2]) 
    
    for i in range(Elem_NDOF): #looping over rows 
        for j in ConstDof: #looping over columns
            if i == j:  Intfc_K[i, i] = 1.0  #Making diagonal members of GlobalStiffnessMatrix = 1, if displacement Boundary Constraint is applied. (So as to make the GlobalStiffnessMatrix invertible)
            else:       Intfc_K[i, j] = 0.0  #Making column vector = 0, if displacement Boundary Constraint is applied.        
         
    DispVec = np.zeros(Elem_NDOF)
    #DispVec[DofId0] = -0.5
    DispVec[DofId1] = 1.0
    
    U = np.dot(inv(Intfc_K),RefLoad)
    U[ConstDof] = 0.0
    print(U)
    """
    
    
    """
    DispVec = np.zeros(GlobNDOF)
    NodeId0 = np.array([1,2,5,6]) - 1
    NodeId1 = np.array([3,4,7,8]) - 1
    NodeId2 = np.array([11,12,15,16]) - 1
    NodeId3 = np.array([9,10,13,14]) - 1
    DofId0 = NodeId0*3+2
    DofId1 = NodeId1*3+2
    DofId2 = NodeId2*3+2
    DofId3 = NodeId3*3+2
    #DispVec[DofId0] = -1.5
    DispVec[DofId1] = 1.0
    DispVec[DofId2] = 2.0
    DispVec[DofId3] = 3.0
    
    RefLoad = np.zeros(GlobNDOF)
    #RefLoad[DofId0] = -1e9/4
    RefLoad[DofId3] = 1e9/4
    
    print(np.dot(GlobSecK,DispVec))
    U = np.dot(inv(GlobSecK),RefLoad)
    U[ConstrainedDOFIdVector] = 0.0
    print(U)
    
    """
    
    
    return GlobSecK, GlobCntCohLoadVec
    
  
def getNodalStress(PenaltyData, IntfcElemList, PlotFileName):

    GapVecData              = PenaltyData['GapVecData']
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
            
            c_p  = (CntCohMatData['css']*CssWeightData[p,i] + \
                    CntCohMatData['cpp']*CppWeightData[p,i])/IntfcElemArea
        
            GapVec_p = GapVecData[p,i,:]
            
            GP_StressVec[p,:] = np.dot(k_p, GapVec_p) + c_p
        
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


def calcNodalGapVec(PenaltyData, Un, NodalGapCalcMatrix, IntfcElemList_DofVector):
    
    IntfcElemList_HasStiffness = PenaltyData['IntfcElemList_HasStiffness']
    
    IntfcElemList_Un =   Un[IntfcElemList_DofVector]
    NodalGapVec =  np.dot(NodalGapCalcMatrix, IntfcElemList_HasStiffness*IntfcElemList_Un).T.ravel()
    
    return NodalGapVec
  

    

def plotDispVecData(PlotFileName, PlotLoadVector, PlotDispVector, PenaltyData = []):

    fig = plt.figure()
    if len(PenaltyData)>0:
        TSL_Param = PenaltyData['TSL_Param']
        f = TSL_Param['fs']
        d0 = TSL_Param['ds0']
        dc = TSL_Param['dsc']
        plt.plot([0, d0, dc], [0, f, 0], '--k')
        
    
    plt.plot(np.abs(PlotDispVector.T), PlotLoadVector, '.-')
    plt.tick_params(direction='in', which='both',top=True, right=True)
    #plt.xlim(0,1.5)
    #plt.ylim(0,1.5)
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    
    np.save(PlotFileName+'.npy', np.array([PlotDispVector, PlotLoadVector], dtype=object))
  

def exportDispVecData(DispVecPath, ExportCount, GlobUnVector, Splits = 1):

    DispVecFileName = DispVecPath + 'U_'+str(ExportCount)+'_'

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
        
        DispData_j = {'U': GlobUnVector[N1:N2]}
        savemat(DispVecFileName+str(j+1)+'.mat', DispData_j)

    savemat(DispVecPath+'J.mat', {'J':J})


if __name__ == "__main__":
    
    ModelName =         sys.argv[1]
    ScratchPath =       sys.argv[2]
    R0 =                sys.argv[3]
    
    #Creating directories
    ResultPath          = ScratchPath + 'Results_Run' + str(R0)
    ResultPath += '/'
        
    PlotPath = ResultPath + 'PlotData/'
    DispVecPath = ResultPath + 'DispVecData/'
    if not os.path.exists(ResultPath):            
        os.makedirs(PlotPath)        
        os.makedirs(DispVecPath)

    PlotFileName = PlotPath + ModelName
    
    #Reading Model Data Files
    RefMeshData_FileName = ScratchPath + 'ModelData/PyData.zpkl'
    Cmpr_RefMeshData = open(RefMeshData_FileName, 'rb').read()
    RefMeshData = pickle.loads(zlib.decompress(Cmpr_RefMeshData))
    
    RefLoadVector =             RefMeshData['RefTransientLoadVector']
    ConstrainedDOFIdVector =    RefMeshData['ConstrainedDOFIdVector']
    IntfcElemList =             RefMeshData['IntfcElemList']
    GlobK =                     RefMeshData['GlobK']
    GlobData =                  RefMeshData['GlobData']
    NodeCoordVec =              RefMeshData['NodeCoordVec']
    
    GlobNDOF =                  GlobData['GlobNDOF']
    ExportFlag =                GlobData['ExportFlag']
    PlotFlag =                  GlobData['PlotFlag']
    RefPlotData =               GlobData['RefPlotData']
    
    DelLambda_ini               = GlobData['del_lambda_ini']
    RelTol                      = GlobData['RelTol']
    m                           = GlobData['m'] 
    Nd0                         = GlobData['Nd0']
    n                           = GlobData['n']
    MaxIterCount                = GlobData['MaxIterCount']
    N_IntegrationPoints         = GlobData['N_IntegrationPoints']
    MaxTimeStepCount            = GlobData['MaxTimeStepCount'] 
    MaxFailedConvergenceCount   = GlobData['MaxFailedConvergenceCount'] 
    
    NodalGapCalcMatrix = initNodalGapCalcMatrix()
    PenaltyData = initPenaltyData(IntfcElemList, N_IntegrationPoints)
    IntfcElemList_DofVector = np.array([IntfcElem['DofIdList'] for IntfcElem in IntfcElemList], dtype=int).T
    
    Un = np.zeros(GlobNDOF, dtype=float)
    
    ExportCount = 0
    FailedConvergenceCount = 0
    
    if PlotFlag == 1:
        RefPlotDofVec       = RefPlotData['RefPlotDofVec']
        qpoint              = RefPlotData['qpoint']
        PlotNDofs = len(RefPlotDofVec)
        if PlotNDofs > 0:
            PlotLoadVector               = np.zeros(MaxTimeStepCount+1)
            PlotGapVector               = np.zeros(MaxTimeStepCount+1)
            PlotDispVector               = np.zeros([PlotNDofs, MaxTimeStepCount+1])
            PlotDispVector[:,0]          = Un[RefPlotDofVec]
        else:   
            PlotDispVector               = []
        
        PlotCntStressData               = []
        #PlotCntStrData               = []
        #PlotNormGapData               = []
        
    
    if ExportFlag == 1:
        exportDispVecData(DispVecPath, ExportCount, Un)
        ExportCount += 1
    
     
        
    #Arc Length Parameters
    Norm_RefLoadVector                   = norm(RefLoadVector)
    TolF                                 = RelTol*Norm_RefLoadVector
    
    """
    #Updating GlobalRefTransientLoadVector to include reactions        
    RefDispVector = np.dot(InvGlobSecK, RefLoadVector)
    RefDispVector[ConstrainedDOFIdVector] = 0.0
    RefLoadVector = np.dot(GlobSecK, RefDispVector)
    """
    
    """
    #print(RefLoadVector)
    
    A = DelUp
    for i in range(GlobNDOF):
        print(i, A[i])
    
    
    Fint_test = np.dot(GlobSecK, DelUp)
    
    
    for i in range(GlobNDOF):
        if i%3==0:  print('---')
        print(int(i/3)+1, Fint_test[i], RefLoadVector[i])
    
    exit()
    """
    #print(DelUp)
    
    #Initializing parameters
    IntfcElemList_Un                = Un[IntfcElemList_DofVector]
    updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un)
    GlobSecK, GlobCntCohLoadVec     = updateGlobalSecK(PenaltyData, IntfcElemList, GlobK, ConstrainedDOFIdVector)
    
    #Initializing Residual
    DeltaLambda = 0.0
    Fext = DeltaLambda*RefLoadVector 
    Fint = np.dot(GlobSecK, Un) + GlobCntCohLoadVec
    R = Fint - Fext
    R[ConstrainedDOFIdVector] = 0.0
    
    LoadDispData = {'Load': [],
                    'Disp': []}
    
    #if NormR<TolF:    raise Exception
    print('-----------')
    
    for TimeStepCount in range(MaxTimeStepCount):
            
        DeltaUn =       np.zeros(GlobNDOF)
        Nintfc =        len(np.where(PenaltyData['IntfcElemList_HasStiffness'] == 1)[0]) 
        Nd =            Nintfc**n + Nd0
        
        InvGlobSecK                     = inv(GlobSecK)
        DelUp                           = np.dot(InvGlobSecK, RefLoadVector)
        DelUp[ConstrainedDOFIdVector]   = 0.0
        LocDelUp = calcNodalGapVec(PenaltyData, DelUp, NodalGapCalcMatrix, IntfcElemList_DofVector)
        NormLocDelUp                    = norm(LocDelUp)
            
        if TimeStepCount == 0:
            Delta_l                         = DelLambda_ini*NormLocDelUp
        else:
            Delta_l *=      (float(Nd)/LastIterCount)**m
            
            if Delta_l < 0.25*DelLambda_ini*NormLocDelUp:
                Delta_l *= 1.5
            
            print('')
            print(TimeStepCount, LastIterCount, Delta_l)
            
        
        for IterCount in range(MaxIterCount):
            DelUf                           = -np.dot(InvGlobSecK, R)
            DelUf[ConstrainedDOFIdVector]   = 0.0
            LocDelUf                        = calcNodalGapVec(PenaltyData, DelUf, NodalGapCalcMatrix, IntfcElemList_DofVector)
            LocDeltaUn                      = calcNodalGapVec(PenaltyData, DeltaUn, NodalGapCalcMatrix, IntfcElemList_DofVector)
                        
            if IterCount == 0:
                DelLambda   = Delta_l/NormLocDelUp
                DelLambda0  = DelLambda
                
            else:
                
                #DelLambda = DelLambda0 - np.dot(LocDelUp, LocDeltaUn + LocDelUf)/np.dot(LocDelUp, LocDelUp)
                #DelLambda = (Delta_l**2 - np.dot(LocDeltaUn, LocDeltaUn + LocDelUf))/np.dot(LocDeltaUn, LocDelUp)
                DelLambda = -np.dot(LocDeltaUn, LocDelUf)/np.dot(LocDeltaUn, LocDelUp)
                    
            
            DelUn = DelUf + DelLambda*DelUp
            
            #Updating increments
            DeltaUn +=  DelUn
            DeltaLambda += DelLambda
            
            Un += DelUn
            Fext = DeltaLambda*RefLoadVector 
            
            IntfcElemList_Un                = Un[IntfcElemList_DofVector]
            updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un)
            GlobSecK, GlobCntCohLoadVec     = updateGlobalSecK(PenaltyData, IntfcElemList, GlobK, ConstrainedDOFIdVector)
            
            Fint = np.dot(GlobSecK, Un) + GlobCntCohLoadVec
            R = Fint - Fext
            R[ConstrainedDOFIdVector] = 0.0
            NormR  = norm(R)
            
            print(IterCount, NormR)
            
            if NormR<TolF:
                LastIterCount = IterCount+1
                print('LastIterCount', LastIterCount)
                print('DeltaLambda', DeltaLambda)
                break
        
        else:
        
            print('Convergence Failed!', IterCount+1,  FailedConvergenceCount, NormR/Norm_RefLoadVector)
            FailedConvergenceCount += 1
            LastIterCount = Nd
            
        if PlotFlag == 1:
            if PlotNDofs > 0:
                """
                p = 1e6
                B = 0.25
                PlotLoadVector[TimeStepCount+1] = p*B*DeltaLambda
                PlotDispVector[:,TimeStepCount+1] = Un[RefPlotDofVec]
                """
                """
                RefGap = PenaltyData['GapVecData'][0, 0, 0] 
                PlotGapVector[TimeStepCount+1] = RefGap
                IntfcElemArea = IntfcElemList[0]['IntfcArea']
                RefK = PenaltyData['KssWeightData'][0,0]/IntfcElemArea
                Tr = RefK*RefGap
                PlotLoadVector[TimeStepCount+1] = Tr
                plotDispVecData(PlotFileName, PlotLoadVector, PlotGapVector, PenaltyData)
                """
                #"""
                p = 1e6
                PlotLoadVector[TimeStepCount+1] = p*0.25*DeltaLambda
                PlotDispVector[:,TimeStepCount+1] = Un[RefPlotDofVec]
                plotDispVecData(PlotFileName, PlotLoadVector, PlotDispVector)
                #"""
            
        if ExportFlag == 1:
            exportDispVecData(DispVecPath, ExportCount, Un)
            ExportCount += 1
            
        
        if TimeStepCount >= MaxTimeStepCount or FailedConvergenceCount >= MaxFailedConvergenceCount:
            AnalysisFinished = True
            break
    
    
    #plotDispVecData(PlotFileName, PlotLoadVector, PlotDispVector)
    
    
    print('Analysis Finished Sucessfully..')
    
    #Extracting Stresses
    """
    IntfcElemList_Un =   Un[IntfcElemList_DofVector]
    updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un)
    getNodalStress(PenaltyData, IntfcElemList, PlotFileName)
    """
    
    
    
    
    
    

