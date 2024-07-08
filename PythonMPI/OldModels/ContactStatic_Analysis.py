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
    Knn     = RefIntfcElem['Knn']
    ft      = 1e3
    GfI     = 10
    wc      = 2*GfI/ft
    wo      = ft/Knn;
    
    Kss     = RefIntfcElem['Kss']
    fs      = 1e3
    GfII    = 10
    sc      = 2*GfII/fs
    so      = fs/Kss;
    
    CntCoh = 0.0e6
    FrCoeff = 0.25
    
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
    
    def getNormSecStiff(NormalGap, ft=ft, wc=wc, Knn=Knn):
         
        wo = ft/Knn;
        if NormalGap <= wo:
            SecStiff = Knn
        elif wo < NormalGap <= wc:
            NormalTraction = (ft/(wc-wo))*(wc-NormalGap)
            SecStiff = NormalTraction/NormalGap 
        elif NormalGap > wc:
            SecStiff = 0.0
            
        return SecStiff
    
    def getShearSecStiff(SlideGap, NormalGap, fs=fs, sc=sc, Kss=Kss):
        
        if NormalGap>0:
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
    
    
    #Saving variables
    PenaltyData = {}
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
    PenaltyData['wc']                       = wc
    PenaltyData['CntCoh']                   = CntCoh
    PenaltyData['FrCoeff']                  = FrCoeff
    PenaltyData['DamageData']               = np.ones([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['GapVecData']               = np.zeros([N_IntegrationPoints**2, N_IntfcElem, 3])
    PenaltyData['KssWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KppWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KnnWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KsnWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['KpnWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['CssWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    PenaltyData['CppWeightData']            = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    
    return PenaltyData


    

   
def updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un):
    
    Mij_List                = PenaltyData['Mij_List']
    getShearSecStiff        = PenaltyData['getShearSecStiff']
    getNormSecStiff         = PenaltyData['getNormSecStiff']
    GapVecData0             = PenaltyData['GapVecData']
    DamageData              = PenaltyData['DamageData']
    N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
    N_IntfcElem             = PenaltyData['N_IntfcElem']
    CntCoh                  = PenaltyData['CntCoh']
    FrCoeff                 = PenaltyData['FrCoeff']
    KssWeightData           = PenaltyData['KssWeightData']
    KppWeightData           = PenaltyData['KppWeightData']
    KnnWeightData           = PenaltyData['KnnWeightData']
    KsnWeightData           = PenaltyData['KsnWeightData']
    KpnWeightData           = PenaltyData['KpnWeightData']
    CssWeightData           = PenaltyData['CssWeightData']
    CppWeightData           = PenaltyData['CppWeightData']
    
    
    GapVecData                      = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 3])
    IntfcElemList_HasStiffness      = np.zeros(N_IntfcElem)
    
    PenaltyData['GapVecData']                   = GapVecData
    PenaltyData['IntfcElemList_HasStiffness']   = IntfcElemList_HasStiffness
    
    
    for p in range(N_IntegrationPoints_2D):
        IntfcElemList_GapVec =  np.dot(Mij_List[p], IntfcElemList_Un)
        GapVecData[p, :, :] = IntfcElemList_GapVec.T
        
        #print(IntfcElemList_GapVec)
    
        
        for i in range(N_IntfcElem):
            SlideGap_s          = IntfcElemList_GapVec[0, i]
            SlideGap_p          = IntfcElemList_GapVec[1, i]
            SlideGap            = norm([SlideGap_s, SlideGap_p])
            NormalGap           = IntfcElemList_GapVec[2, i]
            IntfcElemArea       = IntfcElemList[i]['IntfcArea']
            
            
            
            if SlideGap == 0:   SlideGap=1e-15
            
            KssWeightData[p,i] = 0.0
            KppWeightData[p,i] = 0.0
            KnnWeightData[p,i] = 0.0
            KsnWeightData[p,i] = 0.0
            KpnWeightData[p,i] = 0.0
            
            if NormalGap > 0:
            
                #DEFINE DAMAGE in terms of NORMAL and SHEAR GAP
                Damage = np.min([1.0,NormalGap/wc])
                
                if DamageData[p, i]<Damage: #Loading (Increasing Damage)
                    DamageData[p, i]=Damage
                    Knn = getNormSecStiff(NormalGap)
                    KnnWeightData[p, i] = IntfcElemArea*Knn
                else: #Loading/Unloading
                    ElemList_KnnWeight[i] = (1.0-DamageData[p, i])*IntfcElemArea*getSecStiff(0)
                
                if Damage < 1.0:
                    IntfcElemList_HasStiffness[i] = 1.0
            
            elif NormalGap <= 0: #Keep the condition "<=" 
                IntfcElemList_HasStiffness[i] = 1.0
                
                Ks = getShearSecStiff(SlideGap, NormalGap)
                Knn = getNormSecStiff(NormalGap)
                KnnWeightData[p, i] = IntfcElemArea*Knn
                
                
                IntfcElem_ShearStress = Ks*SlideGap
                IntfcElem_NormalStress = Knn*abs(NormalGap)
                
                
                #SLIP
                KsnWeightData[p,i] = IntfcElemArea*FrCoeff*Knn
                KpnWeightData[p,i] = 0.0
                
                """
                #print(IntfcElem_ShearStress*1e-12 , (CntCoh + FrCoeff*IntfcElem_NormalStress)*1e-12)
                if IntfcElem_ShearStress < CntCoh + FrCoeff*IntfcElem_NormalStress:
                    #STICK
                    KssWeightData[p,i] = IntfcElemArea*Ks*abs(SlideGap_s)/SlideGap
                    KppWeightData[p,i] = IntfcElemArea*Ks*abs(SlideGap_p)/SlideGap
                    
                    #print('STICK')
                    
                    
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
                    
                    #print('SLIP')
                """

                    


def updateContactCondition_Backup(PenaltyData, IntfcElemList_Un, ZeroGapIntfcCond):
    
    Mij_List                = PenaltyData['Mij_List']
    GapVecData0             = PenaltyData['GapVecData']
    N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
    CntCoh                  = PenaltyData['CntCoh']
    FrCoeff                 = PenaltyData['FrCoeff']
    
    N_IntfcElem             = len(IntfcElemList)
    
    ContactConditionData = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
    GapVecData = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 3])
    SlideDirVecData = np.zeros([N_IntegrationPoints_2D, N_IntfcElem, 3])
    
    PenaltyData['ContactConditionData'] = ContactConditionData
    PenaltyData['GapVecData'] = GapVecData
    PenaltyData['SlideDirVecData'] = SlideDirVecData
    
    for p in range(N_IntegrationPoints_2D):
        
        IntfcElemList_GapVec =  np.dot(Mij_List[p], IntfcElemList_Un)
        GapVecData[p, :, :] = IntfcElemList_IntgPntGapVec.T
        
        for i in range(N_IntfcElem):
            SlideGap_s          = IntfcElemList_GapVec[0, i]
            SlideGap_p          = IntfcElemList_GapVec[1, i]
            NormalGap           = IntfcElemList_GapVec[2, i]
            IntfcElemArea       = IntfcElemList[i]['IntfcArea']
            
            if NormalGap == 0:
                IntfcElem_CntCond = ZeroGapIntfcCond
            
            elif NormalGap > 0:
                IntfcElem_CntCond = 0 #FREE
            
            elif NormalGap < 0:
                IntfcElem_ShearStress = Kss*norm([SlideGap_s, SlideGap_p])
                IntfcElem_NormalStress = Knn*abs(NormalGap)
        
                if IntfcElem_ShearStress < CntCoh + FrCoeff*IntfcElem_NormalStress:
                    IntfcElem_CntCond = 2 #STICK
                else:
                    IntfcElem_CntCond = 1 #SLIP
            
            ContactConditionData[p, i] = IntfcElem_CntCond
        
            GapVec0 = GapVecData0[p, i] #GapVector from last load-step
            SlidingDir = np.sign(IntfcElemList_GapVec[:,i] - GapVec0)
            SlideDirVecData[p, i, :] = SlidingDir
    

def updateStiffnessWeightList_Backup(PenaltyData, IntfcElemList, IntfcElemList_Un):
    
    Mij_List                = PenaltyData['Mij_List']
    DamageData              = PenaltyData['DamageData']
    ContactConditionData    = PenaltyData['ContactConditionData']
    SlideDirVecData         = PenaltyData['SlideDirVecData']
    getSecStiff             = PenaltyData['getSecStiff']
    N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
    wc                      = PenaltyData['wc']
    
    N_IntfcElem = len(IntfcElemList)
    IntfcElemList_KssWeight = np.zeros(N_IntfcElem)
    IntfcElemList_KppWeight = np.zeros(N_IntfcElem)
    IntfcElemList_KnnWeight = np.zeros(N_IntfcElem)
    
    StiffnessWeight_Matrix = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
    HasStiffness_Matrix = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
    for p in range(N_IntegrationPoints_2D):
        
        #Calculating PenaltyWeight
        IntfcElemList_IntgPntGapVec =  np.dot(Mij_List[p], IntfcElemList_Un)
        ElemList_Damage = DamageData[p]
         
        for i in range(N_IntfcElem):
            SlideGap0           = IntfcElemList_IntgPntGapVec[0, i]
            SlideGap1           = IntfcElemList_IntgPntGapVec[1, i]
            NormalGap           = IntfcElemList_IntgPntGapVec[2, i]
            IntfcElemArea       = IntfcElemList[i]['IntfcArea']
            
            if NormalGap <= 0:
                ElemList_KnnWeight[i] = IntfcElemArea*getSecStiff(NormalGap)
                
                #Stick/Slip Condition
                
                
                
            else:
                Damage = np.min([1.0,NormalGap/wc])
                
                if ElemList_Damage[i]<Damage: #Loading (Increasing Damage)
                    ElemList_Damage[i]=Damage
                    ElemList_KnnWeight[i] = IntfcElemArea*getSecStiff(NormalGap)
                
                else: #Loading/Unloading
                    ElemList_KnnWeight[i] = (1.0-ElemList_Damage[i])*IntfcElemArea*getSecStiff(0)
            
        
        
        RefStiffnessList = abs(ElemList_KssWeight) + abs(ElemList_KppWeight) + abs(ElemList_KnnWeight)
        ElemList_HasStiffness = np.zeros(N_IntfcElem)
        for i in range(N_IntfcElem):    
            if RefStiffnessList[i]>0:
                ElemList_HasStiffness[i] = 1.0
                
        StiffnessWeight_Matrix[p,:] = ElemList_StiffnessWeight_p
        HasStiffness_Matrix[p,:] = ElemList_HasStiffness_p
     
    return ElemList_StiffnessWeight, ElemList_HasStiffness
    
    


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
    
    #Calculating Inverse
    InvGlobSecK = inv(GlobSecK)
    
    
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
    
    
    return GlobSecK, InvGlobSecK, GlobCntCohLoadVec
    
    


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
  
    



def plotDispVecData(PlotFileName, PlotLoadVector, PlotDispVector):
    fig = plt.figure()
    plt.plot(np.abs(PlotDispVector.T), PlotLoadVector)
    plt.tick_params(direction='in', which='both',top=True, right=True)
    plt.xlim(0,1.5)
    plt.ylim(0,1.5)
    fig.savefig(PlotFileName+'.png', dpi = 480, bbox_inches='tight')
    plt.close()
    




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
    Norm_RefLoadVector                    = norm(RefLoadVector)
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
    
    
    
    
    #if NormR<TolF:    raise Exception
    print('-----------')
    for TimeStepCount in range(MaxTimeStepCount):
    
        IntfcElemList_Un =   Un[IntfcElemList_DofVector]
        updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un)
        GlobSecK, InvGlobSecK, GlobCntCohLoadVec = updateGlobalSecK(PenaltyData, IntfcElemList, GlobK, ConstrainedDOFIdVector)
        DelUp = np.dot(InvGlobSecK, RefLoadVector)
        DelUp[ConstrainedDOFIdVector] = 0.0
        LocDelUp = calcNodalGapVec(PenaltyData, DelUp, NodalGapCalcMatrix, IntfcElemList_DofVector)
         
        DeltaUn =       np.zeros(GlobNDOF)
        Nintfc =        len(np.where(PenaltyData['IntfcElemList_HasStiffness'] == 1)[0]) 
        Nd =            Nintfc**n + Nd0
        
        if TimeStepCount == 0:
            DelLambda       = DelLambda_ini
            NormLocDelUp    = norm(LocDelUp)
            Delta_l         = DelLambda*NormLocDelUp
            
            #Initializing Residual
            DeltaLambda = 0.0
            Fext = DeltaLambda*RefLoadVector 
            Fint = np.dot(GlobSecK, Un) + GlobCntCohLoadVec
            R = Fint - Fext
            R[ConstrainedDOFIdVector] = 0.0
            
        else:  
            Delta_l *=      (float(Nd)/LastIterCount)**m
            print('')
            print(TimeStepCount, LastIterCount, Delta_l)
            
            
        for IterCount in range(MaxIterCount):
            
            DelUf = -np.dot(InvGlobSecK, R)
            DelUf[ConstrainedDOFIdVector] = 0.0
            LocDelUf = calcNodalGapVec(PenaltyData, DelUf, NodalGapCalcMatrix, IntfcElemList_DofVector)
            LocDeltaUn = calcNodalGapVec(PenaltyData, DeltaUn, NodalGapCalcMatrix, IntfcElemList_DofVector)
                        
            if IterCount == 0:
                DelLambda = Delta_l/NormLocDelUp
                DelLambda0 = DelLambda
                
            else:
                
#                del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
                DelLambda = (Delta_l**2 - np.dot(LocDeltaUn, LocDeltaUn + LocDelUf))/np.dot(LocDeltaUn, LocDelUp)
                #DelLambda = -np.dot(LocDeltaUn, LocDelUf)/np.dot(LocDeltaUn, LocDelUp)
                    
            
            DelUn = DelUf + DelLambda*DelUp
            
            #Updating increments
            DeltaUn +=  DelUn
            DeltaLambda += DelLambda
            
            Un += DelUn
            Fext = DeltaLambda*RefLoadVector 
            
            """
            ElemList_Un =   Un[IntfcElemList_DofVector]
            StiffnessWeight_Matrix, HasStiffness_Matrix = getDamageWeightList(IntfcElemList, ElemList_Damage_p, ElemList_IntgPntGapVec, getSecStiff, wc)
            GlobSecK, InvGlobSecK, ElemList_HasStiffness = updateGlobalSecK(PenaltyData, GlobK, IntfcElemList, ElemList_Un, ConstrainedDOFIdVector)
            """
            
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
                PlotLoadVector[TimeStepCount+1] = DeltaLambda
                PlotDispVector[:,TimeStepCount+1] = Un[RefPlotDofVec]
            
        if ExportFlag == 1:
            exportDispVecData(DispVecPath, ExportCount, Un)
            ExportCount += 1
            
        
        
        if TimeStepCount >= MaxTimeStepCount or FailedConvergenceCount >= MaxFailedConvergenceCount:
            AnalysisFinished = True
            break
    
    
    plotDispVecData(PlotFileName, PlotLoadVector, PlotDispVector)
    
    print('Analysis Finished Sucessfully..')
    
    #Extracting Stresses
    IntfcElemList_Un =   Un[IntfcElemList_DofVector]
    updatePenaltyData(PenaltyData, IntfcElemList, IntfcElemList_Un)
    getNodalStress(PenaltyData, IntfcElemList, PlotFileName)
    
    
    
    
    
    
    

