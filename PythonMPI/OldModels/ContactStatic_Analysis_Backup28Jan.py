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
    
    Knn                     = 1.0
    IntfcArea               = 1.0
    
    #Saving stiffness matrix
    knn_ij = np.array([[0,  0,   0],
                       [0,  0,   0],
                       [0,    0, Knn]], dtype=float)
                    
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
    PenaltyData = {}
    PenaltyData['CntStiffMat_List'] = CntStiffMat_List
    PenaltyData['Mij_List'] = Mij_List
    PenaltyData['getSecStiff'] = getSecStiff
    PenaltyData['N_IntegrationPoints_2D'] = N_IntegrationPoints**2
    PenaltyData['Elem_NDOF'] = RefIntfcElem['NDOF']
    PenaltyData['knn_ij'] = knn_ij
    PenaltyData['wc'] = wc
    PenaltyData['DamageMatrix'] = np.zeros([N_IntegrationPoints**2, N_IntfcElem])
    
    return PenaltyData
    
 
 
 

def updateGlobalSecK(PenaltyData, GlobK, IntfcElemList, ElemList_Un, ConstrainedDOFIdVector):
    
    CntStiffMat_List        = PenaltyData['CntStiffMat_List']
    Mij_List                = PenaltyData['Mij_List']
    DamageMatrix            = PenaltyData['DamageMatrix']
    wc                      = PenaltyData['wc']
    getSecStiff             = PenaltyData['getSecStiff']
    N_IntegrationPoints_2D  = PenaltyData['N_IntegrationPoints_2D']
    Elem_NDOF               = PenaltyData['Elem_NDOF']
    N_IntfcElem             = len(IntfcElemList)
    RefIntfcElemLocId       = 0
    RefIntfcElem_SecStiffList = []
    
    
    
    StiffnessWeight_Matrix = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
    HasStiffness_Matrix = np.zeros([N_IntegrationPoints_2D, N_IntfcElem])
    for p in range(N_IntegrationPoints_2D):
        
        #Calculating PenaltyWeight
        ElemList_IntgPntGapVec =  np.dot(Mij_List[p], ElemList_Un)
        
        #ElemList_StiffnessWeight = getPenaltyWeightList(IntfcElemList, ElemList_IntgPntGapVec, getSecStiff)
        ElemList_Damage_p = DamageMatrix[p]
        ElemList_StiffnessWeight_p, ElemList_HasStiffness_p = getDamageWeightList(IntfcElemList, ElemList_Damage_p, ElemList_IntgPntGapVec, getSecStiff, wc)

        StiffnessWeight_Matrix[p,:] = ElemList_StiffnessWeight_p
        HasStiffness_Matrix[p,:] = ElemList_HasStiffness_p
        
    N_IntfcElem = len(IntfcElemList)
    GlobSecK = np.array(GlobK, dtype = float)
    for i in range(N_IntfcElem):
        IntfcElem = IntfcElemList[i]
        Intfc_DofIdList = IntfcElem['DofIdList']
        Intfc_K = np.zeros([Elem_NDOF, Elem_NDOF], dtype=float)
        for p in range(N_IntegrationPoints_2D):
            Intfc_K += CntStiffMat_List[p]*StiffnessWeight_Matrix[p,i]
            
            """
            #Extracting Data for calculating Contact Stress (Post-processing)
            SecStiff = ElemList_StiffnessWeight[RefIntfcElemLocId]/IntfcElemList[RefIntfcElemLocId]['IntfcArea']
            RefIntfcElem_SecStiffList.append(SecStiff)
            """
        
        
        #Calculating Global Secant Stiffness Matrix
        
        I = np.meshgrid(Intfc_DofIdList, Intfc_DofIdList, indexing='ij')
        GlobSecK[tuple(I)] += Intfc_K
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
    
    #Calculating StiffnessBool
    ElemList_HasStiffness = np.array(np.sum(HasStiffness_Matrix, axis=0),dtype=bool).astype(np.int8)
    
    
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
    
    
    
    
    return GlobSecK, InvGlobSecK, ElemList_HasStiffness
    
    



    

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
    


def getDamageWeightList(IntfcElemList, ElemList_Damage, ElemList_IntgPntGapVec, getSecStiff, wc):
    
    N_IntfcElem = len(IntfcElemList)
    ElemList_StiffnessWeight = np.zeros(N_IntfcElem)
    ElemList_HasStiffness = np.zeros(N_IntfcElem)
    
    for i in range(N_IntfcElem):
        NormalGap           = ElemList_IntgPntGapVec[2, i]
        IntfcElemArea    = IntfcElemList[i]['IntfcArea']
        
    
        if NormalGap <= 0:
            ElemList_StiffnessWeight[i] = IntfcElemArea*getSecStiff(NormalGap)
            
        else:
            Damage = np.min([1.0,NormalGap/wc])
            
            if ElemList_Damage[i]<Damage: #Loading (Increasing Damage)
                ElemList_Damage[i]=Damage
                ElemList_StiffnessWeight[i] = IntfcElemArea*getSecStiff(NormalGap)
            
            else: #Loading/Unloading
                ElemList_StiffnessWeight[i] = (1.0-ElemList_Damage[i])*IntfcElemArea*getSecStiff(0)
        
        
        if abs(ElemList_StiffnessWeight[i])>0:
            ElemList_HasStiffness[i] = 1.0
            
    return ElemList_StiffnessWeight, ElemList_HasStiffness
    
    
  
    

def calcNodalGapVec(Un, NodalGapCalcMatrix, IntfcElemList_DofVector):
    
    ElemList_Un =   Un[IntfcElemList_DofVector]
    NodalGapVec =  np.dot(NodalGapCalcMatrix, ElemList_HasStiffness*ElemList_Un).T.ravel()
    
    return NodalGapVec
  
    






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
    PlotFlag =                  0
    
    GlobNDOF =                           GlobData['GlobNDOF']
    ExportFlag =                         GlobData['ExportFlag']
    
    DelLambda_ini       = GlobData['del_lambda_ini']
    RelTol              = GlobData['RelTol']
    m                   = GlobData['m'] 
    Nd0                 = GlobData['Nd0']
    n                   = GlobData['n']
    MaxIterCount        = GlobData['MaxIterCount']
    N_IntegrationPoints = GlobData['N_IntegrationPoints']
    MaxTimeStepCount        = GlobData['MaxTimeStepCount'] 
    MaxFailedConvergenceCount        = GlobData['MaxFailedConvergenceCount'] 
    
    NodalGapCalcMatrix = initNodalGapCalcMatrix()
    PenaltyData = initPenaltyData(IntfcElemList, N_IntegrationPoints)
    
    Un = np.zeros(GlobNDOF, dtype=float)
    
    ExportCount = 1
    FailedConvergenceCount = 0
    
    if PlotFlag == 1:
        PlotLocalDofVec  = RefPlotData['LocalDofVec']
        PlotNDofs       = len(PlotLocalDofVec)
        RefPlotDofVec       = RefPlotData['RefPlotDofVec']
        qpoint              = RefPlotData['qpoint']
        TestPlotFlag        = RefPlotData['TestPlotFlag']
        
        if PlotNDofs > 0:
            PlotDispVector               = np.zeros([PlotNDofs, N_TimeSteps])
            PlotDispVector[:,0]          = Un[PlotLocalDofVec]
        else:   
            PlotDispVector               = []
        
        PlotCntStressData               = []
        #PlotCntStrData               = []
        #PlotNormGapData               = []
    
        
    #Arc Length Parameters
    Norm_RefLoadVector                    = norm(RefLoadVector)
    TolF                                 = RelTol*Norm_RefLoadVector

    
    
    #Initializing solver variables
    #DeformedCoordVec = NodeCoordVec + Un
    IntfcElemList_DofVector = np.array([IntfcElem['DofIdList'] for IntfcElem in IntfcElemList], dtype=int).T
    ElemList_Un =   Un[IntfcElemList_DofVector]
    
    GlobSecK, InvGlobSecK, ElemList_HasStiffness = updateGlobalSecK(PenaltyData, GlobK, IntfcElemList, ElemList_Un, ConstrainedDOFIdVector)
    
    
    #Updating GlobalRefTransientLoadVector to include reactions        
    RefDispVector = np.dot(InvGlobSecK, RefLoadVector)
    RefDispVector[ConstrainedDOFIdVector] = 0.0
    RefLoadVector = np.dot(GlobSecK, RefDispVector)
    
    
    DelUp           = np.dot(InvGlobSecK, RefLoadVector)
    LocDelUp        = calcNodalGapVec(DelUp, NodalGapCalcMatrix, IntfcElemList_DofVector)
    DelLambda       = DelLambda_ini
    NormLocDelUp    = norm(LocDelUp)
    Delta_l         = DelLambda*NormLocDelUp
    DeltaLambda = 0.0
    
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
    
    
    #Initializing Residual
    Fext = DeltaLambda*RefLoadVector
    Fint = np.dot(GlobSecK, Un)
    R = Fint - Fext
    NormR  = norm(R)
    
    
    #if NormR<TolF:    raise Exception
    
    for TimeStepCount in range(MaxTimeStepCount):
            
        DeltaUn =       np.zeros(GlobNDOF)
        Nintfc =            len(np.where(ElemList_HasStiffness == 1)[0]) 
        Nd =                Nintfc**n + Nd0
        
        if TimeStepCount > 0:  
            Delta_l *=      (float(Nd)/LastIterCount)**m
    
        if TimeStepCount>0:
            print('')
            print(TimeStepCount, LastIterCount, Delta_l)
        
        for IterCount in range(MaxIterCount):
            
            DelUp = np.dot(InvGlobSecK, RefLoadVector)
            DelUf = -np.dot(InvGlobSecK, R)
            
            LocDelUp = calcNodalGapVec(DelUp, NodalGapCalcMatrix, IntfcElemList_DofVector)
            LocDelUf = calcNodalGapVec(DelUf, NodalGapCalcMatrix, IntfcElemList_DofVector)
            LocDeltaUn = calcNodalGapVec(DeltaUn, NodalGapCalcMatrix, IntfcElemList_DofVector)
                        
            if IterCount == 0:
                DelLambda = Delta_l/NormLocDelUp
                DelLambda0 = DelLambda
                
            else:
                
#                del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
#                del_lambda = (delta_l**2 - np.dot(Local_delta_u, Local_delta_u + Local_del_uf))/np.dot(Local_delta_u, Local_del_up)
                DelLambda = -np.dot(LocDeltaUn, LocDelUf)/np.dot(LocDeltaUn, LocDelUp)
                    
            
            DelUn = DelUf + DelLambda*DelUp
            
            #Updating increments
            DeltaUn +=  DelUn
            DeltaLambda += DelLambda
            
            Un += DelUn
            Fext = DeltaLambda*RefLoadVector
            
            ElemList_Un =   Un[IntfcElemList_DofVector]
            GlobSecK, InvGlobSecK, ElemList_HasStiffness = updateGlobalSecK(PenaltyData, GlobK, IntfcElemList, ElemList_Un, ConstrainedDOFIdVector)
            
            Fint = np.dot(GlobSecK, Un)
            R = Fint - Fext
            NormR  = norm(R)
    
            if NormR<TolF:
                LastIterCount = IterCount+1
                print('LastIterCount', LastIterCount)
                break
        
        else:
        
            print('Convergence Failed!',  FailedConvergenceCount, NormR/Norm_RefLoadVector)
            FailedConvergenceCount += 1
            LastIterCount = Nd
            
        
            
        if ExportFlag == 1:        
            if TimeStepCount%1==0:
                exportDispVecData(DispVecPath, ExportCount, Un)
                ExportCount += 1
        
        
        if TimeStepCount >= MaxTimeStepCount or FailedConvergenceCount >= MaxFailedConvergenceCount:
            AnalysisFinished = True
            break
    
        
            
    
    
    print('Analysis Finished Sucessfully..')
    
    
    
    
    
    

