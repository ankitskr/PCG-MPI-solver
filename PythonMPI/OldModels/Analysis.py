# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:15:39 2018

@author: z5166762
"""

import numpy as np
from numpy.linalg import inv, eig, norm, pinv
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from SBFEM.ElementMatrices import getCartesianCoordinates, getElementCoefficientMatrices, \
getStrainModes, getNaturalCoordinates, getDisplacementModes, getTractionEquivalentLoadVector, \
getStrainModes_SidefaceTraction, getDisplacementModes_SidefaceTraction, getCrackTipStressFactor

from Modules.GeneralFunctions import GaussLobattoIntegrationTable, GaussIntegrationTable
from scipy.interpolate import interp1d
from SBFEM.Transformations import Cartesian2Spherical, SignedAngleBetweenVectors
import SBFEM.MeshGen, SBFEM.Analysis, SBFEM.PostProcessor
from copy import deepcopy
from Transformations import Cartesian2Spherical, RotationMatrices
from SBFEM.GeneralFunctions import Interp1d_Periodic
import os

plt.rcParams.update({'font.size': 10})


class SubDomain(object):
    
    def __init__(self, SolverObj, RefPolyCellObj):
                
        """
        Calculates DOFs, ElementCoeffMat, ElementStiffnessMat, LoadVector etc for given RefPolyCellObj        
        """
        
        self.SolverObj = SolverObj
        
        #Reading RefPolyCellObj
        for attr, value in RefPolyCellObj.__dict__.iteritems():    self.__setattr__(attr, value)
        
        #Calculating Solver Parameters
        self.initParameters()
        self.getElementMatrices()
        self.calcElementCoefficientMatrix()
        self.calcCoefficientMatrix()
        self.calcStiffnessMatrix()
        self.classifyDOFIndices()
        self.calcTractionLoadVector()
        self.calcModes()
        self.backupCoordinates()
        
    
    def __setattr__(self, Attribute, Value):

        object.__setattr__(self, Attribute, Value)
    
    
    
    def initParameters(self):
        
        self.HasSidefaceTraction =              False
        self.Sideface_CumulativeDispMode =      np.zeros(self.N_DOF)
        
        NodeAngleList = []
            
        for NodeObj in self.NodeObjList:
            
            NodeVec = NodeObj.Coordinate - self.ScalingCenterCoordinate
            r_garbage, NodeAngle, phi_garbage = Cartesian2Spherical(NodeVec)
            NodeAngleList.append(NodeAngle)
        
        self.NodeAngleList = NodeAngleList
        
        
        
        if self.CrackType == 'PartiallyCracked':
                   
            CrackLength, CrackAngle, phi_garbage = Cartesian2Spherical(self.CrackVector)
            self.CrackAngle = CrackAngle
            self.CrackLength = CrackLength
            self.HasCrackPropagation = False
    
    
    
    def getElementMatrices(self):
        
        self.getElementCoefficientMatrices = getElementCoefficientMatrices
        self.getCartesianCoordinates = getCartesianCoordinates
        self.getNaturalCoordinates = getNaturalCoordinates
        self.getTractionEquivalentLoadVector = getTractionEquivalentLoadVector
        self.getStrainModes = getStrainModes
        self.getStrainModes_SidefaceTraction = getStrainModes_SidefaceTraction
        self.getDisplacementModes = getDisplacementModes
        self.getDisplacementModes_SidefaceTraction = getDisplacementModes_SidefaceTraction
        self.getCrackTipStressFactor = getCrackTipStressFactor
    

    def calcElementCoefficientMatrix(self):
                    
        for EdgeObj in self.EdgeObjList:
            
            e0, e1, e2 = self.getElementCoefficientMatrices(EdgeObj.LocalCoordinateList, self.MaterialObj, self.Thickness)
            
            EdgeObj.ElementCoefficientMatrixList = [e0, e1, e2]
            EdgeObj.N_CircumferencialData = len(EdgeObj.LocalCoordinateList)/3 - 1 #No. of data points in Circumferencial direction of SubDomain
        




    def calcCoefficientMatrix(self):
        
        
        self.SubDomainCoefficientMatrixList = []
        
        #Looping over E0, E1, E2
        for io in range(3):
                    
            SubDomainCoefficientMatrix = np.zeros([self.N_DOF, self.N_DOF])                
            self.SubDomainCoefficientMatrixList.append(SubDomainCoefficientMatrix)
            
            #Assembling Element Coefficient Matrix            
            for EdgeObj in self.EdgeObjList:
            
                ElementCoefficientMatrix = EdgeObj.ElementCoefficientMatrixList[io]
                LocalNodeIdList = EdgeObj.LocalNodeIdList
                N_Node = EdgeObj.N_Node
                
                for j1 in range(N_Node):
                    
                    NodeNo1 = LocalNodeIdList[j1]
                    
                    for j2 in range(N_Node):
                        
                        NodeNo2 = LocalNodeIdList[j2]
                        
                        SubDomainCoefficientMatrix[2*NodeNo1:2*NodeNo1+2, 2*NodeNo2:2*NodeNo2+2] += ElementCoefficientMatrix[2*j1:2*j1+2, 2*j2:2*j2+2]
            
            


    def calcStiffnessMatrix(self):
        """
        Currently based on Eigen Value Decomposition.
        TODO: Schur Decomposition for incorporating Power Logarithmic Singularity        
        """
        
        #Reading SubDomainCoefficientMatrixList
        E0 = self.SubDomainCoefficientMatrixList[0]
        E1 = self.SubDomainCoefficientMatrixList[1]
        E2 = self.SubDomainCoefficientMatrixList[2] 
        
        #Pre-Conditioning SubDomainCoefficientMatrix
#            P = np.zeros([N_DOFSubDomain, N_DOFSubDomain])
#            for j in range(N_DOFSubDomain):
#                
#                P[j,j] = 1/np.sqrt(E0[j,j])                
#                
#            E0 = np.dot(np.dot(P, E0), P)
#            E1 = np.dot(np.dot(P, E1), P)
#            E2 = np.dot(np.dot(P, E2), P)
        
        #Calculating Zp Matrix
        InverseE0 = inv(E0)
        TransposeE1 = E1.T
        E1DotInverseE0 = np.dot(E1, InverseE0)
        
        Zp00 = -np.dot(InverseE0, TransposeE1)
        Zp01 = InverseE0
        Zp10 = E2 - np.dot(E1DotInverseE0, TransposeE1)
        Zp11 = E1DotInverseE0
        
        #TODO: remove np.append as it is slower than list.append
        Zp = np.append(np.append(Zp00, Zp01, axis = 1), np.append(Zp10, Zp11, axis = 1), axis = 0)
        

        #Calculating Eigenvalues and Eigenvectors
        d, v = eig(Zp)
        
        #Reversing the pre-conditioning
#            PRev = np.zeros([2*N_DOFSubDomain, 2*N_DOFSubDomain])
#            PRev[0:N_DOFSubDomain, 0:N_DOFSubDomain] += P
#            PRev[N_DOFSubDomain:2*N_DOFSubDomain, N_DOFSubDomain:2*N_DOFSubDomain] += inv(P)
#            v = np.dot(PRev, v)
        
        #Filtering Eigenvalues and Eigenvectors
        d_i = np.argsort(-np.real(d)) #Getting sorted Index in desceding order of d 
        d_i = d_i[0:self.N_DOF] #Selecting first half sorted index
#            np.random.shuffle(d_i)
        
        d = d[d_i]     #Selecting Eigenvalues
        v = v.T[d_i].T #Selecting Eigenvectors
        
#            print(np.real(d))
        v[:,-2:] = 0
        v[::2, -2] = 1
        v[1::2, -1] = 1
        
        vu = v[0:self.N_DOF] #Displacement Mode
        vq = v[self.N_DOF:2*self.N_DOF] #Force Mode        
        
        self.EigValList = d
        self.EigVec_DispModeData = vu
        self.EigVec_ForceModeData = vq        
        
        #Calculating SubDomain Stiffness Matrix
        self.StiffnessMatrix = np.real(np.dot(vq, inv(vu)))
        
            
#        np.set_printoptions(threshold=np.inf)
        
#        print('')
#        
#        print('----------------------------')
#        print('')
#        print('Id', self.Id)
#        print('NodeId', self.NodeIdList)
#        print('')
#        print('Ke')
#        print(self.StiffnessMatrix.tolist())
#        print('')
#        print('E0')
#        print(E0.tolist())
#        print('')
#        print('E1')
#        print(E1.tolist())
#        print('')
#        print('E2')
#        print(E2.tolist())
#        print('')

    
    
    
    def classifyDOFIndices(self):
        
        #TODO: Check if the no. of DOF indices changes for 3D case
        if self.CrackType == 'PartiallyCracked':    
            
            self.FreeBodyDOFIdList = np.array([-2,-1], dtype = int) #EigenValue ~ 0.0
            self.SingularDOFIdList = np.array([-4,-3], dtype = int) #EigenValue ~ 0.5
            self.CrackTipStressDOFIdList = np.array([-5,-6], dtype = int) #EigenValue ~ 0.5
        
        else:
            
            self.FreeBodyDOFIdList = np.array([-2,-1], dtype = int)
            self.SingularDOFIdList = np.array([], dtype = int)
        
            
            



    def calcTractionLoadVector(self):
        
        self.StaticTractionLoadVector = np.zeros(self.N_DOF)
        self.TransientTractionLoadVector = np.zeros(self.N_DOF)
        
        for EdgeObj in self.EdgeObjList:
           
            Static_TrLoadVector = self.getTractionEquivalentLoadVector(EdgeObj.LocalCoordinateList, EdgeObj.StaticTractionVector)
            Transient_TrLoadVector = self.getTractionEquivalentLoadVector(EdgeObj.LocalCoordinateList, EdgeObj.TransientTractionVector)
            LocalNodeIdList = EdgeObj.LocalNodeIdList
            
            for j in range(EdgeObj.N_Node):
        
                LocalNodeId = LocalNodeIdList[j]
                Static_TrLoadVector_2D = Static_TrLoadVector[2*j:2*j+2]
                Transient_TrLoadVector_2D = Transient_TrLoadVector[2*j:2*j+2]
                
                self.StaticTractionLoadVector[2*LocalNodeId:2*LocalNodeId+2] += Static_TrLoadVector_2D
                self.TransientTractionLoadVector[2*LocalNodeId:2*LocalNodeId+2] += Transient_TrLoadVector_2D
    
    
    
    
    def calcModes(self):
    
        D = self.MaterialObj.D #Material Matrix
        
        #Looping over  Edge            
        for EdgeObj in self.EdgeObjList:                

            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
            
            #Selecting the EigenVector (EigVec_DispMode and Sideface_DispMode) correspoding to the DOFs of the  Edge
            Edge_EigVec_DispModeData =  self.EigVec_DispModeData[EdgeObj.LocalDOFIdList]
            
            StressModeData =                    np.zeros([EdgeObj.N_CircumferencialData, self.N_DOF, 3], dtype = complex)
            DisplacementModeData =              np.zeros([EdgeObj.N_CircumferencialData, self.N_DOF, 2], dtype = complex)
            
            EdgeObj.StressModeData = StressModeData
            EdgeObj.DisplacementModeData = DisplacementModeData
            
            #Looping over Gaussian Integration Points
            for k in range(EdgeObj.N_CircumferencialData):                    
                
                #Eigenvector in Strain Mode (Eq 3.62)     
                StrainModeData = self.getStrainModes(EdgeObj.LocalCoordinateList, self.EigValList, Edge_EigVec_DispModeData, ni[k])
                StressModeData[k] = np.dot(D, StrainModeData.T).T
                
                #Displacement Mode
                DisplacementModeData[k] = self.getDisplacementModes(EdgeObj.LocalCoordinateList, Edge_EigVec_DispModeData, ni[k])
            
    
    
    def backupCoordinates(self):
        
        self.Polygon_CoordinateList_2D = deepcopy([NodeObj.Coordinate[:-1] for NodeObj in self.NodeObjList])
        self.Edge_LocalCoordinateList = deepcopy([EdgeObj.LocalCoordinateList for EdgeObj in self.EdgeObjList])
        
                
            
    
    def calcSidefaceTractionParameters(self, TractionPolynomialCoefficientList, InterfaceElementSurfaceArea, SaveAttr = True):        
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        Sideface_PowerList = []
        
        CrackAngle = self.CrackAngle
        
        CrackMouth_LocalDOFIdList = self.CrackMouth_LocalDOFIdList
        CrackMouth_LocalXDOFId0 = CrackMouth_LocalDOFIdList[0]
        CrackMouth_LocalYDOFId0 = CrackMouth_LocalDOFIdList[1]
        CrackMouth_LocalXDOFId1 = CrackMouth_LocalDOFIdList[2]
        CrackMouth_LocalYDOFId1 = CrackMouth_LocalDOFIdList[3]
        
        #Exctracting Element Properties
        E0 = self.SubDomainCoefficientMatrixList[0]
        E1 = self.SubDomainCoefficientMatrixList[1]
        E2 = self.SubDomainCoefficientMatrixList[2] 
        
        Sideface_CumulativeDispMode = np.zeros(self.N_DOF)
        Sideface_CumulativeForceMode = np.zeros(self.N_DOF)
        
            
        #Calculating Sideface Displacement and Force Modes      
        N_SidefacePower = len(TractionPolynomialCoefficientList)
        Sideface_DispModeData = []
        for t in range(N_SidefacePower):
            
            Sideface_PowerList.append(t)
            
            #Updating TractionPolynomialCoefficient
            Fs = TractionPolynomialCoefficientList[t][0]*InterfaceElementSurfaceArea              
            Fn = TractionPolynomialCoefficientList[t][1]*InterfaceElementSurfaceArea                
            
            #Creating Sideface LoadVector
            SidefaceLoadVector = np.zeros(self.N_DOF)
            
            SidefaceLoadVector[CrackMouth_LocalXDOFId0] = -Fs*np.cos(CrackAngle) + Fn*np.sin(CrackAngle)
            SidefaceLoadVector[CrackMouth_LocalYDOFId0] = -Fs*np.sin(CrackAngle) - Fn*np.cos(CrackAngle)
            SidefaceLoadVector[CrackMouth_LocalXDOFId1] =  Fs*np.cos(CrackAngle) - Fn*np.sin(CrackAngle)
            SidefaceLoadVector[CrackMouth_LocalYDOFId1] =  Fs*np.sin(CrackAngle) + Fn*np.cos(CrackAngle)
            
            #Calculating Modes (Yang and Deeks, 2002)
            Do = ((t+1)**2)*E0 + (t+1)*(E1.T - E1) - E2
            Sideface_DispMode = np.dot(pinv(Do), -SidefaceLoadVector)
            Sideface_DispModeData.append(Sideface_DispMode)
            Sideface_CumulativeDispMode += Sideface_DispMode
            
            Fo = (t+1)*E0 + E1.T
            Sideface_ForceMode = np.dot(Fo, Sideface_DispMode)
            Sideface_CumulativeForceMode += Sideface_ForceMode
        
        
        #Calculating Sideface Traction Load Vector (in local DOF)
        SidefaceTractionLoadVector = -Sideface_CumulativeForceMode + np.dot(self.StiffnessMatrix, Sideface_CumulativeDispMode)
        Sideface_CumulativeDispMode = Sideface_CumulativeDispMode
        
        
        #Checking if Saving Attributes is required
        if SaveAttr:
            
            self.HasSidefaceTraction =          True
            self.Sideface_PowerList =           Sideface_PowerList
            self.N_SidefacePower =              N_SidefacePower
            self.SidefaceTractionLoadVector =   SidefaceTractionLoadVector
            self.Sideface_CumulativeDispMode =  Sideface_CumulativeDispMode
            
            #Calculating Modes due to Sideface Traction
            D = self.MaterialObj.D #Material Matrix
            Sideface_DispModeData =            np.array(Sideface_DispModeData, dtype = float).T
            
            for EdgeObj in self.EdgeObjList:
                
                ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
            
                Edge_Sideface_DispModeData =                Sideface_DispModeData[EdgeObj.LocalDOFIdList]
                Sideface_StressModeData =           np.zeros([EdgeObj.N_CircumferencialData, self.N_SidefacePower, 3], dtype = complex)
                EdgeObj.Sideface_StressModeData =   Sideface_StressModeData
                
                #Looping over Gaussian Integration Points
                for k in range(EdgeObj.N_CircumferencialData):                    
                
                    #StrainModes due to Sideface traction -- Yang and Deeks (2002) (Eq 75)
                    Sideface_StrainModeData = self.getStrainModes_SidefaceTraction(EdgeObj.LocalCoordinateList, self.Sideface_PowerList, Edge_Sideface_DispModeData, ni[k])
                    Sideface_StressModeData[k] = np.dot(D, Sideface_StrainModeData.T).T
                    
                    #TODO: Calc Disp mode due to sideface traction
                            
            
        else:   return Sideface_CumulativeDispMode
    
    
    
    
    def calcSidefaceTractionParameters_1(self, TractionPolynomialCoefficientList, InterfaceElementSurfaceArea, SaveAttr = True):        
        
        """
        For penalty method in contact
        """
        
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        Sideface_PowerList = []
        
        CrackAngle = self.CrackAngle
        
        CrackMouth_LocalDOFIdList = self.CrackMouth_LocalDOFIdList
        CrackMouth_LocalXDOFId0 = CrackMouth_LocalDOFIdList[0]
        CrackMouth_LocalYDOFId0 = CrackMouth_LocalDOFIdList[1]
        CrackMouth_LocalXDOFId1 = CrackMouth_LocalDOFIdList[2]
        CrackMouth_LocalYDOFId1 = CrackMouth_LocalDOFIdList[3]
        
        #Exctracting Element Properties
        E0 = self.SubDomainCoefficientMatrixList[0]
        E1 = self.SubDomainCoefficientMatrixList[1]
        E2 = self.SubDomainCoefficientMatrixList[2] 
        
        Sideface_CumulativeDispMode = np.zeros(self.N_DOF)
        Sideface_CumulativeForceMode = np.zeros(self.N_DOF)
        
            
        #Calculating Sideface Displacement and Force Modes      
        N_SidefacePower = len(TractionPolynomialCoefficientList)
        Sideface_DispModeList = []
        for t in range(N_SidefacePower):
            
            Sideface_PowerList.append(t)
            
            #Updating TractionPolynomialCoefficient
            Fs = TractionPolynomialCoefficientList[t][0]*InterfaceElementSurfaceArea              
            Fn = TractionPolynomialCoefficientList[t][1]*InterfaceElementSurfaceArea                
            
            #Creating Sideface LoadVector
            SidefaceLoadVector = np.zeros(self.N_DOF)
            
            SidefaceLoadVector[CrackMouth_LocalXDOFId0] = -Fs*np.cos(CrackAngle) + Fn*np.sin(CrackAngle)
            SidefaceLoadVector[CrackMouth_LocalYDOFId0] = -Fs*np.sin(CrackAngle) - Fn*np.cos(CrackAngle)
            SidefaceLoadVector[CrackMouth_LocalXDOFId1] =  Fs*np.cos(CrackAngle) - Fn*np.sin(CrackAngle)
            SidefaceLoadVector[CrackMouth_LocalYDOFId1] =  Fs*np.sin(CrackAngle) + Fn*np.cos(CrackAngle)
            
            #Calculating Modes (Yang and Deeks, 2002)
            Do = ((t+1)**2)*E0 + (t+1)*(E1.T - E1) - E2
            Sideface_DispMode = np.dot(pinv(Do), -SidefaceLoadVector)
            Sideface_DispModeList.append(Sideface_DispMode)
            Sideface_CumulativeDispMode += Sideface_DispMode
            
            Fo = (t+1)*E0 + E1.T
            Sideface_ForceMode = np.dot(Fo, Sideface_DispMode)
            Sideface_CumulativeForceMode += Sideface_ForceMode
        
        
        #Calculating Sideface Traction Load Vector (in local DOF)
        SidefaceTractionLoadVector = -Sideface_CumulativeForceMode + np.dot(self.StiffnessMatrix, Sideface_CumulativeDispMode)
        Sideface_CumulativeDispMode = Sideface_CumulativeDispMode
        
        
        #Checking if Saving Attributes is required
        if SaveAttr:
            
            self.HasSidefaceTraction =          True
            self.Sideface_PowerList =           Sideface_PowerList
            self.N_SidefacePower =              N_SidefacePower
            self.SidefaceTractionLoadVector =   SidefaceTractionLoadVector
            self.Sideface_CumulativeDispMode =  Sideface_CumulativeDispMode
            
            #Calculating Modes due to Sideface Traction
            D = self.MaterialObj.D #Material Matrix
            Sideface_DispModeList =            np.array(Sideface_DispModeList, dtype = float).T
            
            for EdgeObj in self.EdgeObjList:
                
                ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
            
                Edge_Sideface_DispModeList =                Sideface_DispModeList[EdgeObj.LocalDOFIdList]
                Sideface_StressModeData =           np.zeros([EdgeObj.N_CircumferencialData, self.N_SidefacePower, 3], dtype = complex)
                Sideface_DispModeData =                     np.zeros([EdgeObj.N_CircumferencialData, self.N_SidefacePower, 2], dtype = complex)
                EdgeObj.Sideface_StressModeData =   Sideface_StressModeData
                EdgeObj.Sideface_DisplacementModeData =     Sideface_DispModeData
                
                #Looping over Gaussian Integration Points
                for k in range(EdgeObj.N_CircumferencialData):                    
                
                    #StrainModes due to Sideface traction -- Yang and Deeks (2002) (Eq 75)
                    Sideface_StrainModeData = self.getStrainModes_SidefaceTraction(EdgeObj.LocalCoordinateList, self.Sideface_PowerList, Edge_Sideface_DispModeList, ni[k])
                    Sideface_StressModeData[k] = np.dot(D, Sideface_StrainModeData.T).T
                    Sideface_DispModeData[k] = self.getDisplacementModes_SidefaceTraction(self.Sideface_PowerList, Edge_Sideface_DispModeList, ni[k])
        
            
        else:   return Sideface_CumulativeDispMode
        
            
    
    def initUnitSidefaceTractionData(self, tmax):
    
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        
        #Transforming Unit Normal sideface traction factor into global coordinate system
        CrackAngle = self.CrackAngle
        
        CrackMouth_LocalDOFIdList = self.CrackMouth_LocalDOFIdList
        CrackMouth_LocalXDOFId0 = CrackMouth_LocalDOFIdList[0]
        CrackMouth_LocalYDOFId0 = CrackMouth_LocalDOFIdList[1]
        CrackMouth_LocalXDOFId1 = CrackMouth_LocalDOFIdList[2]
        CrackMouth_LocalYDOFId1 = CrackMouth_LocalDOFIdList[3]
        
        Pnt = np.zeros(self.N_DOF)
        Pnt[CrackMouth_LocalXDOFId0] = -np.sin(CrackAngle)
        Pnt[CrackMouth_LocalYDOFId0] =  np.cos(CrackAngle)
        Pnt[CrackMouth_LocalXDOFId1] =  np.sin(CrackAngle)
        Pnt[CrackMouth_LocalYDOFId1] = -np.cos(CrackAngle)
        
        #Calculating Normal SidefaceTraction Force Vector
        RefInterfaceCellObj = self.CrackTip_InterfaceCellObj
        CrackSurfaceArea = RefInterfaceCellObj.Thickness*self.CrackLength
        Fnt = CrackSurfaceArea*Pnt
        
        #Exctracting Element Properties
        E0 = self.SubDomainCoefficientMatrixList[0]
        E1 = self.SubDomainCoefficientMatrixList[1]
        E2 = self.SubDomainCoefficientMatrixList[2] 
        
        Sideface_PowerList = []
        Sideface_LoadVectorList = []
        Sideface_DispModeList = []
        
        #Calculating Sideface Displacement and Force Modes      
        for t in range(tmax+1):
            
            k_t = self.StiffnessMatrix - (t+1)*E0 - E1.T
            fi_t = -pinv(((t+1)**2)*E0 + (t+1)*(E1.T - E1) - E2)
            
            Sideface_DispMode = np.dot(fi_t, Fnt)
            Sideface_LoadVector = np.dot(k_t, Sideface_DispMode)
            
            Sideface_PowerList.append(t)
            Sideface_DispModeList.append(Sideface_DispMode)
            Sideface_LoadVectorList.append(Sideface_LoadVector)
        
        
        self.UnitSidefaceTractionData = {'UnitDispModeList':         np.array(Sideface_DispModeList, dtype = float),
                                         'UnitLoadVectorList':       np.array(Sideface_LoadVectorList, dtype = float),
                                         'PowerList':                Sideface_PowerList}
        
    
    
    
    def applyLinearSidefaceTraction(self, LinearSidefaceTractionVector):
        
        self.HasSidefaceTraction =          True
        self.Sideface_PowerList =           self.UnitSidefaceTractionData['PowerList']
        self.N_SidefacePower =              len(self.Sideface_PowerList)
        
        #Calculating SidefaceTraction LoadVector
        f0 = self.UnitSidefaceTractionData['UnitLoadVectorList'][0]
        f1 = self.UnitSidefaceTractionData['UnitLoadVectorList'][1]
        CnP = np.array([f1, f0-f1], dtype = float).T
        self.SidefaceTractionLoadVector = np.dot(CnP, LinearSidefaceTractionVector)
        
        #Calculating Modes due to Sideface Traction
        CrackMouth_P = LinearSidefaceTractionVector[0]
        CrackTip_P = LinearSidefaceTractionVector[1]
        phi_t0 = CrackTip_P*self.UnitSidefaceTractionData['UnitDispModeList'][0]
        phi_t1 = (CrackMouth_P - CrackTip_P)*self.UnitSidefaceTractionData['UnitDispModeList'][1]
        Sideface_DispModeList = np.array([phi_t0, phi_t1], dtype=float).T
        self.Sideface_CumulativeDispMode = phi_t0 + phi_t1
        
        #Updating EdgeObj for post processing
        for EdgeObj in self.EdgeObjList:
            
            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
        
            Edge_Sideface_DispModeList =                Sideface_DispModeList[EdgeObj.LocalDOFIdList]
            Sideface_StressModeData =                   np.zeros([EdgeObj.N_CircumferencialData, self.N_SidefacePower, 3], dtype = complex)
            Sideface_DispModeData =                     np.zeros([EdgeObj.N_CircumferencialData, self.N_SidefacePower, 2], dtype = complex)
            EdgeObj.Sideface_StressModeData =           Sideface_StressModeData
            EdgeObj.Sideface_DisplacementModeData =     Sideface_DispModeData
            
            #Looping over Gaussian Integration Points
            for k in range(EdgeObj.N_CircumferencialData):                    
            
                #Strain and Disp Modes due to Sideface traction -- Yang and Deeks (2002) (Eq 68, 75)
                Sideface_StrainModeData = self.getStrainModes_SidefaceTraction(EdgeObj.LocalCoordinateList, self.Sideface_PowerList, Edge_Sideface_DispModeList, ni[k])
                Sideface_StressModeData[k] = np.dot(self.MaterialObj.D, Sideface_StrainModeData.T).T
                Sideface_DispModeData[k] = self.getDisplacementModes_SidefaceTraction(self.Sideface_PowerList, Edge_Sideface_DispModeList, ni[k])
    
    
    
    
    
    

    def getFieldDisplacement(self, RefCoordinate):
        
        RefX = RefCoordinate[0]
        RefY = RefCoordinate[1]
        RefCoordinate_2D = [RefX, RefY]
        
        #Checking if RefCoordinate lies inside the SubDomain
        #TODO: Point inside polygon will not work for higher order elements with curved edges
        if mpltPath.Path(self.Polygon_CoordinateList_2D).contains_points((RefCoordinate_2D,), radius = 1e-15)[0]:
                        
            RefLocalCoordinate = RefCoordinate - self.ScalingCenterCoordinate
            LocalRefX = RefLocalCoordinate[0]
            LocalRefY = RefLocalCoordinate[1]
            
            #Looping over Edges in the polygon
            N_Edge = len(self.EdgeObjList)
            for i in range(N_Edge):
                
                EdgeObj = self.EdgeObjList[i]
                
                #Extracting Natural Coordinates
                n, zi = self.getNaturalCoordinates(self.Edge_LocalCoordinateList[i], LocalRefX, LocalRefY)
                
                if not (n == None or zi == None):
                                
                    Edge_EigVec_DispModeData =  self.EigVec_DispModeData[EdgeObj.LocalDOFIdList]
                    DisplacementModeData = self.getDisplacementModes(self.Edge_LocalCoordinateList[i], Edge_EigVec_DispModeData, n)
                    
                    #Displacement at RefCoordinate
                    RefDisp = np.zeros(2, dtype = complex)
                    
                    if zi == 0.0:
                        
                        for jo in self.FreeBodyDOFIdList:
                            
                            cj = self.IntegrationConstantList[jo]
                            RefDisp += cj*DisplacementModeData[jo]
                        
                    else:
                        
                        for jo in range(self.N_DOF):
                            
                            cj = self.IntegrationConstantList[jo]
                            RefDisp += cj*(zi**self.EigValList[jo])*DisplacementModeData[jo]
                    
                    return np.real(np.array([RefDisp[0], RefDisp[1], 0.0], dtype = complex))
            
            else:   raise Exception
                
        else:   return []
    
    
    
    
    def calcSIF(self, RefIntegrationConstantList):
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
            
        #Selecting Singular Integration Constants
        SingularIntegrationConstantList = RefIntegrationConstantList[self.SingularDOFIdList]
        
        #Calculating Crack Angle
        R1, R2, R3 = RotationMatrices(theta = self.CrackAngle)
        
        #Calculating Parameters at each Gaussian Point in Subdomain
        GP_AngleList = []
        GP_RadialLengthList = []            
        SingularSigXXDataList = []            
        SingularSigYYDataList = []
        SingularTauXYDataList = []
        
        for EdgeObj in self.EdgeObjList:
            
            #Calculating Coordinates at Gauss Points
            GP_X = np.zeros(EdgeObj.N_CircumferencialData)
            GP_Y = np.zeros(EdgeObj.N_CircumferencialData)
            
            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
        
            for k in range(EdgeObj.N_CircumferencialData):
                
                n = ni[k]
                GP_X[k], GP_Y[k] = self.getCartesianCoordinates(EdgeObj.LocalCoordinateList, self.ScalingCenterCoordinate, n, 1.0)
                
            
            #Calculating Radial Boundary Coordinates at Gaussian Points (with Scaling Center at origin)    
            xb = GP_X - self.ScalingCenterCoordinate[0]
            yb = GP_Y - self.ScalingCenterCoordinate[1]
            rb = np.sqrt(xb**2 + yb**2)
            theta_b = np.arctan2(yb, xb)              
            GP_RadialLengthList += list(rb)
            GP_AngleList += list(theta_b)
            
            #Storing Singular Stress Data at Gaussian Points
            StressModeData = EdgeObj.StressModeData         
            
            for k in range(EdgeObj.N_CircumferencialData):    
                
                SingularStressModeData = StressModeData[k][self.SingularDOFIdList].T
                SingularStressData = np.dot(SingularStressModeData, SingularIntegrationConstantList)
                
                SingularSigXXDataList.append(SingularStressData[0])
                SingularSigYYDataList.append(SingularStressData[1])
                SingularTauXYDataList.append(SingularStressData[2])
            
        
        
        #Interpolating Stress
        f_SingularSigXX = Interp1d_Periodic(GP_AngleList, SingularSigXXDataList, 2*np.pi, InterpOrder = 1)
        SingularSigXX = f_SingularSigXX(self.CrackAngle)
        
        f_SingularSigYY = Interp1d_Periodic(GP_AngleList, SingularSigYYDataList, 2*np.pi, InterpOrder = 1)
        SingularSigYY = f_SingularSigYY(self.CrackAngle)            
        
        f_SingularTauXY = Interp1d_Periodic(GP_AngleList, SingularTauXYDataList, 2*np.pi, InterpOrder = 1)
        SingularTauXY = f_SingularTauXY(self.CrackAngle)
        
        
        #Transforming Stresses
        SingularStressTensor = np.array([[SingularSigXX, SingularTauXY,      0],
                                         [SingularTauXY, SingularSigYY,      0],
                                         [            0,             0,      0]], dtype = complex)
        
        SingularStressTensor_Transformed = np.dot(np.dot(R1.T, SingularStressTensor), R1)            
        SingularSigYY_Transformed = SingularStressTensor_Transformed[1, 1]
        SingularTauXY_Transformed = SingularStressTensor_Transformed[0, 1]            
        
        
        #Calculating Characteristic Length
        f_L0 = Interp1d_Periodic(GP_AngleList, GP_RadialLengthList, 2*np.pi, InterpOrder = 1)                
        L0 = f_L0(self.CrackAngle)
        
        #Calculating Stress Intensity Factors
        Ao = np.sqrt(2*np.pi*L0)
        KI = np.real(Ao*SingularSigYY_Transformed)
        KII = np.real(Ao*SingularTauXY_Transformed)
        
        SIF = [KI, KII]
        
        #Calculating Alpha
        th_p = 2*np.arctan2(-2*KII/KI, 1 + np.sqrt(1 + 8*(KII/KI)**2))
        K = (KI*np.cos(th_p/2.0)**2 - (3.0/2.0)*KII*np.sin(th_p))*np.cos(th_p/2.0)
        Alpha = abs(self.MaterialObj.KIC/K)
        
        #Calculating Crack Propagation Direction based on Max Principle stress criterion
        if abs(KI/KII) < 0.015:  KI = 1e-10
        
#        print('KI_KII', KI, KII)
        
        th_p = 2*np.arctan2(-2*KII/KI, 1 + np.sqrt(1 + 8*(KII/KI)**2))
        
#        if not KI == 1e-10:
#            
##            if abs(th_p) > 15*np.pi/180: th_p *= 0.5
#        
#            if abs(th_p) > 10*np.pi/180: 
#                
#                th_d = abs(th_p*180/np.pi)
#                th_p = np.sign(th_p)*(10 + (th_d-10)/6)*np.pi/180
                
        
        CrackPropagationAngle = self.CrackAngle + th_p
#        print('CrackPropagationAngle', CrackPropagationAngle*180/np.pi)
        return SIF, Alpha, CrackPropagationAngle
    
    
    
    def getCrackTipRegionCoordinateParameters(self, RefRadius):
        
#        RelRadius = RefRadius/self.CrackLength
#        if RelRadius > 0.35:    raise Exception('RelRadius = ' + str(RelRadius) + ' (>0.35)') #Max value in case of Quadtree based rosette
        
        
        LocalRefAngleList = np.linspace(-np.pi/2, np.pi/2, 181)
        
        CrackTipRegion_CoordinateParamList = []
        
        for LocalRefAngle in LocalRefAngleList:
            
            RefAngle = LocalRefAngle + self.CrackAngle
            
            #Calculating Coordinates 
            X = self.ScalingCenterCoordinate[0] + RefRadius*np.cos(RefAngle)
            Y = self.ScalingCenterCoordinate[1] + RefRadius*np.sin(RefAngle)
            
            
            CrackTipRegion_CoordinateParam = {'RefCoordinate':      np.array([X, Y, 0.0], dtype=float),
                                              'RefAngle_Local':     LocalRefAngle, 
                                              'RefRadius':          RefRadius}
        
            CrackTipRegion_CoordinateParamList.append(CrackTipRegion_CoordinateParam)
        
        
        
        
        
        return CrackTipRegion_CoordinateParamList
    
    
        
    
    
    def calcMaxHoopStressCriteriaParameters(self, RelRadius = 0.05):
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        if RelRadius > 0.35:    raise Exception #Max value in case of Quadtree based rosette
        
        #Calculating RefRadius
        RefRadius = RelRadius*self.CrackLength
        
        #Calculating Parameters at each Gaussian Point in Subdomain
        GP_AngleList = []         
        GP_SigXXList = []        
        GP_SigYYList = []
        GP_TauXYList = []
        
        for EdgeObj in self.EdgeObjList:
            
            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
        
            #Calculating Coordinates at Gauss Points
            GP_X = np.zeros(EdgeObj.N_CircumferencialData)
            GP_Y = np.zeros(EdgeObj.N_CircumferencialData)
            
            for k in range(EdgeObj.N_CircumferencialData):
                
                n = ni[k]
                GP_X[k], GP_Y[k] = self.getCartesianCoordinates(EdgeObj.LocalCoordinateList, self.ScalingCenterCoordinate, n, 1.0)
                
            
            #Calculating Radial Boundary Coordinates at Gaussian Points (with Scaling Center at origin)    
            xb = GP_X - self.ScalingCenterCoordinate[0]
            yb = GP_Y - self.ScalingCenterCoordinate[1]
            theta_b = np.arctan2(yb, xb)          
            rb = (xb**2 + yb**2)**0.5
            Ref_zi = RefRadius/rb
            GP_AngleList += list(theta_b)
            
            
            #Calculating Stresses          
            for k in range(EdgeObj.N_CircumferencialData):
                
                #Stresses due to Elastic Deformation
                Stress = np.zeros(3, dtype = complex)
                
                for jo in range(self.N_DOF):
                    
                    cj = self.IntegrationConstantList[jo]
                    Stress += cj*(Ref_zi**(self.EigValList[jo] - 1))*EdgeObj.StressModeData[k, jo]
                    
                #Stress due to Sideface traction
                if self.HasSidefaceTraction:
            
                    for jo in range(self.N_SidefacePower):
                        
                        t = self.Sideface_PowerList[jo]
                        zi_t = Ref_zi**t
                        Stress += zi_t*EdgeObj.Sideface_StressModeData[k, jo]
                
                Stress = np.real(Stress)
                GP_SigXXList.append(Stress[0])
                GP_SigYYList.append(Stress[1])
                GP_TauXYList.append(Stress[2])
                

        #Creating Stress interpolation functions
        f_SigXX = Interp1d_Periodic(GP_AngleList, GP_SigXXList, 2*np.pi, InterpOrder = 1)
        f_SigYY = Interp1d_Periodic(GP_AngleList, GP_SigYYList, 2*np.pi, InterpOrder = 1)
        f_TauXY = Interp1d_Periodic(GP_AngleList, GP_TauXYList, 2*np.pi, InterpOrder = 1)
        
        
        
        #Calculating Stresses around Crack-Tip Region (On a circle with radius = RefRadius and Center at crack-tip)
        TestAngleList = np.linspace(-np.pi/2, np.pi/2, 181) + self.CrackAngle
        
        self.CrackTipRegion_HoopStressParamList = []
        
        for TestAngle in TestAngleList:
            
            if TestAngle > np.pi:  TestAngle -= 2*np.pi
            elif TestAngle <-np.pi:  TestAngle += 2*np.pi
            
            RefAngle = TestAngle - self.CrackAngle #Local test Angle wrt to crack angle
            if RefAngle > np.pi:  RefAngle -= 2*np.pi
            elif RefAngle <-np.pi:  RefAngle += 2*np.pi
            
            if not -np.pi/2 <= RefAngle <= np.pi/2: raise Exception 
            
            #Interpolating Stresses
            SigXX = f_SigXX(TestAngle)
            SigYY = f_SigYY(TestAngle)            
            TauXY = f_TauXY(TestAngle)
            
            #Transforming Stresses to Local coordinates wrt to CrackVector
            GlobalStressTensor = np.array([[SigXX,   TauXY,      0],
                                           [TauXY,   SigYY,      0],
                                           [    0,       0,      0]], dtype = complex)
            
    
            R1, R2, R3 = RotationMatrices(theta = TestAngle)
        
            PolarStressTensor = np.real(np.dot(np.dot(R1.T, GlobalStressTensor), R1))          
            SigRR = PolarStressTensor[0, 0]
            SigTT = PolarStressTensor[1, 1]
            TauRT = PolarStressTensor[0, 1]
            
            
            #Calculating Coordinates 
            X = self.ScalingCenterCoordinate[0] + RefRadius*np.cos(TestAngle)
            Y = self.ScalingCenterCoordinate[1] + RefRadius*np.sin(TestAngle)
            
            
            CrackTipRegion_HoopStressParam = {'PolarStressList':    [SigRR, SigTT, TauRT],
                                              'RefCoordinate':      np.array([X, Y, 0.0], dtype=float),
                                              'RefAngle':           RefAngle, 
                                              'RefRadius':          RefRadius}
        
            self.CrackTipRegion_HoopStressParamList.append(CrackTipRegion_HoopStressParam)
        
        
    
    
    
    def calcMaxPStressCriteriaParameters_Analytical(self):
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        CrackTipStress_IntegrationConstantList = self.IntegrationConstantList[self.CrackTipStressDOFIdList]
        
        CrackTip_SigXXList = []            
        CrackTip_SigYYList = []
        CrackTip_TauXYList = []
        
        for EdgeObj in self.EdgeObjList:
            
            StressModeData = EdgeObj.StressModeData         
            
            for k in range(EdgeObj.N_CircumferencialData):
                
                CrackTipStressModeData = StressModeData[k][self.CrackTipStressDOFIdList].T
                CrackTipStress_0 = np.dot(CrackTipStressModeData, CrackTipStress_IntegrationConstantList)
                CrackTipStress_1 = EdgeObj.Sideface_StressModeData[k, 0]
                
                CrackTipStress = CrackTipStress_0 + CrackTipStress_1
                
                CrackTip_SigXXList.append(CrackTipStress[0])
                CrackTip_SigYYList.append(CrackTipStress[1])
                CrackTip_TauXYList.append(CrackTipStress[2])
        
        
        CrackTip_SigXX = np.mean(np.real(CrackTip_SigXXList))
        CrackTip_SigYY = np.mean(np.real(CrackTip_SigYYList))
        CrackTip_TauXY = np.mean(np.real(CrackTip_TauXYList))
        
        
        #Transforming Stresses to Local coordinates wrt to CrackVector
        R1, R2, R3 = RotationMatrices(theta = self.CrackAngle)
        
        CrackTip_GlobalStressTensor = np.array([[CrackTip_SigXX,   CrackTip_TauXY,      0],
                                                [CrackTip_TauXY,   CrackTip_SigYY,      0],
                                                [             0,                0,      0]], dtype = complex)
        
        CrackTip_LocalStressTensor = np.real(np.dot(np.dot(R1.T, CrackTip_GlobalStressTensor), R1))          
        CrackTip_LocalSigXX = CrackTip_LocalStressTensor[0, 0]
        CrackTip_LocalSigYY = CrackTip_LocalStressTensor[1, 1]
        CrackTip_LocalTauXY = CrackTip_LocalStressTensor[0, 1]
        
        CrackTip_MC_Center = (CrackTip_LocalSigXX + CrackTip_LocalSigYY)/2.0
        CrackTip_MC_Radius = np.sqrt(0.25*(CrackTip_LocalSigXX - CrackTip_LocalSigYY)**2 + CrackTip_LocalTauXY**2)
        
        #Calculating Principle Stresses
        CrackTip_PStress1 = CrackTip_MC_Center + CrackTip_MC_Radius
        CrackTip_PStress2 = CrackTip_MC_Center - CrackTip_MC_Radius
            
        
        self.CrackTip_PStressParam =    {'MC_Center':              CrackTip_MC_Center,
                                         'MC_Radius':              CrackTip_MC_Radius,
                                         'LocalStressList':     [CrackTip_LocalSigXX, CrackTip_LocalSigYY, CrackTip_LocalTauXY],
                                         'PStressList':         [CrackTip_PStress1, CrackTip_PStress2]}
        
        
        
        
                
#        
#        RelRadius = 0.05
#        
#        if RelRadius > 0.35:    raise Exception #Max value in case of Quadtree based rosette
#        
#        #Calculating RefRadius
#        RefRadius = RelRadius*self.CrackLength
#        
#        #Calculating Parameters at each Gaussian Point in Subdomain
#        GP_AngleList = []         
#        GP_SigXXList = []        
#        GP_SigYYList = []
#        GP_TauXYList = []
#        
#        for EdgeObj in self.EdgeObjList:
#            
#            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
#        
#            #Calculating Coordinates at Gauss Points
#            GP_X = np.zeros(EdgeObj.N_CircumferencialData)
#            GP_Y = np.zeros(EdgeObj.N_CircumferencialData)
#            
#            for k in range(EdgeObj.N_CircumferencialData):
#                
#                n = ni[k]
#                GP_X[k], GP_Y[k] = self.getCartesianCoordinates(EdgeObj.LocalCoordinateList, self.ScalingCenterCoordinate, n, 1.0)
#                
#            
#            #Calculating Radial Boundary Coordinates at Gaussian Points (with Scaling Center at origin)    
#            xb = GP_X - self.ScalingCenterCoordinate[0]
#            yb = GP_Y - self.ScalingCenterCoordinate[1]
#            theta_b = np.arctan2(yb, xb)          
#            rb = (xb**2 + yb**2)**0.5
#            Ref_zi = RefRadius/rb
#            GP_AngleList += list(theta_b)
#            
#            
#            #Calculating Stresses          
#            for k in range(EdgeObj.N_CircumferencialData):
#                
#                #Stresses due to Elastic Deformation
#                Stress = np.zeros(3, dtype = complex)
#                
#                for jo in range(self.N_DOF):
#                    
#                    cj = self.IntegrationConstantList[jo]
#                    Stress += cj*(Ref_zi**(self.EigValList[jo] - 1))*EdgeObj.StressModeData[k, jo]
#                    
#                #Stress due to Sideface traction
#                if self.HasSidefaceTraction:
#            
#                    for jo in range(self.N_SidefacePower):
#                        
#                        t = self.Sideface_PowerList[jo]
#                        zi_t = Ref_zi**t
#                        Stress += zi_t*EdgeObj.Sideface_StressModeData[k, jo]
#                
#                Stress = np.real(Stress)
#                GP_SigXXList.append(Stress[0])
#                GP_SigYYList.append(Stress[1])
#                GP_TauXYList.append(Stress[2])
#                
#
#        #Creating Stress interpolation functions
#        f_SigXX = Interp1d_Periodic(GP_AngleList, GP_SigXXList, 2*np.pi, InterpOrder = 1)
#        f_SigYY = Interp1d_Periodic(GP_AngleList, GP_SigYYList, 2*np.pi, InterpOrder = 1)
#        f_TauXY = Interp1d_Periodic(GP_AngleList, GP_TauXYList, 2*np.pi, InterpOrder = 1)
#        
#        
#        #Calculating Stresses around Crack-Tip Region (On a circle with radius = RefRadius and Center at crack-tip)
#        TestAngleList = np.linspace(-np.pi, np.pi, 361)
#        SigXXList = [f_SigXX(TestAngle) for TestAngle in TestAngleList]
#        SigYYList = [f_SigYY(TestAngle) for TestAngle in TestAngleList]
#        TauXYList = [f_TauXY(TestAngle) for TestAngle in TestAngleList]
#        
#        
#        #Calculating Mean Stress at the Crack-tip
#        CrackTip_SigXX = np.mean(SigXXList)
#        CrackTip_SigYY = np.mean(SigYYList)
#        CrackTip_TauXY = np.mean(TauXYList)
#        
#        
#        #Transforming Stresses to Local coordinates wrt to CrackVector
#        R1, R2, R3 = RotationMatrices(theta = self.CrackAngle)
#        
#        CrackTip_GlobalStressTensor = np.array([[CrackTip_SigXX,   CrackTip_TauXY,      0],
#                                                [CrackTip_TauXY,   CrackTip_SigYY,      0],
#                                                [             0,                0,      0]], dtype = complex)
#        
#        CrackTip_LocalStressTensor = np.real(np.dot(np.dot(R1.T, CrackTip_GlobalStressTensor), R1))          
#        CrackTip_LocalSigXX = CrackTip_LocalStressTensor[0, 0]
#        CrackTip_LocalSigYY = CrackTip_LocalStressTensor[1, 1]
#        CrackTip_LocalTauXY = CrackTip_LocalStressTensor[0, 1]
#        
#        CrackTip_MC_Center = (CrackTip_LocalSigXX + CrackTip_LocalSigYY)/2.0
#        CrackTip_MC_Radius = np.sqrt(0.25*(CrackTip_LocalSigXX - CrackTip_LocalSigYY)**2 + CrackTip_LocalTauXY**2)
#        
#        #Calculating Principle Stresses
#        CrackTip_PStress1 = CrackTip_MC_Center + CrackTip_MC_Radius
#        CrackTip_PStress2 = CrackTip_MC_Center - CrackTip_MC_Radius
#            
#        
#        self.CrackTip_PStressParam =    {'MC_Center':              CrackTip_MC_Center,
#                                         'MC_Radius':              CrackTip_MC_Radius,
#                                         'LocalStressList':     [CrackTip_LocalSigXX, CrackTip_LocalSigYY, CrackTip_LocalTauXY],
#                                         'PStressList':         [CrackTip_PStress1, CrackTip_PStress2]}
#        
#        
    
    
    
    def calcMaxPStressCriteriaParameters(self, RelRadius = 0.22):
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        if RelRadius > 0.35:    raise Exception #Max value in case of Quadtree based rosette
        
        #Calculating RefRadius
        RefRadius = RelRadius*self.CrackLength
        
        #Calculating Parameters at each Gaussian Point in Subdomain
        GP_AngleList = []         
        GP_SigXXList = []        
        GP_SigYYList = []
        GP_TauXYList = []
        
        for EdgeObj in self.EdgeObjList:
            
            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
        
            #Calculating Coordinates at Gauss Points
            GP_X = np.zeros(EdgeObj.N_CircumferencialData)
            GP_Y = np.zeros(EdgeObj.N_CircumferencialData)
            
            for k in range(EdgeObj.N_CircumferencialData):
                
                n = ni[k]
                GP_X[k], GP_Y[k] = self.getCartesianCoordinates(EdgeObj.LocalCoordinateList, self.ScalingCenterCoordinate, n, 1.0)
                
            
            #Calculating Radial Boundary Coordinates at Gaussian Points (with Scaling Center at origin)    
            xb = GP_X - self.ScalingCenterCoordinate[0]
            yb = GP_Y - self.ScalingCenterCoordinate[1]
            theta_b = np.arctan2(yb, xb)          
            rb = (xb**2 + yb**2)**0.5
            Ref_zi = RefRadius/rb
            GP_AngleList += list(theta_b)
            
            
            #Calculating Stresses          
            for k in range(EdgeObj.N_CircumferencialData):
                
                #Stresses due to Elastic Deformation
                Stress = np.zeros(3, dtype = complex)
                
                for jo in range(self.N_DOF):
                    
                    cj = self.IntegrationConstantList[jo]
                    Stress += cj*(Ref_zi**(self.EigValList[jo] - 1))*EdgeObj.StressModeData[k, jo]
                    
                #Stress due to Sideface traction
                if self.HasSidefaceTraction:
            
                    for jo in range(self.N_SidefacePower):
                        
                        t = self.Sideface_PowerList[jo]
                        zi_t = Ref_zi**t
                        Stress += zi_t*EdgeObj.Sideface_StressModeData[k, jo]
                
                Stress = np.real(Stress)
                GP_SigXXList.append(Stress[0])
                GP_SigYYList.append(Stress[1])
                GP_TauXYList.append(Stress[2])
                

        #Creating Stress interpolation functions
        f_SigXX = Interp1d_Periodic(GP_AngleList, GP_SigXXList, 2*np.pi, InterpOrder = 1)
        f_SigYY = Interp1d_Periodic(GP_AngleList, GP_SigYYList, 2*np.pi, InterpOrder = 1)
        f_TauXY = Interp1d_Periodic(GP_AngleList, GP_TauXYList, 2*np.pi, InterpOrder = 1)
        
        
        #Calculating Stresses around Crack-Tip Region (On a circle with radius = RefRadius and Center at crack-tip)
        TestAngleList = np.linspace(-np.pi, np.pi, 361)
        SigXXList = [f_SigXX(TestAngle) for TestAngle in TestAngleList]
        SigYYList = [f_SigYY(TestAngle) for TestAngle in TestAngleList]
        TauXYList = [f_TauXY(TestAngle) for TestAngle in TestAngleList]
        
        
        #Calculating Mean Stress at the Crack-tip
        CrackTip_SigXX = np.mean(SigXXList)
        CrackTip_SigYY = np.mean(SigYYList)
        CrackTip_TauXY = np.mean(TauXYList)
        
        
        #Transforming Stresses to Local coordinates wrt to CrackVector
        R1, R2, R3 = RotationMatrices(theta = self.CrackAngle)
        
        CrackTip_GlobalStressTensor = np.array([[CrackTip_SigXX,   CrackTip_TauXY,      0],
                                                [CrackTip_TauXY,   CrackTip_SigYY,      0],
                                                [             0,                0,      0]], dtype = complex)
        
        CrackTip_LocalStressTensor = np.real(np.dot(np.dot(R1.T, CrackTip_GlobalStressTensor), R1))          
        CrackTip_LocalSigXX = CrackTip_LocalStressTensor[0, 0]
        CrackTip_LocalSigYY = CrackTip_LocalStressTensor[1, 1]
        CrackTip_LocalTauXY = CrackTip_LocalStressTensor[0, 1]
        
        CrackTip_MC_Center = (CrackTip_LocalSigXX + CrackTip_LocalSigYY)/2.0
        CrackTip_MC_Radius = np.sqrt(0.25*(CrackTip_LocalSigXX - CrackTip_LocalSigYY)**2 + CrackTip_LocalTauXY**2)
        
        #Calculating Principle Stresses
        CrackTip_PStress1 = CrackTip_MC_Center + CrackTip_MC_Radius
        CrackTip_PStress2 = CrackTip_MC_Center - CrackTip_MC_Radius
            
        
        self.CrackTip_PStressParam =    {'MC_Center':              CrackTip_MC_Center,
                                         'MC_Radius':              CrackTip_MC_Radius,
                                         'LocalStressList':     [CrackTip_LocalSigXX, CrackTip_LocalSigYY, CrackTip_LocalTauXY],
                                         'PStressList':         [CrackTip_PStress1, CrackTip_PStress2]}
    
    
    
    
    
    def calcCrackTipMohrCircleParameters_Backup(self, Ref_zi = 0.5):
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        if Ref_zi == 0.0:   raise Exception("Check if number of summation terms is SubDomain['N_DOF'] - 2. The last two summation terms produce singularity when zi ~ 0")
        
        
        #Calculating Parameters at each Gaussian Point in Subdomain
        GP_AngleList = []         
        GP_SigXXList = []        
        GP_SigYYList = []
        GP_TauXYList = []
        
        for EdgeObj in self.EdgeObjList:
            
            ni, wi = GaussIntegrationTable(EdgeObj.N_CircumferencialData)
        
            #Calculating Coordinates at Gauss Points
            GP_X = np.zeros(EdgeObj.N_CircumferencialData)
            GP_Y = np.zeros(EdgeObj.N_CircumferencialData)
            
            for k in range(EdgeObj.N_CircumferencialData):
                
                n = ni[k]
                GP_X[k], GP_Y[k] = self.getCartesianCoordinates(EdgeObj.LocalCoordinateList, self.ScalingCenterCoordinate, n, 1.0)
                
            
            #Calculating Radial Boundary Coordinates at Gaussian Points (with Scaling Center at origin)    
            xb = GP_X - self.ScalingCenterCoordinate[0]
            yb = GP_Y - self.ScalingCenterCoordinate[1]
            theta_b = np.arctan2(yb, xb)              
            GP_AngleList += list(theta_b)
            
            
            #Calculating Stresses          
            for k in range(EdgeObj.N_CircumferencialData):
                
                #Stresses due to Elastic Deformation
                Stress = np.zeros(3, dtype = complex)
                
                for jo in range(self.N_DOF):
                    
                    cj = self.IntegrationConstantList[jo]
                    Stress += cj*(Ref_zi**(self.EigValList[jo] - 1))*EdgeObj.StressModeData[k, jo]
                    
                #Stress due to Sideface traction
                if self.HasSidefaceTraction:
            
                    for jo in range(self.N_SidefacePower):
                        
                        t = self.Sideface_PowerList[jo]
                        zi_t = Ref_zi**t
                        Stress += zi_t*EdgeObj.Sideface_StressModeData[k, jo]
                
                Stress = np.real(Stress)
                GP_SigXXList.append(Stress[0])
                GP_SigYYList.append(Stress[1])
                GP_TauXYList.append(Stress[2])
                

        #Creating Stress interpolation functions
        f_SigXX = Interp1d_Periodic(GP_AngleList, GP_SigXXList, 2*np.pi, InterpOrder = 1)
        f_SigYY = Interp1d_Periodic(GP_AngleList, GP_SigYYList, 2*np.pi, InterpOrder = 1)
        f_TauXY = Interp1d_Periodic(GP_AngleList, GP_TauXYList, 2*np.pi, InterpOrder = 1)
        
        #Calculating CracTip Stresses
        TestAngleList = np.linspace(-np.pi, np.pi, 181)
        
        self.CrackTip_MohrCircleParamList = []
        
        R1, R2, R3 = RotationMatrices(theta = self.CrackAngle)
        
        for TestAngle in TestAngleList:
            
            SigXX = f_SigXX(TestAngle)
            SigYY = f_SigYY(TestAngle)            
            TauXY = f_TauXY(TestAngle)
                
            #Transforming Stresses to Local coordinates wrt to CrackVector
            GlobalStressTensor = np.array([[SigXX,   TauXY,      0],
                                           [TauXY,   SigYY,      0],
                                           [    0,       0,      0]], dtype = complex)
            
            
            LocalStressTensor = np.real(np.dot(np.dot(R1.T, GlobalStressTensor), R1))          
            LocalSigXX = LocalStressTensor[0, 0]
            LocalSigYY = LocalStressTensor[1, 1]
            LocalTauXY = LocalStressTensor[0, 1]
            
            
            #Calculating Mohr Cricle Parameters        
            MC_Center = (LocalSigXX + LocalSigYY)/2.0
            MC_Radius = np.sqrt(0.25*(LocalSigXX - LocalSigYY)**2 + LocalTauXY**2)
            
            CrackTip_MohrCircleParam = {'Center':              MC_Center,
                                        'Radius':              MC_Radius,
                                        'RefAngle':            (TestAngle - self.CrackAngle)%(2*np.pi),
                                        'LocalStressList':     [LocalSigXX, LocalSigYY, LocalTauXY]}
        
            self.CrackTip_MohrCircleParamList.append(CrackTip_MohrCircleParam)
    
    
    
        
    

    def calcNewCrackTipCoordinate(self, CrackIncrementLength, CrackPropagationAngle):
        
        if not self.CrackType == 'PartiallyCracked':    raise Exception
        
        
        #Calculating New CrackTip Coordinate
        NewCrackTipVector = np.array([CrackIncrementLength*np.cos(CrackPropagationAngle), CrackIncrementLength*np.sin(CrackPropagationAngle), 0], dtype = float)
        NewCrackTipCoordinate = self.ScalingCenterCoordinate + NewCrackTipVector
        
        #Updating Crack Curve
        LastCrackTipCoordinate = self.RefCrackCurve['CrackTip_Coordinate']
        self.RefCrackCurve['NewCrackSeg_CoordinateList'] = [LastCrackTipCoordinate, NewCrackTipCoordinate]
        self.RefCrackCurve['CrackTip_Coordinate'] = NewCrackTipCoordinate
        self.RefCrackCurve['HasCrackPropagation'] = True
        
        print('-----', [LastCrackTipCoordinate, NewCrackTipCoordinate])
        
#        #Crack Coordinate List
#        print('X', [self.SolverObj.NodeObjList[NodeId].Coordinate[0] for NodeId in self.RefCrackCurve['NodeIdList']])
#        print('Y', [self.SolverObj.NodeObjList[NodeId].Coordinate[1] for NodeId in self.RefCrackCurve['NodeIdList']])
    
    
    
    
    
    
class InterfaceElement(object):
    
    
    def __init__(self, SolverObj, RefPolyCellObj):
        
        self.SolverObj = SolverObj
        self.N_GaussLobattoPoints = self.SolverObj.N_GaussLobattoPoints
        self.RefPartiallyCrackedSubDomainObj = None
        self.HasZeroNormalStiffness = False
        self.HasZeroWaterPressure = False
        self.ModifiedCrackTip_NormalTraction = 0.0
        self.ModifiedCrackMouth_NormalTraction = 0.0
        self.ModifiedCrackTip_ShearTraction = 0.0
        self.ModifiedCrackMouth_ShearTraction = 0.0
        
        #Reading RefPolyCellObj
        for attr, value in RefPolyCellObj.__dict__.iteritems():    self.__setattr__(attr, value)
        
        #Calculating Solver Parameters
        self.calcParameters()
        self.calcTransformationMatrix()
        self.updateMaterialProperties()
        
    
    
    def __setattr__(self, Attribute, Value):

        object.__setattr__(self, Attribute, Value)
    
    
    
    
    def calcParameters(self):
        
        DirVector = self.NodeObjList[1].Coordinate - self.NodeObjList[0].Coordinate
        Length, Angle, phi_garbage = Cartesian2Spherical(DirVector)
        self.Length = Length
        self.Angle = Angle
        self.SurfaceArea = self.Thickness*self.Length
        
        
        #Reading Contact Properties 
        if self.InterfaceType == 'CONTACT':
                
            self.Kn = self.MaterialObj.Kn
            self.Ks = self.MaterialObj.Ks
            self.FrictionCoefficient = self.MaterialObj.FrictionCoefficient
            self.ContactCohesion = self.MaterialObj.ContactCohesion
        
        self.RefGLPoint = int(self.N_GaussLobattoPoints/2.0)
        self.SlidingDirection = 1.0
        self.LastLoadStep_ShearCOD = 0.0
            
    
    
        
        
    def calcTransformationMatrix(self):
        
        Theta = self.Angle
        
        T = np.array([[ np.cos(Theta),  np.sin(Theta),              0,              0,              0,              0,              0,              0],
                      [-np.sin(Theta),  np.cos(Theta),              0,              0,              0,              0,              0,              0],
                      [             0,              0,  np.cos(Theta),  np.sin(Theta),              0,              0,              0,              0],
                      [             0,              0, -np.sin(Theta),  np.cos(Theta),              0,              0,              0,              0],
                      [             0,              0,              0,              0,  np.cos(Theta),  np.sin(Theta),              0,              0],
                      [             0,              0,              0,              0, -np.sin(Theta),  np.cos(Theta),              0,              0],
                      [             0,              0,              0,              0,              0,              0,  np.cos(Theta),  np.sin(Theta)],
                      [             0,              0,              0,              0,              0,              0, -np.sin(Theta),  np.cos(Theta)]], dtype = float)
      
        self.TransformationMatrix = T
    
    
    
    
    
    def resetContactCondition(self):
        
        self.HasZeroNormalStiffness = False
        self.ModifiedCrackTip_NormalTraction = 0.0
        self.ModifiedCrackMouth_NormalTraction = 0.0
        self.ModifiedCrackTip_ShearTraction = 0.0
        self.ModifiedCrackMouth_ShearTraction = 0.0
        
        self.NormalCODList = np.zeros(self.N_GaussLobattoPoints)
        self.ShearCODList = np.zeros(self.N_GaussLobattoPoints)
        
        self.calcParameters()
        self.updateContactCondition()
        self.calcPenaltyStiffnessMatrix()
        
        
        
    
        
    def updateContactCondition(self, UpdatePenaltyStiffness = False):
        
        
        if self.InterfaceType == 'FREE':
            
            self.ContactConditionList = ['FREE']*self.N_GaussLobattoPoints
            
    
        elif self.InterfaceType == 'CONTACT': 
            
            for p in range(self.N_GaussLobattoPoints):
                
                #Calculating Normal Stiffness
                NormalCOD = self.NormalCODList[p]
                ShearCOD = self.ShearCODList[p]
                
                if NormalCOD > 0:   
                    
                    ContactCondition = 'FREE'
                    
                else:
                    
                    ShearStress = self.Ks*abs(ShearCOD)
                    NormalStress = self.Kn*abs(NormalCOD)
                    if ShearStress < self.ContactCohesion + self.FrictionCoefficient*NormalStress:
                        
                        ContactCondition = 'STICK'
                        
                    else:   ContactCondition = 'SLIP'
#                
                self.ContactConditionList[p] = ContactCondition
                
                #Updating Sliding Direction
#                self.SlidingDirectionList[p] = np.sign(ShearCOD - self.LastLoadStep_ShearCODList[p])
#                self.LastLoadStep_ShearCODList[p] = ShearCOD
            
            self.SlidingDirection = np.sign(self.ShearCODList[self.RefGLPoint] - self.LastLoadStep_ShearCOD)
            self.LastLoadStep_ShearCOD = self.ShearCODList[self.RefGLPoint]
            
            
            if UpdatePenaltyStiffness:
                    
                N_Free = len(['FREE' for ContactCondition0 in self.ContactConditionList if ContactCondition0 == 'FREE'])
                self.Kn = self.MaterialObj.Kn*0.5**N_Free
                self.Ks = self.MaterialObj.Ks*0.5**N_Free
                    
                    

        
    
    
        
    def updateMaterialProperties(self):
        
        
        if self.InterfaceType in ['FREE', 'CONTACT']:
                
            self.calcCOD_1()
            self.calcPenaltyStiffnessMatrix()
        
        
        elif self.InterfaceType == 'COHESIVE':
            
            self.calcCOD()
            self.calcCohesiveStiffnessMatrix()
    
    
    
    
    
    def calcCOD(self):
        """
        Calculates Crack Opening Displacement (COD) at each Integration Point
        """
                
        #Reading Nodes
        NodeObj0 = self.NodeObjList[0]
        NodeObj1 = self.NodeObjList[1]
        
        PairNodeObj0 = self.NodeObjList[3]                
        PairNodeObj1 = self.NodeObjList[2]
        
        NodeObjList = [NodeObj0, NodeObj1]
        PairNodeObjList = [PairNodeObj0, PairNodeObj1]
        
        #Calculating COD at the Nodes
        Nodal_NormalCODList = []
        Nodal_ShearCODList = []
        
        RefVector = [0, 0, 1]
        
#        MidCoord0 = (NodeObj0.DeformedCoordinate + PairNodeObj0.DeformedCoordinate)/2.0
#        MidCoord1 = (NodeObj1.DeformedCoordinate + PairNodeObj1.DeformedCoordinate)/2.0
#        MidCrackVector = MidCoord1 - MidCoord0
#        self.NodalCrackVectorList = [MidCrackVector, MidCrackVector]
        
        
        for n in range(2):
            
            CODVector = PairNodeObjList[n].DeformedCoordinate - NodeObjList[n].DeformedCoordinate
            NodalCrackVector = self.NodalCrackVectorList[n]
            
            #Projecting Half-COD Vectors on Line perpendicular to NodalCrackVector to get nodal COD
            Nodal_NormalCOD = norm(np.cross(CODVector, NodalCrackVector))/norm(NodalCrackVector) #This is equal to norm(CODVector)*np.sin(theta), where theta is angle between CODVector and NodalCrackVector
            Nodal_NormalCODList.append(Nodal_NormalCOD)
                        
            #Projecting COD Vector on CrackVector to get Relative Sliding of CrackPair Node
            Nodal_ShearCOD = np.dot(CODVector, NodalCrackVector)/norm(NodalCrackVector) #This is equal to norm(CODVector)*np.cos(theta), where theta is angle between CODVector and NodalCrackVector
            Nodal_ShearCODList.append(Nodal_ShearCOD)
            
        
        #Interpolating COD at the Gauss-Lobatto Points
        Ni, wi = GaussLobattoIntegrationTable(self.N_GaussLobattoPoints)
        fN = interp1d([-1, 1], Nodal_NormalCODList)
        fS = interp1d([-1, 1], Nodal_ShearCODList)
        
        GLPoint_NormalCODList = []
        GLPoint_ShearCODList = []
        
        for p in range(self.N_GaussLobattoPoints):   
            
            GLPoint_NormalCOD = float(fN(Ni[p]))
            GLPoint_ShearCOD = float(fS(Ni[p]))
            
            #Checking the Max COD obtained
            if GLPoint_NormalCOD > self.MaxNormalCODList[p]:    self.MaxNormalCODList[p] = GLPoint_NormalCOD
            if GLPoint_ShearCOD > self.MaxShearCODList[p]:    self.MaxShearCODList[p] = GLPoint_ShearCOD
            
            GLPoint_NormalCODList.append(GLPoint_NormalCOD)
            GLPoint_ShearCODList.append(GLPoint_ShearCOD)
            
        self.NormalCODList = GLPoint_NormalCODList
        self.ShearCODList = GLPoint_ShearCODList
        
        
        
        
    
    
    def calcCOD_1(self):
        """
        Calculates Crack Opening Displacement (COD) at each Integration Point
        """
        
        #Reading Nodes
        NodeObj0 = self.NodeObjList[0]
        NodeObj1 = self.NodeObjList[1]
        
        PairNodeObj0 = self.NodeObjList[3]                
        PairNodeObj1 = self.NodeObjList[2]
        
        NodeObjList = [NodeObj0, NodeObj1]
        PairNodeObjList = [PairNodeObj0, PairNodeObj1]
        
        #Calculating COD at the Nodes
        Nodal_NormalCODList = []
        Nodal_ShearCODList = []
        
#        MidCoord0 = (NodeObj0.DeformedCoordinate + PairNodeObj0.DeformedCoordinate)/2.0
#        MidCoord1 = (NodeObj1.DeformedCoordinate + PairNodeObj1.DeformedCoordinate)/2.0
#        MidCrackVector = MidCoord1 - MidCoord0
        RefVector = [0, 0, 1]
#        MidCrackPerpVector = np.cross(RefVector, MidCrackVector)
        
        
        for n in range(2):
            
            NodalCrackVector = self.NodalCrackVectorList[n]
            NodalPerpVector = np.cross(RefVector, NodalCrackVector)
            
            #Calculating COD Vectors
            CODVector = PairNodeObjList[n].DeformedCoordinate - NodeObjList[n].DeformedCoordinate
            
            #Projecting Half-COD Vectors on Line perpendicular to MidCrackVector to get Nodal Normal COD
            Nodal_NormalCOD = norm(np.cross(CODVector, NodalCrackVector))/norm(NodalCrackVector) #This is equal to norm(CODVector)*np.sin(theta), where theta is angle between CODVector and MidCrackVector
            
            #Asssinging the sign to Nodal Normal COD
            if SignedAngleBetweenVectors(NodalCrackVector, CODVector, RefVector) < 0:   Nodal_NormalCOD = -Nodal_NormalCOD
            
            Nodal_NormalCODList.append(Nodal_NormalCOD)
            
            
            #Projecting COD Vector on CrackVector to get Relative Sliding of CrackPair Node
            Nodal_ShearCOD = np.dot(CODVector, NodalCrackVector)/norm(NodalCrackVector) #This is equal to norm(CODVector)*np.cos(theta), where theta is angle between CODVector and MidCrackVector
            
            #Asssinging the sign to Nodal Shear COD
            if SignedAngleBetweenVectors(NodalPerpVector, CODVector, RefVector) > 0:   Nodal_ShearCOD = -Nodal_ShearCOD
            
            Nodal_ShearCODList.append(Nodal_ShearCOD)
            
        
        #Interpolating COD at the Gauss-Lobatto Points
        Ni, wi = GaussLobattoIntegrationTable(self.N_GaussLobattoPoints)
        fN = interp1d([-1, 1], Nodal_NormalCODList)
        fS = interp1d([-1, 1], Nodal_ShearCODList)
        
        GLPoint_NormalCODList = []
        GLPoint_ShearCODList = []
        
        for p in range(self.N_GaussLobattoPoints):   
            
            GLPoint_NormalCOD = float(fN(Ni[p]))
            GLPoint_ShearCOD = float(fS(Ni[p]))
            
            #Checking the Max COD obtained
            if GLPoint_NormalCOD > self.MaxNormalCODList[p]:    self.MaxNormalCODList[p] = GLPoint_NormalCOD
            if abs(GLPoint_ShearCOD) > self.MaxShearCODList[p]:    self.MaxShearCODList[p] = abs(GLPoint_ShearCOD)
            
            GLPoint_NormalCODList.append(GLPoint_NormalCOD)
            GLPoint_ShearCODList.append(GLPoint_ShearCOD)
            
        self.NormalCODList = GLPoint_NormalCODList
        self.ShearCODList = GLPoint_ShearCODList
        
        
    
    
    
    
    
    
    
    def calcCohesiveStiffnessMatrix(self):
        
        Ni, wi = GaussLobattoIntegrationTable(self.N_GaussLobattoPoints)
        
        SurfaceArea = self.SurfaceArea
        
        #Reading TSL properties
#        fNT = self.MaterialObj.fNT
        fNS = self.NormalCohesiveProp['fNS']
        wc = self.NormalCohesiveProp['wc']
        
#        fST = self.MaterialObj.fST
        fSS = self.ShearCohesiveProp['fSS']
        sc = self.ShearCohesiveProp['sc']
        
#        LocalTangentStiffnessMatrix = np.zeros([self.N_DOF, self.N_DOF])
        LocalSecantStiffnessMatrix = np.zeros([self.N_DOF, self.N_DOF])
        
        NormalSecantStiffnessList = np.zeros(self.N_GaussLobattoPoints)
        
        
        #Numerically integrating to obtain stiffness matrices
        for p in range(self.N_GaussLobattoPoints):
            
            #Calculating Normal Stiffness
            NormalCOD = self.NormalCODList[p]
            
            if NormalCOD >= wc or self.MaxNormalCODList[p] >= wc:
                
#                NormalTangentStiffness = NormalSecantStiffness = 0.0
                NormalSecantStiffness = 0.0
                self.HasZeroNormalStiffness = True
                
            else:
                        
                if NormalCOD < self.MaxNormalCODList[p]:    
                    
#                    NormalTangentStiffness = NormalSecantStiffness = fNS(self.MaxNormalCODList[p])
                    NormalSecantStiffness = fNS(self.MaxNormalCODList[p])
                    
                else:      
                    
#                    NormalTangentStiffness = fNT(NormalCOD)
                    NormalSecantStiffness = fNS(NormalCOD)
                    
            NormalSecantStiffnessList[p] = NormalSecantStiffness
            
            
            #Calculating Shear Stiffness
            ShearCOD = self.ShearCODList[p]
            
            
            if ShearCOD >= sc or self.MaxShearCODList[p] >= sc:
                
#                ShearTangentStiffness = ShearSecantStiffness = 0.0
                ShearSecantStiffness = 0.0
            
            else:
                    
                if ShearCOD < self.MaxShearCODList[p]: 
                    
#                    ShearTangentStiffness = ShearSecantStiffness = fSS(self.MaxShearCODList[p])
                   ShearSecantStiffness = fSS(self.MaxShearCODList[p])
                   
                else:      
                    
#                    ShearTangentStiffness = fST(ShearCOD)
                    ShearSecantStiffness = fSS(ShearCOD)
                    
            
            
            #Calculating Stiffness Matrices
            ni = Ni[p]
            
            NI = 0.5*(1-ni)
            NII = 0.5*(1+ni)
            
            Mi = np.array([[ -NI,   0, -NII,    0,  NII,    0,   NI,    0],
                           [   0, -NI,    0, -NII,    0,  NII,    0,   NI]], dtype = float)
            
            #Tangent Stiffness Matrix
#            kn_t = NormalTangentStiffness
#            ks_t = ShearTangentStiffness
#            ksn_t = 0.0
            
#            Ki_t = np.array([[ks_t, ksn_t],
#                            [ksn_t, kn_t]], dtype = float)
            
#            LocalTangentStiffnessMatrix += 0.5*SurfaceArea*wi[p]*np.dot(np.dot(Mi.T, Ki_t), Mi)
            
            #Secant Stiffness Matrix
            kn_s = NormalSecantStiffness
            ks_s = ShearSecantStiffness
            ksn_s = 0.0
            
            Ki_s = np.array([[ks_s, ksn_s],
                             [ksn_s, kn_s]], dtype = float)
            
            LocalSecantStiffnessMatrix += 0.5*SurfaceArea*wi[p]*np.dot(np.dot(Mi.T, Ki_s), Mi)
        
        
        #Checking Zero Stiffness
        if np.all(NormalSecantStiffnessList == 0):  
            
            self.HasZeroNormalStiffness = True
            self.InterfaceType = 'FREE'
            self.updateContactCondition()
            
        else:   self.HasZeroNormalStiffness = False
        
            
        
        #Transforming to Global Coordinate System
        T = self.TransformationMatrix
#        self.TangentStiffnessMatrix = np.dot(np.dot(T.T, LocalTangentStiffnessMatrix), T)
        self.SecantStiffnessMatrix = np.dot(np.dot(T.T, LocalSecantStiffnessMatrix), T)
    
    
    
    
    
    
    def calcPenaltyStiffnessMatrix(self):
        
        Ni, wi = GaussLobattoIntegrationTable(self.N_GaussLobattoPoints)
        SurfaceArea = self.SurfaceArea
        LocalSecantStiffnessMatrix = np.zeros([self.N_DOF, self.N_DOF])
        NormalSecantStiffnessList = np.zeros(self.N_GaussLobattoPoints)
        
#        RefSlidingDirection = self.SlidingDirectionList[self.RefGLPoint]
        RefSlidingDirection = self.SlidingDirection
                
        #Numerically integrating to obtain stiffness matrices
        for p in range(self.N_GaussLobattoPoints):
            
            if self.ContactConditionList[p] == 'FREE':
                
                kn_s = 0.0
                ks_s = 0.0
                ksn_s = 0.0
                kns_s = 0.0
                
            elif self.ContactConditionList[p] == 'STICK':
        
                kn_s = self.Kn
                ks_s = self.Ks
                ksn_s = 0.0
                kns_s = 0.0
                
            elif self.ContactConditionList[p] == 'SLIP':
                
                kn_s = self.Kn
                ks_s = 0.0
                ksn_s = RefSlidingDirection*self.Kn*self.FrictionCoefficient
                kns_s = 0.0
            
            
            NormalSecantStiffnessList[p] = kn_s
            
            #Calculating Stiffness Matrices
            ni = Ni[p]
            
            NI = 0.5*(1-ni)
            NII = 0.5*(1+ni)
        
            #Secant Stiffness Matrix
            Ki_s = np.array([[ks_s, ksn_s],
                             [kns_s, kn_s]], dtype = float)
        
            Mi = np.array([[ -NI,   0, -NII,    0,  NII,    0,   NI,    0],
                           [   0, -NI,    0, -NII,    0,  NII,    0,   NI]], dtype = float)
            
            
            LocalSecantStiffnessMatrix += 0.5*SurfaceArea*wi[p]*np.dot(np.dot(Mi.T, Ki_s), Mi)
        
        
        #Checking Zero Stiffness
        if np.all(NormalSecantStiffnessList == 0):  self.HasZeroNormalStiffness = True
        else:                                       self.HasZeroNormalStiffness = False
        
            
        #Transforming to Global Coordinate System
        T = self.TransformationMatrix
        self.SecantStiffnessMatrix = np.dot(np.dot(T.T, LocalSecantStiffnessMatrix), T)
    
    
    
    
    
    
    def calcContactCohesionLoadVector(self):
        
        self.ContactCohesionLoadVector = np.zeros(self.N_DOF)
        
        if self.InterfaceType == 'CONTACT': 
        
            Ni, wi = GaussLobattoIntegrationTable(self.N_GaussLobattoPoints)
            LocalContactCohesionLoadVector = np.zeros(self.N_DOF)
#            RefSlidingDirection = self.SlidingDirectionList[self.RefGLPoint]
            RefSlidingDirection = self.SlidingDirection
                
            #Numerically integrating to obtain LoadVector
            for p in range(self.N_GaussLobattoPoints):
                
                if self.ContactConditionList[p] == 'SLIP':
                    
                    Fc = RefSlidingDirection*self.ContactCohesion
                    
                else:   Fc = 0.0
                    
                #Calculating Local ContactSlip LoadVector
                ni = Ni[p]
                
                NI = 0.5*(1-ni)
                NII = 0.5*(1+ni)
                
                Mi = np.array([[ -NI,   0, -NII,    0,  NII,    0,   NI,    0],
                               [   0, -NI,    0, -NII,    0,  NII,    0,   NI]], dtype = float)
                
                Pw_n = np.array([[Fc],
                                 [0.0]], dtype = float)
                
                LocalContactCohesionLoadVector += 0.5*self.SurfaceArea*wi[p]*np.dot(Mi.T, Pw_n).T[0]
                
            
            #Transforming to Global Coordinate System
            T = self.TransformationMatrix
            self.ContactCohesionLoadVector = np.dot(T.T, LocalContactCohesionLoadVector)
        
    
    
    
    
    
    
    def calcWaterPressureLoadVector_Empirical(self, CrackMouth_WaterPressure):
        
        #Ref: Thesis by Reich 1993, On the marriage of Fracture Mech and Mixed FEM: An application to concrete Dam (Eq 6.5, 6.12)
        self.WaterPressureLoadVector = np.zeros(self.N_DOF)
        self.CrackFluidObj = None
        
        if self.CrackFluidObj:
                
            Ni, wi = GaussLobattoIntegrationTable(self.N_GaussLobattoPoints)
            
            #Reading Empirical Flow properties
            fphi = self.CrackFluidObj.fphi
            ww0 = self.CrackFluidObj.ww0
            
            LocalWaterPressureLoadVector = np.zeros(self.N_DOF)
            PwList = np.zeros(self.N_GaussLobattoPoints)
            
            
            #Numerically integrating to obtain LoadVector
            for p in range(self.N_GaussLobattoPoints):
                
                #Calculating Empirical Water Pressure at given Normal COD
                NormalCOD = self.NormalCODList[p]
                
                if NormalCOD >= ww0 or self.LiesInInitialCrackSegment:   phi = 1.0
                    
                else:                                                    phi = fphi(NormalCOD/ww0)
                      
                Pw = CrackMouth_WaterPressure*phi
                PwList[p] = Pw
                
                #Calculating Local WaterPressure LoadVector
                ni = Ni[p]
                
                NI = 0.5*(1-ni)
                NII = 0.5*(1+ni)
                
                Mi = np.array([[ -NI,   0, -NII,    0,  NII,    0,   NI,    0],
                               [   0, -NI,    0, -NII,    0,  NII,    0,   NI]], dtype = float)
                
                Pw_n = np.array([[0.0],
                                 [Pw ]], dtype = float)
                
                LocalWaterPressureLoadVector += 0.5*self.SurfaceArea*wi[p]*np.dot(Mi.T, Pw_n).T[0]
                
                
            #Checking Zero WaterPressure
            if np.all(PwList == 0):     self.HasZeroWaterPressure = True
            else:                       self.HasZeroWaterPressure = False
            
            
            #Transforming to Global Coordinate System
            T = self.TransformationMatrix
            self.WaterPressureLoadVector = np.dot(T.T, LocalWaterPressureLoadVector)
        
        else:   self.HasZeroWaterPressure = True
    
    
    
    
    
    def calcLinearTractionVector(self):
        
        #Calculating Local Load Vector
        if self.RefPartiallyCrackedSubDomainObj == None: 
            
            RefLoadVector = np.dot(self.SecantStiffnessMatrix, self.GlobDispVector)
            RefLocalLoadVector = np.dot(self.TransformationMatrix, RefLoadVector)
            
            #Calculating Equivalent Linear Traction
            Fx0 = RefLocalLoadVector[0]
            Fy0 = RefLocalLoadVector[1]
            Fx1 = RefLocalLoadVector[2]
            Fy1 = RefLocalLoadVector[3]
            
            fs0 = ( 4*Fx0 - 2*Fx1)/self.SurfaceArea
            fs1 = (-2*Fx0 + 4*Fx1)/self.SurfaceArea
            fn0 = ( 4*Fy0 - 2*Fy1)/self.SurfaceArea
            fn1 = (-2*Fy0 + 4*Fy1)/self.SurfaceArea
            
            
        else:
            
            fs0 = -self.ModifiedCrackMouth_ShearTraction
            fs1 = -self.ModifiedCrackTip_ShearTraction
            fn0 = -self.ModifiedCrackMouth_NormalTraction
            fn1 = -self.ModifiedCrackTip_NormalTraction
            
            
    
        #Transforming to Global Coordinate System
        RefLocalTractionVector = np.array([fs0, fn0, fs1, fn1, -fs1, -fn1, -fs0, -fn0], dtype=float)
        LinearTractionVector = np.dot(self.TransformationMatrix.T, RefLocalTractionVector)
    
        return LinearTractionVector
        

    
    def createTractionPlotCoordinate(self, Scale = 2.5e-9):
        
        if self.RefCrackCurve['Id'] in [1, 2]:   Scale = -Scale
        
#        if self.InterfaceType == 'CONTACT': Scale = 0.5*Scale
        
        #Linearizing Tractions
        LinearCohesiveTractionVector = self.calcLinearTractionVector()
        
        #Reading Nodes
        NodeObj0 = self.NodeObjList[0]
        NodeObj1 = self.NodeObjList[1]
        PairNodeObj0 = self.NodeObjList[3]                
        PairNodeObj1 = self.NodeObjList[2]
        
        #In case where the interface ELement lies within Partially Cracked SubDomain in Pseudo sense
        if not self.RefPartiallyCrackedSubDomainObj == None: 
            
            NodeObj1.ScaledDeformedCoordinate = self.RefPartiallyCrackedSubDomainObj.ScaledDeformed_ScalingCenterCoordinate
            PairNodeObj1.ScaledDeformedCoordinate = self.RefPartiallyCrackedSubDomainObj.ScaledDeformed_ScalingCenterCoordinate
            
                
        #Creating Traction PlotCoordinate
        MidScaledDeformedCoord0 = (NodeObj0.ScaledDeformedCoordinate + PairNodeObj0.ScaledDeformedCoordinate)/2.0
        MidScaledDeformedCoord1 = (NodeObj1.ScaledDeformedCoordinate + PairNodeObj1.ScaledDeformedCoordinate)/2.0
        
        
        #Plot Coordinate for Cohesive Traction
        Coh_Coord0 = MidScaledDeformedCoord0
        Coh_Coord1 = MidScaledDeformedCoord0 + Scale*np.array([LinearCohesiveTractionVector[0], LinearCohesiveTractionVector[1], 0.0], dtype = float)
        Coh_Coord2 = MidScaledDeformedCoord1 + Scale*np.array([LinearCohesiveTractionVector[2], LinearCohesiveTractionVector[3], 0.0], dtype = float)
        Coh_Coord3 = MidScaledDeformedCoord1
        self.CohesiveTractionPlotCoordinateList = [Coh_Coord0, Coh_Coord1, Coh_Coord2, Coh_Coord3]
        
        
        
        
    
    


class SolverBaseClass(): 
    
    
    def evalDOFs(self):
        
        self.N_DOFPerNode = 2
        self.N_Node = len(self.NodeObjList)
        self.N_DOF = self.N_DOFPerNode*self.N_Node
        
        #Evaluating DOFs for each Node
        for NodeObj in self.NodeObjList:
            
            NodeObj.DOFIdList = [self.N_DOFPerNode*NodeObj.Id + k  for k in range(self.N_DOFPerNode)]
                
        
        #Evaluating DOFs for each PolyCell
        for PolyCellObj in self.PolyCellObjList:
            
            if PolyCellObj.Enabled:
                
                #Updating Parameters
                PolyCellObj.N_Node = len(PolyCellObj.NodeIdList)
                PolyCellObj.N_DOF = self.N_DOFPerNode*PolyCellObj.N_Node
                
                #Calculating DOFIdList     
                PolyCellObj.DOFIdList = [0]*PolyCellObj.N_DOF
                
                for j in range(PolyCellObj.N_Node):
                    
                    for k in range(self.N_DOFPerNode):
                        
                        jo = self.N_DOFPerNode*j + k
                        PolyCellObj.DOFIdList[jo] = self.N_DOFPerNode*PolyCellObj.NodeIdList[j] + k
                        
                
                #Calculating DOFIdList For each EdgeObj
                for EdgeObj in PolyCellObj.EdgeObjList:
                    
                    EdgeObj.N_Node = len(EdgeObj.NodeIdList)
                    EdgeObj.N_DOF = EdgeObj.N_Node*self.N_DOFPerNode 
                    EdgeObj.DOFIdList = [0]*EdgeObj.N_DOF         
                    EdgeObj.LocalDOFIdList = [0]*EdgeObj.N_DOF
                    
                    for j in range(EdgeObj.N_Node):
                        
                        for k in range(self.N_DOFPerNode):
                            
                            jo = self.N_DOFPerNode*j + k                    
                            EdgeObj.DOFIdList[jo] = self.N_DOFPerNode*EdgeObj.NodeIdList[j] + k
                            EdgeObj.LocalDOFIdList[jo] = self.N_DOFPerNode*EdgeObj.LocalNodeIdList[j] + k
                
                
                #Calculating DOFs corresponding to the CrackMouth in case of PartiallyCracked SubDomain
                if PolyCellObj.CrackType == 'PartiallyCracked':
                    
                    #Extracting Cracked Node at the Boundary of SubDomain (CrackMouth of the SubDomain)
                    CrackMouth_NodeObj0 = PolyCellObj.CrackMouthNodeObjList[0]
                    CrackMouth_NodeObj1 = PolyCellObj.CrackMouthNodeObjList[1]
                    CrackMouth_NodeId0 = CrackMouth_NodeObj0.Id
                    CrackMouth_NodeId1 = CrackMouth_NodeObj1.Id
                    
                    #Calculating DOFs corresponing to Cracked Nodes to find CrackMouth Displacement
                    CrackMouth_XDOFId0 = self.N_DOFPerNode*CrackMouth_NodeId0
                    CrackMouth_YDOFId0 = self.N_DOFPerNode*CrackMouth_NodeId0 + 1
                    CrackMouth_XDOFId1 = self.N_DOFPerNode*CrackMouth_NodeId1
                    CrackMouth_YDOFId1 = self.N_DOFPerNode*CrackMouth_NodeId1 + 1
                    
                    CrackMouth_LocalXDOFId0 = PolyCellObj.DOFIdList.index(CrackMouth_XDOFId0)
                    CrackMouth_LocalYDOFId0 = PolyCellObj.DOFIdList.index(CrackMouth_YDOFId0)
                    CrackMouth_LocalXDOFId1 = PolyCellObj.DOFIdList.index(CrackMouth_XDOFId1)
                    CrackMouth_LocalYDOFId1 = PolyCellObj.DOFIdList.index(CrackMouth_YDOFId1)
                    
                    PolyCellObj.CrackMouth_DOFIdList = [CrackMouth_XDOFId0, CrackMouth_YDOFId0, CrackMouth_XDOFId1, CrackMouth_YDOFId1]
                    PolyCellObj.CrackMouth_LocalDOFIdList = [CrackMouth_LocalXDOFId0, CrackMouth_LocalYDOFId0, CrackMouth_LocalXDOFId1, CrackMouth_LocalYDOFId1]
                              
                     
            
    
    

    def createSubDomains(self):
        
        self.SubDomainObjList = []
        self.PartiallyCrackedSubDomainObjList = []
        
        for PolyCellObj in self.PolyCellObjList:
            
            if PolyCellObj.Enabled:
                
                if PolyCellObj.ElementType == 'SubDomain':
                    
                    SubDomainObj = SubDomain(self, PolyCellObj)
                    self.SubDomainObjList.append(SubDomainObj)
                    
                    if SubDomainObj.CrackType == 'PartiallyCracked':
                        
                        self.PartiallyCrackedSubDomainObjList.append(SubDomainObj)
                        
    
    
    
    
    

    def assembleStiffnessMatrix(self, NodeIdList, StiffnessMatrix, GlobalStiffnessMatrix):
        
        N_Node = len(NodeIdList)
        
        for j1 in range(N_Node):
            
            NodeId1 = NodeIdList[j1]
            
            for j2 in range(N_Node):
                
                NodeId2 = NodeIdList[j2]                    
                
                GlobalStiffnessMatrix[2*NodeId1:2*NodeId1+2, 2*NodeId2:2*NodeId2+2] += StiffnessMatrix[2*j1:2*j1+2, 2*j2:2*j2+2]
    
    
    
    
    
    def assembleTractionLoadVector(self, NodeIdList, TractionLoadVector, GlobalLoadVector):
        
        N_Node = len(NodeIdList)
        
        for j in range(N_Node):
        
            NodeId = NodeIdList[j]
            TrLoadVector = TractionLoadVector[2*j:2*j+2]
            
            GlobalLoadVector[2*NodeId:2*NodeId+2] += TrLoadVector
        
        
        
        
    def initGlobalStiffnessMatrix(self):
        
        self.GlobalStiffnessMatrix = np.zeros([self.N_DOF, self.N_DOF])
        
        for SubDomainObj in self.SubDomainObjList:
            
            self.assembleStiffnessMatrix(SubDomainObj.NodeIdList, SubDomainObj.StiffnessMatrix, self.GlobalStiffnessMatrix)
                    
          
            
    
    def initGlobalRefLoadVector(self):
        
#        self.GlobalStaticLoadVector = np.zeros(self.N_DOF)
        self.GlobalRefTransientLoadVector = np.zeros(self.N_DOF)
        
        #Assembling Nodal LoadVectors
        for NodeObj in self.NodeObjList:
            
            NodeId = NodeObj.Id  
#            StaticLoadVector_2D = NodeObj.StaticLoadVector[0:2]
            TransientLoadVector_2D = NodeObj.TransientLoadVector[0:2]
            
#            self.GlobalStaticLoadVector[2*NodeId:2*NodeId+2] += StaticLoadVector_2D
            self.GlobalRefTransientLoadVector[2*NodeId:2*NodeId+2] += TransientLoadVector_2D
        
        
        #Assembling Traction LoadVectors
        for SubDomainObj in self.SubDomainObjList:
            
#            self.assembleTractionLoadVector(SubDomainObj.NodeIdList, SubDomainObj.StaticTractionLoadVector, self.GlobalStaticLoadVector)
            self.assembleTractionLoadVector(SubDomainObj.NodeIdList, SubDomainObj.TransientTractionLoadVector, self.GlobalRefTransientLoadVector)
    
        

    
    def initBoundaryConditionVector(self):
        
        #Filtering DOFs where BC is constrained
        GlobalBoundaryConditionVector = np.zeros(self.N_DOF)
        
        for NodeObj in self.NodeObjList:
            
            NodeId = NodeObj.Id            
            BoundaryConditionVector_2D = NodeObj.BoundaryCondition[0:2]
            
            GlobalBoundaryConditionVector[2*NodeId:2*NodeId+2] = BoundaryConditionVector_2D
    
        self.FilteredBCDOFList = np.where(GlobalBoundaryConditionVector >= 1)[0]
        self.NonBCDOFList = np.where(GlobalBoundaryConditionVector == 0)[0]
        
        
        
    
    
    def applyBoundaryConditions(self, GlobalStiffnessMatrix, GlobalRefLoadVector = [], Mode = 'ColumnModification'):
        
        if Mode == 'ColumnModification':
                
            for i in range(self.N_DOF): #looping over rows of GlobalStiffnessMatrix        
                
                for j in self.FilteredBCDOFList: #looping over columns of GlobalStiffnessMatrix            
                    
                    if i == j:  GlobalStiffnessMatrix[i, i] = 1.0  #Making diagonal members of GlobalStiffnessMatrix = 1, if displacement Boundary Constraint is applied. (So as to make the GlobalStiffnessMatrix invertible)
                    else:       GlobalStiffnessMatrix[i, j] = 0.0  #Making column vector = 0, if displacement Boundary Constraint is applied.        
        
        
        elif Mode == 'MatrixSizeReduction':
            
            Reduced_GlobStiffMat = np.delete(GlobalStiffnessMatrix, self.FilteredBCDOFList, 0)
            Reduced_GlobStiffMat = np.delete(Reduced_GlobStiffMat, self.FilteredBCDOFList, 1)
            self.Reduced_GlobStiffMat = Reduced_GlobStiffMat
            
            if len(GlobalRefLoadVector) == 0:   raise Exception
            self.Reduced_GlobRefLoadVec = np.delete(GlobalRefLoadVector, self.FilteredBCDOFList, 0)
            
            
        self.BCApplyMode = Mode
        
        
    
    
    def applyStaticCondensation(self, ConDOFIdList):
        """
        Con = Condensed 
        Rem = Remaining
        Red = Reduced
        Mod = Modified
        Loc = Local
        """
        
        #Checking if ConDOFIdList has any boundary condition
        for ConDOFId in ConDOFIdList:
            
            if ConDOFId in self.FilteredBCDOFList:  raise Exception
        
        #Calculating remaining DOFIdLists
        RemDOFIdList = [j for j in range(self.N_DOF) if not j in ConDOFIdList]
        RedRemDOFIdList = [RemDOFId for RemDOFId in RemDOFIdList if not RemDOFId in self.FilteredBCDOFList]
           
        #Calculating Modfied  DOFIdList
        ModDOFIdList = ConDOFIdList + RedRemDOFIdList
        N_ModDOF = len(ModDOFIdList)
        
        #Calculating Modified Stiffness and Force Matrix (This also applies the Boundary Condition)
        ModGlobStiffMat = np.zeros([N_ModDOF, N_ModDOF])
        for i in range(N_ModDOF): ModGlobStiffMat[i] = self.GlobalStiffnessMatrix[ModDOFIdList[i]][ModDOFIdList]
        ModGlobRefLoadVec = self.GlobalRefLoadVector[ModDOFIdList]
        
        #Partitioning the Modified Matrices
        N_ConDOF = len(ConDOFIdList)
        LocConDOFIdList = [i for i in range(N_ConDOF)]
        LocRemDOFIdList = [i for i in range(N_ConDOF, len(ModDOFIdList))]

        Kcc = ModGlobStiffMat[LocConDOFIdList].T[LocConDOFIdList].T
        Kcr = ModGlobStiffMat[LocConDOFIdList].T[LocRemDOFIdList].T
        Krc = ModGlobStiffMat[LocRemDOFIdList].T[LocConDOFIdList].T
        Krr = ModGlobStiffMat[LocRemDOFIdList].T[LocRemDOFIdList].T
        
        Fc = ModGlobRefLoadVec[LocConDOFIdList]
        Fr = ModGlobRefLoadVec[LocRemDOFIdList]
        
        #Calculating Condensed Matrices
        Kcr_InvKrr = np.dot(Kcr, inv(Krr))
        ConStiffMat = Kcc - np.dot(Kcr_InvKrr, Krc)
        ConLoadVec = Fc - np.dot(Kcr_InvKrr, Fr)
            
        CondensedData = {'ModDOFIdList':        ModDOFIdList,
                         'ConStiffMat':         ConStiffMat,
                         'ConLoadVec':          ConLoadVec,
                         'Kcc':                 Kcc,
                         'Kcr':                 Kcr,
                         'Krc':                 Krc,
                         'Krr':                 Krr,
                         'Fc':                  Fc,
                         'Fr':                  Fr}
        
        return CondensedData
    
    
    
    
    
    def calcGlobDispVecPostCondensation(self, CondensedData):
        
        Uc = CondensedData['ConDispVec']
        Fr = CondensedData['Fr']
        Krc = CondensedData['Krc']
        Krr = CondensedData['Krr']
        ModDOFIdList = CondensedData['ModDOFIdList']
        
        self.GlobalDisplacementVector = np.zeros(self.N_DOF)
        Ur = np.dot(inv(Krr), Fr - np.dot(Krc, Uc))
        U = np.hstack((Uc,Ur))
        
        self.GlobalDisplacementVector[ModDOFIdList] = U
        
        
        
        
#    
#    
#    def applyStaticCondensation_(self, ConNodeIdList):
#        """
#        Con = Condensed 
#        Rem = Remaining
#        Mod = Modified
#        """
#        RemNodeIdList = [i for i in range(self.N_Node) if not i in ConNodeIdList]
#        
#        #Calculating Modfied NodeIdList and DOFIdList
#        ModNodeIdList = ConNodeIdList + RemNodeIdList
#        ModDOFIdList = np.zeros(self.N_DOF, dtype = int)
#        
#        for i in range(self.N_Node):
#            
#            for j in range(self.N_DOFPerNode):
#                
#                ModDOFIdList[self.N_DOFPerNode*i + j] = self.N_DOFPerNode*ModNodeIdList[i] + j
#        
##        print('ModDOFIdList', ModDOFIdList)
##        print(len(ModDOFIdList), self.N_DOF)
#        
#        #Calculating Modified Stiffness Matrix
#        ModGlobStiffMat = np.zeros([self.N_DOF, self.N_DOF])
#        for i in range(self.N_DOF): ModGlobStiffMat[i] = self.GlobalStiffnessMatrix[ModDOFIdList[i]][ModDOFIdList]
#        
#        #Calculating Modified Force Matrix
#        ModGlobRefLoadVec = self.GlobalRefLoadVector[ModDOFIdList]
#        
#        #Partitioning the Modified Matrices
#        N_ConDOF = self.N_DOFPerNode*len(ConNodeIdList)
#        ConDOFIdList = [i for i in range(N_ConDOF)]
#        RemDOFIdList = [i for i in range(N_ConDOF, self.N_DOF)]
#
#        Kcc = ModGlobStiffMat[ConDOFIdList].T[ConDOFIdList].T
#        Kcr = ModGlobStiffMat[ConDOFIdList].T[RemDOFIdList].T
#        Krc = ModGlobStiffMat[RemDOFIdList].T[ConDOFIdList].T
#        Krr = ModGlobStiffMat[RemDOFIdList].T[RemDOFIdList].T
#        
#        Fc = ModGlobRefLoadVec[ConDOFIdList]
#        Fr = ModGlobRefLoadVec[RemDOFIdList]
#        
#        #Calculating Condensed Matrices
#        Kcr_InvKrr = np.dot(Kcr, inv(Krr))
#        ConStiffMat = Kcc - np.dot(Kcr_InvKrr, Krc)
#        ConLoadVec = Fc - np.dot(Kcr_InvKrr, Fr)
#        
#        return ConStiffMat, ConLoadVec
    
    
    
    
    
    def assignDeformationToNodes(self):
        
        #Assigning deformation to Nodes
        for i in range(self.N_Node):
            
            NodeObj = self.NodeObjList[i]
            
            NodeObj.GlobDispVector = self.GlobalDisplacementVector[NodeObj.DOFIdList]
            NodeObj.DeformedCoordinate = NodeObj.Coordinate + np.array([NodeObj.GlobDispVector[0], NodeObj.GlobDispVector[1], 0], dtype=float)
        
    
    
    
    
    def scaleDeformation(self, DeformationScale):
        
        #Scaling Deformation for Nodes
        N_Node = len(self.NodeObjList)
        N_DOFPerNode = self.N_DOFPerNode
        
        for i in range(N_Node): self.NodeObjList[i].ScaledDeformedCoordinate = self.NodeObjList[i].Coordinate + DeformationScale*np.array([self.GlobalDisplacementVector[N_DOFPerNode*i], self.GlobalDisplacementVector[N_DOFPerNode*i+1], 0], dtype=float)
        
        
        #Scaling Deformation for CrackTip
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            SubDomainObj.ScaledDeformed_ScalingCenterCoordinate = SubDomainObj.ScalingCenterCoordinate + DeformationScale*SubDomainObj.getFieldDisplacement(SubDomainObj.ScalingCenterCoordinate)
    
    
    
    
    def assignDeformationToSubDomains(self):
        
        #Updating SubDomain
        for SubDomainObj in self.SubDomainObjList:  
            
            #Disp. Vector
            SubDomain.GlobDispVector = self.GlobalDisplacementVector[SubDomainObj.DOFIdList]
            
            #Calculating Integration Constant (Song Book, Eq. 3.56) (Yang and Deeks 2002, Eq 71)
            SubDomainObj.IntegrationConstantList = np.dot(inv(SubDomainObj.EigVec_DispModeData), SubDomain.GlobDispVector - SubDomainObj.Sideface_CumulativeDispMode)
            
    
    

    def createNodalTractionPlotCoordinate(self, Scale = 2.0e-8):
                
        for CrackCurve in self.CrackCurveList:
            
            
            Nodal_CohesivePlotData = {  'NodeIdList':   [],
                                        'LoadVectorList' : [],
                                        'SurfaceAreaList' : []}
            
            Nodal_ContactPlotData = {  'NodeIdList':   [],
                                       'LoadVectorList' : [],
                                       'SurfaceAreaList' : []}
            
            #Calculating Nodal Cohesive Load Vectors
            RefInterfaceElementObjList = CrackCurve['InterfaceElementObjList'] + [CrackCurve['CrackTip_InterfaceElementObj']] 
            
            for InterfaceElementObj in RefInterfaceElementObjList:
                    
                if not InterfaceElementObj.InterfaceType == 'FREE':
                        
                    LinearCohesiveTractionVector = InterfaceElementObj.calcLinearTractionVector()
                    Local_LinearCohesiveTractionVector = np.dot(InterfaceElementObj.TransformationMatrix, LinearCohesiveTractionVector)
        
                    fx1 = Local_LinearCohesiveTractionVector[0]
                    fy1 = Local_LinearCohesiveTractionVector[1]
                    fx2 = Local_LinearCohesiveTractionVector[2]
                    fy2 = Local_LinearCohesiveTractionVector[3]
                    
                    Fx1 = (fx1/3 + fx2/6)*InterfaceElementObj.SurfaceArea
                    Fy1 = (fy1/3 + fy2/6)*InterfaceElementObj.SurfaceArea
                    Fx2 = (fx1/6 + fx2/3)*InterfaceElementObj.SurfaceArea
                    Fy2 = (fy1/6 + fy2/3)*InterfaceElementObj.SurfaceArea
                    
                    RefLocalLoadVector = np.array([Fx1, Fy1, Fx2, Fy2, -Fx2, -Fy2, -Fx1, -Fy1], dtype=float)
                    RefLoadVector = np.dot(InterfaceElementObj.TransformationMatrix.T, RefLocalLoadVector)
                    RefSurfaceArea = 0.5*InterfaceElementObj.SurfaceArea
                    
                    NodeObj0 = InterfaceElementObj.NodeObjList[0]
                    NodeObj1 = InterfaceElementObj.NodeObjList[1]
                    RefNodeObjList = [NodeObj0, NodeObj1]
                    
                    RefLoadVector0 = np.array([RefLoadVector[0], RefLoadVector[1], 0.0])
                    RefLoadVector1 = np.array([RefLoadVector[2], RefLoadVector[3], 0.0])
                    RefLoadVectorList = [RefLoadVector0, RefLoadVector1]
                    
                    if InterfaceElementObj.InterfaceType == 'COHESIVE': RefPlotData = Nodal_CohesivePlotData
                    elif InterfaceElementObj.InterfaceType == 'CONTACT': RefPlotData = Nodal_ContactPlotData
                    
                    for n in range(2):
                        
                        RefNodeObj = RefNodeObjList[n]
                        RefNodeId = RefNodeObj.Id
                        
                        if not RefNodeId in RefPlotData['NodeIdList']:
                            
                            RefPlotData['NodeIdList'].append(RefNodeId)
                            RefPlotData['LoadVectorList'].append(RefLoadVectorList[n])
                            RefPlotData['SurfaceAreaList'].append(RefSurfaceArea)
                            
                        else:
                            
                            I = RefPlotData['NodeIdList'].index(RefNodeId)
                            RefPlotData['LoadVectorList'][I] += RefLoadVectorList[n]
                            RefPlotData['SurfaceAreaList'][I] += RefSurfaceArea
                            
                
            
            #Converting Nodal Cohesive Load Vectors to plot coordinates
            for InterfaceElementObj in RefInterfaceElementObjList:
                
                if not InterfaceElementObj.InterfaceType == 'FREE':
                        
                    if CrackCurve['Id'] in [1, 2]:          Scale = -abs(Scale)
                    else:                                   Scale = abs(Scale)
                    #Reading Nodes
                    NodeObj0 = InterfaceElementObj.NodeObjList[0]
                    NodeObj1 = InterfaceElementObj.NodeObjList[1]
                    PairNodeObj0 = InterfaceElementObj.NodeObjList[3]                
                    PairNodeObj1 = InterfaceElementObj.NodeObjList[2]
                    
                    #In case where the interface ELement lies within Partially Cracked SubDomain in Pseudo sense
                    if not InterfaceElementObj.RefPartiallyCrackedSubDomainObj == None: 
                        
                        NodeObj1.ScaledDeformedCoordinate = InterfaceElementObj.RefPartiallyCrackedSubDomainObj.ScaledDeformed_ScalingCenterCoordinate
                        PairNodeObj1.ScaledDeformedCoordinate = InterfaceElementObj.RefPartiallyCrackedSubDomainObj.ScaledDeformed_ScalingCenterCoordinate
                        
                            
                    #Creating Traction PlotCoordinate
                    MidScaledDeformedCoord0 = (NodeObj0.ScaledDeformedCoordinate + PairNodeObj0.ScaledDeformedCoordinate)/2.0
                    MidScaledDeformedCoord1 = (NodeObj1.ScaledDeformedCoordinate + PairNodeObj1.ScaledDeformedCoordinate)/2.0
                    
                    
                    if InterfaceElementObj.InterfaceType == 'COHESIVE': RefPlotData = Nodal_CohesivePlotData
                    elif InterfaceElementObj.InterfaceType == 'CONTACT': RefPlotData = Nodal_ContactPlotData
                    
                    NodeId0 = NodeObj0.Id
                    NodeId1 = NodeObj1.Id
                    
                    I0 = RefPlotData['NodeIdList'].index(NodeId0)
                    Node0_RefTractionVector = RefPlotData['LoadVectorList'][I0]/RefPlotData['SurfaceAreaList'][I0]
                    
                    I1 = RefPlotData['NodeIdList'].index(NodeId1)
                    Node1_RefTractionVector = RefPlotData['LoadVectorList'][I1]/RefPlotData['SurfaceAreaList'][I1]
                    
                    Coh_Coord0 = MidScaledDeformedCoord0
                    Coh_Coord1 = MidScaledDeformedCoord0 + Scale*Node0_RefTractionVector
                    Coh_Coord2 = MidScaledDeformedCoord1 + Scale*Node1_RefTractionVector
                    Coh_Coord3 = MidScaledDeformedCoord1
                    
                    InterfaceElementObj.Nodal_CohesiveTractionPlotCoordinateList = [Coh_Coord0, Coh_Coord1, Coh_Coord2, Coh_Coord3]
                    



    def plotNodalDisplacement(self, DeformationScale = 1.0, ShowTractions = True, SaveImage = True, ImageFilePrefix = 'M_'):
        
        #Scaling Deformation
        self.scaleDeformation(DeformationScale)
        
        
        #Initializing Figure
#        fig = plt.figure(figsize=(4,5))
        fig = plt.figure()
        
        
        #Plotting SubDomain
        for SubDomainObj in self.SubDomainObjList:
                       
            Xd = []
            Yd = []
                     
            for EdgeObj in SubDomainObj.EdgeObjList:
                                    
                xd = []
                yd = []
                            
                for NodeObj in EdgeObj.NodeObjList:
                    
                    #Reading Deformed Nodal Coordinate
                    DeformedNodalCoordinate = NodeObj.ScaledDeformedCoordinate                
                    xd.append(DeformedNodalCoordinate[0])                    
                    yd.append(DeformedNodalCoordinate[1])
                
                
                Xd += xd
                Yd += yd
                plt.plot(xd, yd, 'k', linewidth = 0.6)
            
            plt.fill(Xd, Yd, 'k', alpha = 0.1)



        if ShowTractions:
                
            #Plotting Interface Elements
            for InterfaceElementObj in self.InterfaceElementObjList:
                
                for EdgeObj in InterfaceElementObj.EdgeObjList:
                                        
                    xd = []
                    yd = []
                                
                    for NodeObj in EdgeObj.NodeObjList:
                        
                        #Reading Deformed Nodal Coordinate
                        DeformedNodalCoordinate = NodeObj.ScaledDeformedCoordinate                
                        xd.append(DeformedNodalCoordinate[0])                    
                        yd.append(DeformedNodalCoordinate[1])
                    
                    if InterfaceElementObj.HasZeroNormalStiffness:  plt.plot(xd, yd, 'k--', linewidth = 0.2, alpha = 0.1)
                    else:                                           plt.plot(xd, yd, 'k', linewidth = 0.2, alpha = 0.8)
                    
                
        
            
            #Plotting the Cohesive Traction
            if len(self.InterfaceElementObjList) > 0:
                
                self.createNodalTractionPlotCoordinate()
                
                RefInterfaceElementObjList = self.InterfaceElementObjList + [SubDomainObj.CrackTip_InterfaceElementObj for SubDomainObj in self.PartiallyCrackedSubDomainObjList]
                
                #Plotting Thin Lines Data
                for RefInterfaceElementObj in RefInterfaceElementObjList:
                    
                    if not RefInterfaceElementObj.InterfaceType == 'FREE':
                        
#                        RefInterfaceElementObj.createTractionPlotCoordinate()
                        
                        #Cohesive Traction
                        if not RefInterfaceElementObj.HasZeroNormalStiffness:
                                
                            xt = []
                            yt = []
                            
                            
                            for CohesiveTractionPlotCoordinate in RefInterfaceElementObj.Nodal_CohesiveTractionPlotCoordinateList:
                             
#                            for CohesiveTractionPlotCoordinate in RefInterfaceElementObj.CohesiveTractionPlotCoordinateList:
                                
                                xt.append(CohesiveTractionPlotCoordinate[0])                    
                                yt.append(CohesiveTractionPlotCoordinate[1])
                            
                            xt.append(xt[0])
                            yt.append(yt[0])
                            
                            plt.fill(xt, yt, 'k', alpha = 0.6)
                            
                            
                
                
                #Plotting Thick Lines Data
                for RefInterfaceElementObj in RefInterfaceElementObjList:
                    
                    if not RefInterfaceElementObj.InterfaceType == 'FREE':
                            
                        #Cohesive Traction
                        if not RefInterfaceElementObj.HasZeroNormalStiffness:
                            
                            xT = []
                            yT = []
                        
                            for CohesiveTractionPlotCoordinate in RefInterfaceElementObj.Nodal_CohesiveTractionPlotCoordinateList[1:3]:
                              
#                            for CohesiveTractionPlotCoordinate in RefInterfaceElementObj.CohesiveTractionPlotCoordinateList[1:3]:
                                    
                                xT.append(CohesiveTractionPlotCoordinate[0])                    
                                yT.append(CohesiveTractionPlotCoordinate[1])
                            
                            plt.plot(xT, yT, 'k', linewidth = 1.5, alpha = 0.8)
                            
                    
        
        
#        #Marking CrackTips
#        if not DeformationScale == 0:
#                
#            for CrackCurve in self.CrackCurveList:
#                
#                CrackTipCoord = CrackCurve['CrackTip_Coordinate']
#                plt.plot(CrackTipCoord[0], CrackTipCoord[1], 'kx', ms = 5, markeredgewidth = 0.5)
            
            
        #Fitting the plot
#        RefLoadNodeObj = self.NodeObjList[self.RefLoadNodeId]
#        RefXCoord = RefLoadNodeObj.ScaledDeformedCoordinate[0]
#        RefYCoord = RefLoadNodeObj.ScaledDeformedCoordinate[1]
#        plt.xlim( RefXCoord - 0.25, RefXCoord + 0.25)
#        plt.ylim( RefYCoord - 0.25, RefYCoord + 0.05)
#        
        
        
#        plt.title('Mesh with Surface Tractions', y = 0.955)
        plt.axis('equal')
        plt.axis('off')
        
        
#        plt.xlim(0.38, 0.47)
#        plt.ylim(0.02, 0.12)
        
        plt.show()
        
        #Saving Image
        if SaveImage == True:
            
            Path = os.getcwd() + '\\Results'
            FileName = Path + '\\' + ImageFilePrefix + str(self.TimeStepCount) + '.png'
            print('FileName', FileName)
            fig.savefig(FileName, dpi = 480, bbox_inches='tight')
                
                
    
    

    def updateLoadDisplacementCurve(self, GeometryType, Plot = True):
            
        
        if GeometryType == 'SENB':
        
            self.PlotCurve_LoadList[self.TimeStepCount] = -self.GlobalExternalLoadVector[2*self.RefLoadNodeId+1]
            self.PlotCurve_DisplacementList[self.TimeStepCount] = -self.GlobalDisplacementVector[2*self.RefLoadNodeId+1]
            
            
            if Plot:
                    
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3)
#                plt.plot(np.array(self.PlotCurve_DisplacementList)*1e3, np.array(self.PlotCurve_LoadList)/1e3, '.')
                plt.xlim(0, 1.0)
                plt.ylim(0, 0.9)
    #            plt.ylim(0, 1.8)
                plt.show()
                
        
        elif GeometryType == 'SENS':
        
            RefLoadNodeId = self.RefLoadNodeObj.Id
            
            self.PlotCurve_LoadList[self.TimeStepCount] = -self.GlobalExternalLoadVector[2*RefLoadNodeId+1]
            self.PlotCurve_DisplacementList[self.TimeStepCount] = -(self.GlobalDisplacementVector[2*self.RefCrackEdgeNodeId1+1]-self.GlobalDisplacementVector[2*self.RefCrackEdgeNodeId0+1])
            
            if Plot:
                        
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3)
#                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3, '.')
                plt.xlim(0, 0.12)
                plt.ylim(0, 160)
                plt.show()
        

            
        elif GeometryType == 'Test_Penalty':
            
            LineLoad = self.delta_lambda*self.RefLineLoad
            self.PlotCurve_LoadList[self.TimeStepCount] = LineLoad
            self.PlotCurve_DisplacementList[self.TimeStepCount] = -self.GlobalDisplacementVector[2*self.RefLoadNodeId+1]
            
            if Plot:
            
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1]), np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1]))
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1]), np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1]), '.')
                plt.show()
            
        
        
        elif GeometryType == 'test_coh':
            
            LineLoad = 0.15*self.delta_lambda*self.RefLineLoad # 0.15 is the length of specimen
            self.PlotCurve_LoadList[self.TimeStepCount] = LineLoad            
            self.PlotCurve_DisplacementList[self.TimeStepCount] = -self.GlobalDisplacementVector[2*self.RefLoadNodeId+1]
            
            if Plot:
                
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3)
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3, '.')
#                plt.xlim(0, 1.0)
#                plt.ylim(0, 12)
                plt.show()


    
        elif GeometryType == 'WingCrack':
            
            LineLoad = 0.15*self.delta_lambda*self.RefLineLoad # 0.15 is the length of specimen
            self.PlotCurve_LoadList[self.TimeStepCount] = LineLoad
            
            RelDispX = self.GlobalDisplacementVector[2*self.RefCrackEdgeNodeId1] - self.GlobalDisplacementVector[2*self.RefCrackEdgeNodeId0]
            RelDispY = self.GlobalDisplacementVector[2*self.RefCrackEdgeNodeId1+1] - self.GlobalDisplacementVector[2*self.RefCrackEdgeNodeId0+1]
            RelDisp = (RelDispX**2 + RelDispY**2)**0.5
            self.PlotCurve_DisplacementList[self.TimeStepCount] = RelDisp
            
            if Plot:
                
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3)
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3, '.')
#                plt.xlim(0, 1.0)
#                plt.ylim(0, 12)
                plt.show()



        elif GeometryType == 'BrazilianRect':
            
            RefDispNodeId0 = self.CrackCurveList[0]['NodeIdList'][0]
            RefDispNodeId1 = self.NodeObjList[RefDispNodeId0].CrackPairNodeId
            
#            RefDispNodeId0 = self.RefLoadNodeObj0.Id
#            RefDispNodeId1 = self.RefLoadNodeObj1.Id
            
#            LineLoad = 0.04*0.15*self.Alpha*self.RefLoad # 0.15 is the length of specimen
            LineLoad = 0.04*0.075*self.delta_lambda*self.RefLoad # 0.15 is the length of specimen
#            LineLoad = self.delta_lambda*self.RefLoad
            
            self.PlotCurve_LoadList[self.TimeStepCount] = LineLoad
            
            RelDispX = self.GlobalDisplacementVector[2*RefDispNodeId0] - self.GlobalDisplacementVector[2*RefDispNodeId1]
            self.PlotCurve_DisplacementList[self.TimeStepCount] = RelDispX
            
            if Plot:
                
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3)
                plt.plot(np.array(self.PlotCurve_DisplacementList[:self.TimeStepCount+1])*1e3, np.array(self.PlotCurve_LoadList[:self.TimeStepCount+1])/1e3, '.')
#                plt.xlim(0, 1.0)
#                plt.ylim(0, 12)
                plt.show()










class NLSolverBaseClass(SolverBaseClass):

    
    def createInterfaceElements(self):
        
        self.InterfaceElementObjList = []
        CrackTip_InterfaceCellIdList = [SubDomainObj.CrackTip_InterfaceCellObj.Id for SubDomainObj in self.PartiallyCrackedSubDomainObjList]
        
        for CrackCurve in self.CrackCurveList:
            
            CrackCurve['InterfaceElementObjList'] = []
            
            for InterfaceCellId in CrackCurve['InterfaceCellIdList']:
                
                InterfaceCellObj = self.PolyCellObjList[InterfaceCellId]
                InterfaceElementObj = InterfaceElement(self, InterfaceCellObj)
                
                if InterfaceCellId in CrackTip_InterfaceCellIdList:
                    
                    InterfaceCellObj.Enabled = False
                    InterfaceElementObj.Enabled = False
                    
                    I = CrackTip_InterfaceCellIdList.index(InterfaceCellId)
                    RefSubDomainObj = self.PartiallyCrackedSubDomainObjList[I]
                    RefSubDomainObj.CrackTip_InterfaceElementObj = InterfaceElementObj
                    InterfaceElementObj.RefPartiallyCrackedSubDomainObj = RefSubDomainObj
                    CrackCurve['CrackTip_InterfaceElementObj'] = InterfaceElementObj
                    InterfaceElementObj.NeighbourInterfaceElementObj = CrackCurve['InterfaceElementObjList'][-1]
                    
                else:   
                    
                    CrackCurve['InterfaceElementObjList'].append(InterfaceElementObj)
                    self.InterfaceElementObjList.append(InterfaceElementObj)
                    
        
        for InterfaceSegmentObj in self.InterfaceSegmentObjList:
            
            InterfaceSegmentObj.InterfaceElementObjList = []
            
            for InterfaceCellObj in InterfaceSegmentObj.InterfaceCellObjList:
                
                InterfaceElementObj = InterfaceElement(self, InterfaceCellObj)
                InterfaceSegmentObj.InterfaceElementObjList.append(InterfaceElementObj)
                self.InterfaceElementObjList.append(InterfaceElementObj)
                    
    
            
    
    
    
    def removeInterfaceCells(self):
        
        self.InterfaceElementObjList = []
        
        for PolyCellObj in self.PolyCellObjList:
            
            if PolyCellObj.ElementType == 'InterfaceElement':
                
                PolyCellObj.Enabled = False
        
    
    
    
    
    
    def removeCohesiveElements(self):
        
        #Changing InterfaceType from Cohesive to Free
        RefInterfaceElementObjList = self.InterfaceElementObjList + [SubDomainObj.CrackTip_InterfaceElementObj for SubDomainObj in self.PartiallyCrackedSubDomainObjList]
        
        for RefInterfaceElementObj in RefInterfaceElementObjList:
            
            if RefInterfaceElementObj.InterfaceType == 'COHESIVE': 
                
                RefInterfaceElementObj.InterfaceType = 'FREE'
                RefInterfaceElementObj.ContactConditionList = ['FREE']*RefInterfaceElementObj.N_GaussLobattoPoints
    
    
    
    

         
    def mapMesh(self, NewMeshRootObj):
        
        MappedNodeIdList = []
        
        if self.FirstNonLinearStepExecuted: #Mapping is after non-linear analysis starts
            
            #Mapping Nodal Deformation of Old mesh into New mesh
            for NewNodeObj in NewMeshRootObj.NodeObjList:
                
                if NewNodeObj.Enabled and NewNodeObj.RequiresMeshMapping:
                    
                    MappedNodeIdList.append(NewNodeObj.Id)
                    #Mapping Displacement of SubDomain of Old Mesh in which NewNodeObj lies
                    #TODO: Reduce the number of SubDomains by considering only those which were subdivided in last remesh.
                    for SubDomainObj in self.SubDomainObjList:
                        
                        RefDisp = SubDomainObj.getFieldDisplacement(NewNodeObj.Coordinate)
                        
                        if len(RefDisp) > 0:
                            
                            NewNodeObj.DeformedCoordinate = NewNodeObj.Coordinate + RefDisp
                            break
                    
                    else:   raise Exception
                    
#            print('mappedNodes', MappedNodeIdList)
        
        
        #Resetting MeshMapping Flag for Nodes
        for NodeObj in NewMeshRootObj.NodeObjList:    NodeObj.RequiresMeshMapping = False
        
        
        
       
    
        
    
     
    def assignDeformationToInterfaceElements(self):
        
        #Updating Interface Element 
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            #Disp. Vector
            InterfaceElementObj.GlobDispVector = self.GlobalDisplacementVector[InterfaceElementObj.DOFIdList]
            
            InterfaceElementObj.updateMaterialProperties()
    
    
    
    
    def updateContactConditions(self):
        
        #Updating Interface Elements
        RefInterfaceElementObjList = self.InterfaceElementObjList + [SubDomainObj.CrackTip_InterfaceElementObj for SubDomainObj in self.PartiallyCrackedSubDomainObjList]
        
        for RefInterfaceElementObj in RefInterfaceElementObjList:
            
            RefInterfaceElementObj.updateContactCondition()
            
           
        
    
    def resetContactConditions(self):
        
        #Updating Interface Elements
        RefInterfaceElementObjList = self.InterfaceElementObjList + [SubDomainObj.CrackTip_InterfaceElementObj for SubDomainObj in self.PartiallyCrackedSubDomainObjList]
        
        for RefInterfaceElementObj in RefInterfaceElementObjList:
            
            RefInterfaceElementObj.resetContactCondition()
            
           
    
    
    def updateContactCohesions(self):
        
        self.GlobalContactCohesionLoadVector = np.zeros(self.N_DOF)
        
        RefInterfaceElementObjList = self.InterfaceElementObjList + [SubDomainObj.CrackTip_InterfaceElementObj for SubDomainObj in self.PartiallyCrackedSubDomainObjList]
        
        #Updating Interface Element         
        for RefInterfaceElementObj in RefInterfaceElementObjList:
            
            RefInterfaceElementObj.calcContactCohesionLoadVector()
            
            #Updating Load Vector
            self.assembleTractionLoadVector(RefInterfaceElementObj.NodeIdList, RefInterfaceElementObj.ContactCohesionLoadVector, self.GlobalContactCohesionLoadVector)
            
        
        
        
    
    def updateSidefaceTractions(self, CrackMouthUpdateType = 'A'):
        
        alpha0 =      self.SolverParameters['alpha0']
        alpha1 =      self.SolverParameters['alpha1']
        
        self.GlobalSidefaceTractionLoadVector = np.zeros(self.N_DOF)
            
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            if RefInterfaceElementObj.InterfaceType in ['CONTACT', 'COHESIVE']:
                
                
                #Updating Material Properties
                RefInterfaceElementObj.updateMaterialProperties()
                RefInterfaceSurfaceArea = RefInterfaceElementObj.SurfaceArea
                    
                if CrackMouthUpdateType == 'A':
                    
                    #Calculating Forces at CrackMouth
                    NeighbourInterfaceElementObj = RefInterfaceElementObj.NeighbourInterfaceElementObj
                    RefLoadVector = np.dot(NeighbourInterfaceElementObj.SecantStiffnessMatrix, NeighbourInterfaceElementObj.GlobDispVector)
                    RefLocalLoadVector = np.dot(NeighbourInterfaceElementObj.TransformationMatrix, RefLoadVector)
                    
                    Fx0 = RefLocalLoadVector[0]
                    Fy0 = RefLocalLoadVector[1]
                    Fx1 = RefLocalLoadVector[2]
                    Fy1 = RefLocalLoadVector[3]
                    
                    CrackMouth_ShearTraction = -(-2*Fx0 + 4*Fx1)/NeighbourInterfaceElementObj.SurfaceArea
                    CrackMouth_NormalTraction = -(-2*Fy0 + 4*Fy1)/NeighbourInterfaceElementObj.SurfaceArea
                
                elif CrackMouthUpdateType == 'B':
                        
                    #Calculating Displacement Vector for Interface Element
                    RefDOFIdList = RefInterfaceElementObj.DOFIdList
                    RefGlobDispVector = self.GlobalDisplacementVector[RefDOFIdList]
                    
                    ScalingCenterDisplacementVector = SubDomainObj.getFieldDisplacement(SubDomainObj.ScalingCenterCoordinate)
                    RefGlobDispVector[2] = RefGlobDispVector[4] = ScalingCenterDisplacementVector[0]
                    RefGlobDispVector[3] = RefGlobDispVector[5] = ScalingCenterDisplacementVector[1]
                    RefInterfaceElementObj.GlobDispVector = RefGlobDispVector
                    
                    RefLoadVector = np.dot(RefInterfaceElementObj.SecantStiffnessMatrix, RefGlobDispVector)
                    RefLocalLoadVector = np.dot(RefInterfaceElementObj.TransformationMatrix, RefLoadVector)
                    
                    Fx0 = RefLocalLoadVector[0]
                    Fy0 = RefLocalLoadVector[1]
                    Fx1 = RefLocalLoadVector[2]
                    Fy1 = RefLocalLoadVector[3]
                    
                    CrackMouth_ShearTraction = -( 4*Fx0 - 2*Fx1)/RefInterfaceSurfaceArea
                    CrackMouth_NormalTraction = -( 4*Fy0 - 2*Fy1)/RefInterfaceSurfaceArea
                
                
                elif CrackMouthUpdateType == 'C':
                        
                    InterfaceMaterialObj = RefInterfaceElementObj.MaterialObj
                    CrackMouth_NormalCOD = RefInterfaceElementObj.NormalCODList[0]
                    CrackMouth_ShearCOD = RefInterfaceElementObj.ShearCODList[0]
                    
                    #Calculating Forces at CrackMouth
                    if RefInterfaceElementObj.InterfaceType == 'COHESIVE':
                        
                        modify_fNTr_to_InterfaceElementObj_based
                        
                        CrackMouth_ShearTraction = -np.sign(CrackMouth_ShearCOD)*InterfaceMaterialObj.fSTr(abs(CrackMouth_ShearCOD))
                        CrackMouth_NormalTraction = np.sign(CrackMouth_NormalCOD)*InterfaceMaterialObj.fNTr(abs(CrackMouth_NormalCOD))
                        
                    else:
                        
                        CrackMouth_ShearTraction = -InterfaceMaterialObj.Cohesion - np.sign(CrackMouth_ShearCOD)*InterfaceMaterialObj.Ks*abs(CrackMouth_ShearCOD)
                        CrackMouth_NormalTraction = np.sign(CrackMouth_NormalCOD)*InterfaceMaterialObj.Kn*abs(CrackMouth_NormalCOD)
                        
                    
                
                    
                #Modifying Sideface Tractions
                alpha_n0 = alpha0
                alpha_s0 = alpha0
                alpha_n1 = alpha1
                alpha_s1 = alpha1
                
                SIF, Temp1, Temp2 = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
                KI = SIF[0]
                KII = SIF[1]
                
#                print('KI_KII', KI, KII, self.TimeStepCount)
#                print(100*KI/KII)
                
                #Updating Crack Mouth Pressure
                ModifiedCrackMouth_NormalTraction = RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction
                dTr_CrackMouth = alpha_n1*(CrackMouth_NormalTraction - ModifiedCrackMouth_NormalTraction)
                ModifiedCrackMouth_NormalTraction += dTr_CrackMouth
                CrackMouth_NormalTraction = ModifiedCrackMouth_NormalTraction
                RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction = CrackMouth_NormalTraction
#                    
                ModifiedCrackMouth_ShearTraction = RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction
                dTr_CrackMouth = alpha_s1*(CrackMouth_ShearTraction - ModifiedCrackMouth_ShearTraction)
                ModifiedCrackMouth_ShearTraction += dTr_CrackMouth
                CrackMouth_ShearTraction = ModifiedCrackMouth_ShearTraction
                RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction = CrackMouth_ShearTraction
                
                
                #Updating Crack-Tip Pressure
                self.ModeI_SIFList.append(KI)
                dTrN_CrackTip = alpha_n0*KI
                ModifiedCrackTip_NormalTraction_tmp = RefInterfaceElementObj.ModifiedCrackTip_NormalTraction + dTrN_CrackTip
                CrackTip_NormalTraction = ModifiedCrackTip_NormalTraction_tmp
                RefInterfaceElementObj.ModifiedCrackTip_NormalTraction = CrackTip_NormalTraction
                
                
                self.ModeII_SIFList.append(KII)
                dTrS_CrackTip = alpha_s0*KII
                ModifiedCrackTip_ShearTraction_tmp = RefInterfaceElementObj.ModifiedCrackTip_ShearTraction + dTrS_CrackTip
                
                
#                if RefInterfaceElementObj.InterfaceType == 'COHESIVE':    
#                    
#                    if ModifiedCrackTip_ShearTraction_tmp > RefInterfaceElementObj.MaterialObj.Fs:    ModifiedCrackTip_ShearTraction_tmp = RefInterfaceElementObj.MaterialObj.Fs
                
                
                CrackTip_ShearTraction = ModifiedCrackTip_ShearTraction_tmp
                RefInterfaceElementObj.ModifiedCrackTip_ShearTraction = CrackTip_ShearTraction
                
                
#                print('CrackMouth_ShearTraction', CrackMouth_ShearTraction)
#                print('CrackMouth_NormalTraction', CrackMouth_NormalTraction)
#                print('CrackTip_ShearTraction', CrackTip_ShearTraction)
#                print('CrackTip_NormalTraction', CrackTip_NormalTraction)
#                print('---')
                
                #Polynomial coefficient (Linear)
                TrS0 = CrackTip_ShearTraction
                TrN0 = CrackTip_NormalTraction
                TrS1 = CrackMouth_ShearTraction - CrackTip_ShearTraction
                TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
                
                TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
                
                #Calculating Sideface Traction Parameters
                SubDomainObj.calcSidefaceTractionParameters_1(TractionPolynomialCoefficientList, RefInterfaceSurfaceArea)
                
                
                #Updating Load Vector
                self.assembleTractionLoadVector(SubDomainObj.NodeIdList, SubDomainObj.SidefaceTractionLoadVector, self.GlobalSidefaceTractionLoadVector)
                
    
    
    
    def updateSidefaceTractions_Backup(self):
        
        alpha0 =      self.SolverParameters['alpha0']
        alpha1 =      self.SolverParameters['alpha1']
        
        self.GlobalSidefaceTractionLoadVector = np.zeros(self.N_DOF)
            
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            
            if RefInterfaceElementObj.InterfaceType in ['CONTACT', 'COHESIVE']:
                
                #Updating Material Properties
                RefInterfaceElementObj.updateMaterialProperties()
                
                #Calculating Displacement Vector for Interface Element
                RefDOFIdList = RefInterfaceElementObj.DOFIdList
                RefGlobDispVector = self.GlobalDisplacementVector[RefDOFIdList]
                
                ScalingCenterDisplacementVector = SubDomainObj.getFieldDisplacement(SubDomainObj.ScalingCenterCoordinate)
                RefGlobDispVector[2] = RefGlobDispVector[4] = ScalingCenterDisplacementVector[0]
                RefGlobDispVector[3] = RefGlobDispVector[5] = ScalingCenterDisplacementVector[1]
                RefInterfaceElementObj.GlobDispVector = RefGlobDispVector
                
                RefLoadVector = np.dot(RefInterfaceElementObj.SecantStiffnessMatrix, RefGlobDispVector)
                RefLocalLoadVector = np.dot(RefInterfaceElementObj.TransformationMatrix, RefLoadVector)
                
                Fx0 = RefLocalLoadVector[0]
                Fy0 = RefLocalLoadVector[1]
                Fx1 = RefLocalLoadVector[2]
                Fy1 = RefLocalLoadVector[3]
                
                RefInterfaceElementSurfaceArea = RefInterfaceElementObj.SurfaceArea
            
                CrackMouth_ShearTraction = -( 4*Fx0 - 2*Fx1)/RefInterfaceElementSurfaceArea
                CrackTip_ShearTraction = -(-2*Fx0 + 4*Fx1)/RefInterfaceElementSurfaceArea
                CrackMouth_NormalTraction = -( 4*Fy0 - 2*Fy1)/RefInterfaceElementSurfaceArea
                CrackTip_NormalTraction = -(-2*Fy0 + 4*Fy1)/RefInterfaceElementSurfaceArea
                
                #Modifying Sideface Tractions
                alpha_n0 = alpha0
                alpha_s0 = alpha0
                alpha_n1 = alpha1
                alpha_s1 = alpha1
                
                SIF, Temp1, Temp2 = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
                KI = SIF[0]
                KII = SIF[1]
                
                
#                print('KI_KII', KI, KII, self.TimeStepCount)
#                print(100*KI/KII)
                
                #Updating Crack Mouth Pressure
                ModifiedCrackMouth_NormalTraction = RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction
                dTr_CrackMouth = alpha_n1*(CrackMouth_NormalTraction - ModifiedCrackMouth_NormalTraction)
                ModifiedCrackMouth_NormalTraction += dTr_CrackMouth
                CrackMouth_NormalTraction = ModifiedCrackMouth_NormalTraction
                RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction = CrackMouth_NormalTraction
#                    
                ModifiedCrackMouth_ShearTraction = RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction
                dTr_CrackMouth = alpha_s1*(CrackMouth_ShearTraction - ModifiedCrackMouth_ShearTraction)
                ModifiedCrackMouth_ShearTraction += dTr_CrackMouth
                CrackMouth_ShearTraction = ModifiedCrackMouth_ShearTraction
                RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction = CrackMouth_ShearTraction
                
                
                #Updating Crack-Tip Pressure                    
                self.ModeI_SIFList.append(KI)
                dTrN_CrackTip = alpha_n0*KI
                ModifiedCrackTip_NormalTraction_tmp = RefInterfaceElementObj.ModifiedCrackTip_NormalTraction + dTrN_CrackTip
                
                self.ModeII_SIFList.append(KII)
                dTrS_CrackTip = alpha_s0*KII
                ModifiedCrackTip_ShearTraction_tmp = RefInterfaceElementObj.ModifiedCrackTip_ShearTraction + dTrS_CrackTip
                
                
                if RefInterfaceElementObj.InterfaceType == 'COHESIVE':
                    
#                    if abs(ModifiedCrackTip_NormalTraction_tmp) < abs(RefInterfaceElementObj.MaterialObj.Ft):
                        
                    CrackTip_NormalTraction = ModifiedCrackTip_NormalTraction_tmp
                    
#                    else:   CrackTip_NormalTraction = RefInterfaceElementObj.MaterialObj.Ft
                    
#                    if abs(ModifiedCrackTip_ShearTraction_tmp) < abs(RefInterfaceElementObj.MaterialObj.Fs):
#                        
                    CrackTip_ShearTraction = ModifiedCrackTip_ShearTraction_tmp
#                        
#                    else:   CrackTip_ShearTraction = RefInterfaceElementObj.MaterialObj.Fs
                        
                    
                if RefInterfaceElementObj.InterfaceType == 'CONTACT':
                    
                    CrackTip_NormalTraction = ModifiedCrackTip_NormalTraction_tmp
                        
#                    if RefInterfaceElementObj.ContactConditionList[-1] == 'STICK':       
                        
                    CrackTip_ShearTraction = ModifiedCrackTip_ShearTraction_tmp
                    
                    
                RefInterfaceElementObj.ModifiedCrackTip_NormalTraction = CrackTip_NormalTraction
                RefInterfaceElementObj.ModifiedCrackTip_ShearTraction = CrackTip_ShearTraction
                
                
#                print('CrackMouth_ShearTraction', CrackMouth_ShearTraction)
#                print('CrackMouth_NormalTraction', CrackMouth_NormalTraction)
#                print('CrackTip_ShearTraction', CrackTip_ShearTraction)
#                print('CrackTip_NormalTraction', CrackTip_NormalTraction)
#                print('---')
                
                #Polynomial coefficient (Linear)
                TrS0 = CrackTip_ShearTraction
                TrN0 = CrackTip_NormalTraction
                TrS1 = CrackMouth_ShearTraction - CrackTip_ShearTraction
                TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
                
                TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
                
                #Calculating Sideface Traction Parameters
                SubDomainObj.calcSidefaceTractionParameters_1(TractionPolynomialCoefficientList, RefInterfaceElementSurfaceArea)
                
                
                #Updating Load Vector
                self.assembleTractionLoadVector(SubDomainObj.NodeIdList, SubDomainObj.SidefaceTractionLoadVector, self.GlobalSidefaceTractionLoadVector)
                
    
    def updateGlobalTangentStiffnessMatrix(self, UpdateInverse = False):
    
        #Updating Global Stiffness matrix with deformed interface elements
        self.GlobalTangentStiffnessMatrix = np.array(self.GlobalStiffnessMatrix, dtype = float)
        
            
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            #Calculating Global Tangent Stiffness Matrix
            self.assembleStiffnessMatrix(InterfaceElementObj.NodeIdList, InterfaceElementObj.TangentStiffnessMatrix, self.GlobalTangentStiffnessMatrix)
        
        
        self.applyBoundaryConditions(self.GlobalTangentStiffnessMatrix)
        
            
        if UpdateInverse: self.InvGlobalTangentStiffnessMatrix = inv(self.GlobalTangentStiffnessMatrix)
        
    
    
    
    
    
    def updateGlobalSecantStiffnessMatrix(self, UpdateInverse = False):
    
        #Updating Global Stiffness matrix with deformed interface elements
        self.GlobalSecantStiffnessMatrix = np.array(self.GlobalStiffnessMatrix, dtype = float)
        
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            if not InterfaceElementObj.InterfaceType == 'FREE':
                
                #Calculating Global Secant Stiffness Matrix
                self.assembleStiffnessMatrix(InterfaceElementObj.NodeIdList, InterfaceElementObj.SecantStiffnessMatrix, self.GlobalSecantStiffnessMatrix)
            
        self.applyBoundaryConditions(self.GlobalSecantStiffnessMatrix)
            
        if UpdateInverse: self.InvGlobalSecantStiffnessMatrix = inv(self.GlobalSecantStiffnessMatrix)
        
    
    
    
    
    def calcLocalDeformationVector(self, RefDispVector):
        
        Relative_RefDispVector = []
        
        #Calculating Relative Displcement in Interface Elements
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            if not InterfaceElementObj.InterfaceType == 'FREE':
                        
                XDOFId0 = InterfaceElementObj.DOFIdList[0]
                XDOFId1 = InterfaceElementObj.DOFIdList[2]
                XDOFId2 = InterfaceElementObj.DOFIdList[4]
                XDOFId3 = InterfaceElementObj.DOFIdList[6]
                
                YDOFId0 = InterfaceElementObj.DOFIdList[1]
                YDOFId1 = InterfaceElementObj.DOFIdList[3]
                YDOFId2 = InterfaceElementObj.DOFIdList[5]
                YDOFId3 = InterfaceElementObj.DOFIdList[7]
                
                Relative_RefDispX0 = RefDispVector[XDOFId3] - RefDispVector[XDOFId0]
                Relative_RefDispY0 = RefDispVector[YDOFId3] - RefDispVector[YDOFId0]
                
                Relative_RefDispX1 = RefDispVector[XDOFId2] - RefDispVector[XDOFId1]
                Relative_RefDispY1 = RefDispVector[YDOFId2] - RefDispVector[YDOFId1]
                
                
                Relative_RefDispVector += [Relative_RefDispX0, Relative_RefDispY0, Relative_RefDispX1, Relative_RefDispY1]
                
        
        #Calculating Relative Displcement of CrackMouth in Partially Cracked SubDomain
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            if not RefInterfaceElementObj.InterfaceType == 'FREE':
                    
                CrackMouth_XDOFId0 = SubDomainObj.CrackMouth_DOFIdList[0]
                CrackMouth_XDOFId1 = SubDomainObj.CrackMouth_DOFIdList[2]
                
                CrackMouth_YDOFId0 = SubDomainObj.CrackMouth_DOFIdList[1]
                CrackMouth_YDOFId1 = SubDomainObj.CrackMouth_DOFIdList[3]
                
                Relative_RefDispX = RefDispVector[CrackMouth_XDOFId0] - RefDispVector[CrackMouth_XDOFId1]
                Relative_RefDispY = RefDispVector[CrackMouth_YDOFId0] - RefDispVector[CrackMouth_YDOFId1]
                Relative_RefDispVector += [Relative_RefDispX, Relative_RefDispY, 0.0, 0.0]
            
        
        return np.array(Relative_RefDispVector, dtype = float)
    
    
    
    
    def updateContactPressureError(self):
        
        RefContactPressure = np.array([])
        RefContactShear = np.array([])
        
        
        CPErrorList = []
        if len(self.CrackCurveList) > 1:    raise NotImplementedError
        CrackCurve = self.CrackCurveList[0]
    
        RefInterfaceElementObjList = CrackCurve['InterfaceElementObjList'] + [CrackCurve['CrackTip_InterfaceElementObj']]
            
        N_Elem = len(RefInterfaceElementObjList)
        for i in range(N_Elem):
            
            InterfaceElementObj = RefInterfaceElementObjList[i]
            
            LinearCohesiveTractionVector = InterfaceElementObj.calcLinearTractionVector()
            Local_LinearCohesiveTractionVector = np.dot(InterfaceElementObj.TransformationMatrix, LinearCohesiveTractionVector)
    
            ContactShear = Local_LinearCohesiveTractionVector[0]
            CPError_s = ContactShear - RefContactShear[i]
            CPErrorList.append(CPError_s)
            
            ContactPressure = Local_LinearCohesiveTractionVector[1]
            CPError_p = ContactPressure - RefContactPressure[i]
            CPErrorList.append(CPError_p)
            
            if i == N_Elem-1:
                
                ContactShear = Local_LinearCohesiveTractionVector[2]
                CPError_s = ContactShear - RefContactShear[i+1]
                CPErrorList.append(CPError_s)
                
                ContactPressure = Local_LinearCohesiveTractionVector[3]                
                CPError_p = ContactPressure - RefContactPressure[i+1]
                CPErrorList.append(CPError_p)
                
        
        NormCPError = norm(CPErrorList)/norm(np.append(RefContactPressure, RefContactShear))
        self.NormCPErrorList.append(NormCPError)
        
    
     
    def displayContactTraction(self):
        
        XCoord_List = []
        YCoord_List = []
        CST_List = []
        CNT_List = []
        
        if len(self.CrackCurveList) > 1:    raise NotImplementedError
        CrackCurve = self.CrackCurveList[0]
    
        RefInterfaceElementObjList = CrackCurve['InterfaceElementObjList'] + [CrackCurve['CrackTip_InterfaceElementObj']]
            
        N_Elem = len(RefInterfaceElementObjList)
        for i in range(N_Elem):
            
            InterfaceElementObj = RefInterfaceElementObjList[i]
            
            #COORDINATES
            XCoord = InterfaceElementObj.NodeObjList[0].Coordinate[0]
            YCoord = InterfaceElementObj.NodeObjList[0].Coordinate[1]
            XCoord_List.append(XCoord)
            YCoord_List.append(YCoord)
            
            #TRACTION
            LinearCohesiveTractionVector = InterfaceElementObj.calcLinearTractionVector()
            Local_LinearCohesiveTractionVector = np.dot(InterfaceElementObj.TransformationMatrix, LinearCohesiveTractionVector)
    
            
            ContactShear = Local_LinearCohesiveTractionVector[0]
            CST_List.append(ContactShear)
            
            ContactPressure = Local_LinearCohesiveTractionVector[1]
            CNT_List.append(ContactPressure)
            
            #Crack-Tip Tractions
            if i == N_Elem-1:
                
                #COORDINATES
                XCoord = InterfaceElementObj.NodeObjList[1].Coordinate[0]
                YCoord = InterfaceElementObj.NodeObjList[1].Coordinate[1]
                XCoord_List.append(XCoord)
                YCoord_List.append(YCoord)
                
                #TRACTION
                ContactShear = Local_LinearCohesiveTractionVector[2]
                CST_List.append(ContactShear)
                
                ContactPressure = Local_LinearCohesiveTractionVector[3]
                CNT_List.append(ContactPressure)
                
            
#            
#        
#        self.CST_List = CST_List
#        self.CNT_List = CNT_List
#       
        print('----Contact Tractions------')
        print('XCoord', XCoord_List)
        print('YCoord', YCoord_List)
        print('rCoord', np.sqrt(np.array(XCoord_List)**2 + np.array(YCoord_List)**2))
        print('CST', CST_List)
        print('CNT', CNT_List)
        print('---------------------------')
        
        
    
    
    def initLoadStep(self):
        
        #Initializing Global Displacement Vector
        self.GlobalDisplacementVector =     np.zeros(self.N_DOF)
        
        for NodeObj in  self.NodeObjList:
            
            for j in range(self.N_DOFPerNode):
                
                self.GlobalDisplacementVector[NodeObj.DOFIdList[j]] = NodeObj.DeformedCoordinate[j] - NodeObj.Coordinate[j]
        
        
        #Initializing Stiffness Matrices of Interface Elements
        self.assignDeformationToNodes()
        self.assignDeformationToSubDomains()
        self.assignDeformationToInterfaceElements()
        self.updateSidefaceTractions()
        self.updateContactCohesions()
        self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
        
        
        #Updating GlobalRefTransientLoadVector to include reactions        
        RefTransientDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
        RefTransientDisplacementVector[self.FilteredBCDOFList] = 0.0
        self.GlobalRefTransientLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, RefTransientDisplacementVector)
        
        
        #Initializing solver variables
        if not self.FirstNonLinearStepExecuted:
            
            del_lambda_ini =    self.SolverParameters['del_lambda_ini']
            Nd0 =               self.SolverParameters['Nd0']
            n =                 self.SolverParameters['n']
            Nint =              len(self.InterfaceElementObjList)
            Nd =                Nint**n + Nd0
            
            del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
            Local_del_up = self.calcLocalDeformationVector(del_up)
            del_lambda = del_lambda_ini
            self.delta_l = del_lambda*norm(Local_del_up)
            
            self.LastIterCount = Nd
            self.delta_lambda = 0.0
        
    
    
    def resetLoadStep(self):
        
        #Initializing Global Displacement Vector
        self.GlobalDisplacementVector =     np.zeros(self.N_DOF)
        
        
        #Initializing Stiffness Matrices of Interface Elements
        self.assignDeformationToNodes()
        self.assignDeformationToSubDomains()
        self.assignDeformationToInterfaceElements()
        self.updateSidefaceTractions()
        self.resetContactConditions()
        self.updateContactCohesions()
        self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
        
        
        #Updating GlobalRefTransientLoadVector to include reactions        
        RefTransientDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
        RefTransientDisplacementVector[self.FilteredBCDOFList] = 0.0
        self.GlobalRefTransientLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, RefTransientDisplacementVector)
        
            
        #Initializing solver variables
        del_lambda_ini =    self.SolverParameters['del_lambda_ini']
        Nd0 =               self.SolverParameters['Nd0']
        n =                 self.SolverParameters['n']
        Nint =              len(self.InterfaceElementObjList)
        Nd =                Nint**n + Nd0
        
        del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
        Local_del_up = self.calcLocalDeformationVector(del_up)
        del_lambda = del_lambda_ini
        self.delta_l = del_lambda*norm(Local_del_up)
        
        self.LastIterCount = Nd
        self.delta_lambda = 0.0
    
    
    
    
    
    def backupLoadStep(self):
        """
        Useful in changing direction of friction        
        """
        pass
    
    
    
    
    def restoreLoadStep(self):
        
        pass
    
    
    
        
    
    def runLocalDuanSolver(self): 
        
        
        #Arc Length Parameters
        MaxIterCount =      self.SolverParameters['MaxIterCount']
        RelTol =            self.SolverParameters['RelTol']
        m =                 self.SolverParameters['m']
        Nd0 =               self.SolverParameters['Nd0']
        n =                 self.SolverParameters['n']
        p =                 self.SolverParameters['p']
        Nint =              len(self.InterfaceElementObjList)
        Nd =                Nint**n + Nd0
        
        
        #Marking First Non-Linear Step is Executed
        self.FirstNonLinearStepExecuted = True
        
        
        #Initializing Residual
        self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
        NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector + self.GlobalContactCohesionLoadVector   
        GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
        R = GlobalInternalLoadVector - NetExternalLoadVector
        NormR = norm(R)
#        LastNormR = NormR
        
        
#        if self.TimeStepCount == 55:    self.delta_l = self.delta_l/5.0
        
        #Looping over Timesteps
        while self.TimeStepCount < self.MaxTimeStepCount:
            
            self.TimeStepCount += 1
            
            IterCount =         0
            self.delta_u =      np.zeros(self.N_DOF)
            self.delta_l =      self.delta_l*(float(Nd)/self.LastIterCount)**m
            
            if self.TimeStepCount%50 == 0:
            
                print('')
                print('StepCounts', self.CrackIncrementStepCount, self.TimeStepCount)
                print('LastIterCount', self.LastIterCount, self.delta_l)
                
            
            self.updateContactConditions()
            
            while IterCount < MaxIterCount:
                
                
                del_uf = -np.dot(self.InvGlobalSecantStiffnessMatrix, R)
                del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
                
                Local_del_uf = self.calcLocalDeformationVector(del_uf)
                Local_del_up = self.calcLocalDeformationVector(del_up)
                Local_delta_u = self.calcLocalDeformationVector(self.delta_u)
                
                   
                if IterCount == 0:  
                    
                    
                    del_lambda = self.delta_l/norm(Local_del_up)
                    del_lambda0 = del_lambda
                    
                else:
                    
#                    del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
#                    del_lambda = (self.delta_l**2 - np.dot(Local_delta_u, Local_delta_u + Local_del_uf))/np.dot(Local_delta_u, Local_del_up)
                    del_lambda = -np.dot(Local_delta_u, Local_del_uf)/np.dot(Local_delta_u, Local_del_up)
                    
                
                del_u = del_uf + del_lambda*del_up
                del_u[self.FilteredBCDOFList] = 0.0
                
#                    print('dl', del_lambda)
#                    print('du', del_u)
                
                
                #Updating increments
                self.delta_u +=  del_u
                self.delta_lambda += del_lambda
                
#                print('delta_l', self.delta_l)
                
                
                self.GlobalDisplacementVector += del_u
#                self.GlobalExternalLoadVector = self.GlobalStaticLoadVector + self.delta_lambda*self.GlobalRefTransientLoadVector
                self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
                
                #TODO: Bisection
#                self.checkBisection(IterCount, MaxIterCount)
                
                
                self.assignDeformationToNodes()
                self.assignDeformationToSubDomains()
                self.assignDeformationToInterfaceElements()
                self.updateSidefaceTractions()
                self.updateContactCohesions()
                self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
                
                NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector + self.GlobalContactCohesionLoadVector   
                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
                
                R = GlobalInternalLoadVector - NetExternalLoadVector
                NormR = norm(R)
                RelNormR = NormR/norm(NetExternalLoadVector)
                
#                if IterCount > 0 and NormR > LastNormR:   break
#                LastNormR = NormR
                    
#                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#                print 'RelNormR', RelNormR
#                self.plotNodalDisplacement(DeformationScale = 0.0, SaveImage = False)
            
#                self.updateContactPressureError()
                
                if RelNormR < RelTol:    
                    
                    self.LastIterCount = IterCount + 1
                    break
                
                else:   IterCount += 1
                
                
            
            else:
            
                print('Convergence was not achieved;',  self.FailedConvergenceCount, RelNormR)
                self.FailedConvergenceCount += 1
                self.LastIterCount = Nd*p
                
            
            #Plotting LoadDisplcement Curve 
            self.updateLoadDisplacementCurve(GeometryType = 'SENS', Plot = False) 
#            print('LL', self.CrackIncrementStepCount)
#                
###            -------------------------------
#            if self.TimeStepCount <= 200:            TGap = 20
#            elif 200 < self.TimeStepCount <= 300:     TGap = 10
#            else:                                   TGap = 5
#            
#            if self.TimeStepCount%TGap == 0:   
#                
#                self.updateLoadDisplacementCurve(GeometryType = 'BrazilianRect', Plot = True) 
#                self.plotNodalDisplacement(DeformationScale = 1.0e2, SaveImage = True)
#                
#                print('DL' ,self.PlotCurve_DisplacementList[:self.TimeStepCount+1])
#                print('LL', self.PlotCurve_LoadList[:self.TimeStepCount+1])
#                
#                SBFEM.PostProcessor.FieldVariables(self)
##            -------------------------------
            
            
#            print('CPError', self.NormCPErrorList)
#            print('SIF_I', self.ModeI_SIFList)
#            print('SIF_II', self.ModeII_SIFList)
            
#            self.displayContactTraction()
            
            
#            for InterfaceElementObj in self.InterfaceElementObjList:
#                
#                print(InterfaceElementObj.Id, InterfaceElementObj.ShearCODList[2])
#            
#            print('')
            
#            if self.TimeStepCount > 150:   self.plotNodalDisplacement(DeformationScale = 1.0e2, SaveImage = True)
            
#            self.plotNodalDisplacement(DeformationScale = 0.0, SaveImage = True)
#            SBFEM.PostProcessor.FieldVariables(self)
            
            #Checking if Crack Propagates
            if self.checkCrackPropagation() == True: 
                
                self.updateLoadDisplacementCurve(GeometryType = 'SENS', Plot = True)   
            
#                self.TempFDList.append([self.CrackIncrementStepCount, self.PlotCurve_DisplacementList[-1], self.PlotCurve_LoadList[-1]])
#                self.plotNodalDisplacement(DeformationScale = 3.0e2, SaveImage = True)
#                self.plotNodalDisplacement(DeformationScale = 0.0e2, ShowTractions = False, ImageFilePrefix = 'D_')
             
#                print('DL' ,self.PlotCurve_DisplacementList[:self.TimeStepCount+1])
#                print('LL', self.PlotCurve_LoadList[:self.TimeStepCount+1])
#                print('TFD', self.TempFDList)
                break
            
#            else:   self.plotNodalDisplacement(DeformationScale = 0.0, SaveImage = True)

#            self.displayContactTraction()
        
            if self.TimeStepCount >= self.MaxTimeStepCount or \
               self.CrackIncrementStepCount >= self.MaxCrackIncrementStepCount or \
               self.FailedConvergenceCount >= self.MaxFailedConvergenceCount:
                
                self.AnalysisFinished = True
                break
        
            
            
            
    
    
    
    def checkCrackPropagation(self):
        
        CrackPropagates = False
        
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:  
        
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            if RefInterfaceElementObj.InterfaceType == 'FREE':          
                
                if self.checkGCriteria(SubDomainObj):
                    
                    SubDomainObj.HasCrackPropagation = True
                    CrackPropagates = True
            
            elif RefInterfaceElementObj.InterfaceType == 'COHESIVE':    
                
#                if self.checkZeroKICriteria(SubDomainObj):
                
                if self.HasMaterialHeterogeneity:
                        
                    if self.checkMaxPStressCriteria_HeteroMat(SubDomainObj):
    #                if self.checkMaxHoopStressCriteria(SubDomainObj):
                        
                        SubDomainObj.HasCrackPropagation = True
                        CrackPropagates = True
                
                else:
                    
                    if self.checkMaxPStressCriteria_HomoMat(SubDomainObj):
    #                if self.checkMaxHoopStressCriteria(SubDomainObj):
                        
                        SubDomainObj.HasCrackPropagation = True
                        CrackPropagates = True
                    
                
            
            
            elif RefInterfaceElementObj.InterfaceType == 'CONTACT':     
                
                if self.checkMohrCoulombCriteria(SubDomainObj):
#                if self.checkFrictionalSlipCriteria(SubDomainObj):                    
#                if self.checkGCriteria(SubDomainObj):
                    
                    SubDomainObj.HasCrackPropagation = True
                    CrackPropagates = True
                    
        
        return CrackPropagates
        
        
        
    
    
    def checkZeroKICriteria(self, SubDomainObj):
    
        RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
        RefInterfaceElementSurfaceArea = RefInterfaceElementObj.SurfaceArea
        
        InterfaceMaterialObj = RefInterfaceElementObj.MaterialObj
        CrackMouth_NormalCOD = RefInterfaceElementObj.NormalCODList[0]
        
        
        #Calculating Forces at CrackMouth
        CrackMouth_ShearTraction = 0.0
        
        if CrackMouth_NormalCOD > InterfaceMaterialObj.wc:      raise Exception
        elif CrackMouth_NormalCOD <= InterfaceMaterialObj.w0:   CrackMouth_NormalTraction = InterfaceMaterialObj.Ft
        else:                                                   CrackMouth_NormalTraction = InterfaceMaterialObj.fNTr(CrackMouth_NormalCOD)
           
        
        #Calculating Forces at CrackTip
        CrackTip_ShearTraction = 0.0
        CrackTip_NormalTraction = InterfaceMaterialObj.Ft
        
        TrS0 = CrackTip_ShearTraction
        TrN0 = CrackTip_NormalTraction
        TrS1 = CrackMouth_ShearTraction - CrackTip_ShearTraction
        TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
        
        TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
        
        #Calculating IntegrationConstant
        RefSideface_CumulativeDispMode = SubDomainObj.calcSidefaceTractionParameters_1(TractionPolynomialCoefficientList, RefInterfaceElementSurfaceArea, SaveAttr = False)
        SubDomain_GlobDispVector = self.GlobalDisplacementVector[SubDomainObj.DOFIdList]
        RefIntegrationConstantList = np.dot(inv(SubDomainObj.EigVec_DispModeData), SubDomain_GlobDispVector - RefSideface_CumulativeDispMode)
        
        #Checking ZeroKI
        #TODO: Modify for multiple cracks
        SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
        KI = SIF[0]
        
        print('KI', KI)
        
        if KI >= 0:     return True
        else:           return False
            
    
    
    
    
    def checkFrictionalSlipCriteria(self, SubDomainObj):
    
        RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
        
        if RefInterfaceElementObj.ContactConditionList[0] == 'SLIP':
                
            return True
        
        else:   return False
    
    
    
    
    
    
    def checkMaxHoopStressCriteria(self, SubDomainObj):
        
        #Calculating Mohr-Circle Parameters
        SubDomainObj.calcMaxHoopStressCriteriaParameters()
        
        #Reading Material Properties
        Ft = SubDomainObj.MaterialObj.Ft
#        C0 = SubDomainObj.MaterialObj.MaterialCohesion
#        U0 = SubDomainObj.MaterialObj.MaterialFrictionCoefficient
        
        
        #Searching Failure
        N_RefAngle = len(SubDomainObj.CrackTipRegion_HoopStressParamList)
        RefAngleList = []
        SigTTList = []
        for i in range(N_RefAngle):
            
            CrackTipRegion_HoopStressParam_i = SubDomainObj.CrackTipRegion_HoopStressParamList[i]
            
            SigTT_i = CrackTipRegion_HoopStressParam_i['PolarStressList'][1]
            RefAngle_i = CrackTipRegion_HoopStressParam_i['RefAngle']
            
            
#            X_i = CrackTipRegion_MohrCircleParam_i['CoordinateList'][0]
#            Y_i = CrackTipRegion_MohrCircleParam_i['CoordinateList'][1]

#            dFt =   self.Ft_interp2d(X_i, Y_i)
            
            modify_this
            dFt = 0.0
            Ft_i =  Ft + dFt
            
            #CCM
            if SigTT_i >= Ft_i: 
                
                RefAngleList.append(RefAngle_i)
                SigTTList.append(SigTT_i)
                
                
        if len(RefAngleList) > 0:
                
            I = SigTTList.index(max(SigTTList))
            RefAngle_I = RefAngleList[I]
            CrackPropagationAngle = SubDomainObj.CrackAngle + RefAngle_I
            SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
            
            print('Tensile Failure', RefAngle_i*180/np.pi, CrackPropagationAngle*180/np.pi)
                        
            return True
                
            
        else:   return False
            



    
    
    
    def checkMaxPStressCriteria_HomoMat(self, SubDomainObj):
        
        #Reading Crack-tip Tensile Strength
        CrackTip_Ft = SubDomainObj.MaterialObj.Ft
        
        #Calculating Crack-tip P-Stress Parameters
        SubDomainObj.calcMaxPStressCriteriaParameters()
        CrackTip_PStressParam = SubDomainObj.CrackTip_PStressParam
        MC_Center = CrackTip_PStressParam['MC_Center']
        LocalSigXX, LocalSigYY, LocalTauXY = CrackTip_PStressParam['LocalStressList']
        PStress1 = CrackTip_PStressParam['PStressList'][0]
        PStress2 = CrackTip_PStressParam['PStressList'][1]
        
        if PStress2 > PStress1: raise Exception
        
        if PStress1 >= CrackTip_Ft:
            
            Tensile_FailureAngleList = [0.0]*2
            
            Tensile_FailureAngleList[0] = np.pi/2.0 + 0.5*np.arctan2(LocalTauXY, LocalSigXX - MC_Center)
            Tensile_FailureAngleList[1] = -np.pi/2.0 + 0.5*np.arctan2(LocalTauXY, LocalSigXX - MC_Center)
            print('')
            print('Tensile_FailureAngles (MaxPrinci)', Tensile_FailureAngleList)
            print('-------------------------')
#                
            
            for Tensile_FailureAngle in Tensile_FailureAngleList:
            
                EquivalentFailureAngleList = [0.0]*3
                EquivalentFailureAngleList[0] = Tensile_FailureAngle - np.pi
                EquivalentFailureAngleList[1] = Tensile_FailureAngle
                EquivalentFailureAngleList[2] = Tensile_FailureAngle + np.pi
                
                for EquivalentFailureAngle in EquivalentFailureAngleList:
                    
                    if abs(EquivalentFailureAngle) <= np.pi/2:
                                
                        CrackPropagationAngle = SubDomainObj.CrackAngle + EquivalentFailureAngle
                        print('Tensile Failure', EquivalentFailureAngle*180/np.pi, CrackPropagationAngle*180/np.pi)
                        
                        SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
                
                        return True
            
            else:   raise Exception
            
            
        else:   return False
            

            
    
    def checkMaxPStressCriteria_HeteroMat(self, SubDomainObj):
        
        #Calculating Crack-tip Tensile Strength
        Xc = SubDomainObj.ScalingCenterCoordinate[0]
        Yc = SubDomainObj.ScalingCenterCoordinate[1]
        CrackTip_Ft = self.Ft_interp2d(Xc, Yc)
        
        
        #Calculating Crack-tip P-Stress Parameters
        SubDomainObj.calcMaxPStressCriteriaParameters()
        CrackTip_PStressParam = SubDomainObj.CrackTip_PStressParam
        CrackTip_PStress1, CrackTip_PStress2 = CrackTip_PStressParam['PStressList']
        
        if CrackTip_PStress1 < CrackTip_PStress2:   raise Exception
        
        if CrackTip_PStress1 >= CrackTip_Ft:
                
            LocalSigXX, LocalSigYY, LocalTauXY = CrackTip_PStressParam['LocalStressList']
            
            LocalStressTensor = np.array([ [LocalSigXX,   LocalTauXY,      0],
                                           [LocalTauXY,   LocalSigYY,      0],
                                           [         0,            0,      0]], dtype = complex)
            
            CrackTipRegion_CoordinateParamList = SubDomainObj.getCrackTipRegionCoordinateParameters(self.CrackIncrementLength)
            N_RefAngle = len(CrackTipRegion_CoordinateParamList)
            TendencyIndicatorList = [] #Yang 2008, Heterogenous Cohesive Crack model, CMAME
            
            for i in range(N_RefAngle):
                
                CrackTipRegion_CoordinateParam = CrackTipRegion_CoordinateParamList[i]
                
                RefCoordinate = CrackTipRegion_CoordinateParam['RefCoordinate']
                X_i = RefCoordinate[0]
                Y_i = RefCoordinate[1]
                Ft_i =  self.Ft_interp2d(X_i, Y_i)
                
                #Transforming Stresses to RefAngle wrt to CrackVector
                RefAngle_Local = CrackTipRegion_CoordinateParam['RefAngle_Local']
                R1, R2, R3 = RotationMatrices(theta = RefAngle_Local)
                StressTensor_i = np.real(np.dot(np.dot(R1.T, LocalStressTensor), R1)) 
                SigYY_i = StressTensor_i[1,1]
                
                TendencyIndicator_i = SigYY_i/Ft_i
                TendencyIndicatorList.append(TendencyIndicator_i)
            
            MaxTI = max(TendencyIndicatorList)
            
            I = TendencyIndicatorList.index(MaxTI)
            RefAngle_Local = CrackTipRegion_CoordinateParamList[I]['RefAngle_Local']
            CrackPropagationAngle = SubDomainObj.CrackAngle + RefAngle_Local
            print('Tensile Failure', RefAngle_Local*180/np.pi, CrackPropagationAngle*180/np.pi)
            
            SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
    
            return True
                    
                
        else:   return False
            

        
    
    
    def checkMaxPStressCriteria_HeteroMatBackup(self, SubDomainObj):
        
        Mean_Ft = SubDomainObj.MaterialObj.Ft
        
        #Calculating Crack-tip Tensile Strength
        Xc = SubDomainObj.ScalingCenterCoordinate[0]
        Yc = SubDomainObj.ScalingCenterCoordinate[1]
        CrackTip_Ft = self.Ft_interp2d(Xc, Yc)
        
        #Calculating Crack-tip P-Stress Parameters
        SubDomainObj.calcMaxPStressCriteriaParameters()
        CrackTip_PStressParam = SubDomainObj.CrackTip_PStressParam
        LocalSigXX, LocalSigYY, LocalTauXY = CrackTip_PStressParam['LocalStressList']
        PStress1 = CrackTip_PStressParam['PStressList'][0]
        PStress2 = CrackTip_PStressParam['PStressList'][1]
        
        LocalStressTensor = np.array([ [LocalSigXX,   LocalTauXY,      0],
                                       [LocalTauXY,   LocalSigYY,      0],
                                       [         0,            0,      0]], dtype = complex)
                
        
        if PStress2 > PStress1: raise Exception
        
        if PStress1 >= CrackTip_Ft:
            
            CrackTipRegion_CoordinateParamList = SubDomainObj.getCrackTipRegionCoordinateParameters(self.CrackIncrementLength)
            
            N_RefAngle = len(CrackTipRegion_CoordinateParamList)
            
            TendencyIndicatorList = [] #Yang 2008, Heterogenous Cohesive Crack model, CMAME
            
            for i in range(N_RefAngle):
                
                CrackTipRegion_CoordinateParam = CrackTipRegion_CoordinateParamList[i]
                
                RefCoordinate = CrackTipRegion_CoordinateParam['RefCoordinate']
                X_i = RefCoordinate[0]
                Y_i = RefCoordinate[1]
                Ft_i =  self.Ft_interp2d(X_i, Y_i)
                
#                dFt_i_da = (self.lch/Mean_Ft)*(Ft_i-CrackTip_Ft)/self.CrackIncrementLength
                
                
                #Transforming Stresses to RefAngle wrt to CrackVector
                RefAngle_Local = CrackTipRegion_CoordinateParam['RefAngle_Local']
                R1, R2, R3 = RotationMatrices(theta = RefAngle_Local)
                StressTensor_i = np.real(np.dot(np.dot(R1.T, LocalStressTensor), R1)) 
                SigYY_i = StressTensor_i[1,1]
                
#                TendencyIndicatorNew_i = (SigYY_i/Mean_Ft)*(1 - dFt_i_da)
#                TendencyIndicatorList.append(TendencyIndicatorNew_i)
                
                TendencyIndicator_i = SigYY_i/Ft_i
                TendencyIndicatorList.append(TendencyIndicator_i)
            
            
            I = TendencyIndicatorList.index(max(TendencyIndicatorList))
            RefAngle_Local = CrackTipRegion_CoordinateParamList[I]['RefAngle_Local']
            CrackPropagationAngle = SubDomainObj.CrackAngle + RefAngle_Local
            print('Tensile Failure', RefAngle_Local*180/np.pi, CrackPropagationAngle*180/np.pi)
            
            SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
    
            return True
                
                
        else:   return False
            

        
        
            
    
                   
    
    def checkMohrCoulombCriteria_Backup(self, SubDomainObj):
        
        #Calculating Mohr-Circle Parameters
        SubDomainObj.calcCrackTipMohrCircleParameters()
        
        #Reading Material Properties
        Ft = SubDomainObj.MaterialObj.Ft
#        C0 = SubDomainObj.MaterialObj.MaterialCohesion
#        U0 = SubDomainObj.MaterialObj.MaterialFrictionCoefficient
        
        
        #Searching Failure
        N_RefAngle = len(SubDomainObj.CrackTipRegion_MohrCircleParamList)
        for i in range(N_RefAngle):
            
            CrackTipRegion_MohrCircleParam_i = SubDomainObj.CrackTipRegion_MohrCircleParamList[i]
            
            MC_Center_i = CrackTipRegion_MohrCircleParam_i['Center']
            MC_Radius_i = CrackTipRegion_MohrCircleParam_i['Radius']
            
            #Calculating Tensile Failure
            PStress1_i = MC_Center_i + MC_Radius_i
            PStress2_i = MC_Center_i - MC_Radius_i
            
            if PStress2_i > PStress1_i: raise Exception
            
#            #LEFM
#            if PStress1_i >= Ft:    return True
            
            
            #CCM
            if PStress1_i >= Ft:
                        
                CrackTip_MohrCircleParam = SubDomainObj.CrackTip_MohrCircleParam
                MC_Center = CrackTip_MohrCircleParam['Center']
                LocalSigXX, LocalSigYY, LocalTauXY = CrackTip_MohrCircleParam['LocalStressList']
                
                # "Multiple cohesive crack growth in brittle materials by the extended Voronoi cell finite element model"
                # beta (in above paper) is assumed to be zero in calculating crack direction
                Tensile_FailureAngleList = [0.0]*4
                Tensile_FailureAngleList[0] = np.arctan2((-LocalSigXX + LocalSigYY + np.sqrt((LocalSigXX-LocalSigYY)**2 + 4*LocalTauXY**2)),(2*LocalTauXY))
                Tensile_FailureAngleList[1] = np.arctan2((-LocalSigXX + LocalSigYY - np.sqrt((LocalSigXX-LocalSigYY)**2 + 4*LocalTauXY**2)),(2*LocalTauXY))
#                Tensile_FailureAngleList[2] = np.arctan2((-2*LocalTauXY + np.sqrt((LocalSigXX-LocalSigYY)**2 + 4*LocalTauXY**2)),(-LocalSigXX + LocalSigYY))
#                Tensile_FailureAngleList[3] = np.arctan2((-2*LocalTauXY - np.sqrt((LocalSigXX-LocalSigYY)**2 + 4*LocalTauXY**2)),(-LocalSigXX + LocalSigYY))
#                
#                print('Tensile_FailureAngles', np.array(Tensile_FailureAngleList)*180/np.pi)
#                
#                for Tensile_FailureAngle in Tensile_FailureAngleList:
#                
#                    Cos2a = np.cos(2*Tensile_FailureAngle)
#                    Sin2a = np.sin(2*Tensile_FailureAngle)
#                    
#                    print('RefDir', -(-LocalSigXX*Cos2a - 2*LocalTauXY*Sin2a + LocalSigYY*Cos2a))
                
                
                
                Tensile_FailureAngle_0 = np.pi/2.0 + 0.5*np.arctan2(LocalTauXY, LocalSigXX - MC_Center)
                Tensile_FailureAngle_1 = -np.pi/2.0 + 0.5*np.arctan2(LocalTauXY, LocalSigXX - MC_Center)
                print('')
                print('Tensile_FailureAngles (MaxPrinci)', np.array([Tensile_FailureAngle_0, Tensile_FailureAngle_1])*180/np.pi)
                
                print('-------------------------')
#                
                
                for Tensile_FailureAngle in Tensile_FailureAngleList:
                
                    Cos2a = np.cos(2*Tensile_FailureAngle)
                    Sin2a = np.sin(2*Tensile_FailureAngle)
                    
                    if -(-LocalSigXX*Cos2a - 2*LocalTauXY*Sin2a + LocalSigYY*Cos2a) < 0:
                        
                        EquivalentFailureAngleList = [0.0]*3
                        EquivalentFailureAngleList[0] = Tensile_FailureAngle - np.pi
                        EquivalentFailureAngleList[1] = Tensile_FailureAngle
                        EquivalentFailureAngleList[2] = Tensile_FailureAngle + np.pi
                        
                        for EquivalentFailureAngle in EquivalentFailureAngleList:
                            
                            if abs(EquivalentFailureAngle) <= np.pi/2:
                                        
                                CrackPropagationAngle = SubDomainObj.CrackAngle + EquivalentFailureAngle
                                print('Tensile Failure', EquivalentFailureAngle*180/np.pi, CrackPropagationAngle*180/np.pi)
                                
                                SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
                        
                                return True
                        
                        else: raise Exception
                    
                else:   raise Exception
                
            
            
        else:   return False
            

            
            
    
        
            
    
    
    
    def checkGCriteria(self, SubDomainObj):
        
        SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
        
        
        if Alpha <= 1.0:    return True
        else:               return False
        
    
            




class LEFM_Solver(SolverBaseClass):
    
    
    def __init__(self, GuiMainObj = None):
        
        if GuiMainObj:  self.runTimeSteps(GuiMainObj)
                
        
    
    def runTimeSteps(self, GuiMainObj):
        
        CurrentAnalysis = GuiMainObj.CurrentAnalysis
        self.CurrentAnalysis = CurrentAnalysis
        self.CrackIncrementLength = CurrentAnalysis.CrackIncrementLength
        self.MaxTimeStepCount = CurrentAnalysis.MaxTimeStepCount
        self.SolverParameters = CurrentAnalysis.SolverParameters
        
        self.PlotCurve_LoadList = [0.0]*self.MaxTimeStepCount
        self.PlotCurve_DisplacementList = [0.0]*self.MaxTimeStepCount
        self.TimeStepCount = 0    
        self.CrackIncrementStepCount = -1
#        self.delta_lambda = self.SolverParameters['del_lambda_ini']
        
        while 1:
            
            self.CrackIncrementStepCount += 1
            self.TimeStepCount += 1
            
            CurrentAnalysis.OutputVTUFile = GuiMainObj.ResultDirectory + '/' + CurrentAnalysis.Name + '_' + str(self.CrackIncrementStepCount) + '.vtu'
            self.OutputVTUFile = CurrentAnalysis.OutputVTUFile
            
            if self.CrackIncrementStepCount == 0:   MeshRootObj = CurrentAnalysis.MeshRootObj
            else:                                   MeshRootObj = SBFEM.MeshGen.Root(GuiMainObj, self.CrackIncrementStepCount, UpdateProgressBar = False)
            
            self.execLinearSolver(MeshRootObj)
            
            if self.TimeStepCount >= self.MaxTimeStepCount:   break 
        
    
    
    
    def execLinearSolver(self, MeshRootObj):
        
        self.NodeObjList = MeshRootObj.NodeObjList
        self.NodeIdList = MeshRootObj.NodeIdList
        self.PolyEdgeObjList = MeshRootObj.PolyEdgeObjList
        self.PolyCellObjList = MeshRootObj.PolyCellObjList
        self.CrackCurveList = MeshRootObj.CrackCurveList
        self.RefLoad = MeshRootObj.RefLoad
        
        if MeshRootObj.RefLoadNodeObj:
                
            self.RefLoadNodeId = MeshRootObj.RefLoadNodeObj.Id
            if len(MeshRootObj.CrackCurveList) > 0:
                
                self.RefCrackEdgeNodeId0 = MeshRootObj.CrackCurveList[0]['NodeIdList'][0]
                RefCrackEdgeNodeObj0 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0]
                self.RefCrackEdgeNodeId1 = RefCrackEdgeNodeObj0.CrackPairNodeId

            
        self.evalDOFs()
        self.InterfaceElementObjList = []
        self.createSubDomains()
        
        self.initGlobalStiffnessMatrix()
        self.initGlobalRefLoadVector()
        self.initBoundaryConditionVector()
        
#        self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
        self.GlobalExternalLoadVector = self.GlobalRefTransientLoadVector
        
        self.applyBoundaryConditions(self.GlobalStiffnessMatrix)
        self.calcGlobalDisplacementVector()
        
        
        self.assignDeformationToNodes()
        self.assignDeformationToSubDomains()
        self.calcCriticalEquilibriumState()
        
        
        
#        #-----------------------------------------
#        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
#    
#            RefIntegrationConstantList = SubDomainObj.IntegrationConstantList
#            SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
#            SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
#        #-------------------------------------------
        
        
        
#        SBFEM.PostProcessor.FieldVariables(self)
        self.plotNodalDisplacement(DeformationScale = 1.0e2)
        self.updateLoadDisplacementCurve(GeometryType = 'SENS')
        
#        self.delta_lambda += self.SolverParameters['del_lambda']
        
        
        
    
    def testLinear(self):
        
        
        RefSubdomainObj = self.PartiallyCrackedSubDomainObjList[0]
        
        
        #----------------------------------------------------------------------
#        RefInterfaceCellObj = RefSubdomainObj.CrackTip_InterfaceCellObj
#        DirVector = RefInterfaceCellObj.NodeObjList[1].Coordinate - RefInterfaceCellObj.NodeObjList[0].Coordinate
#        Length, Angle, phi_garbage = Cartesian2Spherical(DirVector)
#        RefInterfaceCellObj.Length = Length
#        Le = RefInterfaceCellObj.Length
#        Thk = RefInterfaceCellObj.Thickness
#        RefInterfaceCell_SurfaceArea = Thk*Le
#        
#        #Calculating Sideface Traction Parameters
#        CrackTip_NormalTraction =   -100.0
#        CrackMouth_NormalTraction = -200.0
#        
#        TrS0 = 0.0
#        TrN0 = CrackTip_NormalTraction
#        TrS1 = 0.0
#        TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
#        TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
#        
#        RefSubdomainObj.calcSidefaceTractionParameters(TractionPolynomialCoefficientList, RefInterfaceCell_SurfaceArea)
        
        
        
        CrackTip_NormalTraction = 1000.0
        CrackMouth_NormalTraction = 1000.0
        P = np.array([CrackMouth_NormalTraction, CrackTip_NormalTraction], dtype =float)
        
        RefSubdomainObj.initUnitSidefaceTractionData(1)
        RefSubdomainObj.applyLinearSidefaceTraction(P)
        
        #----------------------------------------------------------------------
        
        
        #Updating Load Vector
        self.assembleTractionLoadVector(RefSubdomainObj.NodeIdList, RefSubdomainObj.SidefaceTractionLoadVector, self.GlobalRefLoadVector)
        
        
        #----------------------------------------------------------------------
        
        #Calc Disp.
#        self.applyBoundaryConditions(self.GlobalStiffnessMatrix, self.GlobalRefLoadVector)
#        self.calcGlobalDisplacementVector()
        
        
        #Applying Condensation
        ConNodeIdList = RefSubdomainObj.NodeIdList + self.CrackCurveList[0]['NodeIdList'][:-2] + self.CrackCurveList[0]['PairNodeIdList'][:-2]
        print('ConNodeIdList', ConNodeIdList)
        ConDofIdList = []
        for ConNodeId in ConNodeIdList: ConDofIdList += self.NodeObjList[ConNodeId].DOFIdList
        CondensedData = self.applyStaticCondensation(ConDofIdList)
        ConStiffMat = CondensedData['ConStiffMat']
        ConLoadVec = CondensedData['ConLoadVec']
        ConDispVec = np.dot(inv(ConStiffMat), ConLoadVec)
        CondensedData['ConDispVec'] = ConDispVec
        self.calcGlobDispVecPostCondensation(CondensedData)
        self.GlobalExternalLoadVector = self.GlobalRefLoadVector
        self.ReactionVector = np.dot(self.GlobalStiffnessMatrix, self.GlobalDisplacementVector) - self.GlobalExternalLoadVector
    
    
    
    
    
    
    def calcGlobalDisplacementVector(self):
        
        if self.BCApplyMode == 'ColumnModification':
                
            #Calculating Displacement
            self.GlobalDisplacementVector = np.dot(inv(self.GlobalStiffnessMatrix), self.GlobalExternalLoadVector)
            
            #Modifying Displacement vector to include Boundary Condition
            self.GlobalDisplacementVector[self.FilteredBCDOFList] = 0.0
            
            #Calc Reaction
            self.ReactionVector = np.dot(self.GlobalStiffnessMatrix, self.GlobalDisplacementVector) - self.GlobalExternalLoadVector
            
            
        
        elif self.BCApplyMode == 'MatrixSizeReduction':
            
            #Calculating Displacement
            self.GlobalDisplacementVector = np.zeros(self.N_DOF)
            
            Reduced_GlobDispVec = np.dot(inv(self.Reduced_GlobStiffMat), self.Reduced_GlobRefLoadVec)
            self.GlobalDisplacementVector[self.NonBCDOFList] = Reduced_GlobDispVec
            
            #Calc Reaction
            self.ReactionVector = np.dot(self.GlobalStiffnessMatrix, self.GlobalDisplacementVector) - self.GlobalExternalLoadVector
    
        
    
    def calcCriticalEquilibriumState(self):
        
        #Calculating Equilibrium State for Crack-Propagation (Alpha)
        AlphaList = []
        SIFList = []
        CrackPropagationAngleList = []
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
    
            RefIntegrationConstantList = SubDomainObj.IntegrationConstantList
            SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
            AlphaList.append(Alpha)
            SIFList.append(SIF)
            CrackPropagationAngleList.append(CrackPropagationAngle)
            
        if len(AlphaList) > 0:
                
            Alpha = min(AlphaList)
            I = AlphaList.index(Alpha)
            
#            if self.CrackIncrementStepCount == 0:   CrackPropagationAngle = 2.8
#            else:                                   CrackPropagationAngle = CrackPropagationAngleList[I]
            
            CrackPropagationAngle = CrackPropagationAngleList[I]
            
            
            self.PartiallyCrackedSubDomainObjList[I].calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
            
            
            
            #Updating Loadvec, Dispvec and Reaction for Equilibrium State
            self.GlobalExternalLoadVector = Alpha*self.GlobalExternalLoadVector
            self.GlobalDisplacementVector = Alpha*self.GlobalDisplacementVector
            self.ReactionVector =           Alpha*self.ReactionVector
            
            #Updating Deformation and element properties
            self.assignDeformationToNodes()
            self.assignDeformationToSubDomains()
            
    
   
    
    
    
    def calcCriticalEquilibriumState_BrazilianSplit(self):
        
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
    
            RefIntegrationConstantList = SubDomainObj.IntegrationConstantList
            SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
            SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
            
            
            
        #Updating Loadvec, Dispvec and Reaction for Equilibrium State
        self.GlobalExternalLoadVector = Alpha*self.GlobalExternalLoadVector
        self.GlobalDisplacementVector = Alpha*self.GlobalDisplacementVector
        self.ReactionVector =           Alpha*self.ReactionVector
        self.Alpha =                    Alpha
        
        #Updating Deformation and element properties
        self.assignDeformationToNodes()
        self.assignDeformationToSubDomains()
        
    
    
    


class CCM_Solver(NLSolverBaseClass):
    
    
    def __init__(self, GuiMainObj = None):
        
        self.SubDomainObjList = []
        self.InterfaceElementObjList = []
        self.GlobalDisplacementVector = []
        self.FirstNonLinearStepExecuted = False
        self.TimeStepCount = 0     
        self.CrackIncrementStepCount = -1
        self.FailedConvergenceCount = 0
        self.AnalysisFinished = False
        
        self.NormCPErrorList = []
        self.ModeI_SIFList = []
        self.ModeII_SIFList = []
        self.TempFDList = []
        
        
        if GuiMainObj:  
            
            GuiMainObj.CurrentSolverObj = self
            self.runCrackIncrementSteps(GuiMainObj)
    
    
    
    def runCrackIncrementSteps(self, GuiMainObj):
        
        CurrentAnalysis = GuiMainObj.CurrentAnalysis
        self.CurrentAnalysis = CurrentAnalysis
        self.CrackIncrementLength = CurrentAnalysis.CrackIncrementLength
        self.MaxTimeStepCount = CurrentAnalysis.MaxTimeStepCount
        self.MaxCrackIncrementStepCount = CurrentAnalysis.MaxCrackIncrementStepCount
        self.MaxFailedConvergenceCount = CurrentAnalysis.MaxFailedConvergenceCount
        self.SolverParameters = CurrentAnalysis.SolverParameters
        
        self.PlotCurve_LoadList = [0.0]*self.MaxTimeStepCount
        self.PlotCurve_DisplacementList = [0.0]*self.MaxTimeStepCount
        
        
        while 1:
            
            self.CrackIncrementStepCount += 1
            
            CurrentAnalysis.OutputVTUFile = GuiMainObj.ResultDirectory + '/' + CurrentAnalysis.Name + '_' + str(self.CrackIncrementStepCount) + '.vtu'
            self.OutputVTUFile = CurrentAnalysis.OutputVTUFile
            
            if self.CrackIncrementStepCount == 0:   MeshRootObj = CurrentAnalysis.MeshRootObj
            else:                                   MeshRootObj = SBFEM.MeshGen.Root(GuiMainObj, self.CrackIncrementStepCount, UpdateProgressBar = False)
                
            
            self.execNonLinearSolver(MeshRootObj)
                
            if self.AnalysisFinished:   break
        
        
        
        
    def execNonLinearSolver(self, MeshRootObj):
        
        self.MeshRootObj = MeshRootObj
        
        self.mapMesh(MeshRootObj)
        
        self.NodeObjList = list(MeshRootObj.NodeObjList)
        self.NodeIdList = list(MeshRootObj.NodeIdList)
        self.PolyEdgeObjList = list(MeshRootObj.PolyEdgeObjList)
        self.PolyCellObjList = list(MeshRootObj.PolyCellObjList)
        self.CrackCurveList = MeshRootObj.CrackCurveList
        self.InterfaceSegmentObjList = MeshRootObj.InterfaceSegmentObjList
        self.N_GaussLobattoPoints = MeshRootObj.N_GaussLobattoPoints
        self.HasMaterialHeterogeneity = MeshRootObj.HasMaterialHeterogeneity
        
        if self.HasMaterialHeterogeneity:
            
            self.Ft_interp2d = MeshRootObj.Ft_interp2d
            self.lch = MeshRootObj.RandomFieldData['lch']
        
        
        
        self.RefLoad = MeshRootObj.RefLoad
        self.RefLoadNodeObj = MeshRootObj.RefLoadNodeObj
        self.RefLoadNodeObj0 = MeshRootObj.RefLoadNodeObj0
        self.RefLoadNodeObj1 = MeshRootObj.RefLoadNodeObj1
        
        
        self.RefCrackEdgeNodeId0 = MeshRootObj.CrackCurveList[0]['NodeIdList'][0]
        RefCrackEdgeNodeObj0 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0]
        self.RefCrackEdgeNodeId1 = RefCrackEdgeNodeObj0.CrackPairNodeId


        self.evalDOFs()
        self.createSubDomains()
        self.initGlobalStiffnessMatrix()
        self.initGlobalRefLoadVector()
        self.initBoundaryConditionVector()
        
        if self.CrackIncrementStepCount == 0:
            
            self.finishCrackIncrementStep()
            
        else:
            
            self.createInterfaceElements()
            self.initLoadStep()
            self.runLocalDuanSolver()
            
        
        
    
    def finishCrackIncrementStep(self):
        
        
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:  
            
            #Reading Avg Material Strength
#            Mean_Ft = SubDomainObj.MaterialObj.Ft
               
            #Calculating Crack-tip Tensile Strength
            Xc = SubDomainObj.ScalingCenterCoordinate[0]
            Yc = SubDomainObj.ScalingCenterCoordinate[1]
#            CrackTip_Ft =   self.Ft_interp2d(Xc, Yc)
            
            if self.HasMaterialHeterogeneity:
                
                KI = 0.0
                KII = -1.0e6
                    
                CrackTipRegion_CoordinateParamList = SubDomainObj.getCrackTipRegionCoordinateParameters(self.CrackIncrementLength)
                
                N_RefAngle = len(CrackTipRegion_CoordinateParamList)
                
                TendencyIndicatorList = [] #Yang 2008, Heterogenous Cohesive Crack model, CMAME
                KiList = []
                FtiList = []
                LocalRefAngleList = []
                
                for i in range(N_RefAngle):
                    
                    CrackTipRegion_CoordinateParam_i = CrackTipRegion_CoordinateParamList[i]
                    
                    RefCoordinate = CrackTipRegion_CoordinateParam_i['RefCoordinate']
                    X_i = RefCoordinate[0]
                    Y_i = RefCoordinate[1]
                    Ft_i =  self.Ft_interp2d(X_i, Y_i)
                    FtiList.append(Ft_i)
                    
#                    dFt_i_da = (self.lch/Mean_Ft)*(Ft_i-CrackTip_Ft)/self.CrackIncrementLength
                    
                    RefAngle_Local_i = CrackTipRegion_CoordinateParam_i['RefAngle_Local']
                    LocalRefAngleList.append(RefAngle_Local_i)
                    
                    K_i = (KI*np.cos(RefAngle_Local_i/2.)**2 - 1.5*KII*np.sin(RefAngle_Local_i))*np.cos(RefAngle_Local_i/2.)
                    KiList.append(K_i)
                    
                    TendencyIndicator_i = K_i/Ft_i
                    TendencyIndicatorList.append(TendencyIndicator_i)
                    
#                    TendencyIndicatorNew_i = (K_i/Mean_Ft)*(1 - dFt_i_da)
#                    TendencyIndicatorList.append(TendencyIndicatorNew_i)
                
                    
                LocalRefAngleList = list(np.array(LocalRefAngleList)*180.0/np.pi)
                FtiList = list(np.array(FtiList)/1e6)
                KiList = list(np.array(KiList)/1e6)
                
                
                io = FtiList.index(max(FtiList))
                plt.plot(LocalRefAngleList, FtiList)
                plt.plot([LocalRefAngleList[io]], [FtiList[io]], 'o')
                plt.show()
                
                io = KiList.index(max(KiList))
                plt.plot(LocalRefAngleList, KiList)
                plt.plot([LocalRefAngleList[io]], [KiList[io]], 'o')
                plt.show()
                
                io = TendencyIndicatorList.index(max(TendencyIndicatorList))
                plt.plot(LocalRefAngleList, TendencyIndicatorList)
                plt.plot([LocalRefAngleList[io]], [TendencyIndicatorList[io]], 'o')
                plt.show()
                
                
                
                I = TendencyIndicatorList.index(max(TendencyIndicatorList))
                RefAngle_Local = CrackTipRegion_CoordinateParamList[I]['RefAngle_Local']
                
                CrackPropagationAngle = SubDomainObj.CrackAngle + RefAngle_Local
                SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
        
            
            
            else:
                
                CrackPropagationAngle = 2.7
                SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
                
    
    
    
    
    def finishCrackIncrementStep_0(self):
        
        #Initializing LEFM Solver
        LEFMSolverObj = LEFM_Solver()
        LEFMSolverObj.CrackIncrementLength = self.CrackIncrementLength 
        LEFMSolverObj.NodeObjList = self.NodeObjList
        LEFMSolverObj.NodeIdList = self.NodeIdList
        LEFMSolverObj.N_DOFPerNode = self.N_DOFPerNode
        LEFMSolverObj.N_Node = self.N_Node
        LEFMSolverObj.N_DOF = self.N_DOF
        LEFMSolverObj.SubDomainObjList = self.SubDomainObjList
        LEFMSolverObj.PartiallyCrackedSubDomainObjList = self.PartiallyCrackedSubDomainObjList
        LEFMSolverObj.InterfaceElementObjList = []
        LEFMSolverObj.GlobalStiffnessMatrix = self.GlobalStiffnessMatrix
        LEFMSolverObj.GlobalExternalLoadVector = self.GlobalRefTransientLoadVector
        LEFMSolverObj.FilteredBCDOFList = self.FilteredBCDOFList
        
        #Running LInear Analysis
        LEFMSolverObj.applyBoundaryConditions(LEFMSolverObj.GlobalStiffnessMatrix)
        LEFMSolverObj.calcGlobalDisplacementVector()
        
        #Calculating Next CrackTip Location
        for SubDomainObj in LEFMSolverObj.PartiallyCrackedSubDomainObjList:  
            
            SubDomain_GlobDispVector = LEFMSolverObj.GlobalDisplacementVector[SubDomainObj.DOFIdList]
            RefIntegrationConstantList = np.dot(inv(SubDomainObj.EigVec_DispModeData), SubDomain_GlobDispVector)
            SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
        
            SubDomainObj.calcNewCrackTipCoordinate(LEFMSolverObj.CrackIncrementLength, CrackPropagationAngle)
                
            

    
    
    

class Contact_LEFM_Solver(NLSolverBaseClass):
    
    
    def __init__(self, GuiMainObj = None):
        
        #Initializing Variables
        self.SubDomainObjList = []
        self.InterfaceElementObjList = []
        self.GlobalDisplacementVector = []
        self.FirstNonLinearStepExecuted = False
        self.TimeStepCount = 0     
        self.CrackIncrementStepCount = -1
        self.AnalysisFinished = False
        
        self.NormCPErrorList = []
        self.ModeI_SIFList = []
        self.ModeII_SIFList = []
        self.TempFDList = []
        self.CrackTipCoordinateList = []
        self.CrackPropagationAngleList = []
        
        
        self.SIFList = []
        
        #Running Analysis
        if GuiMainObj:  self.runCrackIncrementSteps(GuiMainObj)
            
    
    
    def runCrackIncrementSteps(self, GuiMainObj):
        
        self.CurrentAnalysis = GuiMainObj.CurrentAnalysis
        self.CrackIncrementLength = self.CurrentAnalysis.CrackIncrementLength
        self.MaxTimeStepCount = self.CurrentAnalysis.MaxTimeStepCount
        self.SolverParameters = self.CurrentAnalysis.SolverParameters
        
        self.PlotCurve_LoadList = [0.0]*(self.MaxTimeStepCount+1)
        self.PlotCurve_DisplacementList = [0.0]*(self.MaxTimeStepCount+1)
        
        while 1:
            
            self.CrackIncrementStepCount += 1
            
            self.CurrentAnalysis.OutputVTUFile = GuiMainObj.ResultDirectory + '/' + self.CurrentAnalysis.Name + '_' + str(self.CrackIncrementStepCount) + '.vtu'
            self.OutputVTUFile = self.CurrentAnalysis.OutputVTUFile
            
            if self.CrackIncrementStepCount == 0:   MeshRootObj = self.CurrentAnalysis.MeshRootObj
            else:                                   MeshRootObj = SBFEM.MeshGen.Root(GuiMainObj, self.CrackIncrementStepCount, UpdateProgressBar = False)
            
            self.execNonLinearSolver(MeshRootObj)
                
            if self.AnalysisFinished:   break
        
    
    
    def execNonLinearSolver(self, MeshRootObj):
        
                
        self.NodeObjList = list(MeshRootObj.NodeObjList)
        self.NodeIdList = list(MeshRootObj.NodeIdList)
        self.PolyEdgeObjList = list(MeshRootObj.PolyEdgeObjList)
        self.PolyCellObjList = list(MeshRootObj.PolyCellObjList)
        self.CrackCurveList = MeshRootObj.CrackCurveList
        self.InterfaceSegmentObjList = MeshRootObj.InterfaceSegmentObjList
        
        self.N_GaussLobattoPoints = MeshRootObj.N_GaussLobattoPoints
        self.RefLoadNodeId = MeshRootObj.RefLoadNodeObj.Id
        self.RefCrackEdgeNodeId0 = self.CrackCurveList[0]['NodeIdList'][0]
        self.RefCrackEdgeNodeId1 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0].CrackPairNodeId
        self.RefLineLoad = MeshRootObj.RefLineLoad
        
        self.evalDOFs()
        self.createSubDomains()
        self.initGlobalStiffnessMatrix()
        self.initGlobalRefLoadVector()
        self.initBoundaryConditionVector()
            
        self.createInterfaceElements()
        self.removeCohesiveElements()
        
        self.resetLoadStep()
        self.runLocalDuanSolver()
            
        self.finishCrackIncrementStep()
        
    
    
    
    def finishCrackIncrementStep(self):
        
        #Saving Crack-Tip coordinateList
        RefCrackCurve = self.CrackCurveList[0]
        self.CrackTipCoordinateList.append(RefCrackCurve['CrackTip_Coordinate'])
        CrackPropagationAngleList_i = []
        
        #Calculating Next CrackTip Location
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:  
            
            if self.CrackIncrementStepCount == 0:   SubDomainObj.HasCrackPropagation = True
            
            if SubDomainObj.HasCrackPropagation:
                
                if self.CrackIncrementStepCount == 0:
                                        
                    CrackPropagationAngle = SubDomainObj.CrackAngle + 70.52*np.pi/180.0
#                    SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
                    SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
                    
                else:
                        
                    SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
                    SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
                
                CrackPropagationAngleList_i.append(CrackPropagationAngle)
                  
        self.CrackPropagationAngleList.append(CrackPropagationAngleList_i)
        
        
        print('CrackTipX', [CrackTipCoordinate[0] for CrackTipCoordinate in self.CrackTipCoordinateList])
        print('CrackTipY', [CrackTipCoordinate[1] for CrackTipCoordinate in self.CrackTipCoordinateList])

        print('CrackPropagationAngleList', self.CrackPropagationAngleList)



    
    

class Contact_CCM_Solver(NLSolverBaseClass):
    
    
    def __init__(self, GuiMainObj = None):
        
        if GuiMainObj:  self.runCrackIncrementSteps(GuiMainObj)
    
    
    
    def runCrackIncrementSteps(self, GuiMainObj):
        
        CurrentAnalysis = GuiMainObj.CurrentAnalysis
        self.CurrentAnalysis = CurrentAnalysis
        self.CrackIncrementLength = CurrentAnalysis.CrackIncrementLength
        self.MaxTimeStepCount = CurrentAnalysis.MaxTimeStepCount
        self.SolverParameters = CurrentAnalysis.SolverParameters
        
        self.SubDomainObjList = []
        self.InterfaceElementObjList = []
        self.GlobalDisplacementVector = []
        self.PlotCurve_LoadList = [0.0]*self.MaxTimeStepCount
        self.PlotCurve_DisplacementList = [0.0]*self.MaxTimeStepCount
        self.FirstNonLinearStepExecuted = False
        self.TimeStepCount = 0     
        self.CrackIncrementStepCount = -1
        self.AnalysisFinished = False
        
        
        self.NormCPErrorList = []
        self.ModeI_SIFList = []
        self.ModeII_SIFList = []
        self.TempFDList = []
        self.CrackTipCoordinateList0 = []
        self.CrackTipCoordinateList1 = []
        
        while 1:
            
            self.CrackIncrementStepCount += 1
            
            CurrentAnalysis.OutputVTUFile = GuiMainObj.ResultDirectory + '/' + CurrentAnalysis.Name + '_' + str(self.CrackIncrementStepCount) + '.vtu'
            self.OutputVTUFile = CurrentAnalysis.OutputVTUFile
            
            if self.CrackIncrementStepCount == 0:   MeshRootObj = CurrentAnalysis.MeshRootObj
            else:                                   MeshRootObj = SBFEM.MeshGen.Root(GuiMainObj, self.CrackIncrementStepCount, UpdateProgressBar = False)
                
            
            self.execNonLinearSolver(MeshRootObj)
                
            if self.AnalysisFinished:   break

        
        
        
        
        
    def execNonLinearSolver(self, MeshRootObj):
        
        self.mapMesh(MeshRootObj)
        
        self.NodeObjList = list(MeshRootObj.NodeObjList)
        self.NodeIdList = list(MeshRootObj.NodeIdList)
        self.PolyEdgeObjList = list(MeshRootObj.PolyEdgeObjList)
        self.PolyCellObjList = list(MeshRootObj.PolyCellObjList)
        self.CrackCurveList = MeshRootObj.CrackCurveList
        self.InterfaceSegmentObjList = MeshRootObj.InterfaceSegmentObjList
        self.RefLineLoad = MeshRootObj.RefLineLoad
        
        self.N_GaussLobattoPoints = MeshRootObj.N_GaussLobattoPoints
#        self.RefLoadNodeId = MeshRootObj.RefLoadNodeObj.Id
            
#        self.RefCrackEdgeNodeId0 = self.CrackCurveList[0]['NodeIdList'][0]
#        self.RefCrackEdgeNodeId1 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0].CrackPairNodeId
        
        if self.CrackIncrementStepCount == 0:   
            
            self.RefCrackEdgeNodeId0 = self.CrackCurveList[0]['NodeIdList'][-1]
            self.RefCrackEdgeNodeId1 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0].CrackPairNodeId
        
        elif self.CrackIncrementStepCount == 1:   self.RefCrackEdgeNodeId1 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0].CrackPairNodeId
        
        
        
        self.evalDOFs()
        self.createSubDomains()
        self.initGlobalStiffnessMatrix()
        self.initGlobalRefLoadVector()
        self.initBoundaryConditionVector()
        
#        if self.CrackIncrementStepCount == 0:
#            
#            for SubDomainObj in self.PartiallyCrackedSubDomainObjList:  
#            
#                SubDomainObj.HasCrackPropagation = True
#        
#        else:
#                    
        self.createInterfaceElements()         
        self.initLoadStep()
        self.runLocalDuanSolver()
                
        self.finishCrackIncrementStep()
        
    
    
    
    def finishCrackIncrementStep(self):
        
        
        #Saving Crack-Tip coordinateList
        RefCrackCurve0 = self.CrackCurveList[0]
        self.CrackTipCoordinateList0.append(RefCrackCurve0['CrackTip_Coordinate'])
        
        RefCrackCurve1 = self.CrackCurveList[1]
        self.CrackTipCoordinateList1.append(RefCrackCurve1['CrackTip_Coordinate'])
        
#        #Calculated for 0.0045 mm Crack increment length
#        if not self.CrackIncrementLength == 0.0045:  raise Exception
#        CrackPropagationAngleList_18 = [[1.5449654538653768, -1.59662719972441], [1.8062517621634997, -1.3353407873739396], [1.5071486651865558, -1.6344437660490578], [1.7364513812658775, -1.4051409087127782], [1.445673242069367, -1.6959189154747252], [1.7176019454958955, -1.423990240076114], [1.4480699442248812, -1.6935221040682338], [1.7113519784305686, -1.4302399219596613], [1.4582541897839305, -1.683337763735705], [1.7157940075373914, -1.4257978299585112], [1.4589384204613598, -1.6826536455513952], [1.7098771283314798, -1.4317146801614624], [1.4595977072821866, -1.681994421401784], [1.7055181827481571, -1.4360736717101341], [1.4665354362119976, -1.6750579276013733], [1.7064893164892871, -1.4351036841075944], [1.4627491120430354, -1.678844096315415], [1.7028795082325485, -1.438712833842459]]
#        CrackPropagationAngleList_36 = [[1.859124719224361, -1.2824679343654333], [2.0382301619433343, -1.1033623598333475], [1.8034409748257272, -1.338151398695483], [1.6212330485859938, -1.5203592499448177], [1.8200861731832372, -1.3215061314728656], [1.6234233070825013, -1.5181690573829096], [1.7982067031073505, -1.3433858221094708], [1.602099974365643, -1.5394923322661904], [1.7889000249899993, -1.3526922827593688], [1.580590447149604, -1.56100151241633], [1.7442021567960646, -1.397391456203685], [1.5650148774972943, -1.5765783943566758], [1.7633247448084532, -1.378268159622461], [1.6038695019550693, -1.5377211102066421], [1.7796812804373272, -1.3619092101374501], [1.5946704227222033, -1.546921718057643]]
#        CrackPropagationAngleList_54 = [[2.173283984583339, -0.9683086690064544], [2.018088471828554, -1.1235034422797812], [1.8534834580943267, -1.288110357245144], [1.8483522921697868, -1.293237289536997], [1.6515117802509554, -1.4900788069583557], [1.8537336188055502, -1.2878580151869177], [1.634803634641887, -1.5067889014980236], [1.8054114484293566, -1.3361798568687169], [1.5894147010393482, -1.5521789225173637], [1.786746186681417, -1.3548482050189985], [1.5802601515200385, -1.5613339474538968], [1.795282102094656, -1.3463112146511036], [1.5701719106955219, -1.571420681698126], [1.777024837528748, -1.3645672696806406], [1.55270343443725, -1.588889874777685], [1.7848442667879905, -1.3567479581270647]]
#        
#        
#        
#        q = -1
#        for RefSubDomainObj in self.PartiallyCrackedSubDomainObjList:
#            
#            q += 1
#            CrackPropagationAngle = CrackPropagationAngleList_36[self.CrackIncrementStepCount][q]
#                
#            RefSubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)


        print('CrackTipX0', [CrackTipCoordinate0[0] for CrackTipCoordinate0 in self.CrackTipCoordinateList0])
        print('CrackTipY0', [CrackTipCoordinate0[1] for CrackTipCoordinate0 in self.CrackTipCoordinateList0])
        
        print('CrackTipX1', [CrackTipCoordinate1[0] for CrackTipCoordinate1 in self.CrackTipCoordinateList1])
        print('CrackTipY1', [CrackTipCoordinate1[1] for CrackTipCoordinate1 in self.CrackTipCoordinateList1])
        
    




class CCM_Solver_Backup(SolverBaseClass):
    
    
    def __init__(self, GuiMainObj = None):
        
        if GuiMainObj:  self.runCrackIncrementSteps(GuiMainObj)
        
    
    
    def runCrackIncrementSteps(self, GuiMainObj):
        
        CurrentAnalysis = GuiMainObj.CurrentAnalysis
        self.CurrentAnalysis = CurrentAnalysis
        self.CrackIncrementLength = CurrentAnalysis.CrackIncrementLength
        self.MaxTimeStepCount = CurrentAnalysis.MaxTimeStepCount
        self.SolverParameters = CurrentAnalysis.SolverParameters
        
        self.SubDomainObjList = []
        self.InterfaceElementObjList = []
        self.delta_lambda = 0.0
        self.GlobalDisplacementVector = []
        self.PlotCurve_LoadList = [0.0]*self.MaxTimeStepCount
        self.PlotCurve_DisplacementList = [0.0]*self.MaxTimeStepCount
        self.FirstNonLinearStepExecuted = False
        self.TimeStepCount = 0     
        self.CrackIncrementStepCount = -1
        self.AnalysisFinished = False
        
        
        self.NormCPErrorList = []
        self.ModeI_SIFList = []
        self.ModeII_SIFList = []
        
        self.TempFDList = []
        
        
        while 1:
            
            self.CrackIncrementStepCount += 1
            
            CurrentAnalysis.OutputVTUFile = GuiMainObj.ResultDirectory + '/' + CurrentAnalysis.Name + '_' + str(self.CrackIncrementStepCount) + '.vtu'
            self.OutputVTUFile = CurrentAnalysis.OutputVTUFile
            
            if self.CrackIncrementStepCount == 0:   MeshRootObj = CurrentAnalysis.MeshRootObj
            else:                                   MeshRootObj = SBFEM.MeshGen.Root(GuiMainObj, self.CrackIncrementStepCount, UpdateProgressBar = False)
            
            self.execNonLinearSolver(MeshRootObj)
                
            if self.AnalysisFinished:   break
        
    
    def execNonLinearSolver(self, MeshRootObj):
        
        self.mapMesh(MeshRootObj)
        
        self.NodeObjList = list(MeshRootObj.NodeObjList)
        self.NodeIdList = list(MeshRootObj.NodeIdList)
        self.PolyEdgeObjList = list(MeshRootObj.PolyEdgeObjList)
        self.PolyCellObjList = list(MeshRootObj.PolyCellObjList)
        self.CrackCurveList = MeshRootObj.CrackCurveList
        self.InterfaceSegmentObjList = MeshRootObj.InterfaceSegmentObjList
        
        self.N_GaussLobattoPoints = MeshRootObj.N_GaussLobattoPoints
        self.RefLoadNodeId = MeshRootObj.RefLoadNodeObj.Id
        self.RefCrackEdgeNodeId0 = self.CrackCurveList[0]['NodeIdList'][0]
        self.RefCrackEdgeNodeId1 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0].CrackPairNodeId
        
        
        self.evalDOFs()
        self.createSubDomains()
        self.initGlobalStiffnessMatrix()
        self.initGlobalRefLoadVector()
        self.initBoundaryConditionVector()
            
        if self.CrackIncrementStepCount >= 1:
            
            self.createInterfaceElements()
            self.initNonLinearSolverVariables()
            self.runLocalDuanSolver()
            
        self.finishCrackIncrementStep()
        
    
    
    
    def runCrackIncrementSteps_1(self, GuiMainObj):
        
        CurrentAnalysis = GuiMainObj.CurrentAnalysis
        self.CurrentAnalysis = CurrentAnalysis
        self.CrackIncrementLength = CurrentAnalysis.CrackIncrementLength
        self.MaxTimeStepCount = CurrentAnalysis.MaxTimeStepCount
        self.SolverParameters = CurrentAnalysis.SolverParameters
        
        self.SubDomainObjList = []
        self.InterfaceElementObjList = []
        self.delta_lambda = 0.0
        self.GlobalDisplacementVector = []
        self.PlotCurve_LoadList = [0.0]*self.MaxTimeStepCount
        self.PlotCurve_DisplacementList = [0.0]*self.MaxTimeStepCount
        self.FirstNonLinearStepExecuted = False
        self.TimeStepCount = 0
        self.CrackIncrementStepCount = -1
        
        self.NormCPErrorList = []
        self.ModeI_SIFList = []
        self.ModeII_SIFList = []
        
        self.TempFDList = []
        
        
        while 1:
                
            self.CrackIncrementStepCount += 1
            
            if self.TimeStepCount >= self.MaxTimeStepCount:   break 
            
            CurrentAnalysis.OutputVTUFile = GuiMainObj.ResultDirectory + '/' + CurrentAnalysis.Name + '_' + str(self.CrackIncrementStepCount) + '.vtu'
            self.OutputVTUFile = CurrentAnalysis.OutputVTUFile
            
            if self.CrackIncrementStepCount == 0:   MeshRootObj = CurrentAnalysis.MeshRootObj
            else:                                   MeshRootObj = SBFEM.MeshGen.Root(GuiMainObj, self.CrackIncrementStepCount, UpdateProgressBar = False)
            
            self.execNonLinearSolver_1(MeshRootObj)
    
    
    
    
    def execNonLinearSolver_1(self, MeshRootObj):
        """
        For Penalty Stiffness approach in Contact
        """
        
        self.mapMesh(MeshRootObj)
        
        self.NodeObjList = list(MeshRootObj.NodeObjList)
        self.NodeIdList = list(MeshRootObj.NodeIdList)
        self.PolyEdgeObjList = list(MeshRootObj.PolyEdgeObjList)
        self.PolyCellObjList = list(MeshRootObj.PolyCellObjList)
        self.CrackCurveList = MeshRootObj.CrackCurveList
        self.InterfaceSegmentObjList = MeshRootObj.InterfaceSegmentObjList
        
        self.N_GaussLobattoPoints = MeshRootObj.N_GaussLobattoPoints
        self.RefLoadNodeId = MeshRootObj.RefLoadNodeObj.Id
#        self.RefCrackEdgeNodeId0 = self.CrackCurveList[0]['NodeIdList'][0]
#        self.RefCrackEdgeNodeId1 = MeshRootObj.NodeObjList[self.RefCrackEdgeNodeId0].CrackPairNodeId
#        self.RefLineLoad = MeshRootObj.RefLineLoad
        
        self.evalDOFs()
        self.createSubDomains()
        self.initGlobalStiffnessMatrix()
        self.initGlobalRefLoadVector()
        self.initBoundaryConditionVector()
        
        if self.CrackIncrementStepCount >= 1 or MeshRootObj.HasInitWeakInterface:
                
            self.createInterfaceElements()
            self.initNonLinearSolverVariables()
            self.runLocalDuanSolver_1()
            
        self.finishCrackIncrementStep() #<--Change to finishCrackIncrementStep_1
        
        
    
    def mapMesh(self, MeshRootObj):
        
        MappedNodeIdList = []
        
        if self.FirstNonLinearStepExecuted: #Mapping is after non-linear analysis starts
            
            #Mapping Nodal Deformation of Old mesh into New mesh
            for NewNodeObj in MeshRootObj.NodeObjList:
                
                if NewNodeObj.Enabled and NewNodeObj.RequiresMeshMapping:
                    
                    MappedNodeIdList.append(NewNodeObj.Id)
                    #Mapping Displacement of SubDomain of Old Mesh in which NewNodeObj lies
                    #TODO: Reduce the number of SubDomains by considering only those which were subdivided in last remesh.
                    for SubDomainObj in self.SubDomainObjList:
                        
                        RefDisp = SubDomainObj.getFieldDisplacement(NewNodeObj.Coordinate)
                        
                        if len(RefDisp) > 0:
                            
                            NewNodeObj.DeformedCoordinate = NewNodeObj.Coordinate + RefDisp
                            break
                    
                    else:   raise Exception
                    
#            print('mappedNodes', MappedNodeIdList)
        
        
        #Resetting MeshMapping Flag for Nodes
        for NodeObj in MeshRootObj.NodeObjList:    NodeObj.RequiresMeshMapping = False
        
        
  
    
    
    def initNonLinearSolverVariables(self):
        
        #Initializing Global Displacement Vector
        self.GlobalDisplacementVector =     np.zeros(self.N_DOF)
        
        for NodeObj in  self.NodeObjList:
            
            for j in range(self.N_DOFPerNode):
                
                self.GlobalDisplacementVector[NodeObj.DOFIdList[j]] = NodeObj.DeformedCoordinate[j] - NodeObj.Coordinate[j]
        
        
        #Initializing Stiffness Matrices of Interface Elements
        self.assignDeformationToNodes()
        self.updateElementProperties()
        self.updateSidefaceTractionParameters()
#        self.updateWaterPressureOnCrackCurve()
        
#        self.updateGlobalTangentStiffnessMatrix(UpdateInverse = True)
        self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
        
        
    
     
    
    
#    
#    def updateSidefaceTractionParameters(self):
#        
#        self.GlobalSidefaceTractionLoadVector = np.zeros(self.N_DOF)
#            
#        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
#            
#            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
#            RefDOFIdList = RefInterfaceElementObj.DOFIdList
#            RefInterfaceElementSurfaceArea = RefInterfaceElementObj.SurfaceArea
#            
#            
#            RefInterfaceElementObj.calcCOD()
#            RefInterfaceElementObj.calcCohesiveStiffnessMatrix()
#            
#            #Calculating Displacement Vector for Interface Element
#            RefGlobDispVector = self.GlobalDisplacementVector[RefDOFIdList]
#            ScalingCenterDisplacementVector = SubDomainObj.getFieldDisplacement(SubDomainObj.ScalingCenterCoordinate)
#            RefGlobDispVector[2] = RefGlobDispVector[4] = ScalingCenterDisplacementVector[0]
#            RefGlobDispVector[3] = RefGlobDispVector[5] = ScalingCenterDisplacementVector[1]
#            RefInterfaceElementObj.GlobDispVector = RefGlobDispVector
#            
#            RefLoadVector = np.dot(RefInterfaceElementObj.SecantStiffnessMatrix, RefGlobDispVector)
#            RefLocalLoadVector = np.dot(RefInterfaceElementObj.TransformationMatrix, RefLoadVector)
#            
#            Fx0 = RefLocalLoadVector[0]
#            Fy0 = RefLocalLoadVector[1]
#            Fx1 = RefLocalLoadVector[2]
#            Fy1 = RefLocalLoadVector[3]
#            
#            
##            print('SF', SubDomainObj.Id, Fx0, Fy0, Fx1, Fy1)
##            Fy0 = Fy1 = Fx0 = Fx1 = 0.0
#            
##                print('SF', SubDomainObj.Id, Fx0, Fy0, Fx1, Fy1)
#            
#            CrackMouth_ShearTraction = -( 4*Fx0 - 2*Fx1)/RefInterfaceElementSurfaceArea
#            CrackTip_ShearTraction = -(-2*Fx0 + 4*Fx1)/RefInterfaceElementSurfaceArea
#            
#            CrackMouth_NormalTraction = -( 4*Fy0 - 2*Fy1)/RefInterfaceElementSurfaceArea
#            CrackTip_NormalTraction = -(-2*Fy0 + 4*Fy1)/RefInterfaceElementSurfaceArea
##            print('CrackMouth_NormalTraction', CrackMouth_NormalTraction)
##            print('CrackTip_NormalTraction', CrackTip_NormalTraction)
#            
#            
#            RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction = CrackMouth_NormalTraction
#            RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction = CrackMouth_ShearTraction
#            RefInterfaceElementObj.ModifiedCrackTip_NormalTraction = CrackTip_NormalTraction
#            RefInterfaceElementObj.ModifiedCrackTip_ShearTraction = CrackTip_ShearTraction
#                    
#                    
#                    
#            TrS0 = CrackTip_ShearTraction
#            TrN0 = CrackTip_NormalTraction
#            TrS1 = CrackMouth_ShearTraction - CrackTip_ShearTraction
#            TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
#            
#            
#            TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
#            
#            #Calculating Sideface Traction Parameters
#            SubDomainObj.calcSidefaceTractionParameters_1(TractionPolynomialCoefficientList, RefInterfaceElementSurfaceArea)
#                         
#            #Updating Load Vector
#            self.assembleTractionLoadVector(SubDomainObj.NodeIdList, SubDomainObj.SidefaceTractionLoadVector, self.GlobalSidefaceTractionLoadVector)
#            
#    
    
    
    def updateContactConditions(self):
        
        #Updating Interface Element 
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            InterfaceElementObj.updateContactCondition()
            
                
        #Updating Cracktip subdomain
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            SubDomainObj.CrackTip_InterfaceElementObj.updateContactCondition()
            
    
    
    
    
    def updateSidefaceTractionParameters(self):
        
        self.GlobalSidefaceTractionLoadVector = np.zeros(self.N_DOF)
            
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            if RefInterfaceElementObj.InterfaceType in ['CONTACT', 'COHESIVE']:
                
                #Updating Material Properties
                RefInterfaceElementObj.updateMaterialProperties()
                
                
                #Calculating Displacement Vector for Interface Element
                RefDOFIdList = RefInterfaceElementObj.DOFIdList
                RefGlobDispVector = self.GlobalDisplacementVector[RefDOFIdList]
                
                ScalingCenterDisplacementVector = SubDomainObj.getFieldDisplacement(SubDomainObj.ScalingCenterCoordinate)
                RefGlobDispVector[2] = RefGlobDispVector[4] = ScalingCenterDisplacementVector[0]
                RefGlobDispVector[3] = RefGlobDispVector[5] = ScalingCenterDisplacementVector[1]
                RefInterfaceElementObj.GlobDispVector = RefGlobDispVector
                
                RefLoadVector = np.dot(RefInterfaceElementObj.SecantStiffnessMatrix, RefGlobDispVector)
                RefLocalLoadVector = np.dot(RefInterfaceElementObj.TransformationMatrix, RefLoadVector)
                
                Fx0 = RefLocalLoadVector[0]
                Fy0 = RefLocalLoadVector[1]
                Fx1 = RefLocalLoadVector[2]
                Fy1 = RefLocalLoadVector[3]
                
                RefInterfaceElementSurfaceArea = RefInterfaceElementObj.SurfaceArea
            
                CrackMouth_ShearTraction = -( 4*Fx0 - 2*Fx1)/RefInterfaceElementSurfaceArea
                CrackTip_ShearTraction = -(-2*Fx0 + 4*Fx1)/RefInterfaceElementSurfaceArea
                CrackMouth_NormalTraction = -( 4*Fy0 - 2*Fy1)/RefInterfaceElementSurfaceArea
                CrackTip_NormalTraction = -(-2*Fy0 + 4*Fy1)/RefInterfaceElementSurfaceArea
                
                #Modifying Sideface Tractions
                alpha_n0 = 2.0
                alpha_s0 = alpha_n0
                alpha_n1 = 0.2
                alpha_s1 = alpha_n1
                
                SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
                KI = SIF[0]
                KII = SIF[1]
                
#                print('KI', KI)
#                print('KII', KII)
#                print(100*KI/KII)
                
                #Updating Crack Mouth Pressure
                ModifiedCrackMouth_NormalTraction = RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction
                dTr_CrackMouth = alpha_n1*(CrackMouth_NormalTraction - ModifiedCrackMouth_NormalTraction)
                ModifiedCrackMouth_NormalTraction += dTr_CrackMouth
                CrackMouth_NormalTraction = ModifiedCrackMouth_NormalTraction
                RefInterfaceElementObj.ModifiedCrackMouth_NormalTraction = CrackMouth_NormalTraction
#                    
                ModifiedCrackMouth_ShearTraction = RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction
                dTr_CrackMouth = alpha_s1*(CrackMouth_ShearTraction - ModifiedCrackMouth_ShearTraction)
                ModifiedCrackMouth_ShearTraction += dTr_CrackMouth
                CrackMouth_ShearTraction = ModifiedCrackMouth_ShearTraction
                RefInterfaceElementObj.ModifiedCrackMouth_ShearTraction = CrackMouth_ShearTraction
                
                
                #Updating Crack-Tip Pressure                    
                self.ModeI_SIFList.append(KI)
                dTrN_CrackTip = alpha_n0*KI
                ModifiedCrackTip_NormalTraction_tmp = RefInterfaceElementObj.ModifiedCrackTip_NormalTraction + dTrN_CrackTip
                
                self.ModeII_SIFList.append(KII)
                dTrS_CrackTip = alpha_s0*KII
                ModifiedCrackTip_ShearTraction_tmp = RefInterfaceElementObj.ModifiedCrackTip_ShearTraction + dTrS_CrackTip
                
                
                if RefInterfaceElementObj.InterfaceType == 'COHESIVE':
                    
                    CrackTip_NormalTraction = min([ModifiedCrackTip_NormalTraction_tmp, RefInterfaceElementObj.MaterialObj.Ft])
                    CrackTip_ShearTraction = min([ModifiedCrackTip_ShearTraction_tmp, RefInterfaceElementObj.MaterialObj.Fs])
                        
                    
                elif RefInterfaceElementObj.InterfaceType == 'CONTACT':
                    
                    CrackTip_NormalTraction = ModifiedCrackTip_NormalTraction_tmp
                    CrackTip_ShearTraction = ModifiedCrackTip_ShearTraction_tmp
                        
                    if not RefInterfaceElementObj.ContactConditionList[-1] == 'STICK':       raise Exception
                    
                    
                RefInterfaceElementObj.ModifiedCrackTip_NormalTraction = CrackTip_NormalTraction
                RefInterfaceElementObj.ModifiedCrackTip_ShearTraction = CrackTip_ShearTraction
                    
                
#                print('CrackMouth_ShearTraction', CrackMouth_ShearTraction)
#                print('CrackMouth_NormalTraction', CrackMouth_NormalTraction)
#                print('CrackTip_ShearTraction', CrackTip_ShearTraction)
#                print('CrackTip_NormalTraction', CrackTip_NormalTraction)
#                print('---')
                
                TrS0 = CrackTip_ShearTraction
                TrN0 = CrackTip_NormalTraction
                TrS1 = CrackMouth_ShearTraction - CrackTip_ShearTraction
                TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
                
                
                TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
                
                #Calculating Sideface Traction Parameters
                SubDomainObj.calcSidefaceTractionParameters_1(TractionPolynomialCoefficientList, RefInterfaceElementSurfaceArea)
                
                
                #Updating Load Vector
                self.assembleTractionLoadVector(SubDomainObj.NodeIdList, SubDomainObj.SidefaceTractionLoadVector, self.GlobalSidefaceTractionLoadVector)
                
        
    
    
    
    
    def updateContactPressureError(self):
        
        RefContactPressure = np.array([])
        RefContactShear = np.array([])
        
        CPErrorList = []
        if len(self.CrackCurveList) > 1:    raise NotImplementedError
        CrackCurve = self.CrackCurveList[0]
    
        RefInterfaceElementObjList = CrackCurve['InterfaceElementObjList'] + [CrackCurve['CrackTip_InterfaceElementObj']]
            
        N_Elem = len(RefInterfaceElementObjList)
        for i in range(N_Elem):
            
            InterfaceElementObj = RefInterfaceElementObjList[i]
            
            LinearCohesiveTractionVector = InterfaceElementObj.calcLinearTractionVector()
            
            ContactShear = LinearCohesiveTractionVector[0]
            CPError_s = ContactShear - RefContactShear[i]
            CPErrorList.append(CPError_s)
            
            ContactPressure = LinearCohesiveTractionVector[1]
            CPError_p = ContactPressure - RefContactPressure[i]
            CPErrorList.append(CPError_p)
            
            ContactShear = LinearCohesiveTractionVector[0]
             
            if i == N_Elem-1:
                
                ContactShear = LinearCohesiveTractionVector[2]
                CPError_s = ContactShear - RefContactShear[i+1]
                CPErrorList.append(CPError_s)
                
                ContactPressure = LinearCohesiveTractionVector[3]                
                CPError_p = ContactPressure - RefContactPressure[i+1]
                CPErrorList.append(CPError_p)
                
        
        NormCPError = norm(CPErrorList)/norm(np.append(RefContactPressure, RefContactShear))
        self.NormCPErrorList.append(NormCPError)
        
        
                
     
    def displayContactTraction(self):
        
        XCoord_List = []
        YCoord_List = []
        CST_List = []
        CNT_List = []
        
        if len(self.CrackCurveList) > 1:    raise NotImplementedError
        CrackCurve = self.CrackCurveList[0]
    
        RefInterfaceElementObjList = CrackCurve['InterfaceElementObjList'] + [CrackCurve['CrackTip_InterfaceElementObj']]
            
        N_Elem = len(RefInterfaceElementObjList)
        for i in range(N_Elem):
            
            InterfaceElementObj = RefInterfaceElementObjList[i]
            
            #COORDINATES
            XCoord = InterfaceElementObj.NodeObjList[0].Coordinate[0]
            YCoord = InterfaceElementObj.NodeObjList[0].Coordinate[1]
            XCoord_List.append(XCoord)
            YCoord_List.append(YCoord)
            
            #TRACTION
            LinearCohesiveTractionVector = InterfaceElementObj.calcLinearTractionVector()
            
            ContactShear = LinearCohesiveTractionVector[0]
            CST_List.append(ContactShear)
            
            ContactPressure = LinearCohesiveTractionVector[1]
            CNT_List.append(ContactPressure)
            
            #Crack-Tip Tractions
            if i == N_Elem-1:
                
                #COORDINATES
                XCoord = InterfaceElementObj.NodeObjList[1].Coordinate[0]
                YCoord = InterfaceElementObj.NodeObjList[1].Coordinate[1]
                XCoord_List.append(XCoord)
                YCoord_List.append(YCoord)
                
                #TRACTION
                ContactShear = LinearCohesiveTractionVector[2]
                CST_List.append(ContactShear)
                
                ContactPressure = LinearCohesiveTractionVector[3]
                CNT_List.append(ContactPressure)
                
            
#            
#        
#        self.CST_List = CST_List
#        self.CNT_List = CNT_List
#       
        print('----Contact Tractions------')
        print('XCoord', XCoord_List)
        print('YCoord', YCoord_List)
        print('CST', CST_List)
        print('CNT', CNT_List)
        print('---------------------------')
        
        
    
    def updateWaterPressureOnCrackCurve(self):
        
        self.GlobalWaterPressureLoadVector = np.zeros(self.N_DOF)
        
        for CrackCurve in self.CrackCurveList:
            
#            CrackMouth_WaterPressure = CrackCurve['CrackMouth_WaterPressure']
            CrackMouth_WaterPressure = 0.0
        
            for InterfaceElementObj in CrackCurve['InterfaceElementObjList']:
                
                InterfaceElementObj.calcWaterPressureLoadVector_Empirical(CrackMouth_WaterPressure)
                
                #Updating Load Vector
                self.assembleTractionLoadVector(InterfaceElementObj.NodeIdList, InterfaceElementObj.WaterPressureLoadVector, self.GlobalWaterPressureLoadVector)
            
        
            #Calculating Water Pressure Near Crack Tip (The following code does not apply Water Pressure as side-face traction and is for showing plots only. It is assumed that Water does not reach the crack-tip)
            CrackCurve['CrackTip_InterfaceElementObj'].calcWaterPressureLoadVector_Empirical(CrackMouth_WaterPressure)
            
        
        for InterfaceSegmentObj in self.InterfaceSegmentObjList:
            
            InterfaceMouth_WaterPressure = 0.0
            
            for InterfaceElementObj in InterfaceSegmentObj.InterfaceElementObjList:
                
                InterfaceElementObj.calcWaterPressureLoadVector_Empirical(InterfaceMouth_WaterPressure)
                
                #Updating Load Vector
                self.assembleTractionLoadVector(InterfaceElementObj.NodeIdList, InterfaceElementObj.WaterPressureLoadVector, self.GlobalWaterPressureLoadVector)
            
            
            
            
            
    
    def updateGlobalTangentStiffnessMatrix(self, UpdateInverse = False):
    
        #Updating Global Stiffness matrix with deformed interface elements
        self.GlobalTangentStiffnessMatrix = np.array(self.GlobalStiffnessMatrix, dtype = float)
        
            
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            #Calculating Global Tangent Stiffness Matrix
            self.assembleStiffnessMatrix(InterfaceElementObj.NodeIdList, InterfaceElementObj.TangentStiffnessMatrix, self.GlobalTangentStiffnessMatrix)
        
        
        self.applyBoundaryConditions(self.GlobalTangentStiffnessMatrix)
        
            
        if UpdateInverse: self.InvGlobalTangentStiffnessMatrix = inv(self.GlobalTangentStiffnessMatrix)
        
    
    
    
    
    
    def updateGlobalSecantStiffnessMatrix(self, UpdateInverse = False):
    
        #Updating Global Stiffness matrix with deformed interface elements
        self.GlobalSecantStiffnessMatrix = np.array(self.GlobalStiffnessMatrix, dtype = float)
        
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            if not InterfaceElementObj.InterfaceType == 'FREE':
                
                #Calculating Global Secant Stiffness Matrix
                self.assembleStiffnessMatrix(InterfaceElementObj.NodeIdList, InterfaceElementObj.SecantStiffnessMatrix, self.GlobalSecantStiffnessMatrix)
            
        self.applyBoundaryConditions(self.GlobalSecantStiffnessMatrix)
            
        if UpdateInverse: self.InvGlobalSecantStiffnessMatrix = inv(self.GlobalSecantStiffnessMatrix)
        
    
    
    
    
    def calcLocalDeformationVector(self, RefDispVector):
        
        Relative_RefDispVector = []
        
        #Calculating Relative Displcement in Interface Elements
        for InterfaceElementObj in self.InterfaceElementObjList:
            
            if not InterfaceElementObj.InterfaceType == 'FREE':
                        
                XDOFId0 = InterfaceElementObj.DOFIdList[0]
                XDOFId1 = InterfaceElementObj.DOFIdList[2]
                XDOFId2 = InterfaceElementObj.DOFIdList[4]
                XDOFId3 = InterfaceElementObj.DOFIdList[6]
                
                YDOFId0 = InterfaceElementObj.DOFIdList[1]
                YDOFId1 = InterfaceElementObj.DOFIdList[3]
                YDOFId2 = InterfaceElementObj.DOFIdList[5]
                YDOFId3 = InterfaceElementObj.DOFIdList[7]
                
                Relative_RefDispX0 = RefDispVector[XDOFId3] - RefDispVector[XDOFId0]
                Relative_RefDispY0 = RefDispVector[YDOFId3] - RefDispVector[YDOFId0]
                
                Relative_RefDispX1 = RefDispVector[XDOFId2] - RefDispVector[XDOFId1]
                Relative_RefDispY1 = RefDispVector[YDOFId2] - RefDispVector[YDOFId1]
                
                
                Relative_RefDispVector += [Relative_RefDispX0, Relative_RefDispY0, Relative_RefDispX1, Relative_RefDispY1]
                
        
        #Calculating Relative Displcement of CrackMouth in Partially Cracked SubDomain
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:
            
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            if not RefInterfaceElementObj.InterfaceType == 'FREE':
                    
                CrackMouth_XDOFId0 = SubDomainObj.CrackMouth_DOFIdList[0]
                CrackMouth_XDOFId1 = SubDomainObj.CrackMouth_DOFIdList[2]
                
                CrackMouth_YDOFId0 = SubDomainObj.CrackMouth_DOFIdList[1]
                CrackMouth_YDOFId1 = SubDomainObj.CrackMouth_DOFIdList[3]
                
                Relative_RefDispX = RefDispVector[CrackMouth_XDOFId0] - RefDispVector[CrackMouth_XDOFId1]
                Relative_RefDispY = RefDispVector[CrackMouth_YDOFId0] - RefDispVector[CrackMouth_YDOFId1]
                Relative_RefDispVector += [Relative_RefDispX, Relative_RefDispY, 0.0, 0.0]
            
        
        return np.array(Relative_RefDispVector, dtype = float)
    
    
    
#    
#    def runLoadControlled_ModifiedNewtonSolver(self):
#        
#        #Parameters
#        RTol =              5e2
#        MaxIterCount =      100
#        MaxLoadStepCount =  20
#        delta_lambda =      1.0/(MaxLoadStepCount)        
#        
#        #Initializing variables
#        self.GlobalDisplacementVector =     np.zeros(self.N_DOF)
#        GlobalExternalLoadVector =          np.zeros(self.N_DOF)
#        RefExternalLoadVector =             self.GlobalLoadVector
#        R =                                 np.zeros(self.N_DOF)
#        NormR =                             0.0
#        
#        LoadList = []
#        DisplacementList = []
#        
#        self.assignDeformationToNodes()
#        self.updateElementStiffnessMatrix()
#        
#        for i in range(MaxLoadStepCount):
#            
#            print('')
#            print('-------------------------------')
#            print('')
#            
#            IterCount =     0
#            
#            #Updating Tangent Stiffness and External Load Vector (this is the equilibrium position at ith loadstep)
#            self.updateGlobalTangentStiffnessMatrix(CalcInverse = True)
#            delta_ExternalLoadVector = delta_lambda*RefExternalLoadVector #Does not include the nodal reactions
#            
#            #Calculating Displacement due to load increment
#            delta_u = np.dot(self.InvGlobalTangentStiffnessMatrix, delta_ExternalLoadVector)
#            delta_u[self.FilteredBCDOFList] = 0.0
#            
#            #Calculating GlobalExternalLoadVector (includes the reactions)
#            GlobalExternalLoadVector += np.dot(self.GlobalTangentStiffnessMatrix, delta_u)
#            
#            #Assigning Deformation
#            self.GlobalDisplacementVector += delta_u
#            self.assignDeformationToNodes()
#            
#            #Calculating Secant Stiffness at new Disp Vector
#            self.updateElementStiffnessMatrix()
#            self.updateGlobalSecantStiffnessMatrix()
#            self.updateGlobalSidefaceTractionLoadVector()
#            
#            #Calculating Residue at new Disp Vector
#            GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#            NetExternalLoadVector = GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector #Includes the effect of SidefaceTraction
#            R = GlobalInternalLoadVector - NetExternalLoadVector
#            NormR = norm(R)
#                
#            while NormR > RTol: #Iteration steps
#                            
#                if IterCount > MaxIterCount:    break
#                
#                else:                     
#                    
#                    #Calculating Displacement due to load increment
#                    delta_u = -1*np.dot(self.InvGlobalTangentStiffnessMatrix, R)
#                    delta_u[self.FilteredBCDOFList] = 0.0
#                
#                #Assigning Deformation
#                self.GlobalDisplacementVector += delta_u
#                self.assignDeformationToNodes()
#                
#                #Calculating Secant Stiffness at new Disp Vector
#                self.updateElementStiffnessMatrix()
#                self.updateGlobalSecantStiffnessMatrix()
#                self.updateGlobalSidefaceTractionLoadVector()
#                
#                #Calculating Residue at new Disp Vector
#                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#                NetExternalLoadVector = GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector #Includes the effect of SidefaceTraction
#                R = GlobalInternalLoadVector - NetExternalLoadVector
#                NormR = norm(R)
#                
#                
#                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#    #            print('')
#                IterCount += 1
#            
#            
#            if IterCount > MaxIterCount:  
#                    
#                print 'Convergence cannot achieved within', MaxIterCount, 'iterations'
##                break
#                
#            else:
#                        
#                UpperNodeIdList = [79, 78, 80, 81, 82, 83, 84]
#                UpperYDOFIdList = [2*i+1 for i in UpperNodeIdList]
#                InterfaceElementLength = 2.75
#            
#                LoadList.append(sum([GlobalExternalLoadVector[YDOF] for YDOF in UpperYDOFIdList])/InterfaceElementLength)
#                DisplacementList.append(self.GlobalDisplacementVector[193] - self.GlobalDisplacementVector[191])
#                
##        plt.plot(InterfaceElement['TSL']['NormalCODList'], np.array(InterfaceElement['TSL']['NormalTractionList'])/1e6, label='Bilinear TSL')
#        plt.plot(np.array(DisplacementList)*1e3, np.array(LoadList)/1e6,  label='Newtons')
#        plt.xlabel('$\delta$ (mm)')
#        plt.ylabel('$\sigma$ (MPa)')
##        plt.legend()
##        plt.xlim(0, 70)
##        plt.ylim(0, 21)
#    #        plt.axis('equal')
#        
#        print 'The program completed successfully'
#    
#    
    
#    
#    
#    def runDisplacementControlled_ModifiedNewtonSolver(self): 
#        
#            
#        #Parameters
#        RTol =              5e2
#        MaxIterCount =      100
#        MaxDispStepCount =  40
#                
#        
#        #Initializing variables
#        self.GlobalDisplacementVector =     np.zeros(self.N_DOF)
#        GlobalExternalLoadVector =          np.zeros(self.N_DOF)
#        RefExternalLoadVector =             self.GlobalLoadVector
#        R =                                 np.zeros(self.N_DOF)
#        NormR =                             0.0
#            
#            
#        DispControlledDOF =                 79*2+1
#        DispControlVector =                 np.zeros(self.N_DOF)
#        DispControlVector[DispControlledDOF] = 1
#        MaxDisp =                           0.015
#        dDisp =                             MaxDisp/MaxDispStepCount
#        
#        
#        LoadList = []
#        DisplacementList = []
#        
#        for i in range(MaxDispStepCount-1):
#            
#            print('')
#            print('-------------------------------')
#            print('')
#            
#             
#            IterCount =     0
#            self.assignDeformationToNodes()
#            self.updateElementStiffnessMatrix()
#            self.updateGlobalTangentStiffnessMatrix(CalcInverse = True)
#            
#            self.GlobalDisplacementVector += dDisp*DispControlVector
#            
#               
#            
#            while IterCount < MaxIterCount: #Iteration steps
#                            
#                if IterCount > 0 and NormR < RTol:  
#                    
#                    break
#                
#                else:
#                        
#                        
#                    del_ua = -np.dot(self.InvGlobalTangentStiffnessMatrix, R)
#                    del_ub = np.dot(self.InvGlobalTangentStiffnessMatrix, RefExternalLoadVector)
#                    
#                    del_lambda = -del_ua[DispControlledDOF]/del_ub[DispControlledDOF]
#                    
#                    del_u = del_ua + del_lambda*del_ub
#                    
#                    del_u[DispControlledDOF] = 0.0
#                    del_u[self.FilteredBCDOFList] = 0.0
#                
#                
#                self.GlobalDisplacementVector += del_u
#                GlobalExternalLoadVector += del_lambda*RefExternalLoadVector
#                
#                self.assignDeformationToNodes()
#                self.updateElementStiffnessMatrix()
#                self.updateGlobalSecantStiffnessMatrix()
#                self.updateGlobalSidefaceTractionLoadVector()
#                
#                #Incorporating Cohesive SidefaceTraction LoadVector
#                NetExternalLoadVector = GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector
#        
#                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#                
#                if i == 0 and IterCount == 1: print(norm(GlobalInternalLoadVector))
#                
#                R = GlobalInternalLoadVector - NetExternalLoadVector
#                
#                NormR = norm(R)
#                if i == 0 and IterCount == 0: print(NormR)
#                
##                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#    #            print('')
#                IterCount += 1
#                
#                
#                
#            else:
#            
#                print 'Convergence cannot achieved within', MaxIterCount, 'iterations'
##                break
#                
#            
#            
#            UpperNodeIdList = [79, 78, 80, 81, 82, 83, 84]
#            UpperYDOFIdList = [2*i+1 for i in UpperNodeIdList]
#            InterfaceElementLength = 2.75
#        
#            LoadList.append(sum([GlobalExternalLoadVector[YDOF] for YDOF in UpperYDOFIdList])/InterfaceElementLength)
#            DisplacementList.append(self.GlobalDisplacementVector[193] - self.GlobalDisplacementVector[191])
#            
##            LoadList.append(abs(GlobalExternalLoadVector[61]))
##            DisplacementList.append(self.GlobalDisplacementVector[61])
#            
#            
##            
#        plt.plot(np.array(DisplacementList)*1e3, np.array(LoadList)/1e6, '.',)
#        plt.xlabel('$\delta$ (mm)')
#        plt.ylabel('$\sigma$ (MPa)')
#        
##        plt.legend()
#        plt.xlim(0, MaxDisp*1e3)
#        plt.ylim(0, 2.5)
#        
#        print 'The program completed successfully'
#        
#    
#
#    
#    
#    def runCrisfieldSolver(self): 
#                
#        
#        def solveQuadtraticArcLengthEqn(delta_l, delta_u, del_uf, del_up, delta_lambda):
#        
##            du = delta_u + del_uf
#            
#            # Calculate the coefficients of the polynomial
##            c1 = np.dot(del_up, del_up)
##            c2 = 2*(np.dot(du, del_up))
##            c3 = np.dot(du, du) - delta_l*delta_l
#            
#            c1 = np.dot(del_up, del_up)
#            c2 = 2*np.dot(del_uf, del_up) + 2*np.dot(delta_u, del_up)
#            c3 = 2*np.dot(delta_u, del_uf) + np.dot(del_uf, del_uf) + np.dot(delta_u, delta_u) - delta_l*delta_l
#                        
#            ArcLenRoots = np.real(np.roots([c1, c2, c3]))
#            del_lambda1 = ArcLenRoots[0]
#            del_lambda2 = ArcLenRoots[1]
#            
#            return del_lambda1, del_lambda2
#            
#                    
#        def checkForwardDirection(IterCount, GlobalTangentStiffnessMatrix, delta_u, delta_u0, del_u1, del_u2, delta_lambda, del_lambda1, del_lambda2):
#        
#            if IterCount == 0:
#                            
#                detKt = np.linalg.det(GlobalTangentStiffnessMatrix)
#        
#                if np.sign(delta_lambda + del_lambda1) == np.sign(detKt):
#                    
#                    del_u = del_u1
#                    del_lambda = del_lambda1
#                
#                else:
#                    
#                    del_u = del_u2
#                    del_lambda = del_lambda2
#                    
#            else:
#                
#                dot1 = np.dot(delta_u + del_u1, delta_u0)
#                dot2 = np.dot(delta_u + del_u2, delta_u0)
#                
#                if dot1 > dot2:
#                    
#                    del_u = del_u1
#                    del_lambda = del_lambda1
#                    
#                else:
#                    
#                    del_u = del_u2
#                    del_lambda = del_lambda2
#                
#                
#                if del_lambda1 == del_lambda2:
#                    
#                    del_u = del_u1
#                    del_lambda = del_lambda1
#        
#            return del_u, del_lambda
#        
#        
#        
#        #Arc Length Parameters
#        RTol =              1e3
#        MaxArcStepCount =   120
#        Nd =                20
#        m =                 0.5
#        MaxIterCount =      1000
#                
#               
#        
#        #Initializing variables
#        self.GlobalDisplacementVector =     np.zeros(self.N_DOF)
#        GlobalExternalLoadVector =          np.zeros(self.N_DOF)
#        RefExternalLoadVector =             self.GlobalLoadVector
#        delta_u0 =                          np.zeros(self.N_DOF)     
#        R =                                 np.zeros(self.N_DOF)
#        NormR =                             0.0
#        
#        
#        self.assignDeformationToNodes()
#        self.updateElementStiffnessMatrix()
#        self.updateGlobalTangentStiffnessMatrix(UpdateInverse = True)
#                        
#        
#        LoadList = []
#        DisplacementList = []
#        
#        for i in range(MaxArcStepCount):
#            
#            print('')
#            print('-------------------------------')
#            print('')
#            
#            delta_l =       1e-8
#        
#            IterCount =     0
#            delta_u =       np.zeros(self.N_DOF)
#            delta_lambda =  0.0
#            
##            NormR0 =        0.0
#            
#            while IterCount < MaxIterCount: #Iteration steps
#                
#                delta_l = delta_l*(float(Nd)/(IterCount+1))**m
#                  
#                if IterCount > 0 and NormR < RTol:  
#                    
#                    break
#                
#                else:
#                        
#                    del_uf = -np.dot(self.InvGlobalTangentStiffnessMatrix, R)
#                    del_up = np.dot(self.InvGlobalTangentStiffnessMatrix, RefExternalLoadVector)
#                    
#                    Local_del_uf = self.calcLocalDeformationVector(del_uf)
#                    Local_del_up = self.calcLocalDeformationVector(del_up)
#                    Local_delta_u = self.calcLocalDeformationVector(delta_u)
#                    
#                    del_lambda1, del_lambda2 = solveQuadtraticArcLengthEqn(delta_l, Local_delta_u, Local_del_uf, Local_del_up, delta_lambda)
#                    
##                    print('dli', del_lambda1, del_lambda2)
#                    del_u1 = del_uf + del_lambda1*del_up
#                    del_u2 = del_uf + del_lambda2*del_up
#                    del_u1[self.FilteredBCDOFList] = 0.0
#                    del_u2[self.FilteredBCDOFList] = 0.0
#                    
#                                        
#                    
#                    #Selecting the point on circle in correct direction
#                    del_u, del_lambda = checkForwardDirection(IterCount, self.GlobalTangentStiffnessMatrix, delta_u, delta_u0, del_u1, del_u2, delta_lambda, del_lambda1, del_lambda2)
#                    
#                
#                #Updating increments
#                delta_u +=  del_u
#                delta_lambda += del_lambda
#                
#    #            print('dl', del_lambda)
#                
#                self.GlobalDisplacementVector += del_u
#                GlobalExternalLoadVector += del_lambda*RefExternalLoadVector
#                
#                self.assignDeformationToNodes()
#                self.updateElementStiffnessMatrix()
#                self.updateGlobalTangentStiffnessMatrix(UpdateInverse = True)
#                self.updateGlobalSecantStiffnessMatrix()
#                self.updateGlobalSidefaceTractionLoadVector()
#                           
#                #Incorporating Cohesive SidefaceTraction LoadVector
#                NetExternalLoadVector = GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector
#        
#                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#                
#    #            print('GDispVec', GlobalDispVector)
#                                 
##                print('GDispVec', self.GlobalDisplacementVector)
#    #            print('GIntLoad', GlobalInternalLoadVector)
#                R = GlobalInternalLoadVector - NetExternalLoadVector
#                
#                
#                NormR = norm(R)
#                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#    #            print('')
#                IterCount += 1
#            
#            
#            else:
#            
#                print 'Convergence cannot achieved within', MaxIterCount, 'iterations'
#                break
#                
#            
#            #Saving increments in last iteration step
#            delta_u0 = np.array(delta_u, dtype = float)
#            
#            UpperNodeIdList = [NodeObj.Id for NodeObj in self.NodeObjList if abs(NodeObj.Coordinate[1]-2) < 1e-10]
#            UpperYDOFIdList = [2*ii+1 for ii in UpperNodeIdList]
#            InterfaceElementLength = 3.0
#            
##            print(UpperNodeIdList)
#            LoadList.append(sum([GlobalExternalLoadVector[YDOF] for YDOF in UpperYDOFIdList])/InterfaceElementLength)
##            DisplacementList.append(self.GlobalDisplacementVector[77] - self.GlobalDisplacementVector[75])
#            DisplacementList.append(self.GlobalDisplacementVector[2*UpperNodeIdList[0]+1])
#            
#            print(i)
#            self.calcDeformedPlotCoordinate(5e3)
#            self.plotNodalDisplacement()
#            
#        plt.plot(np.array(DisplacementList)*1e3, np.array(LoadList)/1e6, '-',  label='Local Arc Length')
#        plt.xlabel('$\delta$ (mm)')
#        plt.ylabel('$\sigma$ (MPa)')
#        plt.show()
#        print 'The program completed successfully'
#        

    
    
    def runLocalDuanSolver(self): 
        
        #Arc Length Parameters
        del_lambda_ini =    self.SolverParameters['del_lambda_ini']
        MaxIterCount =      self.SolverParameters['MaxIterCount']
        RelTol =            self.SolverParameters['RelTol']
        m =                 self.SolverParameters['m']
        Nd0 =               self.SolverParameters['Nd0']
        n =                 self.SolverParameters['n']
        p =                 self.SolverParameters['p']
        Nint =              len(self.InterfaceElementObjList)
        Nd =                Nint**n + Nd0
        
        
        #Updating GlobalRefTransientLoadVector to include reactions        
        RefTransientDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
        RefTransientDisplacementVector[self.FilteredBCDOFList] = 0.0
        self.GlobalRefTransientLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, RefTransientDisplacementVector)
    
        #Initializing variables
        self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
        NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector      
        GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
        R = GlobalInternalLoadVector - NetExternalLoadVector
        NormR = norm(R)
#        LastNormR = NormR
        
        
        if not self.FirstNonLinearStepExecuted:
            
            self.FirstNonLinearStepExecuted = True
            
            del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
            Local_del_up = self.calcLocalDeformationVector(del_up)
            del_lambda = del_lambda_ini
            self.delta_l = del_lambda*norm(Local_del_up)
            
            self.LastIterCount = Nd
        
        
        #Looping over Timesteps
        while self.TimeStepCount < self.MaxTimeStepCount:
            
            self.TimeStepCount += 1
            
            print('')
            print('-------------------------------')
            print('TimeStepCount', self.TimeStepCount)
            
            
            IterCount =         0
            self.delta_u =      np.zeros(self.N_DOF)
            self.delta_l =      self.delta_l*(float(Nd)/self.LastIterCount)**m
            
            print('dl', self.delta_l, self.LastIterCount)
            
            
            
            while IterCount < MaxIterCount:
                
                
                del_uf = -np.dot(self.InvGlobalSecantStiffnessMatrix, R)
                del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
                
                Local_del_uf = self.calcLocalDeformationVector(del_uf)
                Local_del_up = self.calcLocalDeformationVector(del_up)
                Local_delta_u = self.calcLocalDeformationVector(self.delta_u)
                
                   
                if IterCount == 0:  
                    
                    
                    del_lambda = self.delta_l/norm(Local_del_up)
                    del_lambda0 = del_lambda
                    
                else:
                    
#                    del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
#                    del_lambda = (self.delta_l**2 - np.dot(Local_delta_u, Local_delta_u + Local_del_uf))/np.dot(Local_delta_u, Local_del_up)
                    del_lambda = -np.dot(Local_delta_u, Local_del_uf)/np.dot(Local_delta_u, Local_del_up)
                    
                
                del_u = del_uf + del_lambda*del_up
                del_u[self.FilteredBCDOFList] = 0.0
                
#                    print('dl', del_lambda)
#                    print('du', del_u)
                
                
                #Updating increments
                self.delta_u +=  del_u
                self.delta_lambda += del_lambda
                
#                print('delta_l', self.delta_l)
                
                
                self.GlobalDisplacementVector += del_u
#                self.GlobalExternalLoadVector = self.GlobalStaticLoadVector + self.delta_lambda*self.GlobalRefTransientLoadVector
                self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
                
                #TODO: Bisection
#                self.checkBisection(IterCount, MaxIterCount)
                
                
                self.assignDeformationToNodes()
                self.updateElementProperties()
                self.updateSidefaceTractionParameters()
            
#                self.updateGlobalTangentStiffnessMatrix(UpdateInverse = True)
                self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
                
                NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector
                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
                
                R = GlobalInternalLoadVector - NetExternalLoadVector
                NormR = norm(R)
                RelNormR = NormR/norm(NetExternalLoadVector)
                
#                if IterCount > 0 and NormR > LastNormR:   break
#                LastNormR = NormR
                    
#                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#                print 'RelNormR', RelNormR
            
                if RelNormR < RelTol:    
                    
                    self.LastIterCount = IterCount + 1
                    break
                
                else:   IterCount += 1
                
                
            
            else:
            
                print 'Convergence was not achieved;  RelNormR = ', RelNormR
                self.LastIterCount = Nd*p
                
            
            #Plotting LoadDisplcement Curve
            self.updateLoadDisplacementCurve(GeometryType = 'test_coh', Plot = False)     
#            SBFEM.PostProcessor.FieldVariables(self)
            
#            print('CPError', self.NormCPErrorList)
#            print('SIF_I', self.ModeI_SIFList)
#            print('SIF_II', self.ModeII_SIFList)
            
#            self.displayContactTraction()
            
            #Checking if Crack Propagates
            if self.checkCrackPropagation() == True: 
                
                self.updateLoadDisplacementCurve(GeometryType = 'test_coh')   
            
#                self.TempFDList.append([self.CrackIncrementStepCount, self.PlotCurve_DisplacementList[-1], self.PlotCurve_LoadList[-1]])
                self.plotNodalDisplacement(DeformationScale = 2e2)
             
#                print('DL' ,self.PlotCurve_DisplacementList)
#                print('LL', self.PlotCurve_LoadList)
#                print('TFD', self.TempFDList)
                break
            
#            else:   self.plotNodalDisplacement(DeformationScale = 2e2, SaveImage = True)
                        

        if self.TimeStepCount >= self.MaxTimeStepCount:
            
            self.AnalysisFinished = True
            
#            
            
            
            
            
            
            
            
            
            
#            
#    def runLocalDuanSolver_1(self):
#        
#        #Arc Length Parameters
#        del_lambda_ini =    self.SolverParameters['del_lambda_ini']
#        MaxIterCount =      self.SolverParameters['MaxIterCount']
#        RelTol =            self.SolverParameters['RelTol']
#        m =                 self.SolverParameters['m']
#        Nd0 =               self.SolverParameters['Nd0']
#        n =                 self.SolverParameters['n']
#        p =                 self.SolverParameters['p']
#        Nint =              len([InterfaceElementObj for InterfaceElementObj in self.InterfaceElementObjList if not InterfaceElementObj.InterfaceType == 'FREE'])
#        Nd =                Nint**n + Nd0
#        
#        
#        #Updating Global LoadVectors to include reactions        
##        StaticDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalStaticLoadVector)
##        StaticDisplacementVector[self.FilteredBCDOFList] = 0.0
##        self.GlobalStaticLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, StaticDisplacementVector)
#        
#        RefTransientDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
#        RefTransientDisplacementVector[self.FilteredBCDOFList] = 0.0
#        self.GlobalRefTransientLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, RefTransientDisplacementVector)
#        
#        
#        #Initializing variables
##        self.GlobalExternalLoadVector = self.GlobalStaticLoadVector + self.delta_lambda*self.GlobalRefTransientLoadVector
#        
#        self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
#        NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector     
#        GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#        R = GlobalInternalLoadVector - NetExternalLoadVector
#        NormR = norm(R)
##        LastNormR = NormR
#        
#        if not self.FirstNonLinearStepExecuted:
#            
#            self.FirstNonLinearStepExecuted = True
#            
#            #TODO: Check the formula for del_up
#            del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
#            Local_del_up = self.calcLocalDeformationVector(del_up)
#            del_lambda = del_lambda_ini
#            self.delta_l = del_lambda*norm(Local_del_up)
#            
#            self.LastIterCount = Nd
#        
#        
#        #Looping over Timesteps
#        while self.TimeStepCount < self.MaxTimeStepCount:
#            
#            self.TimeStepCount += 1
#            
#            print('')
#            print('-------------------------------')
#            print('TimeStepCount', self.TimeStepCount)
#            
##            self.updateWaterPressureOnCrackCurve()
#            
#            IterCount =         0
#            self.delta_u =      np.zeros(self.N_DOF)
#            self.delta_l =      self.delta_l*(float(Nd)/self.LastIterCount)**m
#            
#            print('dl', self.delta_l, self.LastIterCount)
#            
#            
##            self.updateContactConditions_1()s
#            
#            while IterCount < MaxIterCount: #Iteration steps
#                                    
#                del_uf = -np.dot(self.InvGlobalSecantStiffnessMatrix, R)
#                
#                #TODO: Check the formula for del_up
#                del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
#                
#                Local_del_uf = self.calcLocalDeformationVector(del_uf)
#                Local_del_up = self.calcLocalDeformationVector(del_up)
#                Local_delta_u = self.calcLocalDeformationVector(self.delta_u)
#                
#                
#                if IterCount == 0:  
#                    
#                    del_lambda = self.delta_l/norm(Local_del_up)
#                    del_lambda0 = del_lambda
#                
#                else:
#                    
##                    del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
##                    del_lambda = (self.delta_l**2 - np.dot(Local_delta_u, Local_delta_u + Local_del_uf))/np.dot(Local_delta_u, Local_del_up)
#                    del_lambda = -np.dot(Local_delta_u, Local_del_uf)/np.dot(Local_delta_u, Local_del_up)
#                
#                del_u = del_uf + del_lambda*del_up
#                del_u[self.FilteredBCDOFList] = 0.0
#                
##                    print('dl', del_lambda)
##                    print('du', del_u)
#                
#                    
#                
#                #Updating increments
#                self.delta_u +=  del_u
#                self.delta_lambda += del_lambda
#                
#    #            print('dl', del_lambda)
#                
#                self.GlobalDisplacementVector += del_u
##                self.GlobalExternalLoadVector = self.GlobalStaticLoadVector + self.delta_lambda*self.GlobalRefTransientLoadVector
#                self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
#                
#                #TODO: Bisection
##                self.checkBisection(IterCount, MaxIterCount)
#                
#                self.assignDeformationToNodes()
#                self.updateElementProperties_1()
#                self.updateSidefaceTractionParameters_1()
#                
##                self.updateWaterPressureOnCrackCurve()
#            
##                self.updateGlobalTangentStiffnessMatrix(UpdateInverse = True)
#                self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
#                
##                self.updateContactPressureError_1()
#                
#                NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector
#                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#                
#                R = GlobalInternalLoadVector - NetExternalLoadVector
#                NormR = norm(R)
#                RelNormR = NormR/norm(NetExternalLoadVector)
#                
##                if IterCount > 0 and NormR > LastNormR:   break
##                LastNormR = NormR
#                    
##                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#                
##                self.plotNodalDisplacement(DeformationScale = 0e6, SaveImage = False)
#                print 'RelNormR', RelNormR
#            
#                if RelNormR < RelTol:    
#                    
#                    self.LastIterCount = IterCount + 1
#                    break
#                
#                else:   IterCount += 1
#                
#                
#            
#            else:
#            
#                print 'Convergence cannot achieved within', MaxIterCount, 'iterations'
#                print 'RelNormR', RelNormR
#            
#            
#            
#            self.LastIterCount = Nd*p
#                
#            
#            #Plotting LoadDisplcement Curve
#            self.plotLoadDisplacementCurve(GeometryType = 'SENB')
#            self.plotNodalDisplacement(DeformationScale = 0e2)
#            
#            SBFEM.PostProcessor.FieldVariables(self)
#        
##            print('CPError', self.NormCPErrorList)
##            print('SIF_I', self.ModeI_SIFList)
##            print('SIF_II', self.ModeII_SIFList)
#            
##            self.displayContactTraction()
#            
#            break
#        
##            #Checking ZeroKI Condition
#            if self.checkZeroKICondition() == True: 
#                
#                self.plotNodalDisplacement(DeformationScale = 2e2)
##                self.TempFDList.append([self.CrackIncrementStepCount, self.PlotCurve_DisplacementList[-1], self.PlotCurve_LoadList[-1]])
#        
##                print('DL' ,self.PlotCurve_DisplacementList)
##                print('LL', self.PlotCurve_LoadList)
##                print('TFD', self.TempFDList)
#                break
#            
#            else:   self.plotNodalDisplacement(DeformationScale = 2e2, SaveImage = False)
#            
#            
#    
#    
#    
#    def runLocalDuanSolver_1_Backup(self):
#        
#        #Arc Length Parameters
#        del_lambda_ini =    self.SolverParameters['del_lambda_ini']
#        MaxIterCount =      self.SolverParameters['MaxIterCount']
#        RelTol =            self.SolverParameters['RelTol']
#        m =                 self.SolverParameters['m']
#        Nd0 =               self.SolverParameters['Nd0']
#        n =                 self.SolverParameters['n']
#        p =                 self.SolverParameters['p']
#        Nint =              len([InterfaceElementObj for InterfaceElementObj in self.InterfaceElementObjList if not InterfaceElementObj.InterfaceType == 'FREE'])
#        Nd =                Nint**n + Nd0
#        
#        
#        #Updating Global LoadVectors to include reactions        
##        StaticDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalStaticLoadVector)
##        StaticDisplacementVector[self.FilteredBCDOFList] = 0.0
##        self.GlobalStaticLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, StaticDisplacementVector)
#        
#        RefTransientDisplacementVector = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
#        RefTransientDisplacementVector[self.FilteredBCDOFList] = 0.0
#        self.GlobalRefTransientLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, RefTransientDisplacementVector)
#        
#        
#        #Initializing variables
##        self.GlobalExternalLoadVector = self.GlobalStaticLoadVector + self.delta_lambda*self.GlobalRefTransientLoadVector
#        
#        self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
#        NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector     
#        GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#        R = GlobalInternalLoadVector - NetExternalLoadVector
#        NormR = norm(R)
##        LastNormR = NormR
#        
#        if not self.FirstNonLinearStepExecuted:
#            
#            self.FirstNonLinearStepExecuted = True
#            
#            #TODO: Check the formula for del_up
#            del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
#            Local_del_up = self.calcLocalDeformationVector(del_up)
#            del_lambda = del_lambda_ini
#            self.delta_l = del_lambda*norm(Local_del_up)
#            
#            self.LastIterCount = Nd
#        
#        
#        #Looping over Timesteps
#        while self.TimeStepCount < self.MaxTimeStepCount:
#            
#            self.TimeStepCount += 1
#            
#            print('')
#            print('-------------------------------')
#            print('TimeStepCount', self.TimeStepCount)
#            
##            self.updateWaterPressureOnCrackCurve()
#            
#            IterCount =         0
#            self.delta_u =      np.zeros(self.N_DOF)
#            self.delta_l =      self.delta_l*(float(Nd)/self.LastIterCount)**m
#            
#            print('dl', self.delta_l, self.LastIterCount)
#            
#            
##            self.updateContactConditions_1()s
#            
#            while IterCount < MaxIterCount: #Iteration steps
#                                    
#                del_uf = -np.dot(self.InvGlobalSecantStiffnessMatrix, R)
#                
#                #TODO: Check the formula for del_up
#                del_up = np.dot(self.InvGlobalSecantStiffnessMatrix, self.GlobalRefTransientLoadVector)
#                
#                Local_del_uf = self.calcLocalDeformationVector(del_uf)
#                Local_del_up = self.calcLocalDeformationVector(del_up)
#                Local_delta_u = self.calcLocalDeformationVector(self.delta_u)
#                
#                
#                if IterCount == 0:  
#                    
#                    del_lambda = self.delta_l/norm(Local_del_up)
#                    del_lambda0 = del_lambda
#                
#                else:
#                    
##                    del_lambda = del_lambda0 - np.dot(Local_del_up, Local_delta_u + Local_del_uf)/np.dot(Local_del_up, Local_del_up)
##                    del_lambda = (self.delta_l**2 - np.dot(Local_delta_u, Local_delta_u + Local_del_uf))/np.dot(Local_delta_u, Local_del_up)
#                    del_lambda = -np.dot(Local_delta_u, Local_del_uf)/np.dot(Local_delta_u, Local_del_up)
#                
#                del_u = del_uf + del_lambda*del_up
#                del_u[self.FilteredBCDOFList] = 0.0
#                
##                    print('dl', del_lambda)
##                    print('du', del_u)
#                
#                    
#                
#                #Updating increments
#                self.delta_u +=  del_u
#                self.delta_lambda += del_lambda
#                
#    #            print('dl', del_lambda)
#                
#                self.GlobalDisplacementVector += del_u
##                self.GlobalExternalLoadVector = self.GlobalStaticLoadVector + self.delta_lambda*self.GlobalRefTransientLoadVector
#                self.GlobalExternalLoadVector = self.delta_lambda*self.GlobalRefTransientLoadVector
#                
#                #TODO: Bisection
##                self.checkBisection(IterCount, MaxIterCount)
#                
#                self.assignDeformationToNodes()
#                self.updateElementProperties_1()
#                self.updateSidefaceTractionParameters_1()
#                
##                self.updateWaterPressureOnCrackCurve()
#            
##                self.updateGlobalTangentStiffnessMatrix(UpdateInverse = True)
#                self.updateGlobalSecantStiffnessMatrix(UpdateInverse = True)
#                
##                self.updateContactPressureError_1()
#                
#                NetExternalLoadVector = self.GlobalExternalLoadVector + self.GlobalSidefaceTractionLoadVector
#                GlobalInternalLoadVector = np.dot(self.GlobalSecantStiffnessMatrix, self.GlobalDisplacementVector)
#                
#                R = GlobalInternalLoadVector - NetExternalLoadVector
#                NormR = norm(R)
#                RelNormR = NormR/norm(NetExternalLoadVector)
#                
##                if IterCount > 0 and NormR > LastNormR:   break
##                LastNormR = NormR
#                    
##                print('NormLoad', norm(GlobalInternalLoadVector), norm(NetExternalLoadVector), NormR)
#                
##                self.plotNodalDisplacement(DeformationScale = 0e6, SaveImage = False)
#                print 'RelNormR', RelNormR
#            
#                if RelNormR < RelTol:    
#                    
#                    self.LastIterCount = IterCount + 1
#                    break
#                
#                else:   IterCount += 1
#                
#                
#            
#            else:
#            
#                print 'Convergence cannot achieved within', MaxIterCount, 'iterations'
#                print 'RelNormR', RelNormR
#            
#            
#            
#            self.LastIterCount = Nd*p
#                
#            
#            #Plotting LoadDisplcement Curve
#            self.plotLoadDisplacementCurve(GeometryType = 'SENB')
#            self.plotNodalDisplacement(DeformationScale = 0e2)
#            
#            SBFEM.PostProcessor.FieldVariables(self)
#        
##            print('CPError', self.NormCPErrorList)
##            print('SIF_I', self.ModeI_SIFList)
##            print('SIF_II', self.ModeII_SIFList)
#            
#            
#            
#            break
#            
#                            
        
        
        
    def checkBisection(self, IterCount, MaxIterCount):
        
        #Assigning mean value in case of non-convergence
        if IterCount == MaxIterCount - 20:  
            
            self.delta_u_0 = np.array(self.delta_u, dtype = float)
            self.delta_lambda_0 = np.array(self.delta_lambda, dtype = float)
            self.GlobDispVec_0 = np.array(self.GlobalDisplacementVector, dtype = float)
            self.GlobExtLoadVec_0 = np.array(self.GlobalExternalLoadVector , dtype = float)
            
            print('-----Bisection Initiated------')
            
            
        elif IterCount == MaxIterCount - 19:
            
            self.delta_u_1 = np.array(self.delta_u, dtype = float)
            self.delta_lambda_1 = np.array(self.delta_lambda, dtype = float)
            self.GlobDispVec_1 = np.array(self.GlobalDisplacementVector, dtype = float)
            self.GlobExtLoadVec_1 = np.array(self.GlobalExternalLoadVector , dtype = float)
        
            
            self.delta_u = (self.delta_u_0 + self.delta_u_1)/2.0
            self.delta_lambda = (self.delta_lambda_0 + self.delta_lambda_1)/2.0
            self.GlobalDisplacementVector = (self.GlobDispVec_0 + self.GlobDispVec_1)/2.0
            self.GlobalExternalLoadVector = (self.GlobExtLoadVec_0 + self.GlobExtLoadVec_1)/2.0
            
            print('-----Bisection Assigned------')
                
    
    
    
    def checkCrackPropagation(self):
        
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:  
        
            RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
            
            if RefInterfaceElementObj.InterfaceType == 'FREE':  pass
            
            elif RefInterfaceElementObj.InterfaceType == 'COHESIVE':  return self.checkZeroKICriteria(SubDomainObj)
        
            elif RefInterfaceElementObj.InterfaceType == 'CONTACT':  return self.checkMaxShearStressCriteria(SubDomainObj)
        
            
    
    

    
    def checkZeroKICriteria(self, SubDomainObj):
    
        RefInterfaceElementObj = SubDomainObj.CrackTip_InterfaceElementObj
        RefInterfaceElementSurfaceArea = RefInterfaceElementObj.SurfaceArea
        
        InterfaceMaterialObj = RefInterfaceElementObj.MaterialObj
        CrackMouth_NormalCOD = RefInterfaceElementObj.NormalCODList[0]
        
        
        #Calculating Forces at CrackMouth
        CrackMouth_ShearTraction = 0.0
        
        if CrackMouth_NormalCOD > InterfaceMaterialObj.wc:      raise Exception
        elif CrackMouth_NormalCOD <= InterfaceMaterialObj.w0:   CrackMouth_NormalTraction = InterfaceMaterialObj.Ft
        else:                                                   CrackMouth_NormalTraction = InterfaceMaterialObj.fNTr(CrackMouth_NormalCOD)
           
        
        #Calculating Forces at CrackTip
        CrackTip_ShearTraction = 0.0
        CrackTip_NormalTraction = InterfaceMaterialObj.Ft
        
        TrS0 = CrackTip_ShearTraction
        TrN0 = CrackTip_NormalTraction
        TrS1 = CrackMouth_ShearTraction - CrackTip_ShearTraction
        TrN1 = CrackMouth_NormalTraction - CrackTip_NormalTraction
        
        TractionPolynomialCoefficientList = [[TrS0, TrN0], [TrS1, TrN1]]
        
        #Calculating IntegrationConstant
        RefSideface_CumulativeDispMode = SubDomainObj.calcSidefaceTractionParameters_1(TractionPolynomialCoefficientList, RefInterfaceElementSurfaceArea, SaveAttr = False)
        SubDomain_GlobDispVector = self.GlobalDisplacementVector[SubDomainObj.DOFIdList]
        RefIntegrationConstantList = np.dot(inv(SubDomainObj.EigVec_DispModeData), SubDomain_GlobDispVector - RefSideface_CumulativeDispMode)
        
        #Checking ZeroKI
        #TODO: Modify for multiple cracks
        SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
        KI = SIF[0]
        
        print('KI', KI)
        
        if KI >= 0:     return True
        else:           return False
            
                
    
    
    def checkMaxShearStressCriteria(self, SubDomainObj):
        
        pass
    
    
    
    
    
    
    def finishCrackIncrementStep(self):
        
        #Initializing LEFM Solver
        LEFMSolverObj = LEFM_Solver()
        LEFMSolverObj.CrackIncrementLength = self.CrackIncrementLength 
        LEFMSolverObj.NodeObjList = self.NodeObjList
        LEFMSolverObj.NodeIdList = self.NodeIdList
        LEFMSolverObj.N_DOFPerNode = self.N_DOFPerNode
        LEFMSolverObj.N_Node = self.N_Node
        LEFMSolverObj.N_DOF = self.N_DOF
        LEFMSolverObj.SubDomainObjList = self.SubDomainObjList
        LEFMSolverObj.PartiallyCrackedSubDomainObjList = self.PartiallyCrackedSubDomainObjList
        LEFMSolverObj.InterfaceElementObjList = []
        LEFMSolverObj.GlobalStiffnessMatrix = self.GlobalStiffnessMatrix
        LEFMSolverObj.GlobalExternalLoadVector = self.GlobalRefTransientLoadVector
        LEFMSolverObj.FilteredBCDOFList = self.FilteredBCDOFList
        
        #Running LInear Analysis
        LEFMSolverObj.applyBoundaryConditions(LEFMSolverObj.GlobalStiffnessMatrix)
        LEFMSolverObj.calcGlobalDisplacementVector()
        
        #Calculating Next CrackTip Location
        for SubDomainObj in LEFMSolverObj.PartiallyCrackedSubDomainObjList:  
        
            SubDomain_GlobDispVector = LEFMSolverObj.GlobalDisplacementVector[SubDomainObj.DOFIdList]
            RefIntegrationConstantList = np.dot(inv(SubDomainObj.EigVec_DispModeData), SubDomain_GlobDispVector)
            SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(RefIntegrationConstantList)
            SubDomainObj.calcNewCrackTipCoordinate(LEFMSolverObj.CrackIncrementLength, CrackPropagationAngle)
            
#            SubDomainObj.RefCrackCurve['NewCrackSeg_CoordinateList'][1] = SubDomainObj.RefCrackCurve['NewCrackSeg_CoordinateList'][0] + np.array([0.0, 0.030 , 0.        ], dtype = float)
#            SubDomainObj.RefCrackCurve['CrackTipCoordinate'] = SubDomainObj.RefCrackCurve['NewCrackSeg_CoordinateList'][1]
                 
            
#            if self.CrackIncrementStepCount == 0:
#                
#                SubDomainObj.RefCrackCurve['NewCrackSeg_CoordinateList'][1] = np.array([0.44397185, 0.0873114 , 0.        ], dtype = float)
#                SubDomainObj.RefCrackCurve['CrackTipCoordinate'] = SubDomainObj.RefCrackCurve['NewCrackSeg_CoordinateList'][1]
                 
        

    
    def finishCrackIncrementStep_1(self):
        
        #TODO: Re-run the CZM_Solver without cohesion, and with contact enabled to get direction.
        
        #Calculating Next CrackTip Location
        for SubDomainObj in self.PartiallyCrackedSubDomainObjList:  
        
            SIF, Alpha, CrackPropagationAngle = SubDomainObj.calcSIF(SubDomainObj.IntegrationConstantList)
            SubDomainObj.calcNewCrackTipCoordinate(self.CrackIncrementLength, CrackPropagationAngle)
            
        
    



