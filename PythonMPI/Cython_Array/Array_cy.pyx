# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:20:37 2017

@author: Ankit
"""


import numpy as np 
cimport numpy as cnp
cimport cython

ctypedef cnp.uint8_t uint8

"""
cdef extern from "math.h":
    double sin(double arg)


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef cnp.ndarray[cnp.double_t,ndim=1] apply_sin(cnp.ndarray[cnp.double_t, ndim=1] a):
    
    cdef int i
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] out
    out = cnp.ndarray(len(a), dtype=float)
    
    for i in range(len(a)):
        
        out[i] = sin(a[i])
        
    return out
"""  


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef apply_sum(long[:,:] ElemList_LocDOFIdVector, double[:] MP_LocFintVector, double[:,:] ElemList_Fint):

    cdef int q0 = 0
    cdef int q1 = 0
    cdef int N_DOF=ElemList_LocDOFIdVector.shape[0]
    cdef int N_Elem=ElemList_LocDOFIdVector.shape[1]
    
    for q0 in range(N_DOF):
        
        for q1 in range(N_Elem):
                    
            MP_LocFintVector[ElemList_LocDOFIdVector[q0, q1]] += ElemList_Fint[q0, q1]
            
            
            

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef updateLocFint(long[:,:] ElemList_LocDOFIdVector, double[:] MP_UnVector_1, uint8[:,:] ElemList_SignVector, double[:] MP_LocFintVector, double[:] ElemList_Level, double[:,:] Ke):

    cdef int q0 = 0
    cdef int q0_ = 0
    cdef int q1 = 0
    cdef int N_DOF=ElemList_LocDOFIdVector.shape[0]
    cdef int N_Elem=ElemList_LocDOFIdVector.shape[1]
    
    ElemUn = np.zeros(N_DOF, dtype=np.double)
    cdef double[:] ElemList_Un_1 = ElemUn
    
    for q1 in range(N_Elem):
        
        for q0 in range(N_DOF):
        
            ElemList_Un_1[q0] = MP_UnVector_1[ElemList_LocDOFIdVector[q0,q1]]*ElemList_Level[q0]
            
            if ElemList_SignVector[q0,q1]:    ElemList_Un_1[q0] *= -1
        
        
        for q0 in range(N_DOF):
        
            Fint_q0 = 0.0
        
            for q0_ in range(N_DOF):
        
                Fint_q0 += Ke[q0, q0_]*ElemList_Un_1[q0_]
            
            if ElemList_SignVector[q0,q1]:    Fint_q0 *= -1
        
            MP_LocFintVector[ElemList_LocDOFIdVector[q0, q1]] += Fint_q0
            
        
        
        
