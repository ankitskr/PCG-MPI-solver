# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:38:51 2020

@author: z5166762
"""

import numpy as np
from numpy.linalg import inv
from time import time

from numpy.random import rand

def dot(A,X,N):
    
    B = [0.0]*N
    
    i = 0
    while i < N:
        
        Ai = A[i]
        
        j = 0
        while j < N:
            
            B[i] += Ai[j]*X[j]
            j += 1            
            
        i += 1
        
    return B


N = 2000
A = rand(N,N).tolist()
X = rand(N).tolist()

t1 = time()
dot(A,X,N)
t2 = time()
print(t2-t1)

t1 = time()
np.dot(A,X)
t2 = time()
print(t2-t1)