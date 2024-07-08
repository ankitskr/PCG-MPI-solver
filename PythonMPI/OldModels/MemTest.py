# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:10:24 2020

@author: z5166762
"""

import os
from time import time, sleep
import numpy as np
import os.path
import shutil
import scipy.io

import pickle
import zlib

import mpi4py
from mpi4py import MPI



if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------
    #print('Initializing MPI..')
    Comm = MPI.COMM_WORLD
    N_Workers = Comm.Get_size()
    Rank = Comm.Get_rank()
    
    if Rank==0:    print('N_Workers', N_Workers)
    
    N = 500;
    
    for i in [2, 1, 0]:
        A=np.ones([i*N,N])
        B=np.ones([i*N,N])
        C=np.ones([N,N])
    
    i = 0
    while(1):
        i = i+1
        if Rank==0 and i%1e8==0:
            print(i)
    
    
    
    
    

