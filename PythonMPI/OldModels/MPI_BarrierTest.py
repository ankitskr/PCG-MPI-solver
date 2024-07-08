# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import numpy as np
import os, sys
from time import time, sleep

import mpi4py
from mpi4py import MPI

Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()

def doWork(t):
    sleep(t)
    

t = 10*np.random.rand();
doWork(t)

print('.', Rank, t)

x = Comm.gather(Rank)

print('---', Rank, x)