# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import os
import numpy as np
import sys
import mpi4py
from mpi4py import MPI


#Initializing MPI
Comm = MPI.COMM_WORLD
N_Workers = Comm.Get_size()
Rank = Comm.Get_rank()

os.system('matlab -nodisplay -nodesktop -nosplash -r "config_linux(); FWI_Test(%d); exit"'%Rank)
