# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:45:13 2020

@author: z5166762
"""

import numpy as np
import scipy.io
import os, sys
from os import listdir
from os.path import isfile, join
    
from mgmetis import metis


cells = [[0, 1],
         [1, 2],
         [3, 4],
         [4, 5],
         [6, 7],
         [7, 8],
         [0, 3],
         [1, 4],
         [2, 5],
         [3, 6],
         [4, 7],
         [5, 8]]


objval, epart, npart = metis.part_mesh_dual(2, cells)

print(objval)
print(epart)
print(npart)
