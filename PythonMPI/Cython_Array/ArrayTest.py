# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:23:55 2017

@author: ankit
"""


#create pyx files
import os
os.system("python setup.py build_ext --inplace")



from Array_cy import apply_sum, updateLocFint

"""
from time import time
import numpy as np



a = np.linspace(0, 10, int(1e6), dtype=np.double)
out = apply_sin(a)

print(len(a))
"""
