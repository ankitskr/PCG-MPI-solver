# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:06:18 2017

@author: Ankit
"""

import setuptools
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('Array_cy.pyx'), include_dirs=[numpy.get_include()])
#setup(ext_modules = cythonize('Test_py.pyx'))
