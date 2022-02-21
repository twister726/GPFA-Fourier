# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:23:25 2021

@author: Sourabh
"""

import numpy as np
from numpy.random import multivariate_normal, randn, exponential

q = 23
p = 3
T = 2000

SNR = 50
nseVar = p**2 / SNR

# Gen data
C = randn(q, p)
x = randn(p, T)

lambdas = exponential(1, (q, 1))
L = np.diag(np.sqrt(1/lambdas))

Y = C@x + L@randn(q, T)
