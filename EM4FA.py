# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:23:25 2021

@author: Sourabh
"""

import numpy as np
import scipy
from scipy.stats import multivariate_normal
from numpy.linalg import lstsq, inv
import matplotlib.pyplot as plt
from numpy.random import randn, exponential

q = 23
p = 3
T = 2000

SNR = 50
nseVar = p**2 / SNR

# Gen data
C = randn(q, p)
lambdas = nseVar*exponential(1, (q,))

# x = randn(p, T)
# Linv = np.diag(np.sqrt(1/lambdas))
Linv = np.diag(1/lambdas)

# Y = C@x + Linv @ randn(q, T)

# Gen data 2
# C = randn(q, p)
# lambdas = nseVar*exponential(1, (q,))
# Linv = np.diag(np.sqrt(1/lambdas))
x = np.zeros((p, T))
mu = np.zeros((T,))

for i in range(p):
    x[i, :] = np.random.multivariate_normal(mu.ravel(), np.eye(T))
    
E = np.random.multivariate_normal(np.zeros((q,)), Linv,
                                    size=T).T

Y = C@x + E
# Y = C@x + Linv @ randn(q, T)

# EM
I = 1200
Cn = randn(q, p)
lambnew = nseVar*exponential(1, (q,))
S = Y@Y.T/T
lls = []

for itr in range(I):
    
    # Reset
    Co = Cn.copy()
    lambold = lambnew.copy()
    
    # E step
    Linvold = np.diag(lambold)
    # invsigma = np.eye(p) + Co.T @ Linvold @ Co
    # Ex = Co.T @ Linvold @ Y
    # # print('Ex shape: ', Ex.shape)
    # Ex = lstsq(invsigma, Ex)[0]
    G = inv(np.eye(p) + Co.T @ Linvold @ Co)
    Ex = G @ Co.T @ Linvold @ Y
    # print('Ex shape: ', Ex.shape)
    sumExx = T * inv(invsigma) + Ex @ Ex.T
    
    
    # M step
    Cn = Y @ Ex.T
    # Cn = lstsq(sumExx, Cn.T)[0].T
    Cn = Cn @ inv(sumExx)
    lambnew = 1 / np.diag(S - Cn @ Ex @ (Y.T)/T)
    
    # Document progress
    var = scipy.stats.multivariate_normal(mean=np.zeros((q,)),
                              cov=Cn@Cn.T + np.diag(1/lambnew))
    lls.append(np.sum(np.log(var.pdf(Y.T))))
    
    
CC = (C @ C.T).reshape((-1,))
CnCn = (Cn @ Cn.T).reshape((-1,))

plt.figure()
plt.plot([np.min(lambdas), np.max(lambdas)], [np.min(lambdas), np.max(lambdas)], 'r--')
plt.scatter(lambdas, lambnew)
plt.xlabel('True')
plt.ylabel('Estimated')
plt.title('Lambda')
plt.show()

plt.figure()
plt.plot([np.min(CC), np.max(CC)], [np.min(CC), np.max(CC)], 'r--')
plt.scatter(CC, CnCn)
plt.xlabel('True')
plt.ylabel('Estimated')
plt.title('C')
plt.show()

plt.figure()
plt.plot(lls)
plt.title('LLs')
plt.show()