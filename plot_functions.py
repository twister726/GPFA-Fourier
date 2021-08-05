# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:55:06 2021

@author: Sourabh
"""

import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(suppress=True)

def plot(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.reshape(-1)
    mu = mu.reshape(-1)
    uncertainty = 2.0 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label='Sample '+str(i+1))
    if X_train is not None:
        # plt.plot(X_train, Y_train, 'rx')
        plt.scatter(X_train, Y_train, s=0.5, color='red')
        
def plotmat(mat, name=None):
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    if name:
        plt.title(name + ' ' + str(mat.shape))
    plt.show()
    
def plotvec(vec, name='', xvalues=None):
    if xvalues is None:
        xvalues = range(len(vec))
    
    plt.figure()
    plt.plot(xvalues, vec, label=name, c=(random.random(), random.random(), random.random()))
    plt.legend()
    plt.show()