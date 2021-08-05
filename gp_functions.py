# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:56:48 2021

@author: Sourabh
"""

import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import circulant, toeplitz, dft, inv
from numpy.linalg import det
np.set_printoptions(suppress=True)

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    X1: (m x d).
    X2: (n x d).

    Returns (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l, sigma_f: Kernel parameters
        sigma_y: Noise parameter.
    
    Returns:
        output means (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.identity(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.identity(len(X_s))
    K_inv = np.linalg.inv(K)
    
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s