# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:10:23 2021

@author: Sourabh
"""

import numpy as np
from matplotlib import pyplot as plt

from gp_functions import kernel, posterior
from plot_functions import plot
import gpconfig

np.set_printoptions(suppress=True)


def drawdatafromgp(num_obs):
    nsevar = gpconfig.noise_coeff
    l = gpconfig.l
    sigma_f = gpconfig.sigma_f
    
    X = np.arange(0, num_obs).reshape((-1,1))
    
    mu = np.zeros(X.shape)
    cov = kernel(X, X, l=l, sigma_f=sigma_f) + nsevar**2 * np.identity(len(X))
    
    drawn_samples = np.random.multivariate_normal(mu.ravel(),
                                                  cov,
                                                  size=3)
    
    # plot(mu, cov, X, samples=drawn_samples)

    # Noisy training data
    X_train = X.copy()
    Y_train = drawn_samples[0].reshape((-1,1))
    
    print("xtrain shape: ", X_train.shape)
    # print('xtrain: ', X_train)
    print("ytrain shape: ", Y_train.shape)
    plt.figure()
    plt.plot(X_train, Y_train)
    plt.title('training data')
    plt.show()
    
    # Mean and covariance of the posterior distribution
    mu_train, cov_train = posterior(X, X_train, Y_train, l=2.0, sigma_f=2.0, sigma_y=nsevar)
    
    samples = np.random.multivariate_normal(mu_train.ravel(), cov_train, 3)
    # plot(mu_train, cov_train, X, X_train=X_train, Y_train=Y_train, samples=samples)
    print('samples0 shape: ', samples[0].shape)
    print('mu_train shape: ', mu_train.shape)
    
    return X_train, Y_train