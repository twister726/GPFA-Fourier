# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 09:29:29 2022

@author: Sourabh
"""

#%% Imports

# import numpy as np
import autograd.numpy as np
from autograd import grad
# from autograd.misc.optimizers import adam
from autograd.scipy.linalg import sqrtm
# from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from scipy.linalg import circulant, toeplitz, dft, inv
# import scipy
import autograd.scipy as sp
from numpy.linalg import det
import pickle

from collections import defaultdict
import time

from gen_data import drawdatafromgp
from autograd_mod import myadam
from log_likelihood import ll_fn
from plot_functions import plot, plotvec, plotmat
from gp_functions import kernel, posterior
# from misc_functions import vec, vecinv
import gpconfig

np.set_printoptions(suppress=True)

#%% Set constants

q = 11
p = 3
T = 76

SNR = 50
nseVar = p**2/SNR

#%% Utility functions

def vec(arr):
    return np.reshape(arr, (-1,1), order='F')

def vecinv(arr, M, N):
    return np.reshape(np.concatenate([arr[M*i:M*i + M, 0] for i in range(N)]), (N,M)).T

def circulant(row):
    # Return a circulant matrix given the first row
    n = row.shape[0]
    rows = [np.roll(row, i) for i in range(n)]
    ans = np.array(rows)
    return ans

def circdet(row):
    # Return determinant of circulant matrix with given first row
    n = row.shape[0]
    return np.real(np.prod(n * np.fft.ifft(row)))

def directsum(matrices):
    # Only works if all the matrices are the same size
    
    m = matrices[0].shape[0]
    n = len(matrices)
    
    ans = np.zeros((m*n, m*n))
    for i in np.arange(n):
        ans = ans + np.pad(matrices[i], ((m*i, m*(n-1-i)), (m*i, m*(n-1-i))),
                           'constant')
        
    return ans

# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# print(directsum([a,b]))
# print(circulant(np.array([1,2,3])))

#%% Generate data

C_orig = np.random.randn(q, p)
lamb_orig = np.random.exponential(1,size=(q,)) / nseVar # noise precisions
tau_orig = np.random.rand(p,) + 0.5

def getkernels(timescales):
    X = np.arange(0, T).reshape((-1,1))
    kernels = [kernel(X, X, l=ts) for ts in timescales]
    return kernels

kernels = getkernels(tau_orig)
# K = sp.linalg.block_diag(*kernels)
K = directsum(kernels)
plotmat(K, 'K')

sqrtkernels = [sqrtm(ker) for ker in kernels]
plotmat(sqrtkernels[0], 'sqrtk0')

x = np.random.multivariate_normal(np.zeros((p*T,)), K, size=1).reshape((-1,1))
X = vecinv(x, T, p).T
E = np.random.multivariate_normal(np.zeros((q,)),
                                  np.diag(1.0/lamb_orig),
                                  size=T).T
Y = C_orig@X + E
y = vec(Y.T)

print('E shape: ', E.shape)
print('Y shape: ', Y.shape)
print('x shape: ', x.shape)
print('y shape: ', y.shape)
print('X shape: ', X.shape)

#%% Define variational loss

def lossfn(constparams, which='variational', debug=False):
    
    def varloss(params, t):
        # vn, phin, circcolsfourier are used for initialization
        if which == 'variational':
            v = np.reshape(params[:T*p], (T, p))
            phineg = np.reshape(params[T*p:2*T*p], (T, p))
            circcolsfourier = np.reshape(params[2*T*p:2*T*p + p*(T//2 + 1)],
                                         (T//2 + 1, p))
            C = np.reshape(constparams[:q*p], (q,p))
            lamb = np.reshape(constparams[q*p:], (q,))
        else:
            C = np.reshape(params[:q*p], (q,p))
            lamb = np.reshape(params[q*p:], (q,))
            v = np.reshape(constparams[:T*p], (T, p))
            phineg = np.reshape(constparams[T*p:2*T*p], (T, p))
            # circrows = np.reshape(constparams[2*T*p:3*T*p], (T, p))
            circcolsfourier = np.reshape(constparams[2*T*p:2*T*p + p*(T//2 + 1)],
                                         (T//2 + 1, p))
            
        # Forcing phi and circcols fourier to be positive
        phi = np.exp(phineg)
        circcolsfourier = np.exp(circcolsfourier)
            
        # print('v: ', v)
        # print('phi: ', phi)
        # print('circrows: ', circrows)
        # print('C: ', C)
        # print('lamb: ', lamb)
        
        Linv = np.diag(1.0/lamb)
        L = np.diag(lamb)
        U = np.array([sqrtkernels[i]@v[:,i] for i in np.arange(p)]).T
        circcols = np.array([np.fft.irfft(circcolsfourier[:,i], T) for i in np.arange(p)]).T
        
        # print('phi shape: ', phi.shape)
        # print('circcols shape: ', circcols.shape)
        # print('circcolsfourier shape: ', circcolsfourier.shape)
        lambdas = [np.diag(phi[:,i])@circulant(circcols[:,i]).T
                    for i in np.arange(p)]
        sigmas = [sqrtkernels[i]@lambdas[i]@lambdas[i].T@sqrtkernels[i].T
                    for i in np.arange(p)]
        
        # sgn, logdet = np.linalg.slogdet(np.kron(Linv, np.eye(T)))
        # term1 = 0.5*sgn*logdet + 0.5*np.trace(Y@Y.T@L)
        term1 = 0.5*T*np.sum(np.log(1.0/lamb)) + 0.5*np.trace(Y@Y.T@L)
        term2 = -np.trace(Y@U@C.T@L)
        
        # exxt = sp.linalg.block_diag(*sigmas) + vec(U)@vec(U).T
        exxt = directsum(sigmas) + vec(U)@vec(U).T
        
        term3 = 0.5*np.trace(np.kron(C.T@L@C, np.eye(T)) @ exxt)
        
        # KL Divergence
        term4 = 0.0
        for i in np.arange(p):
            # term4 = term4 + np.linalg.norm(lambdas[i])**2
            # sgn, logdet = np.linalg.slogdet(lambdas[i])
            # term4 = term4 - 2*sgn*logdet
            
            # temp1 = np.sum(np.log(phi[:,i]))
            # temp2 = np.log(circdet(circrows[:,i]))
            # print('phi i: ', phi[:,i])
            # print('temp1: ', temp1)
            # print('temp2: ', temp2)
            # logdetlambda = temp1 + temp2
            # term4 = term4 - 2*logdetlambda
            
            # term4 = term4 + np.linalg.norm(v[:,i])**2 - T
            
            temp1 = np.sum(circcols[:,i]**2) * np.sum(phi[:,i]**2)
            temp2 = -2*np.sum(np.log(phi[:,i]))
            temp3 = np.linalg.norm(v[:,i])**2 - T
            
            chat = circcolsfourier[:,i]
            if T % 2 == 0:
                logdetC = np.log(chat[0]) + np.log(chat[T//2]) + 2*np.sum(np.log(chat[1:T//2]))
            else:
                logdetC = np.log(chat[0]) + 2*np.sum(np.log(chat[1:(T+1)//2]))
            
            term4 = term4 + temp1 + temp2 + temp3 - 2*logdetC
            
        term4 = term4 / 2
            
        loss = term1 + term2 + term3 + term4
        
        if debug:
            print('term1 shape: ', term1)
            print('term2 shape: ', term2)
            print('term3 shape: ', term3)
            print('term4 shape: ', term4)        
        
        return loss
    
    return varloss
        
#%% Initialize params for EM

# Initialize variational params
vn = np.random.randn(T, p)
phin = np.random.randn(T, p)
circcolsfourier = np.random.randn(T//2 + 1, p)

varparams = np.concatenate([np.reshape(vn, (-1,)),
                            np.reshape(phin, (-1,)),
                            np.reshape(circcolsfourier, (-1,))])

# Initialize GPFA params
Cn = np.random.randn(q, p)
lambn = np.random.exponential(1,size=(q,)) / nseVar # noise precisions

gpfaparams = np.concatenate([np.reshape(Cn, (-1,)),
                           np.reshape(lambn, (-1,))])

# Track log likelihoods
overall_elbos = []
overall_marglls = []

#%% Plot functions

def plotparams(gpfaparams):
    Cf = np.reshape(gpfaparams[:q*p], (q,p))
    lambf = np.reshape(gpfaparams[q*p:], (q,))
    
    plt.figure()
    plt.scatter((Cf@Cf.T).reshape((-1,1)), (C_orig@C_orig.T).reshape((-1,1)))
    plt.plot([0, np.max(C_orig@C_orig.T)], [0, np.max(C_orig@C_orig.T)])
    plt.title('C comparison')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Learned')
    plt.ylabel('Original')
    plt.show()
    
    plt.figure()
    plt.scatter(lambf, lamb_orig)
    plt.plot([0, np.max(lamb_orig)], [0, np.max(lamb_orig)])
    plt.title('lamb comparison')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
plotparams(gpfaparams)

#%% Do EM

# Adam optimization callback function
def get_callback(obj, elbos, which, marglls):
    def callback(params, t, g):
        objval = obj(params, t)
        if t % 10 == 0:
            if which == 'GPFA':
                margval = marg_ll(params)
                print('{} Opt: Iteration {}, elbo {}, marg {}'.format(which, 
                                                                      t, 
                                                                      objval,
                                                                      margval))
                marglls.append(margval)
            else:
                print('{} Opt: Iteration {}, elbo {}'.format(which, 
                                                             t, 
                                                             objval))
        elbos.append(objval)
        
    return callback

def marg_ll(gpfaparams):
    Cf = np.reshape(gpfaparams[:q*p], (q,p))
    lambf = np.reshape(gpfaparams[q*p:], (q,))
    Linv = np.diag(1.0/lambf)
    
    cov = np.kron(Linv, np.eye(T)) + np.kron(Cf,np.eye(T))@K@np.kron(Cf.T,np.eye(T))
    
    sgn, logdet = np.linalg.slogdet(cov)
    return sgn*logdet + np.squeeze(y.T@np.linalg.inv(cov)@y)

# print('margll: ', marg_ll(gpfaparams))

# Do EM

startI = 20
I = 22
for ii in range(startI, I):
    print('Starting EM Iter ' + str(ii))
    
    # Optimize wrt GP parameters
    # constparams = varparams, initparams = gpfaparams
    
    obj = lossfn(varparams, 'gpfa', debug=False)
    gradobj = grad(obj)
    print('test: ', obj(gpfaparams, 0))
    
    elbos = []
    marglls = []
    callback = get_callback(obj, elbos, 'GPFA', marglls)
    newgpfaparams = myadam(gradobj, gpfaparams, callback=callback)
    
    plotvec(elbos, name='GPFA params opt ELBOs iter ' + str(ii))
    plotvec(marglls, name='GPFA params opt marglls iter ' + str(ii))
    
    overall_elbos.append(elbos[-1])
    # overall_elbos.extend(elbos)
    overall_marglls.extend(marglls)
    
    gpfaparams = np.array(newgpfaparams, copy=True)
    
    # Optimize wrt variational parameters
    # constparams = gpfaparams, initparams = varparams
    
    obj = lossfn(gpfaparams, 'variational')
    gradobj = grad(obj)
    print('test: ', obj(varparams, 0))
    
    elbos = []
    marglls = []
    callback = get_callback(obj, elbos, 'Variational', marglls)
    newvarparams = myadam(gradobj, varparams, callback=callback)
    
    plotvec(elbos, name='Var params opt ELBOs iter ' + str(ii))
    
    overall_elbos.append(elbos[-1])
    # overall_elbos.extend(elbos)
    
    varparams = np.array(newvarparams, copy=True)
    
    print()
    
    plotparams(gpfaparams)
    
#%% Plot overall LLs

# plotparams()
    
plt.figure()
plt.scatter(np.arange(len(overall_elbos)), 
            overall_elbos, c=['green' if i%2 else 'red' for i in range(len(overall_elbos))])
plt.plot(np.arange(len(overall_elbos)), overall_elbos)
plt.title('ELBO. Red = GP opt, Green = var opt')
plt.show()

plt.figure()
plt.plot(overall_marglls)
plt.title('Overall Marg LL')
plt.show()