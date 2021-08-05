# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:03:10 2021

@author: Sourabh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant, toeplitz, dft, inv
from numpy.linalg import det, cond, slogdet

from gp_functions import kernel
from fourier_functions import expand_matrix, realfftbasis, realfftbasis2
from plot_functions import plot, plotmat, plotvec
from kron_functions import kronmult
import gpconfig

np.set_printoptions(suppress=True)


def ll_fn(X_train, Y_train, noise=None, which='normal', debug=True):

    def ll(theta):
        if noise != None:
            K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise**2 * np.eye(len(X_train))
        else:
            K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + theta[2]**2 * np.eye(len(X_train))
        
        # print('det K: ', det(K))
        # print('cond K: ', cond(K))
        
        sign, logdet = slogdet(K)

        temp1 = 0.5 * sign * logdet
        # print('ytrain shape: ', Y_train.shape)
        # print('Kshape: ', np.linalg.inv(K).shape)
        temp2 = 0.5 * Y_train.T.dot(inv(K).dot(Y_train)).squeeze()
        # temp2 = 0.5 * Y_train.T.dot(np.linalg.inv(K).dot(Y_train))
        temp3 = 0.5 * len(X_train) * np.log(2*np.pi)
        
        if debug:
            print(which)
            print('---------------------------')
            plotmat(K, 'K')
            print('term1 of normal ll: ' ,temp1)
            print('term2 of normal ll: ' ,temp2)
            # print('temp2 shape: ', temp2.shape)
            print('term3 of normal ll: ' ,temp3)
            print()
        
        return temp1 + temp2 + temp3

    def ll_fourier(theta):
        length = theta[0]
        rho = theta[1]
        
        if noise != None:
            K = kernel(X_train, X_train, l=length, sigma_f=rho) + noise**2 * np.eye(len(X_train))
        else:
            K = kernel(X_train, X_train, l=length, sigma_f=rho) + theta[2]**2 * np.eye(len(X_train))
        ncirc = int((length * 4.0) / gpconfig.WIDTH)

        # Expand covariance and targets
        Kexp = expand_matrix(K, ncirc)
        Y_trainexp = np.append(Y_train, np.zeros((ncirc,))).reshape((-1,1))
        # print('yt shape: ', Y_trainexp.shape)
        # print('yt: ', Y_trainexp)

        # Project covariance and targets to Fourier domain
        B = dft(len(Kexp), scale='sqrtn')
        # print('detb: ', det(B))
        Ctilde = B @ Kexp @ B.conj().T
        # Ctilde = B @ Kexp @ B.T
        Ytilde = B @ Y_trainexp

        # term1 = 0.5 * np.log(det(Ctilde))
        term1 = 0.5 * np.log(np.prod(np.diag(Ctilde)))
        term2 = 0.5 * (Ytilde.conj().T @ inv(Ctilde) @ Ytilde).squeeze()
        # term2 = 0.5 * (Ytilde.T @ inv(Ctilde) @ Ytilde).squeeze()
        term3 = 0.5 * len(X_train) * np.log(2*np.pi)

        final = term1 + term2 + term3
        
        if debug:
            print(which)
            print('---------------------------')
            print('ncirc: ', ncirc)
            print('length: ', length)
            print('ytilde shape: ', Ytilde.shape)
            plotmat(K, 'K fourier')
            plotmat(Kexp, 'Kexp fourier')
            plotmat(np.abs(Ctilde), 'abs ctilde fourier')
            # print('bshape: ', B.shape)
            # print('ctshape: ', Ctilde.shape)
            # print('Ytshape: ', Ytilde.shape)
            print('term1: ', term1)
            print('term2: ', term2)
            print('term3: ', term3)
            print('final imag part: ', np.imag(final))
            
            plotvec(np.diag(Ctilde), 'diag ctilde')
            
            print()

        return np.real(final)
    
    def ll_fourier_implicit(theta):
        condthresh = 1e8
        length = theta[0]
        rho = theta[1]
        
        if noise != None:
            K = kernel(X_train, X_train, l=length, sigma_f=rho) + noise**2 * np.eye(len(X_train))
            nsevar = noise
        else:
            K = kernel(X_train, X_train, l=length, sigma_f=rho) + theta[2]**2 * np.eye(len(X_train))
            nsevar = theta[2]
        ncirc = int((length * 4.0) / gpconfig.WIDTH)
        nx = len(X_train)

        # Expand covariance and targets
        Kexp = expand_matrix(K, ncirc)
        nxcirc = len(Kexp)
        
        Y_trainexp = np.append(Y_train, np.zeros((ncirc,))).reshape((-1,1))
        
        maxfreq = np.floor(nxcirc/(np.pi*length)*np.sqrt(0.5*np.log(condthresh)))
        # print('length: ', length)
        # print('maxfreq: ', maxfreq)
        
        # Assuming maxfreq < nxcirc/2
        if maxfreq < nxcirc/2:
            wvec = np.append(np.arange(0, maxfreq), np.arange(-maxfreq, -1+1))
        else:
            # ncos = np.ceil((nxcirc-1)/2)
            # nsin = np.floor((nxcirc-1)/2)
            ncos = np.ceil((nxcirc+1)/2) # number of cosine terms (positive freqs)
            nsin = np.floor((nxcirc-1)/2) # number of sine terms (negative freqs)
            # wvec = np.append(np.arange(0, ncos+1), np.arange(-nsin, -1+1))
            wvec = np.append(np.arange(ncos), np.arange(-nsin, -1+1))
            
        wvecsq = np.square(wvec)
        
        # Get fourier basis and transform targets to Fourier domain
        # Bfft = realfftbasis(nx, nxcirc, wvec)
        Bfft = realfftbasis2(nx, nxcirc, wvec)
        # print('Bfft: ', Bfft.T @ Bfft)
        # print('Bfft: ', Bfft.shape)
        # yf = Bfft @ Y_train
        yf = kronmult([Bfft], Y_train)
        
        # mkcovdiag
        const = np.square(2*np.pi/nxcirc)
        wwnrm = wvecsq*const
        # cdiag = np.sqrt(2*np.pi)*rho*length*np.exp(-0.5*ww*length**2)
        # cdiag = np.sqrt(2*np.pi)*rho*length*np.exp(-0.5*ww*(length**2))
        # cdiag = np.sqrt(2*np.pi)*rho*np.exp(-0.5*ww*(length**2))
        
        # wwnrm = ((2*np.pi/nxcirc)**2) * (wvec**2)
        
        # wwthresh = 2*np.log(condthresh) / (length**2)
        wwthresh = np.inf
        
        ii = np.where(wwnrm < wwthresh)
        ni = np.sum(ii)
        cdiag = np.sqrt(2*np.pi)*rho*np.exp(-0.5*wwnrm[ii]*(length**2)) + nsevar
        # cdiag = np.sqrt(2*np.pi)*rho*np.exp(-0.5*wwnrm[ii]*(length**2)) + gpconfig.adddiag
        
        term1 = 0.5 * np.sum(np.log(cdiag))
        term2 = 0.5 * np.sum((1/cdiag).dot(yf[ii] ** 2))
        term3 = 0.5 * len(X_train) * np.log(2*np.pi)
        
        if debug:
            print(which)
            print('---------------------------')
            print('xtrain shape: ', X_train.shape)
            print('ytrain shape: ', Y_train.shape)
            print('length: ', length)
            print('kexp shape: ', Kexp.shape)
            print('cdiag shape: ', cdiag.shape)
            # print('cdiag: ', cdiag)
            print('ii shape: ', ii[0].shape)
            print('wwthresh: ', wwthresh)
            print('ytrainexp shape: ', Y_trainexp.shape)
            print('Bfft shape: ', Bfft.shape)
            # print('Bfft: ', Bfft)
            print('yf shape: ', yf.shape)
            # print('yf: ', yf)
            print('maxfreq: ', maxfreq)
            print('ncirc: ', ncirc)
            print('nxcirc: ', nxcirc)
            print('nx: ', nx)
            print('wvec shape: ', wvec.shape)
            print('wwnrm shape: ', wwnrm.shape)
            # print('wvec: ', wvec)
            print('term1: ', term1)
            print('term2: ', term2)
            print('term3: ', term3)
            
            # Plot covariance
            plotmat(K, 'K implicit')
            plotmat(Kexp, "Kexp implicit")
            
            # Plot rows of covariance
            firstrow = Kexp[0,:]
            lastrow = Kexp[-1,:]
            secondrow = Kexp[1,:]
            tenthrow = Kexp[9,:]
            # plt.plot(firstrow, lastrow)
            plt.plot(range(len(firstrow)), firstrow, label='firstrow')
            plt.plot(range(len(lastrow)), lastrow, label='lastrow')
            plt.plot(range(len(lastrow)), np.flip(lastrow), label='flip lastrow')
            # plt.plot(range(len(lastrow)), secondrow, label='secondrow')
            # plt.plot(range(len(lastrow)), tenthrow, label='tenthrow')
            plt.title('firstrow vs lastrow Kexp')
            plt.legend()
            plt.show()
            
            # Plot vectors
            plotvec(cdiag, 'cdiag implicit')
            plotvec(wvec, 'wvec')
            plotvec(1/cdiag, '1/cdiag')
            plotvec(yf**2, 'yf**2')
            plotvec(wwnrm[ii], 'wwnrm ii')
            
            print()
            
        return term1 + term2 + term3
        
    
    if which == 'fourier':
        return ll_fourier
    elif which == 'implicit':
        return ll_fourier_implicit
    
    return ll