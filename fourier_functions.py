# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:58:40 2021

@author: Sourabh
"""

import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import circulant, toeplitz, dft, inv
from numpy.linalg import det
np.set_printoptions(suppress=True)


def expand_matrix(C, ncirc):
    """
    Expand C into a circulant matrix
    """
    row1 = C[0, :]
    row1 = np.append(row1, np.flip(row1[:ncirc]))
    return circulant(row1).T

def realfftbasis(nx, nn, wvec):
    """
    Basis of sines+cosines for nn-point discrete fourier transform (DFT)
    % [B,wvec] = realfftbasis(nx,nn,w)
    %
    % For real-valued vector x, realfftbasis(nx,nn)*x is equivalent to realfft(x,nn) 
    %
    % INPUTS:
    %  nx - number of coefficients in original signal
    %  nn - number of coefficients for FFT (should be >= nx, so FFT is zero-padded)
    %  wvec (optional) - frequencies: positive = cosine
    %
    % OUTPUTS:
    %   B [nn x nx] or [nw x nx] - DFT basis 
    %   wvec - frequencies associated with rows of B
    """
    # print("In realfftbasis:")
    # print('---------------------------')
    
    wcos = wvec[np.where(wvec>=0.0)].reshape((-1,1))
    wsin = wvec[np.where(wvec<0.0)].reshape((-1,1))
    
    # print('nx: ', nx)
    # print('nn: ', nn)
    # print('wvec shape: ', wvec.shape)
    # print('wcos shape: ', wcos.shape)
    # print('wsin shape: ', wsin.shape)
    
    x = np.arange(nx).reshape(-1,1)
    # print('x shape: ', x.shape)
    
    if wsin.size != 0:
        B = np.append(np.cos((wcos*2*np.pi/nn) @ x.T), np.sin((wsin*2*np.pi/nn) @ x.T),
                      axis=0)
        B = B/np.sqrt(nn/2.0)
        # print('B shape: ', B.shape)
    else:
        B = np.cos((wcos*2*np.pi/nn) @ x.T) / np.sqrt(nn/2.0)
        
    # make DC term into a unit vector
    zero_inds = np.where(wvec == 0.0)
    B[zero_inds,:] = B[zero_inds,:] / np.sqrt(2.0)
    
    if (nn/2 == np.max(wvec)):
        ncos = np.ceil((nn+1)/2).astype(int)
        B[ncos,:] = B[ncos,:] / np.sqrt(2.0)
        
    # print()
    return B

def realfftbasis2(nx,nn,wvec):
#  Basis of sines+cosines for nn-point discrete fourier transform (DFT)
# 
#  [B,wvec] = realfftbasis(nx,nn,w)
# 
#  For real-valued vector x, realfftbasis(nx,nn)*x is equivalent to realfft(x,nn) 
# 
#  INPUTS:
#   nx - number of coefficients in original signal
#   nn - number of coefficients for FFT (should be >= nx, so FFT is zero-padded)
#   wvec (optional) - frequencies: positive = cosine
# 
#  OUTPUTS:
#    B [nn x nx]  - DFT basis 
#    wvec - frequencies associated with rows of B
# 
	

	if nn is None:
	    nn = nx

	if nn<nx:
	    print('realfftbasis: nxcirc < nx. SOMETHING IS WRONG')

	# Divide into pos (for cosine) and neg (for sine) frequencies
	wcos = wvec[wvec>=0] 
	wsin = wvec[wvec<0]  

	x = np.arange(nx) # spatial np.pixel indices
	if wsin.any():
	    B = np.concatenate((np.cos(np.outer(wcos*2*np.pi/nn,x)), np.sin(np.outer(wsin*2*np.pi/nn,x))),axis = 0)/np.sqrt(nn/2)
	else:
	    B = np.cos(np.outer(wcos*2*np.pi/nn,x))/np.sqrt(nn/2)


	# make DC term into a unit vector
	izero = [wvec==0][0] # index for DC term... careful of indexing here!
	inds = [i for i, x in enumerate(izero) if x]
	newthing = B[inds]/np.sqrt(2)
	B[inds] = newthing
	# if nn is even, make Nyquist term (highest cosine term) a unit vector
	if (nn/2 == np.max(wvec)):
	    ncos = np.int(np.ceil((nn)/2)) # this is the index of the nyquist freq
	    B[ncos] = B[ncos]/np.sqrt(2)

	return B