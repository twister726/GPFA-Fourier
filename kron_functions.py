# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:33:11 2021

@author: Sourabh
"""

import numpy as np
from functools import reduce
import warnings

def kronmult(Amats,x,ii=None):
	#  Multiply matrix (A{2} kron A{1} kron ... )(:,ii) by x
	# 
	#  y = kronmult(Amats,x,ii);
	#  
	#  INPUT
	#    Amats - cell array with matrices {A1, ..., An}
	#        x - matrix to multiply with kronecker matrix formed from Amats
	#       ii - binary vector indicating sparse locations of x rows (OPTIONAL)
	# 
	#  OUTPUT
	#     y - vector output 
	# 
	#  Equivalent to (for 3rd-order example)
	#     y = (A3 \kron A2 \kron A1) * x
	#  or in matlab:
	#     y = kron(A3,kron(A2,A1)))*x
	# 
	#  For two A's and vector x, equivalent to left and right matrix multiply:
	#     y = vec( A1 * reshape(x,m,n) * A2' ); 
	# 
	#  Computational cost: 
	#     Given A1 [n x n] and A2 [m x m], and x a vector of length nm, 
	#     standard implementation y = kron(A2,A1)*x costs O(n^2m^2)
	#     whereas this algorithm costs O(nm(n+m))
	#
	# Dimensionality explained through an example
	# 	if there are three matrices:
	# 	A is a x b
	# 	B is c x d
	# 	C is e x f
	# 	then (A kron B kron C) is ace x bdf
	# 	an input vector, x, must be a column matrix thats bdf by k
	# 	function output is abc by k. (k is often 1)

	# Additional technical points
	# ii can be a python list, it will be used as a numpy array
	# ii cannot be boolean array!
	# x must be a numpy array with the correct dimensionality.
	# this code will handle if x is a 1-d numpy array (though best if 2-D column array)

	#if x is a numpy array of dim 1,
	#make sure x is a column vector with proper length 
	if len(x.shape) is 1:
		x = np.reshape(x,(-1,1))

	#extract number of columns from x
	ncols = x.shape[1]


	# Check if 'ii' indices passed in for inserting x into larger vector
	if ii is not None:
		x0 = np.zeros([len(ii),ncols])
		x0[np.asarray(ii) == 1] = x
		x = x0
	
	#extra, now potentially larger, number of rows from x	
	nrows = x.shape[0]
	

	#multiply number of rows in each matrix up
	vec_length = reduce(lambda x, y: x*y, [j.shape[1] for j in Amats]) 

	#make sure the number of rows in x matches with the correct dimensions of the matrices
	if vec_length != nrows:
		 warnings.warn('x is not the correct length!')
		 print('is', vec_length, 'should be', nrows)


	# Number of matrices
	nA = len(Amats)

	if nA is 1:
	    # If only 1 matrix, standard matrix multiply
	    y = np.matmul(Amats[0],x)
	else:
	    # Perform remaining matrix multiplies
	    y = x # initialize y with x
	    for jj in np.arange(nA):
	        [ni,nj] = np.shape(Amats[jj]) 
	        y =  np.matmul(Amats[jj],np.reshape(y,(nj,-1), order = 'F')) #reshape & multiply 
	        y =  np.transpose(np.reshape(y,(ni,int(nrows/nj),-1)), (1, 0, 2)) # send cols to 3rd dim & permute
	        nrows = int(ni*nrows/nj) # update number of rows after matrix multiply
	
	    
	    # reshape to column vector
	    y = np.reshape(y,(nrows,ncols), order = 'F')

	
	return y