# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:49:50 2021

@author: Sourabh
"""

# import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.linalg as slinalg
from autograd import grad
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from gen_data import drawdatafromgp, drawmatrixfromgp
from gp_functions import kernel
from plot_functions import plotmat, plotvec
from scipy.optimize import minimize, approx_fprime
import time
from misc_functions import Timer, vec, vecinv, commmat
import torch
from torch.autograd import Variable

#%% Generate data

p = 5
q = 10
T = 20

#nsevar = 0.0
nsevar = 0.1
sigma_f = 1 - nsevar**2

def genlambdas():
    # return np.random.randint(1, 20, (q,))
    return nsevar * np.random.exponential(1, (q,))

lambdas_orig = genlambdas()
# lambdas_orig = np.ones((q,))
L_orig = np.diag(lambdas_orig)


timesteps = np.arange(0, T).reshape((-1,1))
#timescales = np.random.randint(1, 100, (p,))
timescales = np.random.rand(p)*50.0
# kernels = [kernel(timesteps, timesteps, l=ts, sigma_f=sigma_f) + nsevar**2 * np.identity(T)
#            for ts in timescales]
kernels = [np.eye(T) for ts in timescales]
invkernels = [np.linalg.inv(ker) for ker in kernels]
K = slinalg.block_diag(*kernels)
Kinv = slinalg.block_diag(*invkernels)

X = drawmatrixfromgp(T, p, kernels)
x = vec(np.transpose(X))
xquad = np.dot(np.dot(np.transpose(x), K), x).item()

def genC():
    # return np.random.randint(1, 10, (q, p)).astype(float)
    return np.random.randn(q, p)

def gend():
    # return np.random.randint(1, 20, (q, 1)).astype(float)
    return np.random.randn(q,1)

C_orig = genC()
d_orig = gend()

E = np.random.multivariate_normal(np.zeros((q,)), np.diag(1.0/lambdas_orig),
                                   size=T).T

ones = np.ones((T, 1))
Y = np.dot(C_orig, X) + E + np.transpose(np.dot(ones, np.transpose(d_orig)))
y = vec(np.transpose(Y))

print('X shape: ', X.shape)
print('x shape: ', x.shape)
print('C_orig shape: ', C_orig.shape)
print('Y shape: ', Y.shape)
print('y shape: ', y.shape)
print('L_orig shape: ', L_orig.shape)
print('E shape: ', E.shape)
print('K shape: ', K.shape)
print('K cond: ', np.linalg.cond(K))
print('Kinv cond: ', np.linalg.cond(Kinv))
for i in range(len(kernels)):
    plotmat(kernels[i], 'kernel ' + str(i))
    print('kernel {} cond: {}'.format(i, np.linalg.cond(kernels[i])))
    
plotmat(K, 'K')
plotmat(Kinv, 'Kinv')

testinds = [1, 2, 3]
for i in testinds:
    plt.figure()
    plt.plot(X[i,:])
    plt.title('X dim ' + str(i))
    plt.show()
    
testinds = [0, 3, 4]
for i in testinds:
    plt.figure()
    plt.plot(E[:,i])
    plt.title('E dim ' + str(i))
    plt.show()

testinds = [0, 3, 4]
for i in testinds:
    plt.figure()
    plt.plot(Y[i,:])
    plt.title('Y dim ' + str(i))
    plt.show()
    
cm = commmat(T,p)
print('cm shape: ', cm.shape)
print(np.all(cm @ x == vec(X)))

def params2theta(C, d, lambdas=np.array([])):
    return np.concatenate([np.reshape(C, (-1,)), np.reshape(d, (-1,)), lambdas])

def theta2params(theta):
    C = np.reshape(theta[:q*p], (q, p))
    if len(theta) > q*p + q:
        d = np.reshape(theta[q*p: q*p + q], (-1,1))
        lambdas = theta[q*p + q:]
        return C, d, lambdas
    else:
        d = np.reshape(theta[q*p:], (-1,1))
        return C, d

#%% Define log likelihoods

def ll_fn(debug=True, which='joint'):

    def ll(theta):
        C = np.reshape(theta[:q*p], (q, p))
        lambdas = np.reshape(theta[q*p: q*p + q], (q,))
        d = np.reshape(theta[q*p + q:], (q,1))
        
        L = np.diag(lambdas)
        
        with Timer('term1', debug):
            term1 = -T*np.sum(np.log(lambdas))
            # term1 = -T*np.linalg.slogdet(L)[1]
        
        with Timer('term2', debug):
            term2 = 0.0
            for k in kernels:
                sgn, val = np.linalg.slogdet(k)
                term2 = term2 + sgn*val
        
        with Timer('term3', debug):
            term3 = np.trace(np.matmul(np.matmul(Y, np.transpose(Y)), L))
        
        # These are actually for the ELL
        
        # with Timer('term4'):
        #     temp1 = np.reshape(np.ravel(np.dot(np.dot(np.transpose(Y),L),C)), (-1,1))
        #     sigma = np.linalg.inv(Kinv + np.kron(np.dot(np.dot(np.transpose(C),L),C),np.eye(T)))
        #     term4 = np.squeeze(np.dot(np.dot(np.transpose(temp1),sigma),temp1))
        
        # temp = np.arange(p*T)
        # bytime = np.ravel(np.reshape(temp, (p, T)))
        # bylatent = np.ravel(np.transpose(np.reshape(temp, (p, T))))
        
        # sigmaprime = sigma[bylatent,:][:,bylatent]
        
        with Timer('term4', debug):
            term4 = -2.0 * np.trace(np.dot(np.dot(np.dot(np.transpose(X), np.transpose(C)), L), Y))
        
        # with Timer('term5'):
        #     term5 = 0.0
        #     for t in range(T):
        #         term5 = term5 + np.trace(np.dot(np.dot(np.dot(np.transpose(C), L), C), sigmaprime[p*t : p*(t+1), p*t: p*(t+1)]))
        
        with Timer('term5', debug):
            term5 = np.trace(np.dot(np.transpose(X), np.dot(np.transpose(C), np.dot(L, np.dot(C, X)))))
            
        tempd = np.dot(ones, np.transpose(d))
        with Timer('term6', debug):
            term6 = -2*np.trace(np.dot(Y, np.dot(tempd, L)))
            
        with Timer('term7', debug):
            term7 = -2*np.trace(np.dot(X, np.dot(tempd, np.dot(L, C))))
            
        with Timer('term8', debug):
            term8 = np.trace(np.dot(d, np.dot(np.transpose(d), np.transpose(L))))
            
        term9 = xquad
        
        if debug:
            print('term1: {}'.format(term1))
            print('term2: {}'.format(term2))
            print('term3: {}'.format(term3))
            # print('sigma shape: ', sigma.shape)
            print('term4: {}'.format(term4))
            # print('sigmaprime shape ', sigmaprime.shape)
            print('term5: {}'.format(term5))
            print('term6: {}'.format(term6))
            print('term7: {}'.format(term7))
            print('term8: {}'.format(term8))
            print('term9: {}'.format(term9))
        
        return (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)
    
    def ll_y1(theta):
        C = np.reshape(theta[:q*p], (q, p))
        lambdas = np.reshape(theta[q*p: q*p + q], (q,))
        d = np.reshape(theta[q*p + q:], (q,1))
        
        L = np.diag(lambdas)
        Linv = np.diag(1.0/lambdas)
        
        eyeT = np.eye(T)
        
        temp1 = np.dot(K, np.kron(np.transpose(C), eyeT))
        sigma = np.kron(Linv, eyeT) + np.dot(np.kron(C, eyeT), temp1)
        sgn, val = np.linalg.slogdet(sigma)
        term1 = sgn * val
        
        invsigma = np.linalg.inv(sigma)
        
        
        ymean = y - np.dot(np.kron(np.eye(q), ones), d)
        
        term2 = np.dot(np.dot(np.transpose(ymean), invsigma), ymean)[0][0]
        
        # print('term2 shape: ' , term2.shape)
        
        return term1 + term2
    
    def ll_y2(theta):
        C = np.reshape(theta[:q*p], (q, p))
        d = np.reshape(theta[q*p: q*p + q], (q,1))
        lambdas = np.reshape(theta[q*p + q:], (q,))
        
        L = np.diag(lambdas)
        Linv = np.diag(1.0/lambdas)
        
        eyeT = np.eye(T)
        
        temp1 = np.dot(K, np.kron(np.transpose(C), eyeT))
        sigma = np.kron(Linv, eyeT) + np.dot(np.kron(C, eyeT), temp1)
        sgn, val = np.linalg.slogdet(sigma)
        term1 = sgn * val
        
        temp2 = np.linalg.inv(Kinv + np.kron(np.dot(np.dot(np.transpose(C), L),C), eyeT))
        invsigma = np.kron(L, eyeT) - np.kron(np.dot(L,C), eyeT) @ temp2 @ np.kron(np.dot(np.transpose(C), L), eyeT)
        
        
        ymean = y - np.dot(np.kron(np.eye(q), ones), d)
        
        term2 = np.dot(np.dot(np.transpose(ymean), invsigma), ymean)[0][0]
        
        # print('term2 shape: ' , term2.shape)
        
        return term1 + term2
    
    def ll_y3(theta):
        C = np.reshape(theta[:q*p], (q, p))
        d = np.reshape(theta[q*p: q*p + q], (q,1))
        lambdas = np.reshape(theta[q*p + q:], (q,))
        
        L = np.diag(lambdas)
        Linv = np.diag(1.0/lambdas)
        
        eyeT = np.eye(T)
        
        mu = np.kron(np.eye(q), ones) @ d
        mu = mu.reshape((-1,))
        
        print('Linv shape: ', Linv.shape)
        print('eyeT shape: ', eyeT.shape)
        cov1 = np.kron(Linv, eyeT)
        # cov2 = np.kron(C, eyeT) @ K @ np.kron(C.T, eyeT)
        cov2 = np.kron(C @ C.T, eyeT)
        
        cov = cov1 + cov2
            
        print('cov1check: ', cov1[np.where(cov1 != cov1.T)])
        print('cov2check: ', cov2[np.where(cov2 != cov2.T)])
        print('covcheck: ', np.min(np.linalg.eigvals(cov)))
        # print('mu shape: ', mu.shape)
        # print('cov shape: ', cov.shape)

        var = sp.stats.multivariate_normal(mean=mu, cov=cov,
                                           allow_singular=False)
        varvalue = var.pdf(y.reshape((-1,)))
        print('varvalue: ', varvalue)

        return np.log(varvalue)
    
    if which == 'joint':
        return ll
    elif which == 'y1':
        return ll_y1
    elif which == 'y2':
        return ll_y2
    elif which == 'y3':
        return ll_y3
    
    
#%% Directly minimize log likelihood


llfn = ll_fn(which='y2')

print('ll value: ', llfn(params2theta(C_orig, d_orig, lambdas_orig)))

grad_ll = grad(llfn)
# grad_autograd = grad_ll(params2theta(C_orig, d_orig, lambdas_orig))
# grad_scipy = approx_fprime(params2theta(C_orig, d_orig, lambdas_orig),
#                                           ll_fn(debug=False, which='y'), 0.001)
# print('grad value autograd: ', grad_autograd)
# print('grad value scipy: ', grad_scipy)
# print('Diff bw auto and scipy: ', grad_autograd-grad_scipy)

C_init = genC()
lambdas_init = genlambdas()
d_init = gend()
theta_init = params2theta(C_init, d_init, lambdas_init)

print('Starting optimization...')

minimizeResults = []
numIters = 0

def callback(theta):
    global minimizeResults, numIters
    minimizeResults.append(ll_fn(debug=False)(theta))
    numIters += 1

res = minimize(llfn,
                x0=theta_init,
                jac=grad_ll,
                callback=callback,
                method='L-BFGS-B')

C_opt, d_opt, lambdas_opt = theta2params(res.x)
print('C_opt: ', C_opt)
print('lambdas_opt: ', lambdas_opt)
print('d_opt: ', d_opt)
print('lambdas_orig: ', lambdas_orig)
print('minimizeresults: ', minimizeResults)
print('numIters: ', numIters)

# plt.figure()
# plt.title('LL value')
# plt.plot(minimizeResults)
# plt.show()

#%% Closed form EM

num_iters = 10

Cn = genC()
dn = gend()
lambdasn = genlambdas()

margll = ll_fn(debug=False, which='y2')
marglls = []
derrs = []
Cerrs = []
lambdaerrs = []

param_store = []

debug = True

for itr in range(num_iters):
    
    print('Iteration {}...'.format(itr))
    
    """
    Reset
    """
    Co = Cn.copy()
    do = dn.copy()
    lambdaso = lambdasn.copy()
    
    Lo = np.diag(lambdaso)
    
    """
    E step
    """
    ident = np.eye(T)
    sigma = np.linalg.inv(Kinv + np.kron(Co.T @ Lo @ Co, ident))
    ex = sigma @ np.kron(Co.T @ Lo, ident) @ (y - np.kron(np.eye(q), ones) @ do)
    """ A = vecinv(ex) """
#    A = np.concatenate([ex[T*i:T*i + T, 0] for i in range(p)]).reshape((p,T)).T
    A = vecinv(ex, T, p)
    U = A.T
    
    # temp = np.arange(p*T)
    # bytime = np.ravel(np.reshape(temp, (p, T)))
    # bylatent = np.ravel(np.transpose(np.reshape(temp, (p, T))))
    # bytime = np.reshape(np.reshape(temp, (p, T)), order='F')
    # bylatent = np.reshape(np.transpose(np.reshape(temp, (p, T))), order='F')
    # sigmaprime = sigma[bylatent,:][:,bylatent]
    sigmaprime = cm @ sigma @ cm.T
    B = np.zeros((p, p))
    for t in range(T):
        B += sigmaprime[p*t : p*t + p, p*t : p*t + p]
        B += U[:, t] @ U[:, t].T
    
    V = sigma
    
    binv = np.linalg.inv(B)
    uones = U @ ones
    
    
    if debug:
        print('ex shape: ', ex.shape)
        print('sigma shape: ', sigma.shape)
        print('sigma cond: ', np.linalg.cond(sigma))
        print('A shape: ', A.shape)
        print('A cond: ', np.linalg.cond(A))
        print('sigmaprime shape: ', sigmaprime.shape)
        print('B shape: ', B.shape)
        print('B cond: ', np.linalg.cond(B))
        # print('B - Bt: ', B-B.T)
        print('U shape: ', U.shape)
        print('V shape: ', V.shape)
#        print('lambdas: ', lambdas)
    
    """
    M step
    """
    dn = (Y @ (ones - (U.T @ binv @ uones))) / (T - (uones.T @ binv @ uones).item())
    Cn = ((Y@U.T) - dn @ uones.T) @ binv
    # newInvL = (1/T) * (Y@Y.T - 2*C@A.T@Y.T + C@B@C.T - 2*d@ones.T@(Y.T - U.T@C.T)) + d@d.T
    # print('tempshape: ', temp.shape)
    newInvL = (1/T) * (Y@Y.T + (T*dn@dn.T) - Y@ones@dn.T - dn@ones.T@Y.T - Y@U.T@Cn.T + dn@uones.T@Cn.T)
    lambdasn = 1.0/np.diag(newInvL)
#    lambdas = np.diag(np.linalg.inv(newInvL))
    
    if debug:
#        print('newd shape: ', d)
#        print('newC shape: ', C)
#        print('newInvL shape: ', newInvL)
#        print('newlambdas shape: ', lambdas)
        print()
        
    """
    Print marginal likelihood and estimation errors
    """
    llvalue = margll(params2theta(Cn,dn,lambdasn))
    print('Neg Marginal likelihood: ', llvalue)
    marglls.append(llvalue)
    
    derr = np.abs(np.mean(dn - d_orig))
    derrs.append(derr)
    print('d error: ', derr)
    
    Cerr = np.abs(np.mean(Cn@Cn.T - C_orig@C_orig.T))
    Cerrs.append(Cerr)
    print('C error: ', Cerr)
    
    lambdaerr = np.abs(np.mean(lambdasn - lambdas_orig))
    lambdaerrs.append(lambdaerr)
    print('lambda error: ', lambdaerr)
    print()
    
    param_store.append((Cn, dn, lambdasn))
    
    

plt.figure()
# plt.plot(range(1,len(marglls)), marglls[1:])
plt.plot(marglls)
plt.xlabel('Iteration')
plt.title('Neg marg ll vs iter')
plt.show()

plt.figure()
plt.plot(derrs)
plt.xlabel('Iteration')
plt.title('Avg error in d')
plt.show()

plt.figure()
plt.plot(Cerrs)
plt.xlabel('Iteration')
plt.title('Avg error in C')
plt.show()

plt.figure()
plt.plot(lambdaerrs)
plt.xlabel('Iteration')
plt.title('Avg error in lambdas')
plt.show()

#%% Print final params

Cfinal, dfinal, lambdasfinal = param_store[4]
    
print('lambdas orig: ', lambdas_orig)
print('final lambdas: ', lambdasfinal)
print('C orig: ', C_orig)
print('final C: ', Cfinal)
print('d orig: ', d_orig)
print('final d: ', dfinal)

#%% EM with torch autograd

num_iters = 10

C_init = np.random.randint(1, 10, (q, p)).astype(float)
d_init = np.random.randint(1, 20, (q, 1)).astype(float)
lambdas_init = np.random.randint(1, 20, (q,)).astype(float)

C = C_init.copy()
d = d_init.copy()
lambdas = lambdas_init.copy()

margll = ll_fn(debug=False, which='y2')
marglls = []
derrs = []
Cerrs = []
lambdaerrs = []

param_store = []

debug = True

X = torch.tensor(X)
Y = torch.tensor(Y)
yt = torch.tensor(y)
xt = torch.tensor(x)

def ell(Ct, dt, lambdast, A, B, U, V):
    L = torch.diag(lambdast)
    
    term1 = -T*torch.sum(torch.log(lambdast))
    # term1 = -T*np.linalg.slogdet(L)[1]
    
    # term2 = 0.0
    # for k in kernels:
    #     sgn, val = np.linalg.slogdet(k)
    #     term2 = term2 + sgn*val
    
    
    term3 = torch.trace(torch.matmul(torch.matmul(Y, torch.transpose(Y,0,1)), L))
    
    term4 = -2.0 * torch.trace(torch.matmul(torch.matmul(torch.matmul(A, torch.transpose(Ct,0,1)), L), Y))
    
    # with Timer('term5'):
    #     term5 = 0.0
    #     for t in range(T):
    #         term5 = term5 + np.trace(np.dot(np.dot(np.dot(np.transpose(C), L), C), sigmaprime[p*t : p*(t+1), p*t: p*(t+1)]))
    
    # term5 = torch.trace(torch.matmul(torch.transpose(X), torch.matmul(torch.transpose(Ct), torch.matmul(L, torch.matmul(Ct, X)))))
    term5 = torch.trace(torch.matmul(torch.matmul(torch.matmul(torch.transpose(Ct,0,1), L), Ct), B))    
    
    
    tempd = torch.matmul(torch.tensor(ones), torch.transpose(dt,0,1))
    term6 = -2*torch.trace(torch.matmul(Y, torch.matmul(tempd, L)))
        
    term7 = -2*torch.trace(torch.matmul(U, torch.matmul(tempd, torch.matmul(L, Ct))))
        
    term8 = torch.trace(torch.matmul(dt, torch.matmul(torch.transpose(dt,0,1), torch.transpose(L,0,1))))
        
    # term9 = xquad
    
    # if debug:
    #     print('term1: {}'.format(term1))
    #     # print('term2: {}'.format(term2))
    #     print('term3: {}'.format(term3))
    #     # print('sigma shape: ', sigma.shape)
    #     print('term4: {}'.format(term4))
    #     # print('sigmaprime shape ', sigmaprime.shape)
    #     print('term5: {}'.format(term5))
    #     print('term6: {}'.format(term6))
    #     print('term7: {}'.format(term7))
    #     print('term8: {}'.format(term8))
    #     # print('term9: {}'.format(term9))
    
    return (term1 + term3 + term4 + term5 + term6 + term7 + term8)

for itr in range(num_iters):
    
    L = np.diag(lambdas)
    
    """
    Print marginal likelihood and estimation errors
    """
    llvalue = margll(params2theta(C,d,lambdas))
    print('Neg Marginal likelihood: ', llvalue)
    marglls.append(llvalue)
    
    derr = np.abs(np.mean(d - d_orig))
    derrs.append(derr)
    print('d error: ', derr)
    
    Cerr = np.abs(np.mean( - C_orig))
    Cerrs.append(Cerr)
    print('C error: ', Cerr)
    
    lambdaerr = np.abs(np.mean(lambdas - lambdas_orig))
    lambdaerrs.append(lambdaerr)
    print('lambda error: ', lambdaerr)
    
    param_store.append((C, d, lambdas))
    
    
    """
    E step
    """
    ident = np.eye(T)
    sigma = np.linalg.inv(Kinv + np.kron(C.T @ L @ C, ident))
    ex = sigma @ np.kron(C.T @ L, ident) @ (y - np.kron(np.eye(q), ones) @ d)
    """ A = vecinv(ex) """
#    A = np.concatenate([ex[T*i:T*i + T, 0] for i in range(p)]).reshape((p,T)).T
    A = vecinv(ex, T, p)
    
    temp = np.arange(p*T)
    bytime = np.ravel(np.reshape(temp, (p, T)))
    bylatent = np.ravel(np.transpose(np.reshape(temp, (p, T))))
    # bytime = np.reshape(np.reshape(temp, (p, T)), order='F')
    # bylatent = np.reshape(np.transpose(np.reshape(temp, (p, T))), order='F')
    sigmaprime = sigma[bylatent,:][:,bylatent]
    B = np.zeros((p, p))
    for t in range(T):
        B += sigmaprime[p*t : p*t + p, p*t : p*t + p]
        
    U = A.T
    
    V = sigma
    
    
    if debug:
        print('ex shape: ', ex.shape)
        print('sigma shape: ', sigma.shape)
        print('sigma cond: ', np.linalg.cond(sigma))
        print('A shape: ', A.shape)
        print('A cond: ', np.linalg.cond(A))
        print('sigmaprime shape: ', sigmaprime.shape)
        print('B shape: ', B.shape)
        print('B cond: ', np.linalg.cond(B))
        # print('B - Bt: ', B-B.T)
        print('U shape: ', U.shape)
        print('V shape: ', V.shape)
#        print('lambdas: ', lambdas)
    
    """
    M step
    """
    Ct = Variable(torch.tensor(C), requires_grad=True)
    dt = Variable(torch.tensor(d), requires_grad=True)
    lambdast = Variable(torch.tensor(lambdas), requires_grad=True)
    
    print(Ct.requires_grad)
    
    A = torch.tensor(A)
    B = torch.tensor(B)
    U = torch.tensor(U)
    V = torch.tensor(V)
    
    optimizer = torch.optim.Adam([Ct, dt, lambdast])
    
    num_steps = 300
    losses = []
    for s in range(num_steps):
        optimizer.zero_grad()
        loss = ell(Ct, dt, lambdast, A, B, U, V)
        loss.backward()
        losses.append(loss.detach().numpy())
        optimizer.step()
        
    plt.figure()
    plt.plot(losses)
    plt.title('Pytorch ELL loss for iter ' + str(itr))
    plt.show()
    

    C = Ct.detach().numpy()
    d = dt.detach().numpy()
    lambdas = lambdast.detach().numpy()
    
    if debug:
#        print('newd shape: ', d)
#        print('newC shape: ', C)
#        print('newInvL shape: ', newInvL)
#        print('newlambdas shape: ', lambdas)
        print()

plt.figure()
plt.plot(marglls)
plt.xlabel('Iteration')
plt.title('Neg marg ll vs iter')
plt.show()

plt.figure()
plt.plot(derrs)
plt.xlabel('Iteration')
plt.title('Avg error in d')
plt.show()

plt.figure()
plt.plot(Cerrs)
plt.xlabel('Iteration')
plt.title('Avg error in C')
plt.show()

plt.figure()
plt.plot(lambdaerrs)
plt.xlabel('Iteration')
plt.title('Avg error in lambdas')
plt.show()