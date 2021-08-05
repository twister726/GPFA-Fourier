# -*- coding: utf-8 -*-

#%% Import
import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import circulant, toeplitz, dft, inv
from numpy.linalg import det

np.set_printoptions(suppress=True)

#%% Plot functions
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
    
def plotvec(vec, name=''):
    plt.plot(range(len(vec)), vec, label=name)
    plt.legend()
    plt.show()
        
#%% Kernel
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    X1: (m x d).
    X2: (n x d).

    Returns (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

#%% Draw data from GP

noise_coeff = 0.8
# noise_coeff = 1e-8

# X = np.arange(-5, 5, 0.2).reshape(-1, 1)
X = np.arange(0, 200).reshape((-1,1))

mu = np.zeros(X.shape)
cov = kernel(X, X, l=50, sigma_f=10) + noise_coeff**2 * np.identity(len(X))

drawn_samples = np.random.multivariate_normal(mu.ravel(),
                                              cov,
                                              size=3)

plot(mu, cov, X, samples=drawn_samples)

#%% Posterior
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

#%% Generate training data

WIDTH = 1.0

# Noisy training data
# X_train = np.arange(-3, 8, WIDTH).reshape(-1, 1)
# X_train = np.arange(0, 35, 1.0).reshape(-1, 1)
# X_train = np.arange(-3, 4, 1).reshape(-1, 1)
# X_train = np.arange(-5, 5, 0.2).reshape(-1, 1)
# X_train = np.arange(0, 57).reshape((-1,1)).astype(float)
# Y_train = np.sin(X_train) + noise_coeff * np.random.randn(*X_train.shape)
X_train = X.copy()
Y_train = drawn_samples[0].reshape((-1,1))

print("xtrain shape: ", X_train.shape)
# print('xtrain: ', X_train)
print("ytrain shape: ", Y_train.shape)
plt.plot(X_train, Y_train)
plt.title('training data')
plt.show()

# X_train = X
# Y_train = samples[0]

# Mean and covariance of the posterior distribution
mu_train, cov_train = posterior(X, X_train, Y_train, l=2.0, sigma_f=2.0, sigma_y=noise_coeff)

samples = np.random.multivariate_normal(mu_train.ravel(), cov_train, 3)
plot(mu_train, cov_train, X, X_train=X_train, Y_train=Y_train, samples=samples)
print('samples0 shape: ', samples[0].shape)
print('mu_train shape: ', mu_train.shape)

#%% Fourier functions
def expand_matrix(C, ncirc):
    """
    Expand C into a circulant matrix
    """
    row1 = C[0, :]
    row1 = np.append(row1, np.flip(row1[:ncirc]))
    return circulant(row1).T

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
r = np.array([1,2,3])
a = toeplitz(r).T
print(a)
print(expand_matrix(a, 2))

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
        ncos = np.ceil((nn+1)/2)
        B[ncos,:] = B[ncos,:] / np.sqrt(2.0)
        
    # print()
    return B

#%% Log Likelihood
def ll_fn(X_train, Y_train, noise=None, which='normal', debug=True):

    def ll(theta):
        if noise != None:
            K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise**2 * np.eye(len(X_train))
        else:
            K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + theta[2]**2 * np.eye(len(X_train))
        # print('det K: ', det(K))

        temp1 = 0.5 * np.log(np.linalg.det(K))
        # print('ytrain shape: ', Y_train.shape)
        # print('Kshape: ', np.linalg.inv(K).shape)
        temp2 = 0.5 * Y_train.T.dot(np.linalg.inv(K).dot(Y_train)).squeeze()
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
        ncirc = int((length * 4.0) / WIDTH)

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
        ncirc = int((length * 4.0) / WIDTH)
        nx = len(X_train)

        # Expand covariance and targets
        Kexp = expand_matrix(K, ncirc)
        nxcirc = len(Kexp)
        
        Y_trainexp = np.append(Y_train, np.zeros((ncirc,))).reshape((-1,1))
        
        maxfreq = np.floor(nxcirc/(np.pi*length)*np.sqrt(0.5*np.log(condthresh)))
        
        # Assuming maxfreq < nxcirc/2
        
        wvec = np.append(np.arange(0, maxfreq+1), np.arange(-maxfreq, -1+1))
        wvecsq = wvec**2
        
        # Get fourier basis and transform targets to Fourier domain
        Bfft = realfftbasis(nx, nxcirc, wvec)
        yf = Bfft @ Y_train
        # yf = (Bfft @ Y_train)/nsevar
        
        # mkcovdiag
        const = (2*np.pi/nxcirc)**2
        wwnrm = wvecsq*const
        # cdiag = np.sqrt(2*np.pi)*rho*length*np.exp(-0.5*ww*length**2)
        # cdiag = np.sqrt(2*np.pi)*rho*length*np.exp(-0.5*ww*(length**2))
        # cdiag = np.sqrt(2*np.pi)*rho*np.exp(-0.5*ww*(length**2))
        # TODO - add noise variance 
        
        # wwnrm = ((2*np.pi/nxcirc)**2) * (wvec**2)
        
        wwthresh = 2*np.log(condthresh) / (length**2)
        
        ii = np.where(wwnrm < wwthresh)
        ni = np.sum(ii)
        cdiag = np.sqrt(2*np.pi)*rho*np.exp(-0.5*wwnrm[ii]*(length**2)) + nsevar
        
        term1 = 0.5 * np.sum(np.log(cdiag))
        term2 = 0.5 * np.sum((1/cdiag).dot(yf[ii] ** 2))
        # term2 = 0.0
        term3 = 0.5 * len(X_train) * np.log(2*np.pi)
        
        if debug:
            print(which)
            print('---------------------------')
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

#%% Check LL values

theta = np.array([10.0, 10.0])
ans_normal = ll_fn(X_train, Y_train, noise_coeff, which='normal')(theta)
ans_fourier = ll_fn(X_train, Y_train, noise_coeff, which='fourier')(theta)
ans_fourier_implicit = ll_fn(X_train, Y_train, noise_coeff, which='implicit')(theta)
print('ans normal: ', ans_normal)
print('ans fourier: ', ans_fourier)
print('ans fourier implicit: ', ans_fourier_implicit)

#%% Normal LL param recovery without noise

res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='normal', debug=False),
                [10, 5.0], 
                bounds=((1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

# res = minimize(ll_fn(X_train, Y_train, noise=None, which='normal', debug=False),
#                 [2.0, 2.0, 0.05], 
#                 bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
#                 method='L-BFGS-B')

l_opt, sigma_f_opt = res.x
# l_opt, sigma_f_opt, noise_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)
# print('noise_opt: ', noise_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
# mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)

#%% Fourier LL param recovery

# res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='normal', debug=False), 
#                 [5, 5], 
#                 bounds=((1e-5, None), (1e-5, None)),
#                 method='L-BFGS-B')

res = minimize(ll_fn(X_train, Y_train, noise=None, which='fourier', debug=False), 
                [5, 5, 0.1], 
                bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

# l_opt, sigma_f_opt = res.x
l_opt, sigma_f_opt, noise_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)
print('noise_opt: ', noise_opt)

# Plot the results using optimized parameters
# mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)


#%% Normal LL param recovery with noise

# res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='normal', debug=False), 
#                 [5, 5], 
#                 bounds=((1e-5, None), (1e-5, None)),
#                 method='L-BFGS-B')

res = minimize(ll_fn(X_train, Y_train, noise=None, which='normal', debug=False), 
                [5, 5, 0.1], 
                bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

# l_opt, sigma_f_opt = res.x
l_opt, sigma_f_opt, noise_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)
print('noise_opt: ', noise_opt)

# Plot the results using optimized parameters
# mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)

#%% Implicit Fourier LL param recovery without noise

res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='implicit', debug=False), 
                [8, 8], 
                bounds=((1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

l_opt, sigma_f_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
plot(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)

#%% Implicit Fourier LL param recovery with noise

res = minimize(ll_fn(X_train, Y_train, noise=None, which='implicit', debug=False), 
                [8, 8, 0.1], 
                bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

l_opt, sigma_f_opt, noise_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)
print('noise_opt: ', noise_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)

#%% Implicit Fourier estimation accuracy wrt number of observations



#%% GPy experiments

rbf = GPy.kern.RBF(input_dim=1, variance=5.0, lengthscale=5.0)
# gpr = GPy.models.GPRegression(X_train, Y_train, rbf)
gpr = GPy.models.GPRegression(X_train, Y_train, rbf)

# Fix the noise variance to known value 
# gpr.Gaussian_noise.variance = noise_coeff**2
gpr.Gaussian_noise.variance = 0.05
# gpr.Gaussian_noise.variance.fix()

# Run optimization
gpr.optimize();

# Obtain optimized kernel parameters
l = gpr.rbf.lengthscale.values[0]
sigma_f = np.sqrt(gpr.rbf.variance.values[0])
gprnoise = gpr.Gaussian_noise.variance.values[0]
print('l: ', l)
print('sigma_f: ', sigma_f)
print('gprnoise: ', gprnoise)

# Compare with previous results
# assert(np.isclose(l_opt, l))
# assert(np.isclose(sigma_f_opt, sigma_f))

# Plot the results with the built-in plot function
gpr.plot();

#%% Do this multiple times

l_opts = []
sigma_f_opts = []

for i in range(20):
    X_train_orig = np.arange(-3, 4, 0.2).reshape(-1, 1)
    # X_train = np.arange(-5, 5, 0.2).reshape(-1, 1)
    Y_train_orig = np.sin(X_train_orig) + noise_coeff * np.random.randn(*X_train_orig.shape)

    mu_s, cov_s = posterior(X, X_train_orig, Y_train_orig, sigma_y=noise_coeff)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)

    X_train = X
    Y_train = samples[0].reshape((-1,1))

    res = minimize(ll_fn(X_train, Y_train, 0.0, fourier=False), [2.0, 2.0], 
               bounds=((1e-5, None), (1e-5, None)),
               method='BFGS')
    
    l_opt, sigma_f_opt = res.x
    l_opts.append(l_opt)
    sigma_f_opts.append(sigma_f_opt)
    
#%%
plt.scatter(list(range(len(l_opts))), l_opts)
plt.plot([0, 30], [1, 1])
plt.show()
plt.plot([0, 30], [1, 1])
plt.scatter(list(range(len(sigma_f_opts))), sigma_f_opts)
plt.show()