# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:56:28 2021

@author: Sourabh
"""
#%% Imports

import numpy as np
import GPy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import circulant, toeplitz, dft, inv
from numpy.linalg import det
import pickle

from collections import defaultdict
import time

from gen_data import drawdatafromgp
from log_likelihood import ll_fn
from plot_functions import plot, plotvec
from gp_functions import posterior
import gpconfig

np.set_printoptions(suppress=True)

#%% Get training data

num_obs = 1000
noise_coeff = gpconfig.noise_coeff

X_train, Y_train = drawdatafromgp(num_obs)

#%% Check LL values

theta = np.array([10.0, 10.0])
ans_normal = ll_fn(X_train, Y_train, noise_coeff, which='normal')(theta)
# ans_fourier = ll_fn(X_train, Y_train, noise_coeff, which='fourier')(theta)
ans_fourier_implicit = ll_fn(X_train, Y_train, noise_coeff, which='implicit')(theta)
print('ans normal: ', ans_normal)
# print('ans fourier: ', ans_fourier)
print('ans fourier implicit: ', ans_fourier_implicit)

#%% Normal LL param recovery without noise

res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='normal', debug=False),
                [10, 5.0], 
                bounds=((1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

l_opt, sigma_f_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

#%% Normal LL param recovery with noise

res = minimize(ll_fn(X_train, Y_train, noise=None, which='normal', debug=False), 
                [10, 5, 0.1], 
                bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

l_opt, sigma_f_opt, noise_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)
print('noise_opt: ', noise_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

#%% Implicit Fourier LL param recovery without noise

res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='implicit', debug=False), 
                [10, 5], 
                bounds=((1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

l_opt, sigma_f_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

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
mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

#%% temp

mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

#%% Fourier LL param recovery without noise

res = minimize(ll_fn(X_train, Y_train, noise=noise_coeff, which='fourier', debug=False), 
                [5, 5, 0.1], 
                bounds=((1e-5, None), (1e-5, None)),
                method='L-BFGS-B')

l_opt, sigma_f_opt = res.x
print('l_opt: ', l_opt)
print('sigma_f_opt: ', sigma_f_opt)

# Plot the results using optimized parameters
mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

#%% Fourier LL param recovery with noise

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
mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
plot(mu_s, cov_s, X_train, X_train=X_train, Y_train=Y_train)

#%% Test the code wrt number of observations

# List of sizes of training data
min_n = 50
max_n = 1000
step = 30
trials = np.arange(min_n, max_n + 1, step)

# names = ['ll_normal_nonoise', 'll_normal_noise', 'll_implicit_nonoise', 'll_implicit_noise']
# inputs = [(noise_coeff, 'normal'), (None, 'normal'), (noise_coeff, 'implicit'),
#           (None, 'implicit')]
names = ['ll_implicit_nonoise', 'll_implicit_noise']
inputs = [(noise_coeff, 'implicit'),
          (None, 'implicit')]
outputs = {name: defaultdict(list) for name in names}

initialparams_nonoise = [8, 8]
initialparams_noise = [8, 8, 0.1]
bounds_nonoise = ((1e-5, None), (1e-5, None))
bounds_noise = ((1e-5, None), (1e-5, None), (1e-5, None))

for num_obs in trials:
    X_train, Y_train = drawdatafromgp(num_obs)
    
    for name_idx in range(len(names)):
        name = names[name_idx]
        inp = inputs[name_idx]
        nonoise = name_idx % 2 == 0
        
        # Choose initial search params and bounds based on if solving for noise
        if nonoise:
            initialparams = initialparams_nonoise
            bounds = bounds_nonoise
        else:
            initialparams = initialparams_noise
            bounds = bounds_noise
            
        # LL function
        ll = ll_fn(X_train, Y_train, noise=inp[0], which=inp[1], debug=False)
        
        # Conduct the optimization
        start_time = time.time()
        res = minimize(ll, 
                initialparams, 
                bounds=bounds,
                method='L-BFGS-B')
        comp_time = time.time() - start_time
        
        if nonoise:
            l_opt, sigma_f_opt = res.x
            mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
        else:
            l_opt, sigma_f_opt, noise_opt = res.x    
            mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
            
        # Store and print the outputs to be measured
        avgpostvar = np.mean(np.diag(cov_s))
        outputs[name]['avgpostvar'].append(avgpostvar)
        print('avgpostvar: ', avgpostvar)
            
        ldiff = np.abs(l_opt - gpconfig.l)
        outputs[name]['ldiff'].append(ldiff)
        print('ldiff: ', ldiff)
        
        sigmafdiff = np.abs(sigma_f_opt - gpconfig.sigma_f)
        outputs[name]['sigmafdiff'].append(sigmafdiff)
        print('sigmafdiff: ', sigmafdiff)
        
        if not nonoise:
            noisediff = np.abs(noise_opt - noise_coeff)
            outputs[name]['noisediff'].append(noisediff)
            print('noisediff: ', noisediff)
            
            llvalue_opt = ll([l_opt, sigma_f_opt, noise_opt])
        else:
            llvalue_opt = ll([l_opt, sigma_f_opt])
            
        outputs[name]['llvalueopt'].append(llvalue_opt)
        print('llvalueopt: ', llvalue_opt)
        
        outputs[name]['comptime'].append(comp_time)
        print('comptime: ', comp_time)
        
        print('{}: Finished {}. Last is {}'.format(name, num_obs, max_n))
        print()
        
#%% Test the code wrt number of observations multiple times per trial

# List of sizes of training data
min_n = 50
max_n = 1000
step = 30
repeats = 15
trials = np.arange(min_n, max_n + 1, step)

# names = ['ll_normal_nonoise', 'll_normal_noise', 'll_implicit_nonoise', 'll_implicit_noise']
# inputs = [(noise_coeff, 'normal'), (None, 'normal'), (noise_coeff, 'implicit'),
#           (None, 'implicit')]
names = ['ll_implicit_nonoise', 'll_implicit_noise']
inputs = [(noise_coeff, 'implicit'),
          (None, 'implicit')]
outputs = {name: defaultdict(list) for name in names}

initialparams_nonoise = [8, 8]
initialparams_noise = [8, 8, 0.1]
bounds_nonoise = ((1e-5, None), (1e-5, None))
bounds_noise = ((1e-5, None), (1e-5, None), (1e-5, None))

for num_obs in trials:
    X_train, Y_train = drawdatafromgp(num_obs)
    
    for name_idx in range(len(names)):
        name = names[name_idx]
        inp = inputs[name_idx]
        nonoise = name_idx % 2 == 0
        
        # Choose initial search params and bounds based on if solving for noise
        if nonoise:
            initialparams = initialparams_nonoise
            bounds = bounds_nonoise
        else:
            initialparams = initialparams_noise
            bounds = bounds_noise
            
        # LL function
        ll = ll_fn(X_train, Y_train, noise=inp[0], which=inp[1], debug=False)
        
        # Conduct the optimization
        start_time = time.time()
        res = minimize(ll, 
                initialparams, 
                bounds=bounds,
                method='L-BFGS-B')
        comp_time = time.time() - start_time
        
        if nonoise:
            l_opt, sigma_f_opt = res.x
            mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_coeff)
        else:
            l_opt, sigma_f_opt, noise_opt = res.x    
            mu_s, cov_s = posterior(X_train, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_opt)
            
        avgpostvar = np.mean(np.diag(cov_s))
        outputs[name]['avgpostvar'].append(avgpostvar)
        print('avgpostvar: ', avgpostvar)
        
        # Store and print the outputs to be measured
        ldiff = np.abs(l_opt - gpconfig.l)
        outputs[name]['ldiff'].append(ldiff)
        print('ldiff: ', ldiff)
        
        sigmafdiff = np.abs(sigma_f_opt - gpconfig.sigma_f)
        outputs[name]['sigmafdiff'].append(sigmafdiff)
        print('sigmafdiff: ', sigmafdiff)
        
        if not nonoise:
            noisediff = np.abs(noise_opt - noise_coeff)
            outputs[name]['noisediff'].append(noisediff)
            print('noisediff: ', noisediff)
            
            llvalue_opt = ll([l_opt, sigma_f_opt, noise_opt])
        else:
            llvalue_opt = ll([l_opt, sigma_f_opt])
            
        outputs[name]['llvalueopt'].append(llvalue_opt)
        print('llvalueopt: ', llvalue_opt)
        
        outputs[name]['comptime'].append(comp_time)
        print('comptime: ', comp_time)
        
        print('{}: Finished {}. Last is {}'.format(name, num_obs, max_n))
        print()

#%% Plot results

# print('outputs: ', list(outputs['ll_normal_nonoise'].keys()))

for name_idx in range(len(names)):
    name = names[name_idx]
    nonoise = name_idx % 2 == 0
    
    plotvec(outputs[name]['ldiff'], name + ' ldiff', xvalues=trials)
    plotvec(outputs[name]['sigmafdiff'], name + ' sigmafdiff', xvalues=trials)
    if not nonoise:
        plotvec(outputs[name]['noisediff'], name + ' noisediff', xvalues=trials)
    plotvec(outputs[name]['avgpostvar'], name + ' avgpostvar', xvalues=trials)
    plotvec(outputs[name]['comptime'], name + ' comptime', xvalues=trials)
    plotvec(outputs[name]['llvalueopt'], name + ' llvalueopt', xvalues=trials)
    
    # plt.figure()
    # plt.title(name)
    # plt.plot(trials, outputs[name]['ldiff'], label='ldiff')
    # plt.plot(trials, outputs[name]['sigma_f_diff'], label='sigma_f_diff')
    # if not nonoise:
    #     plt.plot(trials, outputs[name]['noisediff'], label='noisediff')
    # plt.plot(trials, outputs[name]['comptime'], label='comptime')
    # plt.plot(trials, outputs[name]['llvalueopt'], label='llvalueopt')
    # plt.legend()
    # plt.show()
    
#%% Save output results

run_no = 5
fname = 'outputplots_run_' + str(run_no) + '.pkl'

pickle.dump(outputs, open(fname, 'wb'))
print('Saved outputs to ' + fname)

#%% Load saved output

run_no = 3
fname = 'outputplots_run_' + str(run_no) + '.pkl'

outputs = pickle.load(open(fname, 'rb'))
print('Loaded outputs from ' + fname)

print('outputs new: ', len(outputs['ll_normal_nonoise']['sigmafdiff']))