#!/usr/bin/env python
# coding: utf-8

# Import stuff
import numpy as np
from matplotlib import pyplot as plt
import george
from george import kernels
import emcee
import scipy.optimize as op


# ## Define all functions
def get_gp(profile):
    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    
    # And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)
    
    kernel = 1.0 * kernels.ExpSquaredKernel(metric=10)
    y = profile
    noise_start = np.std(profile[0:50])
    #    print('noise start: ', noise_start)
    t0 = np.arange(len(profile))
    t = t0
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
                   white_noise=np.log(noise_start), fit_white_noise=True)
    # You need to compute the GP once before starting the optimization.
    gp.compute(t)
    
    # Print the initial ln-likelihood.
    #    print(gp.log_likelihood(y))

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    
    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    return gp

def get_boundaries(mu, noise, snr):
    bins = len(mu)
    left = bins
    right = 0
    index = 0
    
    #    print('noise: ', noise)
    #    baseline = np.min(mu)
    baseline = 0.0
    while left == bins and index < bins:
        if mu[index] - baseline > snr * noise and \
           mu[index+1] - baseline > snr * noise and \
           mu[index+2] - baseline > snr * noise and \
           mu[index+3] - baseline > snr * noise:
            left = index
        index += 1
    index = 1
    while right == 0 and index < bins:
        if mu[-index] - baseline > snr * noise and\
           mu[-(index+1)] - baseline > snr * noise and\
           mu[-(index+2)] - baseline > snr * noise and\
           mu[-(index+3)] - baseline > snr * noise:
            right = bins - index
        index += 1

    return left, right

def get_wX(mu, rms, X):
    bins = len(mu)
    #baseline = np.min(mu)
    baseline = 0.0
    mu = mu - baseline
    peak = np.max(mu)
    wX_level = X*peak /100.
    wX_level1sp = wX_level + rms
    wX_level1sn = wX_level - rms
    #   print ('Wx report: ',wX_level, wX_level1sp, wX_level1sn, peak) 
    left = bins
    left1p = bins
    left1n = bins
    right = 0
    right1p = 0
    right1n = 0
    index = 0
    while left == bins and index < bins:
        if mu[index] > wX_level:
            left = index
        index += 1
    index = 0
    while left1p == bins and index < bins:
        if mu[index] > wX_level1sp:
            left1p = index
        index += 1
    index = 0
    while left1n == bins and index < bins:
        if mu[index] > wX_level1sn:
            left1n = index
        index += 1
    index = 1
    while right == 0 and index < bins:
        if mu[-index] > wX_level:
            right = bins - index
        index += 1
    index = 0
    while right1p == 0 and index < bins:
        if mu[-index] > wX_level1sp:
            right1p = bins -index
        index += 1
    index = 0
    while right1n == 0 and index < bins:
        if mu[-index] > wX_level1sn:
            right1n = bins - index
        index += 1


    wX = right - left + 1
    wXp = right1n - left1n - wX + 1
    wXn = right1p - left1p - wX + 1
    return wX, wXp, wXn
     
# ## Define the main callable function pulseshape(1D_data_array)
def pulseshape(data,mysnr,mcmc):
    originalbins = len(data)
    # roll the profile to bring the peak to the centre
    binmax = np.argmax(data)
    profile = np.roll(data, np.int(originalbins/2 - binmax))
    noise_start = np.std(profile[0:50])
    profile = profile/noise_start
    margin = np.int(originalbins/10)
    # in units of SNR from here
    
    # ## Start the GP 

    gp1 = get_gp(profile)

    # ## Find a window of data around the pulse and produce noiseless data
    mu_b, var_b = gp1.predict(profile, np.arange(len(profile)), return_var=True)
    #    print('First GP')
    noise = np.sqrt(np.exp(gp1.get_parameter_vector()[1]))
    left, right = get_boundaries(mu_b, noise, mysnr)
    #    print('SNR boundaries found: ', left, right, noise)
    if right == 0 and left == originalbins:
        return 0,margin,originalbins-margin,np.zeros((originalbins))
    bins = right + margin - (left - margin) 
    #    print('boundaries: ', left, right)
    if left > margin and right < originalbins-margin:
        base1 = np.int(left/2)
        base2 = np.int(right + 0.5*(originalbins-right)) 
        baseline = 0.5 * (np.mean(profile[0:base1]) + np.mean(profile[base2:]))
        std_baseline = 0.5 * (np.std(profile[0:base1]) + np.std(profile[base2:]))
    else:
        baseline = 0
        
    # ## Return the standard deviation of the noise, the left and
    # ## right pulse boundaries referring to the input array, and the
    # ## noiseless profile 
    leftout = left - (originalbins/2 - binmax)
    rightout = right - (originalbins/2 - binmax)
    profileout = np.roll(mu_b, -np.int(originalbins/2 - binmax))
    if not mcmc:
        return noise*noise_start, leftout, rightout, profileout*noise_start

    # Get a more accurate noiseless profile with mcmc for hyperparameter exploration
    #    print('Off pulse baseline to subtract is: ', baseline, std_baseline)
    yn = profile[left-min(left,margin):min(right+margin,len(profile))] #- baseline
    gp_last = get_gp(yn)
    t = np.arange(len(yn))
    mu, var = gp_last.predict(yn, t, return_var=True)
    #    print('Second GP')
    noise = np.sqrt(np.exp(gp_last.get_parameter_vector()[1]))
    #    print('GP Noise standard deviation: ', noise)

    def lnprob(p):
        # Trivial uniform prior. gp_last is a global
        if np.any((-100 > p[1:]) + (p[1:] > 100)):
                return -np.inf
        
        # Update the kernel and compute the lnlikelihood.
        gp_last.set_parameter_vector(p)
        return gp_last.lnlikelihood(yn, quiet=True)


    # Set up the sampler.
    nwalkers, ndim = 36, len(gp_last)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    
    # Initialize the walkers.
    p0 = gp_last.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)
    
    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 200)
    
    print("Running production chain")
    sampler.run_mcmc(p0, 200)
    nsamples = 50
    allsamples  = np.zeros((nsamples, len(yn)))
    for i in range(nsamples):
        # Choose a random walker and step.
        w = np.random.randint(sampler.chain.shape[0])
        n = np.random.randint(sampler.chain.shape[1])
        gp_last.set_parameter_vector(sampler.chain[w, n])
        
        # Plot a single sample.
        # plt.plot(t, gp_last.sample_conditional(yn, t), "g", alpha=0.1)
        allsamples[i,:] = gp_last.sample_conditional(yn, t)
    noiseless=np.mean(allsamples,0)
    mu_b[left-min(left,margin):min(right+margin,len(profile))] = noiseless
    profileout = np.roll(mu_b, -np.int(originalbins/2 - binmax))
    return noise*noise_start, leftout, rightout, profileout*noise_start
#    plt.plot(t, yn, ".k")
#    plt.plot(t, noiseless, "-b")
#    plt.show()
def getflux(profile,left,right):
    flux = np.sum(profile[left:right])
    return flux

