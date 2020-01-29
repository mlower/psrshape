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
    print('noise start: ', noise_start)
    t = np.arange(len(y))
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(noise_start), fit_white_noise=True)
# You need to compute the GP once before starting the optimization.
    gp.compute(t)

# Print the initial ln-likelihood.
    print(gp.log_likelihood(y))

# Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    return gp

def get_boundaries(profile, gp):
    mu, var = gp.predict(profile, np.arange(len(profile)), return_var=True)
    bins = len(profile)
    left = bins
    right = 0
    index = 0
    noise = np.sqrt(np.exp(gp.get_parameter_vector()[1]))
    print('noise: ', noise)
    baseline = np.mean(profile[1:20])
    while left == bins:
        if mu[index] - baseline > 5.0 * noise        and mu[index+1] - baseline > 5.0 * noise        and mu[index+2] - baseline > 5.0 * noise        and mu[index+3] - baseline > 5.0 * noise:
            left = index
        index += 1
    index = 1
    while right == 0:
        if mu[-index] - baseline > 5.0 * noise        and mu[-(index+1)] - baseline > 5.0 * noise        and mu[-(index+2)] - baseline > 5.0 * noise        and mu[-(index+3)] - baseline > 5.0 * noise:
            right = bins - index
        index += 1
    return left, right

# ## Define the main callable function pulseshape(1D_data_array)
def pulseshape(data):
    originalbins = len(data)
    # roll the profile to bring the peak to the centre
    binmax = np.argmax(data)
    profile = np.roll(data, np.int(originalbins/2 - binmax))
#    peak = np.max(data)
#    profile = rolled/peak
    # plt.plot(profile) plot for debugging

    # ## Start the GP 

    gp1 = get_gp(profile)
    print(gp1.log_likelihood(profile), gp1.get_parameter_names(),gp1.get_parameter_vector())
    # ## Find a window of data around the pulse and produce noiseless data

    left, right = get_boundaries(profile, gp1)
    bins = right + 60 - (left - 60) 
    print('boundaries: ', left, right)
    base1 = np.int(left/2)
    base2 = np.int(right + 0.5*(originalbins-right)) 
    baseline = 0.5 * (np.mean(profile[0:base1]) + np.mean(profile[base2:]))
    std_baseline = 0.5 * (np.std(profile[0:base1]) + np.std(profile[base2:]))
    print('Off pulse baseline to subtract is: ', baseline, std_baseline)
    yn = profile[left-60:right+60] - baseline
    gp_last = get_gp(yn)
    t = np.arange(len(yn))
    mu, var = gp_last.predict(yn, t, return_var=True)
    noise = np.sqrt(np.exp(gp_last.get_parameter_vector()[1]))
    print('GP Noise standard deviation: ', noise)
#    std = np.sqrt(np.mean(var)+noise**2)
    std = noise
# ## Return the standard deviation of the noise, the left and
    # ## right pulse boundaries referring to the input array, and the
    # ## noiseless profile cut at 60 bins either side of the pulse
    left -= originalbins/2 - binmax
    right -= originalbins/2 - binmax


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
#        plt.plot(t, gp_last.sample_conditional(yn, t), "g", alpha=0.1)
        allsamples[i,:] = gp_last.sample_conditional(yn, t)
    noiseless=np.mean(allsamples,0)
#    plt.plot(t, yn, ".k")
#    plt.plot(t, noiseless, "-b")
#    plt.show()
    return std, left, right, noiseless
