#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import PulseShape as ps
import sys
import argparse

parser = argparse.ArgumentParser(description='input SNR cutoff')
parser.add_argument('-s','--snr', help='signal to noise cut', type=float, required=True)
args = parser.parse_args()
snr =  args.snr
#data_raw = np.loadtxt('1735-0724_20cmWBC_oct05.txt', comments='F')
data_raw = np.loadtxt('data.dat', comments='F')
channels = np.int(data_raw[-1,1]) + 1
bins = np.int(data_raw[-1,2]) + 1
print('channels and bins in file:', channels, bins)
idx = 0
bflux = np.zeros((channels))
errorflux = np.zeros((channels))
for chan in np.arange(channels):
    print('################## Channel ', chan, '##################')
    data = data_raw[idx:idx+bins,3]
    rawx = np.arange(len(data))
    margin = np.int(len(data)/20)
    mcmc = False
    #mcmc = True
    #rotate the data to centre the peak
    binmax = np.argmax(data)
    shift = np.int(len(data)/2 - binmax)
    profile = np.roll(data, shift)
    std, left, right, mu = ps.pulseshape(profile,snr,mcmc)
    flux = ps.getflux(profile,np.int(left),np.int(right))
    #baseline =  np.mean((np.mean(profile[0:np.int(left)]),np.mean(profile[np.int(right):-1])))
    maskedprofile = profile
    maskedprofile[np.int(left):np.int(right)] = 0.0
    baseline = np.mean(maskedprofile)
    bflux[chan] = flux-baseline*(right-left)
    errorflux[chan] =  np.sqrt(right-left)*std
    print('flux: ', flux, baseline )
    print('baselined flux: ', bflux[chan])
    w50, w50p, w50n = ps.get_wX(mu, std, 50)
    w10, w10p, w10n = ps.get_wX(mu, std, 10)
    w5, w5p, w5n = ps.get_wX(mu, std, 5)
    w1, w1p, w1n = ps.get_wX(mu, std, 1)
    print('noise: ', std)
    print(w50, w10, w5, w1)
    print('size of mu: ', len(mu))
    if w50 + w50p < right - left + 1 :
        print('W50 params: ', w50, w50p, w50n)
    else:
        print('cannot measure w50')
        print('W50 params: ', w50, w50p, w50n)
    if w10 + w10p < right - left + 1 :
        print('W10 params: ', w10, w10p, w10n)
    else:
        print('cannot measure w10')
        print('W10 params: ', w10, w10p, w10n)
    if w5 + w5p < right - left + 1:
        print('W5 params: ', w5, w5p, w5n)
    else:
        print('cannot measure w5')
        print('W5 params: ', w5, w5p, w5n)
    if w1 + w1p < right - left + 1:
        print('W1 params: ', w1, w1p, w1n)
    else:
        print('cannot measure w1')
        print('W1 params: ', w1, w1p, w1n)

    plt.plot(rawx, data, 'b-')
    new_mu = np.roll(mu, -shift)
    plt.plot(rawx, new_mu, 'r-')
    residual = data - new_mu 
    print("residual RMS: ", np.std(residual[0:100])) 
    plt.plot(rawx, residual, 'g-')
    left = left - shift
    right = right - shift
    print( 'left, right above threshold: ',left, right)
    vxl = left if left > 0 else len(data)+left
    vxr = right if right < len(data) else right - len(data)
    plt.axvline(x=vxl,color='red')
    plt.axvline(x=vxr,color='red')
    #plt.show()
    idx += bins
plt.show()
plt.plot(bflux, 'ro')
plt.errorbar(np.arange(channels), bflux, errorflux, fmt='none')
plt.show()
print('Measured fluxes')
print(bflux)
