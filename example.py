#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import PulseShape as ps

#data_raw = np.loadtxt('1735-0724_20cmWBC_oct05.txt', comments='F')
data_raw = np.loadtxt('data.dat', comments='F')
data = data_raw[:,3]
rawx = np.arange(len(data))
margin = np.int(len(data)/20)
mcmc = False
std, left, right, mu = ps.pulseshape(data,2.0,mcmc)
w50, w50p, w50n = ps.get_wX(mu, std, 50)
w10, w10p, w10n = ps.get_wX(mu, std, 10)
w5, w5p, w5n = ps.get_wX(mu, std, 5)
w1, w1p, w1n = ps.get_wX(mu, std, 1)
print('noise: ', std)
print( 'left, right above threshold: ',left, right)
smoothx = np.arange(left,right,1)
print('size of mu: ', len(mu))
if w50 + w50p < right - left + 1 :
    print('W50 params: ', w50, w50p, w50n)
else:
    print('cannot measure w50')
if w10 + w10p < right - left + 1 :
    print('W10 params: ', w10, w10p, w10n)
else:
    print('cannot measure w10')
if w5 + w5p < right - left + 1:
    print('W5 params: ', w5, w5p, w5n)
else:
    print('cannot measure w5')
if w1 + w1p < right - left + 1:
    print('W1 params: ', w1, w1p, w1n)
else:
    print('cannot measure w1')

plt.plot(rawx, data, 'b-')
plt.plot(rawx, mu, 'r-')
plt.show()
