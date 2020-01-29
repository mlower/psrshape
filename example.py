#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import PulseShape as ps

#data_raw = np.loadtxt('1735-0724_20cmWBC_oct05.txt', comments='F')
data_raw = np.loadtxt('J1809.txt', comments='F')
data = data_raw[:,3]
rawx = np.arange(len(data))
std, left, right, mu = ps.pulseshape(data,3.0)
print('noise: ', std)
print( 'left, right: ',left, right)
smoothx = np.arange(left-60,right+60,1)
print('size of mu: ', len(mu))
plt.plot(rawx, data, 'b-')
plt.plot(smoothx, mu, 'r-')
plt.show()
