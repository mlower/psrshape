#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import PulseShape as ps

import argparse
import psrchive

from scipy.signal.windows import tukey


def get_arguments():
    parser = argparse.ArgumentParser(description="input SNR cutoff")
    parser.add_argument("-s", "--snr", help="signal to noise cut", type=float, required=True)
    parser.add_argument("-f", "--file", help="signal to noise cut", type=str, required=True)

    return parser.parse_args()

args = get_arguments()

print("Reading in data...")

# Read in the archive file
a = psrchive.Archive_load(args.file)
clone = a.clone()
clone.fscrunch()
clone.pscrunch()
clone.remove_baseline()

nbin = clone.get_nbin()
data = clone.get_data()[0,0,0,:]

print("nbins in file:", nbin)

print("Starting the fit...")

rawx = np.arange(len(data))
margin = int(len(data)/20)
mcmc = False
#mcmc = True

#Rotate the data to centre the peak
binmax = np.argmax(data)
shift = int(len(data)/2 - binmax)
profile = np.roll(data, shift)

# Run Gaussian process regression
std, left, right, mu = ps.pulseshape(profile,args.snr,mcmc)
bflux, errorflux = ps.getflux(profile,int(left),int(right),std)
print("flux: ", bflux, "mJy" )

# Make various profile width measurements (ignore for now)
w50, w50p, w50n = ps.get_wX(mu, std, 50)
w10, w10p, w10n = ps.get_wX(mu, std, 10)
w5, w5p, w5n = ps.get_wX(mu, std, 5)
w1, w1p, w1n = ps.get_wX(mu, std, 1)
print("noise: ", std)
print(w50, w10, w5, w1)
print("size of mu: ", len(mu))
if w50 + w50p < right - left + 1 :
    print("W50 params: ", w50, w50p, w50n)
else:
    print("cannot measure w50")
    print("W50 params: ", w50, w50p, w50n)
if w10 + w10p < right - left + 1 :
    print("W10 params: ", w10, w10p, w10n)
else:
    print("cannot measure w10")
    print("W10 params: ", w10, w10p, w10n)
if w5 + w5p < right - left + 1:
    print("W5 params: ", w5, w5p, w5n)
else:
    print("cannot measure w5")
    print("W5 params: ", w5, w5p, w5n)
if w1 + w1p < right - left + 1:
    print("W1 params: ", w1, w1p, w1n)
else:
    print("cannot measure w1")
    print("W1 params: ", w1, w1p, w1n)

# Pad the 'left-' and 'right-most' limits of the profile by 20 bins
right += 20
left -= 20

# Apply a Tukey window to the on-pulse region -- currently only works for non-interpulse/complex profile pulsars!
window = tukey(int(right-left), alpha=0.5)

mu[int(left):int(right)] = mu[int(left):int(right)] * window
mu[:int(left)] = 0.0
mu[int(right):] = 0.0

# Shift the Gaussian process template back to the original pulse phase
new_mu = np.roll(mu, -shift)

# Compute residual between the data and the template, and then report the RMS 
residual = data - new_mu 
print("residual RMS: ", np.std(residual[0:100])) 
left = left - shift
right = right - shift
print( "left, right above threshold: ",left, right)
vxl = left if left > 0 else len(data)+left
vxr = right if right < len(data) else right - len(data)

# Overwrite profile in cloned archive and save to disc
print("Saving to output file: {0}.std".format(args.file))

profile = clone.get_Integration(0).get_Profile(0,0)
amps = profile.get_amps()

for i in range(nbin):
    amps[i] = new_mu[i]

clone.unload("{0}.std".format(args.file))

print("Done!")

# Make a nice plot showing the resulting fit
plt.plot(rawx, data, "b-", alpha=0.5)
plt.plot(rawx, new_mu, "r-")
plt.plot(rawx, residual, "g-", alpha=0.5)
# plt.axvline(x=vxl,color="red")
# plt.axvline(x=vxr,color="red")
plt.show()
