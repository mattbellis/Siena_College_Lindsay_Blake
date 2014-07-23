# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:13:38 2014

@author: Lindsay
"""

import numpy as np
import matplotlib.pylab as plt
import math as math
from scipy.spatial.distance import cdist

from iminuit import Minuit

# It turns out, we need to define data at the beginning so that
# the different functions know about it. 
data = None
signal_template = None
background_template = None

signal_densities = None
background_densities = None

#Takes two arrays as inputs and returns an array of
# the number of points on y 
# can handle up to a 10k length
#x value and any size y value

def numInRange(x, y, radius, npts):
    X = np.vstack((x.T))
    n = np.zeros(npts)    
    top = 10000
    bottom = 0
    done = False
    while done == False:
        ylength = len(y.T)
        if bottom >= ylength:
            n /= (ylength*radius) # added to try and conform to bellis
            return n
        elif top >= ylength:
            top = ylength
        
        temp = y[bottom:top]
        Y = np.vstack((temp.T))
        values = cdist(X, Y)
        i = 0
        for toCheck in values:
            n[i] += len(toCheck[toCheck < radius])
            i += 1
        top += 10000
        bottom += 10000
        
################################################################################
# Total probability for each number/event/value.
################################################################################
def tot_prob(densities=[],frac=0.5):

    sig_densities = densities[0]
    bkg_densities = densities[1]

    tot = sig_densities*(1-frac) + bkg_densities*frac

    return tot

################################################################################
# Negative log-likelihood function.
################################################################################
def negative_log_likelihood(frac):

    ########################################################
    # Write this part so that is uses data (we've already definted it
    # so it knows about it) and loops over all the data points
    # in this function.
    # 
    # Don't forget to take the *negative* of the sum of the logs
    # when you're done. Minuit wants to minimize the function
    # you pass in, not maximize it.
    #
    ########################################################

    probs = tot_prob([GarrettsigFDY,GarrettbkgFDY],frac)
    #probs = tot_prob([GarrettsigFDX,GarrettbkgFDX],frac)
        
    probs = probs[probs!=0]
        
    nll = (-np.log(probs)).sum()
    
    return nll

################################################################################
# Generate your fake data!
################################################################################

print "Generating the fake experimental data!"

# Here's your signal data!
Nsig = 100
sig_mean = 5.0; sig_width = 0.5
signalx = np.random.normal(sig_mean,sig_width,Nsig)
sig_mean = 17.0; sig_width = 1.0
signaly = np.random.normal(sig_mean,sig_width,Nsig)
#print signal

# So here's your background data!
Nbkg = 900
backgroundx = 0.0+(10*np.random.random(Nbkg))
backgroundy = 12.0+(9*np.random.random(Nbkg))

# Combine the background and signal, because when we run the experiment, we actually
# don't know which is which!
data = [None,None]
data[0] = signalx.copy()
data[0] = np.append(data[0],backgroundx.copy())
data[1] = signaly.copy()
data[1] = np.append(data[1],backgroundy.copy())

minvalx = min(data[0])
maxvalx = max(data[0])
xwidth = maxvalx-minvalx

minvaly = min(data[1])
maxvaly = max(data[1])
ywidth = maxvaly-minvaly

data[0] = (data[0] - minvalx)/xwidth
data[1] = (data[1] - minvaly)/ywidth

Xdata = data[0]
Ydata = data[1]

plt.figure()
plt.hist(data[0],bins=25)
plt.figure()
plt.hist(data[1],bins=25)
#plt.show()


data = np.array(data)
Ndata = len(data[0])

print "Generated the fake experimental data!"

################################################################################
# Generate the templates that we will use to fit the data.
################################################################################

print "Generating the templates!"

# Here's your signal template!
N = 10000
sig_mean = 5.0; sig_width = 0.5
signalx = np.random.normal(sig_mean,sig_width,N)
sig_mean = 17.0; sig_width = 1.0
signaly = np.random.normal(sig_mean,sig_width,N)
signal_template = [signalx.copy(),signaly.copy()]
signal_template = np.array(signal_template)
signal_template[0] = (signal_template[0] - minvalx)/xwidth
signal_template[1] = (signal_template[1] - minvaly)/ywidth

sig_tempX = signal_template[0]
sig_tempY = signal_template[1]

#print signal

# So here's your background data!
N = 10000
backgroundx = 0.0+(10*np.random.random(N))
backgroundy = 12.0+(9*np.random.random(N))
background_template = [backgroundx.copy(),backgroundy.copy()]
background_template = np.array(background_template)
background_template[0] = (background_template[0] - minvalx)/xwidth
background_template[1] = (background_template[1] - minvaly)/ywidth

bkg_tempX = background_template[0]
bkg_tempY = background_template[1]

fig_template = plt.figure(figsize=(12,6))
fig_template.add_subplot(1,2,1)
H, xedges, yedges = np.histogram2d(signal_template[1],signal_template[0],bins=25)
imsig = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

fig_template.add_subplot(1,2,2)
H, xedges, yedges = np.histogram2d(background_template[1],background_template[0],bins=25)
imbkg = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

print "Generated the templates!"

print "Calculating the densities!!!"
d_radius = 0.05
GarrettsigFDX = numInRange(Xdata, sig_tempX,d_radius,Ndata)
GarrettbkgFDX = numInRange(Xdata, bkg_tempX, d_radius,Ndata)

GarrettsigFDY = numInRange(Ydata, sig_tempY,d_radius,Ndata)
GarrettbkgFDY = numInRange(Ydata, bkg_tempY, d_radius,Ndata)
print "Calculated the densities!!!"

m = Minuit(negative_log_likelihood,frac=0.9,limit_frac=(0,1.0), \
                                   errordef = 0.5 )
                                   
m.migrad()

m.hesse()

print 'fval', m.fval
print m.get_fmin()

values = m.values
print values

errors = m.errors
print errors
