# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:08:35 2014

@author: Lindsay
"""

import numpy as np
import matplotlib.pylab as plt
import math as math

from iminuit import Minuit

# It turns out, we need to define data at the beginning so that
# the different functions know about it. 
data = None


################################################################################
# Genergate your fake data!
################################################################################

# Here's your signal data!
Nsig = 1000

sig_mean = 10.0
sig_width = 0.5
xvals = np.random.normal(sig_mean,sig_width,Nsig)

sig_mean = 5.0
sig_width = 1.0
yvals = np.random.normal(sig_mean,sig_width,Nsig)

# For every point in the dataset, calculate how many other points
# are within 0.2

# The zip function is very useful!
for x,y in zip(xvals,yvals):
    print x,y

#signal = signal.sort()
#print len(signal)
function_density = []


plt.figure()
H, xedges, yedges = np.histogram2d(yvals, xvals,bins=25)
im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

plt.show()
