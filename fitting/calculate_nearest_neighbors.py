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
signal = np.random.normal(sig_mean,sig_width,Nsig)
signal.sort()
# For every point in the dataset, calculate how many other points
# are within 0.1

#signal = signal.sort()
#print len(signal)
function_density = []

for i in range(len(signal)):
    point = signal[i]
    density = 0
    for j in range(len(signal)-1):
        if i != j:
            dist = abs(point - signal[j])
            
            if dist < .1 :
                density = density + 1
            
    function_density.append(density)
    
# Here's a very simple plot of our data.
plt.figure()
plt.hist(signal,bins=50)
plt.figure()
x = signal
y = function_density
plt.plot(x,y)
plt.axis([signal[0],signal[999],0,200])

plt.show()
