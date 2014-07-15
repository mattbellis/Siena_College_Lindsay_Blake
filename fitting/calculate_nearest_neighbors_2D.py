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
# are within 0.2\\ a distance of .2 d = sqrt((x1-x2)**2 + (y1-y2)**2)

def Distance(x1,y1,x2,y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# The zip function is very useful!
zipped = zip(xvals,yvals)
density_function2D = []


for x1,y1 in zipped:
    density = 0
    for x2,y2 in zipped:
        dist = Distance(x1,y1,x2,y2)
        if dist < .2:
            if dist != 0:
                density = density +1
    density_function2D.append(density)

print density_function2D


plt.figure()
H, xedges, yedges = np.histogram2d(yvals, xvals,bins=25)
im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

plt.show()
