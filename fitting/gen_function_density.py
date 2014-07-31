# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:55:10 2014

@author: Lindsay
"""

import numpy as np
import matplotlib.pylab as plt
import math as math
from scipy.spatial.distance import cdist

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
        if(len(y) > 1):
            yPrime = y.T
            temp = np.array(yPrime[bottom:top], copy = True)
            temp = temp.T
        else:
            temp =np.array( y[bottom:top], copy = True)
        temp = y[bottom:top]
        Y = np.vstack((temp.T))
        print "here"
        print X
        print Y
        values = cdist(X, Y)
        i = 0
        for toCheck in values:
            n[i] += len(toCheck[toCheck < radius])
            i += 1
        top += 10000
        bottom += 10000
