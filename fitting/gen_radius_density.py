# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:55:10 2014

@author: Lindsay
"""

import numpy as np
import matplotlib.pylab as plt
import math as math
from scipy.spatial.distance import cdist

def radiusToK(x, y, k, npts):
    X = np.vstack((x.T))
    n = np.zeros(npts)    
    top = 10000
    bottom = 0
    done = False
    while done == False:
        ylength = len(y.T)
        if bottom >= ylength:
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
        values = cdist(X, Y)
        values.sort()
        print values
        
        for i,toCheck in enumerate(values):
            n[i] = toCheck[k-1]
        top += 10000
        bottom += 10000

