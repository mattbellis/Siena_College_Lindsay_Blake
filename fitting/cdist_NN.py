#For use for all of the distances between two 2-d arrays
#Prints the number of points in y that 
import numpy as np
import math
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist


npts = 10000 # number of ordered pairs
index = 0 # Index of the first array whose number of points in the radius you would like to know
radius = .1 # how far away the points can be to still be included


x = 10*np.random.random(npts)
y = 10*np.random.random(npts)



x = np.float32(x)
y = np.float32(y)
"""
#cdist version
X = np.vstack((x.T)).T
Y = np.vstack((y.T)).T

ans = cdist(X, Y)
toCheck = ans[index]

n = len(x[x<radius])
print n



"""
"""
#iterative version for time tests only

z = np.zeros((npts, npts), dtype = np.float32)
for i in range(0, npts):
    for j in range(0, npts):
        z[i][j] = np.abs((x[i] - y[j])) # for 2 1d arrays
       

"""

"""
#Equality test (comment out for times)
stuff = []
if z.all() == ans.all():
    print "hooray"
else:
    print "WRONG"

for i in range(0, npts):
    for j in range(0, npts):
        if not z[i][j] == ans[i][j]:
            stuff.append((i, j))


"""



"""
    if len(y) <= 10000:             #working takeTwo code (fast) for 10k or less
        Y = np.vstack((y.T))
        values = cdist(X, Y)
        i = 0
        for toCheck in values:
            n[i] += len(toCheck[toCheck < radius])
            i += 1
        return n
"""
    
#Takes two arrays as inputs and returns an array of
# the number of points on y 
# can handle up to a 10k length
#x value and any size y value

def numInRange(x, y, radius):
    n = np.zeros_like(x)
    X = np.vstack((x.T))
    top = 10000
    bottom = 0
    done = False
    while done == False:
        ylength = len(y)
        if bottom >= ylength:
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


