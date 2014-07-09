import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

################################################################################
# First thing I'm going to do is to generate some fake data for us to work with.
# 
# Imagine that at approximately 20 separate times, you measured the radioactive 
# output from some 
# sample and each time, you recorded how many radioactive decays you measure. 
# You might also have some uncertainty (error) associated with each measurement. 
# Ignore for the moment how you are estimating this uncertainty.
################################################################################

# So here's your fake data!
# ``times" from 1-20
x = np.linspace(1,20,20)
print x

# Random (Poisson) data, generated according to an exponential.
exp_slope = -0.3

# Estimate the central value of the data.
mu = (200*np.exp(x*exp_slope)).astype('int')
# Add some experimental uncertainty.
x = x[mu>0]
mu = mu[mu>0]
y = stats.poisson.rvs(mu)

# And your uncertainty.


# Here's a very simple plot.
plt.plot(x,y,'o',markersize=5)
plt.show()


################################################################################
# Question #1
################################################################################
# Replot the data, using the errorbar function in matplotlib, and using yerr as
# uncertainties on the y-values. 
#
# http://matplotlib.org/examples/statistics/errorbar_demo.html
#
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.errorbar


# YOUR WORK HERE.
x = np.linspace(1,20,20)
exp_slope = -0.3
mu = (200*np.exp(x*exp_slope)).astype('int')
x = x[mu>0]
mu = mu[mu>0]
y = stats.poisson.rvs(mu)
plt.plot(x,y,'o',markersize=5)
plt.errorbar(x, y, xerr=0, yerr = np.sqrt(y))
plt.show()


################################################################################
# Question #2
################################################################################
# We know that the data comes from an exponential of the form
#
#    y = N*e^{x*lambda}
#
# but we don't know N or lambda without cheating and looking at the code above. :)
# uncertainties on the y-values. 
#
# Suppose for a moment that N=150 and lambda=0.2. Do two things:
#
# 1) Calculate chisq for this assumption and with this data.
# 2) Overlay on the data a plot of the exponential with that N and lambda.


# YOUR WORK HERE.
print "1.) Chi Squared for N = 150 and lamda = .2"

N=150
lamda=.2
chidat_obs = stats.poisson.rvs(mu)
chidat_assum = N*np.exp(x*-lamda)
yerr = np.sqrt(chidat_obs)

#chi observed and assume are the same length
#for each element, subract observed and assumed and divide by yerr and square

x = len(chidat_obs)

sumDat=[]

for i in range(x):
    if yerr[i] != 0:
        h = ((chidat_obs[i]-chidat_assum[i])/yerr[i])**2
        sumDat.append(h)
    else:
        sumDat.append(0)
        
chisquare = sum(sumDat)
print chisquare

print "2.) Plot Overlay"

x = np.linspace(1,20,20)
exp_slope = -0.3
mu = (200*np.exp(x*exp_slope)).astype('int')
x = x[mu>0]
mu = mu[mu>0]
y = stats.poisson.rvs(mu)
plt.plot(x,y,'o',markersize=5)
plt.errorbar(x, y, xerr=0, yerr = np.sqrt(y),fmt = "o")

N=150
lamda=.2
exp_Y = N*np.exp(x*-lamda)
plt.plot(exp_Y)
plt.show()

################################################################################
# Question #3
################################################################################
# Write some code to loop over reasonable values for N and lambda calculating
# chisq each time. Keep track of when you find the *minimal value* of chisq. 
# 
# Using those values of N and lambda that give you the minimal value of chisq,
# plot the exponential function over the data. Does this look better than your 
# first guess?

# YOUR WORK HERE.
print "3.)"

chisquare = []
minchisquare = 1e6
bestn=-999
bestmy_lamda=-999

for N in range(150,351):
    for my_lamda in np.arange(.1,2,.01):
        sumDat = []
        chidat_obs = stats.poisson.rvs(mu)
        chidat_assum = N*np.exp(x*-my_lamda)
        yerr = np.sqrt(chidat_obs)

        for i in range(len(chidat_obs)):
            if yerr[i] != 0:
                h = ((chidat_obs[i]-chidat_assum[i])/yerr[i])**2
                sumDat.append(h)
            else:
                sumDat.append(0)
        chi_index = sum(sumDat)
        if chi_index < minchisquare:
            minchisquare = chi_index
            bestn = N
            bestmy_lamda = my_lamda
        
    
print minchisquare
print bestn
print bestmy_lamda
        

plt.errorbar(x, y, xerr=0, yerr = np.sqrt(y),fmt = "o")

N=bestn
lamda=bestmy_lamda
xpts = np.linspace(min(x),max(x),1000)
exp_Y = N*np.exp(xpts*-lamda)
plt.plot(xpts,exp_Y,'b-')
plt.show()