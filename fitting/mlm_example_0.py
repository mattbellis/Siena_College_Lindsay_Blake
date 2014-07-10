import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

################################################################################
# This is how we define our own function
################################################################################
def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*width)**2)

    return y


################################################################################
# First thing I'm going to do is to generate some fake data for us to work with.
# 
# Suppose I have test scores from 200 students. 
################################################################################

# So here's your fake data!
Nstudents = 200
mean = 80
width = 5
scores = np.random.normal(mean,width,Nstudents)

# Here's a very simple plot.
plt.figure()
plt.hist(scores,bins=25)#,range=(60,100))

# Functional form for a Gaussian
# y = e^{(x-mean)^2/(2*width)^2
plt.figure()
x = np.linspace(60,100,1000)
y = Gaussian(x,mean,width)
plt.plot(x,y)


################################################################################
# Question #1
################################################################################
# Calculate the probabilities of measuring all these points if they came from a 
# Gaussian of mean=90 and width=10.

# YOUR WORK HERE.

################################################################################
# Question #2
################################################################################
# What is the product of those probabilities?

# YOUR WORK HERE.

################################################################################
# Question #3
################################################################################
# Vary the mean and width to find the maximum probability of measuring those
# particular test scores.

# YOUR WORK HERE.


plt.show()
