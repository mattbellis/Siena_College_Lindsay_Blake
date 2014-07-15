import numpy as np
import matplotlib.pylab as plt
import math as math

from iminuit import Minuit

# It turns out, we need to define data at the beginning so that
# the different functions know about it. 
data = None

################################################################################
# Define the signal function.
################################################################################
def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*(width**2)))

    return y

################################################################################
# Define the background function.
################################################################################
def flat(x):

    return 0.5


################################################################################
# Total probability for each number/event/value.
################################################################################
def tot_prob(x,mean,width,frac):

    ######################################################
    ############### YOUR TASK HERE #######################
    ######################################################
    # Write a function that defines the total probability
    # using the values passed in in the function definition
    # above. You should have your function call the Gaussian
    # and flat functions above.

    tot = #################

    return tot

################################################################################
# Negative log-likelihood function.
################################################################################
def negative_log_likelihood(mean,width,frac):

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

    nll = ##################################################

    return nll

################################################################################
# Genergate your fake data!
################################################################################

# Here's your signal data!
Nsig = 50
sig_mean = 10.1
sig_width = 0.05
signal = np.random.normal(sig_mean,sig_width,Nsig)
print signal

# So here's your background data!
Nbkg = 950
background = 9.0+(2*np.random.random(Nbkg))

# Combine the background and signal, because when we run the experiment, we actually
# don't know which is which!
data = signal.copy()
data = np.append(data,background.copy())

# Here's a very simple plot of our data.
plt.figure()
plt.hist(data,bins=50)


################################################################################
# Set up Minuit!
################################################################################

# First we have to initialize Minuit. You tell it the function you minimize,
# the names and starting values of your parameters (the things you will vary)
# and then the range of those values. For instance, frac should never be less
# than 0 or greater than 1, because it's a fraction.

m = Minuit(negative_log_likelihood,mean=10.0,limit_mean=(9.0,11.0), \
                                   width=1.0,limit_width=(0,3.0), \
                                   frac=0.5,limit_frac=(0,1.0) \
                                   )
# This does the minimization
m.migrad()

# This calculates errors
m.hesse()

# There's different ways to get the final value of the 
# negative log likelihood.
print 'fval', m.fval
print m.get_fmin()

# Here the final fit values! Should you need them.
values = m.values
print values

# Here the final fit errors! Should you need them.
errors = m.errors
print errors

plt.show()
