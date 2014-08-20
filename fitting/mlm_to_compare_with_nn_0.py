import numpy as np
import matplotlib.pylab as plt
import math as math
import scipy.stats as stats


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

    tot = Gaussian(x,mean,width)*(1-frac)+flat(x)*frac

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
    probabilities = []
    
    for d in data:
        comp = tot_prob(d,mean,width,frac)
        
        if comp>0:
            logd = -math.log(comp)
            probabilities.append(logd)
    
    nll = sum(probabilities)

    return nll

################################################################################
# Genergate your fake data!
################################################################################

# Here's your signal data!
Nsig = 100
Nsig = stats.poisson(Nsig).rvs(1)[0]

sig_mean = 10.1
sig_width = 0.05
signal = np.random.normal(sig_mean,sig_width,Nsig)
#print signal

# So here's your background data!
Nbkg = 900
Nbkg = stats.poisson(Nbkg).rvs(1)[0]
background = 9.0+(2*np.random.random(Nbkg))

true_frac = Nbkg/float(Nsig+Nbkg)

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

m = Minuit(negative_log_likelihood,mean=10.1,limit_mean=(9.0,11.0),fix_mean=True, \
                                   width=.05,limit_width=(0,1.0),fix_width=True, \
                                   frac=0.8,limit_frac=(0,1.0), \
                                   errordef = 0.5 )
# This does the minimization
m.migrad()

# This calculates errors
m.hesse()
eh = m.errors['frac']
m.minos()
em = m.get_merrors()['frac']

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

if values['frac']<=true_frac:
    print "FIT VALS: %f %f\t\tTRUE VAL: %f" % (values['frac'],em['upper'],true_frac)
else:
    print "FIT VALS: %f %f\t\tTRUE VAL: %f" % (values['frac'],em['lower'],true_frac)

#print "FIT VALS: %f %f\t\tTRUE VAL: %f" % (values['frac'],errors['frac'],true_frac)
print Nsig
print Nbkg

print eh,em


#plt.show()
