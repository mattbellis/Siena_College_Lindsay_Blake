import numpy as np
import matplotlib.pylab as plt
import math as math

from iminuit import Minuit

# It turns out, we need to define data at the beginning so that
# the different functions know about it. 
data = None
signal_template = None
background_template = None

signal_densities = None
background_densities = None

################################################################################
# Return number of nearest neigbors within some radius. 
################################################################################
def nn_within_radius(values0,values1,same=False,radius=1):

    nvals0 = len(values0)
    nvals1 = len(values1)

    function_density = np.zeros(nvals0)

    for i in range(nvals0):
        point = values0[i]
        for j in range(nvals1):
            if i == j and same==True:
                continue
            else:
                dist = abs(point - values1[j])

                #print point, values1[j]
                
                if dist < radius :
                    function_density[i] += 1
                
    print function_density[0:10]
    function_density /= (nvals1*radius)
    print function_density[0:10]
    #exit()
    #function_density /= function_density.sum()

    return function_density

################################################################################
# Total probability for each number/event/value.
################################################################################
def tot_prob(data,densities=[],frac=0.5):

    sig_densities = densities[0]
    bkg_densities = densities[1]

    tot = sig_densities*(1-frac) + bkg_densities*frac

    return tot

################################################################################
# Negative log-likelihood function.
################################################################################
def negative_log_likelihood(frac):

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
    probs = tot_prob(data,[signal_densities,background_densities],frac)

    probs = probs[probs!=0]
        
    nll = (-np.log(probs)).sum()
    
    return nll

################################################################################
# Generate your fake data!
################################################################################

print "Generating the fake experimental data!"

# Here's your signal data!
Nsig = 100
sig_mean = 10.1
sig_width = 0.05
signal = np.random.normal(sig_mean,sig_width,Nsig)
#print signal

# So here's your background data!
Nbkg = 900
background = 9.0+(2*np.random.random(Nbkg))

# Combine the background and signal, because when we run the experiment, we actually
# don't know which is which!
data = signal.copy()
data = np.append(data,background.copy())

# Here's a very simple plot of our data.
plt.figure()
plt.hist(data,bins=50)

print "Generated the fake experimental data!"



################################################################################
# Generate the templates that we will use to fit the data.
################################################################################

print "Generating the templates!"

# Here's your signal template!
Nsig = 10000
sig_mean = 10.1
sig_width = 0.05
signal_template = np.random.normal(sig_mean,sig_width,Nsig)
#print signal_template

# So here's your background data!
Nbkg = 10000
background_template = 9.0+(2*np.random.random(Nbkg))

fig_template = plt.figure(figsize=(12,6))
fig_template.add_subplot(1,2,1)
plt.hist(signal_template,bins=100,range=(9,11))
plt.xlim(9,11)

fig_template.add_subplot(1,2,2)
plt.hist(background_template,bins=100,range=(9,11))
plt.xlim(9,11)

print "Generated the templates!"


######################################################
print "Calculating the densities!!!"
signal_densities = nn_within_radius(data,signal_template,False,radius=0.02)
background_densities = nn_within_radius(data,background_template,False,radius=0.02)
print "Calculated the densities!!!"

#print signal_densities 
#print background_densities 
#exit()
################################################################################
# Set up Minuit!
################################################################################

# First we have to initialize Minuit. You tell it the function you minimize,
# the names and starting values of your parameters (the things you will vary)
# and then the range of those values. For instance, frac should never be less
# than 0 or greater than 1, because it's a fraction.

m = Minuit(negative_log_likelihood,frac=0.9,limit_frac=(0,1.0), \
                                   errordef = 0.5 )
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

#plt.show()