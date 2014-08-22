import numpy as np
import matplotlib.pylab as plt
import math as math
import gen_data_and_templates as genD
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

    return 0.1*np.ones(len(x))


################################################################################
# Total probability for each number/event/value.
################################################################################
def tot_prob(x,mean0,width0,mean1,width1,frac):

    ######################################################
    ############### YOUR TASK HERE #######################
    ######################################################
    # Write a function that defines the total probability
    # using the values passed in in the function definition
    # above. You should have your function call the Gaussian
    # and flat functions above.

    #print "x[0]"
    #print x[0]
    #print "x[1]"
    #print x[1]

    sig = Gaussian(x[0],mean0,width0)*Gaussian(x[1],mean1,width1)
    #print "SIG:"
    #print sig
    
    bkg = flat(x[0]) * flat(x[1])
    #print "BKG:"
    #print bkg

    tot = (1-frac)*sig + frac*bkg

    #print "TOT:"
    #print tot
    
    return tot

################################################################################
# Negative log-likelihood function.
################################################################################
def negative_log_likelihood(mean0,mean1,width0,width1,frac):

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

    comp = tot_prob(data,mean0,width0,mean1,width1,frac)
    
    comp = comp[comp!=0]
    #print "COMP:"
    #print type(comp)
    #print comp
    logd = -np.log(comp)
     
    nll = logd.sum()

    return nll

################################################################################
# Genergate your fake data!
################################################################################

# 2D mlm
Ntemplates = 0

Nsig = 100
Nsig = stats.poisson(Nsig).rvs(1)[0]
sig_mean1 = 5.0
sig_width1 = 0.5
sig_mean2 = 15.0
sig_width2 = 1.0

Nbkg = 900
Nbkg = stats.poisson(Nbkg).rvs(1)[0]
bkglo1 = 0
bkghi1 = 10
bkglo2 = 10
bkghi2 = 20

true_frac = Nbkg/float(Nsig+Nbkg)

# actually generating data and templates
data, sig_template, bkg_template = genD.sig_ngauss_bkg_flat([sig_mean1, sig_mean2],[sig_width1, sig_width2],Nsig,[bkglo1, bkglo2],[bkghi1,bkghi2],Nbkg,Ntemplates,verbose=False,scale=False)

################################################################################
# Set up Minuit!
################################################################################

# First we have to initialize Minuit. You tell it the function you minimize,
# the names and starting values of your parameters (the things you will vary)
# and then the range of those values. For instance, frac should never be less
# than 0 or greater than 1, because it's a fraction.

m = Minuit(negative_log_likelihood,mean0=5.0,limit_mean0=(4.0,11.0),fix_mean0=True, \
                                   width0=.5,limit_width0=(0,1.0),fix_width0=True, \
                                   mean1=15.0,limit_mean1=(9.0,21.0),fix_mean1=True, \
                                   width1=1.0,limit_width1=(0,2.0),fix_width1=True, \
                                   frac=0.8,limit_frac=(0,1.0), \
                                   errordef = 0.5 )
# This does the minimization
m.migrad()

#calculate errors
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

'''
if values['frac']<=true_frac:
    print "FIT VALS: %f %f\t\tTRUE VAL: %f" % (values['frac'],em['upper'],true_frac)
else:
    print "FIT VALS: %f %f\t\tTRUE VAL: %f" % (values['frac'],em['lower'],true_frac)
'''

print "FIT VALS: %f %f\t\tTRUE VAL: %f" % (values['frac'],errors['frac'],true_frac)
print Nsig
print Nbkg

print eh,em
print "FINAL VAL: ",values['frac']

plt.show()
