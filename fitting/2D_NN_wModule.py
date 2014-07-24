# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:32:20 2014

@author: Lindsay
"""

import gen_data_and_templates as genD
import gen_function_density as genFD
import numpy as np
import matplotlib.pylab as plt
from iminuit import Minuit

################################################################################
# Generate the templates and data from Bellis' module.
################################################################################

# arguments to put into the generating function
Ntemplates = 100000
Nsig = 100
sig_mean1 = 5.0
sig_width1 = 0.5
sig_mean2 = 17.0
sig_width2 = 1.0

Nbkg = 900
bkglo1 = 0
bkghi1 = 10
bkglo2 = 12
bkghi2 = 21

# actually generating data and templates
data, sig_template, bkg_template = genD.sig_ngauss_bkg_flat([sig_mean1, sig_mean2],[sig_width1, sig_width2],Nsig,[bkglo1, bkglo2],[bkghi1,bkghi2],Nbkg,Ntemplates,verbose=False,scale=True)

################################################################################
# Generate the function densities wtih Garrett's code in 
# Lindsay's module.
################################################################################

# arguments for Garrett's numInRange function
d_radius = 0.1
Ndata = len(data[0])

# generating the signal and background function densities
sigFD = genFD.numInRange(data, sig_template,d_radius,Ndata)
bkgFD = genFD.numInRange(data, bkg_template, d_radius,Ndata)

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
    probs = tot_prob(data,[sigFD,bkgFD],frac)

    probs = probs[probs!=0]
        
    nll = (-np.log(probs)).sum()
    
    return nll
    
################################################################################
# Finding frac with Minuit
################################################################################

m = Minuit(negative_log_likelihood,frac=0.9,limit_frac=(0,1.0), \
                                   errordef = 0.5 )

m.migrad()
m.hesse()
print 'fval', m.fval
print m.get_fmin()
values = m.values
print values
errors = m.errors
print errors

plt.figure()
H, xedges, yedges = np.histogram2d(data[1],data[0],bins=25)
im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
