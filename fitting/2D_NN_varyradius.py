# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:32:20 2014

@author: Lindsay
"""

import gen_data_and_templates as genD
import gen_radius_density as genRD
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
import math as math
from iminuit import Minuit
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys

################################################################################
# Generate the templates and data from Bellis' module.
################################################################################

# arguments to put into the generating function

Nsig = 100
if len(sys.argv)>=2:
    Nsig = int(sys.argv[1])

Nsig = stats.poisson(Nsig).rvs(1)[0]
sig_mean1 = 5.0
sig_width1 = 0.5
sig_mean2 = 15.0
sig_width2 = 1.0

Nbkg = 900
if len(sys.argv)>=3:
    Nbkg = int(sys.argv[2])
Nbkg = stats.poisson(Nbkg).rvs(1)[0]
bkglo1 = 0
bkghi1 = 10
bkglo2 = 10
bkghi2 = 20

#Ntemplates = Nsig + Nbkg
#Ntemplates = 9900
Ntemplates = 100000

true_frac = Nbkg/float(Nsig+Nbkg)

# actually generating data and templates
data, sig_template, bkg_template = genD.sig_ngauss_bkg_flat([sig_mean1, sig_mean2],[sig_width1, sig_width2],Nsig,[bkglo1, bkglo2],[bkghi1,bkghi2],Nbkg,Ntemplates,verbose=False,scale=True)

################################################################################
# Generate the function densities wtih Garrett's code in 
# Lindsay's module.
################################################################################

# arguments for Lindsay's radiusToK function
k = 20

Ndata = len(data[0])
unSphere = math.pi

# generating the signal and background function densities

presigFD = genRD.radiusToK(data, sig_template,k)
prebkgFD = genRD.radiusToK(data, bkg_template,k)
#sigFD = np.zeros(Ndata)
#bkgFD = np.zeros(Ndata)
sigFD = k/(Ntemplates*unSphere*(presigFD**2))
bkgFD = k/(Ntemplates*unSphere*(prebkgFD**2))
'''
for i,R in enumerate(presigFD):
    sigFD[i] = k/(Ntemplates*unSphere*(R**2))
for i,R in enumerate(prebkgFD):
    bkgFD[i] = k/(Ntemplates*unSphere*(R**2))
'''    

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
#minimization
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

#final fit values
values = m.values
print values

#final fit errors
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


H, xedges, yedges = np.histogram2d(data[1],data[0],bins=25)
im = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

fig_template = plt.figure(figsize=(15,6))
fig_template.add_subplot(1,2,1)
plt.xlabel('Signal Dataset #1')
plt.ylabel('Signal Dataset #2')
plt.title('2D Signal Frequency', fontsize=14)
H, xedges, yedges = np.histogram2d(sig_template[1],sig_template[0],bins=50,range=[[0,1],[0,1]])
imsig = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
imsig = plt.imshow(H, interpolation='nearest', origin='low', extent=[0, 1.0, 0.0, 1.0])
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Frequency')
#plt.ylim(0,10)
#plt.xlim(12,21)

fig_template.add_subplot(1,2,2)
plt.xlabel('Background Dataset #1')
plt.ylabel('Background Dataset #2')
plt.title('2D Background Frequency', fontsize=14)
H, xedges, yedges = np.histogram2d(bkg_template[1],bkg_template[0],bins=50,range=[[0,1],[0,1]])
imbkg = plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(cax=cax)
cbar.ax.set_ylabel('Frequency')
plt.savefig('2D_Histograms.png')
#plt.ylim(0,10)
#plt.xlim(12,21)

txt = 'Figure A: The location of a sphere in both radius density plots indidcates the location of a data point for the background and signal datasets.  \nThe radius of the sphere indicates the neighbor density of the particular datapoint.  Note how the radii of the background points are \ngenerally the same and the points are scattered throughout the plot in the signal plot the points are concentrated in the center and those \nclosest to the center have the largest radii indicated that they have the largest neighbor densities in the center of the plot.'
fig_template = plt.figure(figsize=(12,6))
fig_template.add_subplot(1,2,1)
plt.figtext(.5, -.01,txt, horizontalalignment='center',verticalalignment='center')
plt.scatter(data[0],data[1],s=sigFD*50,alpha=0.1)
plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel('Signal Dataset #1')
plt.ylabel('Signal Dataset #2')
plt.title('Signal Function Density', fontsize=14)
fig_template.add_subplot(1,2,2)
plt.scatter(data[0],data[1],s=bkgFD*50,alpha=0.1)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Background Dataset 1')
plt.ylabel('Background Dataset 2')
plt.title('Background Function Density', fontsize=14)
plt.savefig('bubble_plots.png')


txt = 'Figure B: The radius of this plot idicates how large a circle around a datapoint was needed to enclose 20 neighbors around a data point. \nNote how at the center of the signal plot there appears to be no spheres.  This is because the radii to enclose 20 neighbors at the center \nof the plot are so small it makes the shperes difficult to see.'   
fig_template = plt.figure(figsize=(12,6))
fig_template.add_subplot(1,2,1)
plt.figtext(.5, 0,txt, horizontalalignment='center',verticalalignment='center')
plt.xlabel('Signal Dataset #1')
plt.ylabel('Signal Dataset #2')
plt.title('Signal Radius Density', fontsize=14)
plt.scatter(data[0],data[1],s=presigFD,alpha=0.9)
plt.xlim(0,1)
plt.ylim(0,1)
fig_template.add_subplot(1,2,2)
plt.xlabel('Signal Dataset #1')
plt.ylabel('Signal Dataset #2')
plt.title('Background Radius Density', fontsize=14)
plt.scatter(data[0],data[1],s=prebkgFD,alpha=0.9)
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('inverse_bubble_plots.png')



#xpts = np.random.random(3)
#ypts = np.random.random(3)
#size = [1,20,30]
#print xpts
#print ypts
#print size
#plt.scatter(xpts,ypts,s=size)


#print 'SIGFD:'
#print sigFD
#print 'BKGFD:'
#print bkgFD
#print 'PRESIGFD:'
#print presigFD
#print 'PREBKGFD:'
#print prebkgFD

#print min(data[0]),max(data[0])
#print min(data[1]),max(data[1])
#print min(sig_template[0]),max(sig_template[0])
#print min(sig_template[1]),max(sig_template[1])
#print min(bkg_template[0]),max(bkg_template[0])
#print min(bkg_template[1]),max(bkg_template[1])

#plt.show()
