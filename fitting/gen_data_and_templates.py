import numpy as np
import matplotlib.pylab as plt
import math as math

################################################################################
# Generate your fake data!
################################################################################
def sig_ngauss_bkg_flat(sigmean,sigwidth,Nsig,bkglo,bkghi,Nbkg,Ntemplates,verbose=False,scale=False):

    data = []
    sig_template = []
    bkg_template = []

    if verbose:
        print "Generating the fake experimental data!"

    # Here's your signal data!
    # Each time through the loop is a different variable (ie, x,y,z)
    for m,w,lo,hi in zip(sigmean,sigwidth,bkglo,bkghi):
        sig = np.random.normal(m,w,Nsig)
        bkg = lo+((hi-lo)*np.random.random(Nbkg))
        d = sig.copy()
        d = np.append(d,bkg.copy())
        data.append(d)

    if verbose:
        print "Generating the templates!"

    # Here's your signal template!
    # Each time through the loop is a different variable (ie, x,y,z)
    for m,w in zip(sigmean,sigwidth):
        sig = np.random.normal(m,w,Ntemplates)
        sig_template.append(sig)

    # Here's your background template!
    # Each time through the loop is a different variable (ie, x,y,z)
    for lo,hi in zip(bkglo,bkghi):
        bkg = lo+((hi-lo)*np.random.random(Nbkg))
        bkg_template.append(bkg)

    
    return data,sig_template,bkg_template
