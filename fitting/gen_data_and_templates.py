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
        
    data = np.array(data)
    
    if verbose:
        print "Generating the templates!"

    # Here's your signal template!
    # Each time through the loop is a different variable (ie, x,y,z)
    for m,w in zip(sigmean,sigwidth):
        sig = np.random.normal(m,w,Ntemplates)
        sig_template.append(sig)

    sig_template = np.array(sig_template)
    
    # Here's your background template!
    # Each time through the loop is a different variable (ie, x,y,z)
    ### Changed line 39 Ntemplates was Nbkg and was making the bkg_template
    # only have 900 (Nbkg) points    
    for lo,hi in zip(bkglo,bkghi):
        bkg = lo+((hi-lo)*np.random.random(Ntemplates))
        bkg_template.append(bkg)

    bkg_template = np.array(bkg_template)
    
    if scale:
        minvalx = min(data[0])
        maxvalx = max(data[0])
        xwidth = maxvalx-minvalx

        minvaly = min(data[1])
        maxvaly = max(data[1])
        ywidth = maxvaly-minvaly 
        
        data[0] = (data[0] - minvalx)/xwidth
        data[1] = (data[1] - minvaly)/ywidth        
        
        sig_template[0] = (sig_template[0] - minvalx)/xwidth
        sig_template[1] = (sig_template[1] - minvaly)/ywidth
    
        bkg_template[0] = (bkg_template[0] - minvalx)/xwidth
        bkg_template[1] = (bkg_template[1] - minvaly)/ywidth
    
    return data,sig_template,bkg_template
