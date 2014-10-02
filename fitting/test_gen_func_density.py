#import gen_radius_density as genRD
import gen_function_density as genRD
import gen_data_and_templates as genD
import scipy.stats as stats
import numpy as np

np.random.seed(1)

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

Ntemplates = 10000

true_frac = Nbkg/float(Nsig+Nbkg)

# actually generating data and templates
data, sig_template, bkg_template = genD.sig_ngauss_bkg_flat([sig_mean1, sig_mean2],[sig_width1, sig_width2],Nsig,[bkglo1, bkglo2],[bkghi1,bkghi2],Nbkg,Ntemplates,verbose=False,scale=True)

k=20
radius = 0.01

presigFD = genRD.numInRange(data, sig_template,radius,len(data[0]))
print presigFD[0:10]
#prebkgFD = genRD.numInRange(data, bkg_template,radius,len(data[0]))
#print prebkgFD[0:10]


