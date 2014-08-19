import numpy as np
import matplotlib.pylab as plt

infilename = "output_of_tests.dat"
infile = open(infilename)

fracs =[]
errors = []
for line in infile:
    #print line
    vals = line.split()
    fracs.append(float(vals[2]))
    errors.append(float(vals[3]))


#print fracs
#print errors

Nfrac = len(fracs)
frac_mean = .9
frac_width = 0.05
fracGauss = np.random.normal(frac_mean,frac_width,Nfrac)

plt.figure()
plt.hist(fracs,bins=25)
plt.figure()
plt.hist(errors,bins=25)
plt.show()
#fracGauss = np.random.normal(sig_mean,sig_width,Nsig)
#uncertGauss = np.random.normal(sig_mean,sig_width,Nsig)
