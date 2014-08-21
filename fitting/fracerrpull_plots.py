import numpy as np
import matplotlib.pylab as plt

infilename = "output_of_tests_1k.dat"
infile = open(infilename)

fracs =[]
errors = []
for line in infile:
    vals = line.split()
    fracs.append(float(vals[2]))
    errors.append(float(vals[3]))

print errors
print fracs

errors = np.array(errors)
fracs = np.array(fracs)


def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-((x-mean)**2)/(2*(width**2)))

    return y


Nfrac = len(fracs)
frac_mean = .9
frac_width = 0.0138

pulls = (fracs-frac_mean)/errors

plt.figure()
nbins = 25
lo = 0.87
hi = 0.93
x = np.linspace(lo,hi,Nfrac)
fracGauss = Gaussian(x,frac_mean,frac_width)
binwidth = (hi-lo)/float(nbins)
plt.hist(fracs,bins=nbins,range=(lo,hi))

fracGauss *= len(fracs)*binwidth
plt.plot(x,fracGauss)


Nerr = len(errors)
err_mean = .01
err_width = 0.05

plt.figure()
x = np.linspace(min(errors),max(errors),1000)
errGauss = np.random.normal(err_mean,err_width,Nerr)
plt.hist(errors,bins=25)



plt.figure()
pullhi = 4
pulllo = -4
Npull = len(pulls)
pullbinwidth = (pullhi-pulllo)/float(nbins)

x = np.linspace(pulllo,pullhi,Npull)
pullGauss = Gaussian(x,0,1)
pullGauss *= len(pulls)*pullbinwidth
plt.plot(x,pullGauss)
plt.hist(pulls,bins=nbins,range=(pulllo,pullhi))


plt.show()
