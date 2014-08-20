import numpy as np
import matplotlib.pylab as plt

#infilename = "output_of_tests.dat"
#infilename = "output_1k.dat"
#infilename = "output_1k_both_pois.dat"
#infilename = "output_1k_both_pois_minos.dat"
#infilename = "output_1k_nn_0.dat"
infilename = "output_1k_nn_0_r01.dat"
infile = open(infilename)

fracs =[]
errors = []
true_fracs =[]
for line in infile:
    vals = line.split()
    fracs.append(float(vals[2]))
    errors.append(float(vals[3]))
    true_fracs.append(float(vals[6]))

print errors[0:10]
print fracs[0:10]
print true_fracs[0:10]

errors = np.array(errors)
fracs = np.array(fracs)
true_fracs = np.array(true_fracs)

def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-((x-mean)**2)/(2*(width**2)))

    return y


Nfrac = len(fracs)
frac_mean = .9
frac_width = 0.01377

print "Fracs:"
print np.mean(fracs)
print np.std(fracs)

print "Errors:"
print np.mean(errors)
print np.std(errors)

#pulls = abs(fracs-true_fracs)/(errors/1.4)
#pulls = abs(fracs-0.9)/(errors/1.0)
pulls = (fracs-0.9)/(errors/1.0)

print "Pulls:"
print np.mean(pulls)
print np.std(pulls)

plt.figure()
nbins = 50
lo = 0.85
hi = 0.95
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
plt.hist(np.abs(errors),bins=nbins)



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
